"""Headless, config-driven NumPyro MCMC runner for the astrophysical GWB.

This is the SLURM-friendly port of ``notebooks/mcmc.py``: it importance-reweights a
fixed polarization-power catalog through NUTS to infer cosmological / population
hyperparameters, reading every setting from a TOML or JSON config file and emitting
only ``logging`` progress (no plots). It saves an ArviZ ``InferenceData`` NetCDF plus
a JSON run-config record, exactly like the notebook.

Design constraint (do not "tidy" away): config parsing lives in
``astrogwb.sampling.config`` (stdlib + pydantic only). ``OMP_NUM_THREADS`` /
``XLA_FLAGS`` and ``numpyro.set_host_device_count(...)`` must be set *before* JAX
initializes its backend, so the heavy imports (jax, numpyro, astrogwb, gwmock_pop)
happen inside functions that run only after :func:`configure_runtime`. See
``configure_runtime`` for the ordering.

Usage::

    uv run python scripts/run_mcmc.py --config configs/mcmc.example.toml
    uv run python scripts/run_mcmc.py --config configs/mcmc/sweep/ET-2L-aligned__H0.json

The script is meant to back a SLURM job array with one config per task; see
``scripts/submit_mcmc.sh -i configs/mcmc/sweep``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

from astrogwb.sampling.config import (
    RunConfig,
    RuntimeConfig,
    build_run_config,
    load_config,
)

logger = logging.getLogger("run_mcmc")

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Headless NumPyro MCMC runner for the astrophysical GWB. Reads all "
            "settings from a TOML or JSON config; saves an ArviZ NetCDF + JSON record."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML or JSON config file for this run / array task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the config seed (handy for array sweeps).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Override the config [output] outdir.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Override the config [output] label.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Log at WARNING instead of INFO.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Runtime / device configuration (MUST run before any JAX import elsewhere)
# --------------------------------------------------------------------------- #
def configure_runtime(rc: RuntimeConfig, num_chains: int):
    """Configure CPU threads, device platform, and host device count, then import jax.

    Returns the imported ``jax`` module and the resolved ``chain_method``. This is
    the only place allowed to set the env vars / host device count, and it must run
    before anything else triggers JAX backend initialization.
    """
    # 1. CPU thread limits (no-op when cpu_threads <= 0).
    if rc.cpu_threads and rc.cpu_threads > 0:
        n = str(rc.cpu_threads)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = n
        xla_flags = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = (
            f"{xla_flags} --xla_cpu_multi_thread_eigen=true "
            f"intra_op_parallelism_threads={rc.cpu_threads}"
        ).strip()

    # 2. Force a platform when requested; "auto" lets JAX pick (GPU if present).
    platform = rc.platform.lower()
    if platform in ("cpu", "gpu"):
        os.environ["JAX_PLATFORMS"] = platform

    # 3. Host device count for CPU parallel chains -- must precede jax init.
    host_device_count = rc.host_device_count or num_chains
    import numpyro

    numpyro.set_host_device_count(host_device_count)

    # 4. Now JAX may initialize. Enable x64 before any array is created.
    import jax

    jax.config.update("jax_enable_x64", True)

    devices = jax.devices()
    resolved_platform = devices[0].platform
    logger.info(
        "JAX x64=%s | platform=%s | devices=%s | host_device_count=%d",
        jax.config.read("jax_enable_x64"),
        resolved_platform,
        devices,
        host_device_count,
    )

    # 5. Resolve chain_method. On a single GPU vectorized chains are best; CPU
    #    chains go on separate host devices via "parallel".
    chain_method = rc.chain_method.lower()
    if chain_method == "auto":
        chain_method = "vectorized" if resolved_platform == "gpu" else "parallel"
    logger.info("chain_method=%s (num_chains=%d)", chain_method, num_chains)

    return jax, chain_method


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def run(config: RunConfig, jax, chain_method: str):
    """Replicate the notebook inference cells headlessly and return the MCMC object."""
    from functools import partial

    import jax.numpy as jnp
    from gwmock_pop.distributions.madau_dickinson import madau_dickinson_redshift_pdf
    from numpyro.infer import MCMC, NUTS

    from astrogwb.detector import effective_psd, load_sensitivity_map
    from astrogwb.gwb import frequency_mask as make_frequency_mask
    from astrogwb.gwb import spectral_density
    from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
        make_merger_rate_and_log_weights_fn,
    )
    from astrogwb.sampling.numpyro_model import numpyro_model
    from astrogwb.sampling.priors import build_prior
    from astrogwb.waveform import load_polarization_power_catalog

    cat = config.catalog
    cosmo = config.cosmology

    # --- Load proposal catalog ------------------------------------------------
    catalog = load_polarization_power_catalog(cat.path)
    frequencies = jnp.asarray(catalog.frequencies)
    polarization_power = jnp.asarray(catalog.polarization_power)
    samples = {name: jnp.asarray(v) for name, v in catalog.samples.items()}
    assert "redshift" in samples, (
        "catalog samples must include 'redshift' for the weights"
    )
    assert "luminosity_distance" in samples, (
        "catalog samples must include 'luminosity_distance' for the weights"
    )
    n_freq, n_samples = polarization_power.shape
    logger.info(
        "Loaded catalog %s: n_frequency_bins=%d n_proposal_samples=%d",
        cat.path,
        n_freq,
        n_samples,
    )

    # Importance weights divide by the proposal redshift PDF, which is normalized on
    # [z_min, z_max] and is exactly zero outside it. Any catalog sample beyond the
    # range gives log(0) = -inf weights (a cryptic "invalid loc" downstream), so guard
    # it here with a clear message instead.
    z_lo = float(jnp.min(samples["redshift"]))
    z_hi = float(jnp.max(samples["redshift"]))
    if z_lo < cosmo.z_min or z_hi > cosmo.z_max:
        raise ValueError(
            f"catalog redshifts span [{z_lo:.4g}, {z_hi:.4g}] but [z_min, z_max] is "
            f"[{cosmo.z_min:.4g}, {cosmo.z_max:.4g}]; widen the cosmology range to "
            "bracket all catalog samples (the proposal PDF is zero outside it)."
        )

    # --- Effective PSD and analysis band -------------------------------------
    sensitivities = load_sensitivity_map(cat.detectors)
    effective_psd_arr = jnp.asarray(
        effective_psd(frequencies, list(cat.detectors), sensitivities)
    )
    freq_mask = make_frequency_mask(frequencies, fmin=cat.f_min, fmax=cat.f_max)
    logger.info(
        "Analysis band: %d of %d bins (%.1f-%.1f Hz)",
        int(jnp.sum(freq_mask)),
        frequencies.shape[0],
        cat.f_min,
        cat.f_max,
    )

    # --- Precompute fiducial proposal log-density ----------------------------
    z_samples = jnp.asarray(samples["redshift"])
    log_p_proposal = jnp.log(
        madau_dickinson_redshift_pdf(
            z_samples,
            z_max=cosmo.z_max,
            z_min=cosmo.z_min,
            gamma=config.fiducials["gamma"],
            kappa=config.fiducials["kappa"],
            z_peak=config.fiducials["z_peak"],
            hubble_constant=config.fiducials["H0"],
            omega_m=config.fiducials["Omega_m"],
            n_grid=cosmo.n_grid,
        )
    )

    z_grid = jnp.linspace(cosmo.z_min, cosmo.z_max, cosmo.n_grid)
    merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
        z_grid=z_grid,
        proposal_log_pdf=log_p_proposal,
        local_merger_rate=config.local_merger_rate,
        fiducial_xi_0=config.fiducials["xi_0"],
        fiducial_xi_n=config.fiducials["xi_n"],
    )

    # --- Inject the fiducial spectrum as observed data (no plot) -------------
    ones_weights = jnp.ones((n_samples,))
    rate0, _ = merger_rate_and_log_weights_fn(config.fiducials, samples)
    observed_spectral_density = spectral_density(
        polarization_power,
        ones_weights,
        rate0,
        average_mode="analytic_inclination",
    )
    logger.info("Injected fiducial spectrum as observed data (rate0=%.4e /s)", rate0)

    # --- Build the model and sampler -----------------------------------------
    priors = {name: build_prior(spec) for name, spec in config.priors.items()}
    model = partial(
        numpyro_model,
        observation_time=config.observation_time,
        average_mode="analytic_inclination",
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
        priors=priors,
        constants=config.constants,
        frequency_mask=freq_mask,
    )

    sampler = config.sampler
    kernel = NUTS(
        model,
        target_accept_prob=sampler.target_accept,
        forward_mode_differentiation=sampler.forward_mode_differentiation,
        dense_mass=True,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=sampler.num_warmup,
        num_samples=sampler.num_samples,
        num_chains=sampler.num_chains,
        chain_method=chain_method,
        progress_bar=sampler.progress_bar,
        jit_model_args=sampler.jit_model_args,
    )

    logger.info(
        "Sampling: warmup=%d samples=%d chains=%d target_accept=%.3f forward_mode=%s",
        sampler.num_warmup,
        sampler.num_samples,
        sampler.num_chains,
        sampler.target_accept,
        sampler.forward_mode_differentiation,
    )
    rng_key = jax.random.PRNGKey(config.seed)
    mcmc.run(
        rng_key,
        frequencies=frequencies,
        polarization_power=polarization_power,
        samples=samples,
        observed_spectral_density=observed_spectral_density,
        effective_psd=effective_psd_arr,
        extra_fields=("num_steps", "accept_prob", "diverging"),
    )

    # numpyro prints the summary to stdout; route it through logging for SLURM logs.
    logger.info("Sampling complete; summary follows")
    mcmc.print_summary()
    return mcmc


def _git_revision() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def save(mcmc, config: RunConfig) -> Path:
    """Write the ArviZ NetCDF + JSON run record, and log the IS health check."""
    import arviz as az

    config.outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    params_suffix = "-".join(config.sampled_params)
    det_suffix = ",".join(config.catalog.detectors)
    base = config.label or (
        f"mcmc-{params_suffix}-det={det_suffix}-seed{config.seed}-{timestamp}"
    )

    idata = az.from_numpyro(mcmc)
    nc_path = config.outdir / f"{base}.nc"
    idata.to_netcdf(nc_path)

    run_record = {
        "catalog_path": str(config.catalog.path),
        "detectors": list(config.catalog.detectors),
        "seed": config.seed,
        "observation_time": config.observation_time,
        "local_merger_rate": config.local_merger_rate,
        "sampled_params": list(config.sampled_params),
        "fiducials": config.fiducials,
        "constants": config.constants,
        "priors": config.priors,
        "cosmology": config.cosmology.model_dump(mode="json"),
        "band": {"f_min": config.catalog.f_min, "f_max": config.catalog.f_max},
        "sampler": config.sampler.model_dump(mode="json"),
        "runtime": config.runtime.model_dump(mode="json"),
        "git_revision": _git_revision(),
        "timestamp": timestamp,
    }
    json_path = config.outdir / f"{base}.json"
    json_path.write_text(json.dumps(run_record, indent=2, default=str))

    # Importance-sampling health: relative ESS near 1 means the proposal catalog
    # still reweights well at the posterior.
    post = idata.posterior
    ress = post["importance_relative_ess"].values.ravel()
    rate = post["total_merger_rate"].values.ravel()
    logger.info("importance_relative_ess: mean=%.3f min=%.3f", ress.mean(), ress.min())
    logger.info("total_merger_rate [/s]: mean=%.4e", rate.mean())
    logger.info("Saved %s and %s", nc_path, json_path)
    return nc_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw = load_config(args.config)
    config = build_run_config(
        raw,
        seed=args.seed,
        outdir=args.outdir,
        label=args.label,
    )

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info("Config: %s", args.config)
    logger.info(
        "Sampling %s | fixed %s",
        tuple(config.sampled_params),
        tuple(config.constants),
    )

    jax, chain_method = configure_runtime(config.runtime, config.sampler.num_chains)
    mcmc = run(config, jax, chain_method)
    save(mcmc, config)


if __name__ == "__main__":
    main()
