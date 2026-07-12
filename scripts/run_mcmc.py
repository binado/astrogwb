"""Headless, config-driven NumPyro MCMC runner for the astrophysical GWB.

This is the SLURM-friendly port of ``notebooks/mcmc.py``: it importance-reweights a
fixed polarization-power catalog through NUTS to infer cosmological / population
hyperparameters, reading every setting from a TOML or JSON config file and emitting
only ``logging`` progress (no plots). It saves an ArviZ ``InferenceData`` NetCDF plus
a JSON run-config record, exactly like the notebook.

Design constraint (do not "tidy" away): config parsing lives in
``astrogwb.config.mcmc`` (stdlib + pydantic only). ``OMP_NUM_THREADS`` /
``XLA_FLAGS`` and ``numpyro.set_host_device_count(...)`` must be set *before* JAX
initializes its backend, so the heavy imports (jax, numpyro, astrogwb, gwmock_pop)
happen inside functions that run only after :func:`configure_runtime`. See
``configure_runtime`` for the ordering.

Usage::

    uv run --extra mcmc python scripts/run_mcmc.py \
        --config configs/mcmc.example.toml --catalog out/catalogs/bns-n16384-df1.h5

Requires the ``mcmc`` optional extra (pydantic plus ArviZ NetCDF output support)
for ``RunConfig`` validation and chain serialization.
Batch runs are dispatched by the Snakemake ``run_mcmc`` rule (one config per
job); see ``workflow/mcmc.smk`` and ``profiles/slurm/config.yaml``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from astrogwb.config.hashing import file_sha256
from astrogwb.config.loading import load_mapping
from astrogwb.config.mcmc import RunConfig, build_run_config, config_sha256

logger = logging.getLogger("run_mcmc")

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def positive_int(value: str) -> int:
    """Parse a strictly positive integer for runtime CLI controls."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add deployment/runtime controls shared by runner entrypoints."""
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="JAX platform (default: auto; gpu selects the CUDA backend).",
    )
    parser.add_argument(
        "--host-device-count",
        type=positive_int,
        default=None,
        help="Logical host devices (default: sampler.num_chains).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=positive_int,
        default=None,
        help="Cap OMP/BLAS/XLA CPU threads; omitted leaves inherited settings.",
    )
    parser.add_argument(
        "--chain-method",
        choices=("auto", "parallel", "sequential", "vectorized"),
        default="auto",
        help="NumPyro chain method (default: auto).",
    )


@dataclass(frozen=True)
class RuntimeOptions:
    """Runtime choices made at the CLI boundary, separate from scientific config."""

    platform: str = "auto"
    host_device_count: int | None = None
    cpu_threads: int | None = None
    chain_method: str = "auto"


def runtime_options_from_args(args: argparse.Namespace) -> RuntimeOptions:
    return RuntimeOptions(
        platform=args.platform,
        host_device_count=args.host_device_count,
        cpu_threads=args.cpu_threads,
        chain_method=args.chain_method,
    )


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
        "--catalog",
        type=Path,
        required=True,
        help="Waveform catalog to reweight for this run.",
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing NetCDF chain and/or JSON sidecar for this run.",
    )
    add_runtime_arguments(parser)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Runtime / device configuration (MUST run before any JAX import elsewhere)
# --------------------------------------------------------------------------- #
def configure_runtime(options: RuntimeOptions, num_chains: int) -> tuple:
    """Configure CPU threads, device platform, and host device count, then import jax.

    Returns the imported ``jax`` module and the resolved chain method. This is the
    only place allowed to set the env vars / host device count, and it must run
    before anything else triggers JAX backend initialization.
    """
    # 1. CPU thread limits (no-op when omitted).
    if options.cpu_threads is not None:
        n = str(options.cpu_threads)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = n
        xla_flags = os.environ.get("XLA_FLAGS", "")
        xla_flags = re.sub(
            r"(?<!\S)(?:--)?intra_op_parallelism_threads(?:=\S+|\s+\S+)",
            "",
            xla_flags,
        )
        xla_flags = re.sub(
            r"(?<!\S)--xla_cpu_multi_thread_eigen=\S+",
            "",
            xla_flags,
        )
        updated_xla_flags = (
            f"{xla_flags} --xla_cpu_multi_thread_eigen=true "
            f"intra_op_parallelism_threads={options.cpu_threads}"
        )
        os.environ["XLA_FLAGS"] = " ".join(updated_xla_flags.split())

    # 2. Force a platform when requested; "auto" lets JAX pick (GPU if present).
    if options.platform == "auto":
        os.environ.pop("JAX_PLATFORMS", None)
    else:
        os.environ["JAX_PLATFORMS"] = (
            "cuda" if options.platform == "gpu" else options.platform
        )

    # 3. Host device count for CPU parallel chains -- must precede jax init.
    host_device_count = options.host_device_count or num_chains
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
    chain_method = options.chain_method
    if chain_method == "auto":
        chain_method = "vectorized" if resolved_platform == "gpu" else "parallel"
    logger.info("chain_method=%s (num_chains=%d)", chain_method, num_chains)

    return jax, chain_method


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def run(config: RunConfig, catalog_path: Path, jax, chain_method: str):
    """Replicate the notebook inference cells headlessly and return the MCMC object."""
    from functools import partial

    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.initialization import init_to_value

    from astrogwb.detector import effective_psd, load_sensitivity_map
    from astrogwb.gwb import frequency_mask as make_frequency_mask
    from astrogwb.gwb import spectral_density
    from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
        compute_proposal_logpdf,
        make_merger_rate_and_log_weights_fn,
    )
    from astrogwb.sampling.numpyro_model import numpyro_model
    from astrogwb.sampling.priors import build_prior
    from astrogwb.waveform import polarization_power as compute_polarization_power
    from pluscross import load_catalog

    analysis = config.analysis
    cosmo = config.cosmology

    # --- Load proposal catalog ------------------------------------------------
    catalog = load_catalog(catalog_path)
    frequencies = jnp.asarray(catalog.frequencies)
    polarization_power = jnp.asarray(compute_polarization_power(catalog))
    samples = {name: jnp.asarray(v) for name, v in catalog.source_parameters.items()}
    del catalog
    assert "redshift" in samples, (
        "catalog samples must include 'redshift' for the weights"
    )
    assert "luminosity_distance" in samples, (
        "catalog samples must include 'luminosity_distance' for the weights"
    )
    n_freq, n_samples = polarization_power.shape
    logger.info(
        "Loaded catalog %s: n_frequency_bins=%d n_proposal_samples=%d",
        catalog_path,
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
    sensitivities = load_sensitivity_map(analysis.detectors)
    effective_psd_arr = jnp.asarray(
        effective_psd(frequencies, list(analysis.detectors), sensitivities)
    )
    freq_mask = make_frequency_mask(
        frequencies, fmin=analysis.f_min, fmax=analysis.f_max
    )
    logger.info(
        "Analysis band: %d of %d bins (%.1f-%.1f Hz)",
        int(jnp.sum(freq_mask)),
        frequencies.shape[0],
        analysis.f_min,
        analysis.f_max,
    )

    # --- Precompute fiducial proposal log-density ----------------------------
    z_samples = jnp.asarray(samples["redshift"])
    z_grid = jnp.linspace(cosmo.z_min, cosmo.z_max, cosmo.n_grid)
    log_p_proposal = compute_proposal_logpdf(
        z_samples, z_grid=z_grid, fiducials=config.fiducials
    )

    merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
        z_grid=z_grid,
        proposal_log_pdf=log_p_proposal,
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
    init_strategy = init_to_value(
        values={name: config.fiducials[name] for name in config.sampled_params}
    )
    kernel = NUTS(
        model,
        target_accept_prob=sampler.target_accept,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        forward_mode_differentiation=sampler.forward_mode_differentiation,
        dense_mass=sampler.dense_mass,
        max_tree_depth=sampler.max_tree_depth,
        init_strategy=init_strategy,
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


def verify_catalog(path: Path) -> str:
    """Hash a catalog before JAX starts, returning its content digest."""
    if not path.is_file():
        raise FileNotFoundError(f"catalog not found: {path}")
    return file_sha256(path)


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


def output_paths(
    config: RunConfig, *, timestamp: str | None = None
) -> tuple[Path, Path]:
    """Return the chain and sidecar paths a run will write.

    Campaign configs always have a label.  Auto-labelled ad-hoc runs retain the
    timestamped naming convention, while still allowing a collision check before
    expensive sampling starts.
    """
    if config.label:
        base = config.label
    else:
        timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
        params_suffix = "-".join(config.sampled_params)
        det_suffix = ",".join(config.analysis.detectors)
        base = f"mcmc-{params_suffix}-det={det_suffix}-seed{config.seed}-{timestamp}"
    return config.outdir / f"{base}.nc", config.outdir / f"{base}.json"


def ensure_output_paths_available(
    config: RunConfig, *, timestamp: str | None = None, force: bool = False
) -> tuple[Path, Path]:
    """Fail before sampling if either output artifact already exists."""
    nc_path, json_path = output_paths(config, timestamp=timestamp)
    existing = [path for path in (nc_path, json_path) if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to replace existing MCMC output(s): {names}. "
            "Pass --force only for an intentional replacement."
        )
    return nc_path, json_path


def build_run_record(
    config: RunConfig,
    *,
    catalog_path: Path,
    timestamp: str,
    catalog_sha256: str | None,
) -> dict:
    """Assemble the JSON sidecar recording the run's inputs and provenance."""
    return {
        "catalog_path": str(catalog_path),
        "catalog_sha256": catalog_sha256,
        "config_sha256": config_sha256(config),
        "detectors": list(config.analysis.detectors),
        "seed": config.seed,
        "observation_time": config.observation_time,
        "sampled_params": list(config.sampled_params),
        "fiducials": config.fiducials,
        "constants": config.constants,
        "priors": config.priors,
        "cosmology": config.cosmology.model_dump(mode="json"),
        "band": {
            "f_min": config.analysis.f_min,
            "f_max": config.analysis.f_max,
        },
        "sampler": config.sampler.model_dump(mode="json"),
        "git_revision": _git_revision(),
        "timestamp": timestamp,
    }


def save(
    mcmc,
    config: RunConfig,
    *,
    catalog_path: Path,
    timestamp: str | None = None,
    force: bool = False,
    catalog_sha256: str | None = None,
) -> Path:
    """Write the ArviZ NetCDF + JSON run record, and log the IS health check."""
    import arviz as az

    config.outdir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    nc_path, json_path = ensure_output_paths_available(
        config, timestamp=timestamp, force=force
    )

    idata = az.from_numpyro(mcmc)
    idata.to_netcdf(nc_path)

    run_record = build_run_record(
        config,
        catalog_path=catalog_path,
        timestamp=timestamp,
        catalog_sha256=catalog_sha256,
    )
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
    raw = load_mapping(args.config)
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

    # Pick a single timestamp now and check both artifacts before JAX starts.
    # Labelled campaign runs are deterministic; auto-labelled runs preserve the
    # existing timestamp convention.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ensure_output_paths_available(config, timestamp=timestamp, force=args.force)

    # Hash the catalog before JAX claims a device.
    catalog_sha256 = verify_catalog(args.catalog)
    logger.info("Catalog SHA-256: %s", catalog_sha256)

    runtime_options = runtime_options_from_args(args)
    jax, chain_method = configure_runtime(runtime_options, config.sampler.num_chains)
    mcmc = run(config, args.catalog, jax, chain_method)
    save(
        mcmc,
        config,
        catalog_path=args.catalog,
        timestamp=timestamp,
        force=args.force,
        catalog_sha256=catalog_sha256,
    )


if __name__ == "__main__":
    main()
