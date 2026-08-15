"""Headless, config-driven NumPyro MCMC runner for the astrophysical GWB.

This is the SLURM-friendly port of ``notebooks/mcmc.py``: it importance-reweights a
fixed polarization-power catalog through NUTS to infer cosmological / population
hyperparameters, reading every setting from a TOML or JSON config file and emitting
only ``logging`` progress (no plots). It saves an ArviZ ``InferenceData`` NetCDF plus
a JSON run-config record, exactly like the notebook.

Design constraint (do not "tidy" away): config parsing lives in
``astrogwb_paper.config.mcmc`` (stdlib + pydantic only). ``OMP_NUM_THREADS`` /
``XLA_FLAGS`` and ``numpyro.set_host_device_count(...)`` must be set *before* JAX
initializes its backend, so the heavy imports (jax, numpyro, astrogwb, gwmock_pop)
happen inside functions that run only after
:func:`astrogwb_paper.runtime.configure_runtime`. See that function for the ordering.

Usage::

    uv run astrogwb-run-mcmc \
        --config packages/astrogwb-paper/configs/mcmc.example.toml \
        --catalog outputs/catalogs/bns-n16384-df1.h5

Use ``uv run --package astrogwb-paper --extra cuda`` (or ``--extra tpu``) for
the matching JAX accelerator plugin.
Batch runs are dispatched by the Snakemake ``run_mcmc`` rule (one config per
job); see the paper project's ``workflow/mcmc.smk`` and SLURM profiles.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrogwb_paper.config.hashing import file_sha256
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.config.mcmc import RunConfig, build_run_config, config_sha256
from astrogwb_paper.runtime import add_runtime_arguments, configure_runtime

if TYPE_CHECKING:
    from astrogwb_paper.amplitude import AmplitudeMarginalization

logger = logging.getLogger("run_mcmc")


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
# Inference
# --------------------------------------------------------------------------- #
def run(config: RunConfig, catalog_path: Path, jax, chain_method: str):
    """Replicate the notebook inference cells headlessly and return the MCMC object.

    Returns ``(mcmc, marginalization)``, where ``marginalization`` is the
    :class:`~astrogwb_paper.amplitude.AmplitudeMarginalization` built for an
    amplitude-marginalized run, or ``None`` for the default likelihood.
    """
    from functools import partial

    import jax.numpy as jnp
    from astrogwb.detector import effective_psd, load_sensitivity_map
    from astrogwb.frequency import (
        apply_frequency_mask,
    )
    from astrogwb.frequency import (
        frequency_mask as make_frequency_mask,
    )
    from astrogwb.gwb import spectral_density
    from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
        compute_merger_rate_distance_and_logprob,
        make_merger_rate_and_log_weights_fn,
    )
    from astrogwb.sampling.models import (
        amplitude_marginalized_model,
        spectral_density_model,
    )
    from astrogwb.waveform import polarization_power as compute_polarization_power
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.initialization import init_to_value
    from pluscross import load_catalog

    from astrogwb_paper.amplitude import build_amplitude_marginalization
    from astrogwb_paper.catalog import apply_gw_distance_at_fiducial
    from astrogwb_paper.priors import build_prior

    analysis = config.analysis
    cosmo = config.cosmology

    # --- Load proposal catalog ------------------------------------------------
    # Rescale the catalog to live-GW distances at the fiducial modified-
    # propagation parameters before reducing to polarization power, so the
    # importance weights below carry only the EM-distance ratio.
    catalog = load_catalog(catalog_path)
    catalog = apply_gw_distance_at_fiducial(
        catalog,
        xi_0=float(config.fiducials["xi_0"]),
        xi_n=float(config.fiducials["xi_n"]),
    )
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
    z_grid = jnp.linspace(cosmo.z_min, cosmo.z_max, cosmo.n_grid)
    _, _, proposal_logprob = compute_merger_rate_distance_and_logprob(
        config.fiducials, samples, redshift_grid=z_grid
    )

    merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
        fiducials=config.fiducials,
        redshift_grid=z_grid,
        proposal_logprob=proposal_logprob,
    )

    # --- Inject the fiducial spectrum as observed data (no plot) -------------
    rate0, log_weights0 = merger_rate_and_log_weights_fn(config.fiducials, samples)
    weights0 = jnp.exp(log_weights0)
    observed_spectral_density = spectral_density(
        polarization_power,
        weights0,
        rate0,
        average_mode="analytic_inclination",
    )
    logger.info("Injected fiducial spectrum as observed data (rate0=%.4e /s)", rate0)

    (
        frequencies,
        polarization_power,
        observed_spectral_density,
        effective_psd_arr,
    ) = apply_frequency_mask(
        freq_mask,
        frequencies,
        polarization_power,
        observed_spectral_density,
        effective_psd_arr,
    )

    # --- Build the model and sampler -----------------------------------------
    priors = {name: build_prior(spec) for name, spec in config.priors.items()}
    marginalization = None
    if analysis.likelihood == "amplitude_marginalized":
        assert analysis.amplitude_parameter is not None
        marginalization = build_amplitude_marginalization(config)
        model = partial(
            amplitude_marginalized_model,
            observation_time=config.observation_time,
            average_mode="analytic_inclination",
            merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
            amplitude_parameter=analysis.amplitude_parameter,
            fiducials=config.fiducials,
            amplitude_fn=marginalization.amplitude_fn,
            amplitude_prior=marginalization.prior,
            amplitude_grid=marginalization.grid,
            priors=priors,
            constants=config.constants,
        )
    else:
        model = partial(
            spectral_density_model,
            observation_time=config.observation_time,
            average_mode="analytic_inclination",
            merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
            priors=priors,
            constants=config.constants,
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
    return mcmc, marginalization


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
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
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
        timestamp = timestamp or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
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
    record: dict[str, Any] = {
        "catalog_path": str(catalog_path),
        "catalog_sha256": catalog_sha256,
        "config_sha256": config_sha256(config),
        "detectors": list(config.analysis.detectors),
        "seed": config.seed,
        "observation_time": config.observation_time,
        "likelihood": config.analysis.likelihood,
        # `sampled_params` is the sampler's latents; `posterior_params` is what
        # the saved chain actually carries. They differ by the reconstructed
        # amplitude parameter under a marginalized likelihood.
        "sampled_params": list(config.sampled_params),
        "posterior_params": list(config.posterior_params),
        "fiducials": config.fiducials,
        "constants": config.constants,
        "priors": {
            name: spec.model_dump(mode="json") for name, spec in config.priors.items()
        },
        "cosmology": config.cosmology.model_dump(mode="json"),
        "band": {
            "f_min": config.analysis.f_min,
            "f_max": config.analysis.f_max,
        },
        "sampler": config.sampler.model_dump(mode="json"),
        "git_revision": _git_revision(),
        "timestamp": timestamp,
    }
    if config.analysis.likelihood == "amplitude_marginalized":
        record["amplitude_parameter"] = config.analysis.amplitude_parameter
        record["amplitude_prior"] = (
            config.amplitude_prior.model_dump(mode="json")
            if config.amplitude_prior is not None
            else None
        )
        record["amplitude_num_nodes"] = config.analysis.amplitude_num_nodes
        record["amplitude_prior_span_sigma"] = (
            config.analysis.amplitude_prior_span_sigma
        )
    return record


def save(
    mcmc,
    config: RunConfig,
    *,
    catalog_path: Path,
    timestamp: str | None = None,
    force: bool = False,
    catalog_sha256: str | None = None,
    marginalization: AmplitudeMarginalization | None = None,
) -> Path:
    """Write the ArviZ NetCDF + JSON run record, and log the IS health check."""
    from functools import partial

    import arviz as az
    import numpy as np
    import xarray as xr

    config.outdir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    nc_path, json_path = ensure_output_paths_available(
        config, timestamp=timestamp, force=force
    )

    idata = az.from_numpyro(mcmc)

    if marginalization is not None:
        import jax
        from astrogwb.sampling.models import amplitude_reconstruction_model
        from numpyro.infer import Predictive

        amplitude_parameter = marginalization.parameter

        # Reconstruction runs here, in-process, against the very objects the
        # chain was marginalized with -- which is why nothing about the
        # quadrature needs persisting to the NetCDF.
        # The sufficient statistics form the AmplitudeConditional batch shape;
        # one Predictive invocation draws one amplitude per (chain, draw).
        posterior_samples = mcmc.get_samples(group_by_chain=True)
        draws = Predictive(
            partial(
                amplitude_reconstruction_model,
                amplitude_parameter=amplitude_parameter,
                amplitude_fn=marginalization.amplitude_fn,
                merger_rate_amplitude_fn=marginalization.merger_rate_fn,
                prior=marginalization.prior,
                fiducial=marginalization.fiducial,
                grid=marginalization.grid,
            ),
            num_samples=1,
            return_sites=[
                amplitude_parameter,
                "total_merger_rate",
                "quadrature_effective_nodes",
            ],
        )(
            jax.random.fold_in(jax.random.PRNGKey(config.seed), 1),
            amplitude_mle=posterior_samples["amplitude_mle"],
            template_optimal_snr=posterior_samples["template_optimal_snr"],
            template_merger_rate=posterior_samples["template_merger_rate"],
        )
        draws = {name: values[0] for name, values in draws.items()}

        # `az.from_numpyro` returns an xarray DataTree, whose __setitem__ does
        # not accept a Dataset-style `(dims, values)` tuple: it would store the
        # tuple as an object scalar and fail at `to_netcdf`. Assign DataArrays.
        for name, values in draws.items():
            idata.posterior[name] = xr.DataArray(
                np.asarray(values), dims=("chain", "draw")
            )

        effective_nodes = draws["quadrature_effective_nodes"]

        min_effective_nodes = float(np.min(effective_nodes))
        if min_effective_nodes < 30:
            logger.warning(
                "quadrature_effective_nodes min=%.1f is below 30; the amplitude "
                "grid may not resolve the conditional posterior. Consider "
                "raising analysis.amplitude_num_nodes.",
                min_effective_nodes,
            )

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
    config_path = args.config.resolve()
    catalog_path = args.catalog.resolve()
    raw = load_mapping(config_path)
    configured_outdir = args.outdir
    if configured_outdir is None:
        configured_outdir = Path(raw.get("output", {}).get("outdir", "chains"))
    config = build_run_config(
        raw,
        seed=args.seed,
        outdir=configured_outdir.resolve(),
        label=args.label,
    )

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info("Config: %s", config_path)
    logger.info(
        "Sampling %s | fixed %s",
        tuple(config.sampled_params),
        tuple(config.constants),
    )

    # Pick a single timestamp now and check both artifacts before JAX starts.
    # Labelled experiment runs are deterministic; auto-labelled runs preserve the
    # existing timestamp convention.
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    ensure_output_paths_available(config, timestamp=timestamp, force=args.force)

    # Hash the catalog before JAX claims a device.
    catalog_sha256 = verify_catalog(catalog_path)
    logger.info("Catalog SHA-256: %s", catalog_sha256)

    jax, chain_method = configure_runtime(
        num_chains=config.sampler.num_chains,
        platform=args.platform,
        host_device_count=args.host_device_count,
        cpu_threads=args.cpu_threads,
        chain_method=args.chain_method,
    )
    mcmc, marginalization = run(config, catalog_path, jax, chain_method)
    save(
        mcmc,
        config,
        catalog_path=catalog_path,
        timestamp=timestamp,
        force=args.force,
        catalog_sha256=catalog_sha256,
        marginalization=marginalization,
    )


if __name__ == "__main__":
    main()
