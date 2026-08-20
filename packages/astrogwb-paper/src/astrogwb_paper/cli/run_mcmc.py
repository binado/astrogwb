"""Headless, config-driven NumPyro MCMC runner for the astrophysical GWB.

This is the SLURM-friendly port of ``notebooks/mcmc.py``: it importance-reweights a
fixed polarization-power catalog through NUTS to infer cosmological / population
hyperparameters, reading every setting from a TOML or JSON config file and emitting
only ``logging`` progress (no plots). It saves an ArviZ ``InferenceData`` NetCDF;
the assembled config it was given is the record of the run's settings.

Design constraint (do not "tidy" away): config parsing lives in
``astrogwb_paper.config.mcmc``, which imports only stdlib + pydantic at module
load and materializes priors without evaluating any JAX op ("is the backend
still uninitialized?" is guarded by a subprocess test). ``OMP_NUM_THREADS`` /
``XLA_FLAGS`` and ``numpyro.set_host_device_count(...)`` must be set *before* JAX
initializes its backend, so the heavy imports (jax, astrogwb, gwmock_pop)
happen inside functions that run only after
:func:`astrogwb_paper.runtime.configure_runtime`. See that function for the ordering.

Usage::

    uv run astrogwb-run-mcmc \
        --config outputs/configs/cosmological-parameters/ET-2L-aligned-CE-Hanford.json \
        --injection-catalog outputs/catalogs/injection-bns-n32768-eps=0-df1.h5 \
        --proposal-catalog outputs/catalogs/bns-n16384-eps=0.1-df1.h5

Configs are assembled from the base and run overlays in ``inputs/experiments.yaml``
by the ``assemble_config`` workflow rule or by
``astrogwb-validate-config``; see docs/running-inference.md.

Use ``uv run --package astrogwb-paper --extra cuda`` (or ``--extra tpu``) for
the matching JAX accelerator plugin.
Batch runs are dispatched by the Snakemake ``run_mcmc`` rule (one config per
job); see the paper project's ``Snakefile`` and SLURM profiles.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.config.mcmc import RunConfig, build_run_config
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
            "settings from a TOML or JSON config; saves an ArviZ NetCDF."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML or JSON config file for this run / array task.",
    )
    parser.add_argument(
        "--injection-catalog",
        type=Path,
        required=True,
        help="Independent fiducial waveform catalog used to construct observed data.",
    )
    parser.add_argument(
        "--proposal-catalog",
        type=Path,
        required=True,
        help="Waveform catalog used by the importance estimator.",
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
        help="Replace an existing NetCDF chain for this run.",
    )
    add_runtime_arguments(parser)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def run(
    config: RunConfig,
    injection_catalog_path: Path,
    proposal_catalog_path: Path,
    jax,
    chain_method: str,
):
    """Replicate the notebook inference cells headlessly and return the MCMC object.

    Returns ``(mcmc, marginalization)``, where ``marginalization`` is the
    :class:`~astrogwb_paper.amplitude.AmplitudeMarginalization` built for an
    amplitude-marginalized run, or ``None`` for the default likelihood.
    """
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.initialization import init_to_value

    from astrogwb_paper.inference import (
        build_model,
        initial_values,
        prepare_inference_inputs,
    )

    inputs = prepare_inference_inputs(
        injection_catalog_path,
        proposal_catalog_path,
        fiducials=config.fiducials,
        proposal_config=config.proposal,
        grid=config.analysis_grid,
        detectors=config.analysis.detectors,
    )
    model, marginalization = build_model(
        config,
        merger_rate_and_log_weights_fn=inputs.merger_rate_and_log_weights_fn,
    )

    sampler = config.sampler
    init_strategy = init_to_value(values=initial_values(config))
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
        **inputs.masked_model_kwargs(),
        extra_fields=("num_steps", "accept_prob", "diverging"),
    )

    # numpyro prints the summary to stdout; route it through logging for SLURM logs.
    logger.info("Sampling complete; summary follows")
    mcmc.print_summary()
    return mcmc, marginalization


def chain_output_path(config: RunConfig, *, timestamp: str | None = None) -> Path:
    """Return the NetCDF chain path a run will write.

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
    return config.outdir / f"{base}.nc"


def ensure_chain_path_available(
    config: RunConfig, *, timestamp: str | None = None, force: bool = False
) -> Path:
    """Fail before sampling if the chain already exists."""
    nc_path = chain_output_path(config, timestamp=timestamp)
    if nc_path.exists() and not force:
        raise FileExistsError(
            f"refusing to replace existing MCMC output: {nc_path}. "
            "Pass --force only for an intentional replacement."
        )
    return nc_path


def save(
    mcmc,
    config: RunConfig,
    *,
    timestamp: str | None = None,
    force: bool = False,
    marginalization: AmplitudeMarginalization | None = None,
) -> Path:
    """Write the ArviZ NetCDF chain, and log the IS health check."""
    from functools import partial

    import arviz as az
    import numpy as np
    import xarray as xr

    config.outdir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    nc_path = ensure_chain_path_available(config, timestamp=timestamp, force=force)

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

    # Importance-sampling health: relative ESS near 1 means the proposal catalog
    # still reweights well at the posterior.
    post = idata.posterior
    ress = post["importance_relative_ess"].values.ravel()
    rate = post["total_merger_rate"].values.ravel()
    logger.info("importance_relative_ess: mean=%.3f min=%.3f", ress.mean(), ress.min())
    logger.info("total_merger_rate [/s]: mean=%.4e", rate.mean())
    logger.info("Saved %s", nc_path)
    return nc_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config.resolve()
    injection_catalog_path = args.injection_catalog.resolve()
    proposal_catalog_path = args.proposal_catalog.resolve()
    raw = load_mapping(config_path)
    config = build_run_config(
        raw,
        seed=args.seed,
        outdir=args.outdir.resolve() if args.outdir else None,
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

    # Pick a single timestamp now and check the artifact before JAX starts.
    # Labelled experiment runs are deterministic; auto-labelled runs preserve the
    # existing timestamp convention.
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    ensure_chain_path_available(config, timestamp=timestamp, force=args.force)

    # Fail on missing catalogs before JAX claims a device.
    for label, path in (
        ("injection", injection_catalog_path),
        ("proposal", proposal_catalog_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} catalog not found: {path}")

    jax, chain_method = configure_runtime(
        num_chains=config.sampler.num_chains,
        platform=args.platform,
        host_device_count=args.host_device_count,
        cpu_threads=args.cpu_threads,
        chain_method=args.chain_method,
    )
    mcmc, marginalization = run(
        config,
        injection_catalog_path,
        proposal_catalog_path,
        jax,
        chain_method,
    )
    save(
        mcmc,
        config,
        timestamp=timestamp,
        force=args.force,
        marginalization=marginalization,
    )


if __name__ == "__main__":
    main()
