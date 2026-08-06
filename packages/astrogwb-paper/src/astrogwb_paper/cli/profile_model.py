"""Profile the production NumPyro model's log-density under ``jax.profiler.trace``.

This runs the exact model used by ``astrogwb-run-mcmc`` (the
``astrogwb.sampling.models`` compared against a fiducial injection), but
instead of sampling it isolates the model's potential-energy function and traces
its forward pass + gradient in a hot loop. The result is a Perfetto trace that
shows which XLA ops dominate the model math (the cosmology grid integrals,
``jnp.interp`` / ``jnp.trapezoid``, ``madau_dickinson_rate``, and the ``(F, N)``
``spectral_density`` contraction).

The model inputs are rebuilt here (not imported from ``run_mcmc``) so this stays
a self-contained profiling entrypoint; only
:func:`astrogwb_paper.runtime.configure_runtime` is reused so the JAX device / x64
setup matches production exactly.

Usage::

    uv run astrogwb-profile-model \
        --config packages/astrogwb-paper/configs/mcmc.example.toml \
        --catalog out/catalogs/bns-n16384-df1.h5

Open the generated ``perfetto_trace.json.gz`` at https://ui.perfetto.dev
(no TensorBoard install required).
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from functools import partial
from pathlib import Path

from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.config.mcmc import RunConfig, build_run_config
from astrogwb_paper.runtime import add_runtime_arguments, configure_runtime

logger = logging.getLogger("profile_model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the production NumPyro model's potential (log-density) forward "
            "pass and gradient with jax.profiler.trace, emitting a Perfetto trace."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML or JSON config file used by astrogwb-run-mcmc.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="Waveform catalog to profile against the configured inference model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the config seed.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of forward+gradient evaluations to trace (default 50).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("profiles"),
        help="Directory for the Perfetto trace (default: profiles/).",
    )
    parser.add_argument(
        "--reverse-ad",
        action="store_true",
        help="Force reverse-mode AD (default follows the config's forward_mode flag).",
    )
    add_runtime_arguments(parser)
    return parser.parse_args(argv)


def build_potential(config: RunConfig, catalog_path: Path, jax):
    """Rebuild the production model inputs and return (potential_fn, init_params).

    Mirrors the production runner up to (but excluding) the NUTS/MCMC step, then
    extracts the potential-energy function via NumPyro's public ``initialize_model``.
    """
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
    from astrogwb.sampling.models import spectral_density_model
    from astrogwb.waveform import polarization_power as compute_polarization_power
    from numpyro.infer.initialization import init_to_value
    from numpyro.infer.util import initialize_model
    from pluscross import load_catalog

    from astrogwb_paper.priors import build_prior

    analysis = config.analysis
    cosmo = config.cosmology

    catalog = load_catalog(catalog_path)
    frequencies = jnp.asarray(catalog.frequencies)
    polarization_power = jnp.asarray(compute_polarization_power(catalog))
    samples = {name: jnp.asarray(v) for name, v in catalog.source_parameters.items()}
    del catalog
    n_freq, n_samples = polarization_power.shape
    logger.info(
        "Loaded catalog %s: n_frequency_bins=%d n_proposal_samples=%d",
        catalog_path,
        n_freq,
        n_samples,
    )

    sensitivities = load_sensitivity_map(analysis.detectors)
    effective_psd_arr = jnp.asarray(
        effective_psd(frequencies, list(analysis.detectors), sensitivities)
    )
    freq_mask = make_frequency_mask(
        frequencies, fmin=analysis.f_min, fmax=analysis.f_max
    )

    z_grid = jnp.linspace(cosmo.z_min, cosmo.z_max, cosmo.n_grid)
    _, _, proposal_logprob = compute_merger_rate_distance_and_logprob(
        config.fiducials, samples, redshift_grid=z_grid
    )

    merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
        fiducials=config.fiducials,
        redshift_grid=z_grid,
        proposal_logprob=proposal_logprob,
    )

    rate0, log_weights0 = merger_rate_and_log_weights_fn(config.fiducials, samples)
    weights0 = jnp.exp(log_weights0)
    observed_spectral_density = spectral_density(
        polarization_power,
        weights0,
        rate0,
        average_mode="analytic_inclination",
    )

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

    priors = {name: build_prior(spec) for name, spec in config.priors.items()}
    model = partial(
        spectral_density_model,
        observation_time=config.observation_time,
        average_mode="analytic_inclination",
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
        priors=priors,
        constants=config.constants,
    )

    model_kwargs = {
        "frequencies": frequencies,
        "polarization_power": polarization_power,
        "samples": samples,
        "observed_spectral_density": observed_spectral_density,
        "effective_psd": effective_psd_arr,
    }

    init_strategy = init_to_value(
        values={name: config.fiducials[name] for name in config.sampled_params}
    )
    info = initialize_model(
        jax.random.PRNGKey(config.seed),
        model,
        init_strategy=init_strategy,
        model_kwargs=model_kwargs,
        forward_mode_differentiation=config.sampler.forward_mode_differentiation,
        dynamic_args=False,
    )
    return info.potential_fn, info.param_info.z


def _bench(fn, x, iters: int) -> float:
    """Mean wall-clock ms per call over ``iters`` (result blocked each time)."""
    import jax

    t0 = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(fn(x))
    return 1e3 * (time.perf_counter() - t0) / iters


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config.resolve()
    catalog_path = args.catalog.resolve()
    outdir = args.outdir.resolve()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    raw = load_mapping(config_path)
    config = build_run_config(raw, seed=args.seed)
    logger.info("Config: %s", config_path)

    jax, _ = configure_runtime(
        num_chains=config.sampler.num_chains,
        platform=args.platform,
        host_device_count=args.host_device_count,
        cpu_threads=args.cpu_threads,
        chain_method=args.chain_method,
    )

    potential_fn, init_params = build_potential(config, catalog_path, jax)

    forward_mode = config.sampler.forward_mode_differentiation and not args.reverse_ad
    ad_mode = "forward" if forward_mode else "reverse"
    grad_transform = jax.jacfwd if forward_mode else jax.grad

    forward = jax.jit(potential_fn)
    grad = jax.jit(grad_transform(potential_fn))

    logger.info("Compiling forward + %s-mode gradient...", ad_mode)
    t0 = time.perf_counter()
    jax.block_until_ready(forward(init_params))
    jax.block_until_ready(grad(init_params))
    logger.info("Compile (forward+grad): %.1f ms", 1e3 * (time.perf_counter() - t0))

    fwd_ms = _bench(forward, init_params, args.iters)
    grad_ms = _bench(grad, init_params, args.iters)
    logger.info("Forward:  %.3f ms/iter", fwd_ms)
    logger.info("Gradient: %.3f ms/iter (%s mode)", grad_ms, ad_mode)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    logdir = outdir / f"model-{ad_mode}-{timestamp}"
    logdir.mkdir(parents=True, exist_ok=True)

    logger.info("Tracing %d forward+gradient iterations -> %s", args.iters, logdir)
    with jax.profiler.trace(str(logdir), create_perfetto_trace=True):
        for _ in range(args.iters):
            with jax.profiler.TraceAnnotation("forward"):
                jax.block_until_ready(forward(init_params))
            with jax.profiler.TraceAnnotation("grad"):
                jax.block_until_ready(grad(init_params))

    traces = list(logdir.rglob("*.pb.gz")) + list(logdir.rglob("*.json.gz"))
    logger.info("Trace written under %s", logdir)
    for path in traces:
        logger.info("  %s", path)
    logger.info("Open the *.json.gz trace at https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
