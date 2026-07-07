"""Profile the production NumPyro model's log-density under ``jax.profiler.trace``.

This runs the *exact* model used by ``scripts/run_mcmc.py`` (the
``astrogwb.sampling.numpyro_model`` compared against a fiducial injection), but
instead of sampling it isolates the model's potential-energy function and traces
its forward pass + gradient in a hot loop. The result is a Perfetto trace that
shows which XLA ops dominate the model math (the cosmology grid integrals,
``jnp.interp`` / ``jnp.trapezoid``, ``madau_dickinson_rate``, and the ``(F, N)``
``spectral_density`` contraction).

The model inputs are rebuilt here (not imported from ``run_mcmc``) so this stays
a self-contained profiling entrypoint; only :func:`run_mcmc.configure_runtime`
is reused so the JAX device / x64 setup matches production exactly.

Usage::

    uv run python scripts/profile_model.py --config configs/mcmc.example.toml
    uv run python scripts/profile_model.py --config configs/mcmc.example.toml --iters 100

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

from astrogwb.sampling.config import RunConfig, build_run_config, load_config
from run_mcmc import configure_runtime

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
        help="Path to the TOML or JSON config file (same format as run_mcmc.py).",
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
    return parser.parse_args(argv)


def build_potential(config: RunConfig, jax):
    """Rebuild the production model inputs and return (potential_fn, init_params).

    Mirrors ``scripts.run_mcmc.run`` up to (but excluding) the NUTS/MCMC step, then
    extracts the potential-energy function via NumPyro's public ``initialize_model``.
    """
    import jax.numpy as jnp
    from numpyro.infer.initialization import init_to_value
    from numpyro.infer.util import initialize_model

    from astrogwb.detector import effective_psd, load_sensitivity_map
    from astrogwb.gwb import frequency_mask as make_frequency_mask
    from astrogwb.gwb import spectral_density
    from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
        compute_proposal_log_pdf,
        make_merger_rate_and_log_weights_fn,
    )
    from astrogwb.sampling.numpyro_model import numpyro_model
    from astrogwb.sampling.priors import build_prior
    from astrogwb.waveform import load_polarization_power_catalog

    cat = config.catalog
    cosmo = config.cosmology

    catalog = load_polarization_power_catalog(cat.path)
    frequencies = jnp.asarray(catalog.frequencies)
    polarization_power = jnp.asarray(catalog.polarization_power)
    samples = {name: jnp.asarray(v) for name, v in catalog.samples.items()}
    n_freq, n_samples = polarization_power.shape
    logger.info(
        "Loaded catalog %s: n_frequency_bins=%d n_proposal_samples=%d",
        cat.path,
        n_freq,
        n_samples,
    )

    sensitivities = load_sensitivity_map(cat.detectors)
    effective_psd_arr = jnp.asarray(
        effective_psd(frequencies, list(cat.detectors), sensitivities)
    )
    freq_mask = make_frequency_mask(frequencies, fmin=cat.f_min, fmax=cat.f_max)

    z_samples = jnp.asarray(samples["redshift"])
    z_grid = jnp.linspace(cosmo.z_min, cosmo.z_max, cosmo.n_grid)
    log_p_proposal = compute_proposal_log_pdf(
        z_samples, z_grid=z_grid, fiducials=config.fiducials
    )

    merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
        z_grid=z_grid,
        proposal_log_pdf=log_p_proposal,
        fiducial_xi_0=config.fiducials["xi_0"],
        fiducial_xi_n=config.fiducials["xi_n"],
    )

    ones_weights = jnp.ones((n_samples,))
    rate0, _ = merger_rate_and_log_weights_fn(config.fiducials, samples)
    observed_spectral_density = spectral_density(
        polarization_power,
        ones_weights,
        rate0,
        average_mode="analytic_inclination",
    )

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    raw = load_config(args.config)
    config = build_run_config(raw, seed=args.seed)
    logger.info("Config: %s", args.config)

    jax, _ = configure_runtime(config.runtime, config.sampler.num_chains)

    potential_fn, init_params = build_potential(config, jax)

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

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = args.outdir / f"model-{ad_mode}-{timestamp}"
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
