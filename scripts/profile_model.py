"""Profile the production NumPyro model's log-density under ``jax.profiler.trace``.

This runs the exact model used by ``scripts/run_mcmc.py`` (the
``astrogwb.sampling.models`` compared against a fiducial injection), but
instead of sampling it isolates the model's potential-energy function and traces
its forward pass + gradient in a hot loop. The result is a Perfetto trace that
shows which XLA ops dominate the model math (the cosmology grid integrals,
``jnp.interp`` / ``jnp.trapezoid``, ``madau_dickinson_rate``, and the ``(F, N)``
``spectral_density`` contraction).

The model inputs come from :mod:`astrogwb.paper.inference`, the same pipeline
``scripts/run_mcmc.py`` feeds NUTS, and the runtime from
:func:`astrogwb.paper.runtime.configure_runtime` -- so what is profiled here is
the production model on production inputs, with the JAX device / x64 setup
matching production exactly.

Usage -- one ``--config`` per layer, in merge order, exactly as
``scripts/run_mcmc.py`` takes them::

    uv run --extra paper python scripts/profile_model.py \
        --config config/analysis/base/model.toml \
        --config config/fiducials.json \
        --config config/priors.json \
        --config config/networks.json \
        --config config/analysis/base/sampling.toml \
        --config config/analysis/runs/cosmological-parameters/_base.toml \
        --config config/analysis/runs/cosmological-parameters/ET-2L-aligned-CE-Hanford.toml \
        --injection-catalog outputs/catalogs/md-imrphenom-s41-n32768.h5 \
        --proposal-catalog outputs/catalogs/md-imrphenom-s42-n16384.h5

See docs/running-inference.md for the layer tree.

Open the generated ``perfetto_trace.json.gz`` at https://ui.perfetto.dev
(no TensorBoard install required).
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from astrogwb.paper.config.mcmc import RunConfig, build_run_config
from astrogwb.paper.config.runs import add_config_arguments, load_merged_config
from astrogwb.paper.runtime import add_runtime_arguments, configure_runtime

if TYPE_CHECKING:
    from astrogwb.catalog import PolarizationPowerCatalog

logger = logging.getLogger("profile_model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the production NumPyro model's potential (log-density) forward "
            "pass and gradient with jax.profiler.trace, emitting a Perfetto trace."
        )
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--injection-catalog",
        type=Path,
        required=True,
        metavar="PATH",
        help="The catalog file this run's [catalog].injection names.",
    )
    parser.add_argument(
        "--proposal-catalog",
        type=Path,
        required=True,
        metavar="PATH",
        help="The catalog file this run's [catalog].proposal names.",
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


def build_potential(
    config: RunConfig,
    injection_catalog: PolarizationPowerCatalog,
    proposal_catalog: PolarizationPowerCatalog,
    jax,
):
    """Build the production model inputs and return (potential_fn, init_params).

    Shares the inference-input pipeline with ``scripts/run_mcmc.py`` and stops
    just short of the NUTS/MCMC step, extracting the potential-energy function
    via NumPyro's public ``initialize_model``.
    """
    from numpyro.infer.initialization import init_to_value
    from numpyro.infer.util import initialize_model

    from astrogwb.paper.inference import (
        build_model,
        initial_values,
        prepare_inference_inputs,
        target_population,
    )

    inputs = prepare_inference_inputs(
        injection_catalog,
        proposal_catalog,
        grid=config.analysis.grid,
        detectors=config.analysis.detectors,
        target=target_population(config),
        density_sites=config.analysis.population.density_sites,
    )
    model, _ = build_model(
        config,
        spectral_density_fn=inputs.spectral_density_fn,
    )

    init_strategy = init_to_value(values=initial_values(config))
    info = initialize_model(
        jax.random.PRNGKey(config.sampler.seed),
        model,
        init_strategy=init_strategy,
        model_kwargs=inputs.model_kwargs(),
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
    outdir = args.outdir.resolve()
    config = build_run_config(load_merged_config(args), seed=args.seed)

    # Load both catalogs before JAX starts, the same way scripts/run_mcmc.py
    # does -- what is profiled must be the production model on production
    # inputs, including the proposal density each file records for itself.
    from astrogwb.paper.catalogs import load_run_catalog

    injection_catalog = load_run_catalog(
        args.injection_catalog.resolve(), label="injection"
    )
    proposal_catalog = load_run_catalog(
        args.proposal_catalog.resolve(), label="proposal"
    )

    jax, _ = configure_runtime(
        num_chains=config.sampler.num_chains,
        platform=args.platform,
        host_device_count=args.host_device_count,
        cpu_threads=args.cpu_threads,
        chain_method=args.chain_method,
    )

    potential_fn, init_params = build_potential(
        config, injection_catalog, proposal_catalog, jax
    )

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
