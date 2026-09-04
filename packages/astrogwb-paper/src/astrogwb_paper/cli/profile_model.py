"""Profile the production NumPyro model's log-density under ``jax.profiler.trace``.

This runs the exact model used by ``astrogwb-run-mcmc`` (the
``astrogwb.sampling.models`` compared against a fiducial injection), but
instead of sampling it isolates the model's potential-energy function and traces
its forward pass + gradient in a hot loop. The result is a Perfetto trace that
shows which XLA ops dominate the model math (the cosmology grid integrals,
``jnp.interp`` / ``jnp.trapezoid``, ``madau_dickinson_rate``, and the ``(F, N)``
``spectral_density`` contraction).

The model inputs come from :mod:`astrogwb_paper.inference`, the same pipeline
``astrogwb-run-mcmc`` feeds NUTS, and the runtime from
:func:`astrogwb_paper.runtime.configure_runtime` -- so what is profiled here is
the production model on production inputs, with the JAX device / x64 setup
matching production exactly.

Usage -- one ``--config`` per layer, in merge order, exactly as
``astrogwb-run-mcmc`` takes them::

    uv run astrogwb-profile-model \
        --config config/analysis/base/model.toml \
        --config config/analysis/base/parameters.toml \
        --config config/analysis/base/sampling.toml \
        --config config/analysis/runs/cosmological-parameters/_base.toml \
        --config config/analysis/runs/cosmological-parameters/ET-2L-aligned-CE-Hanford.toml \
        --bank md-imrphenom-s41=outputs/banks/md-imrphenom-s41.h5 \
        --bank md-imrphenom-s42=outputs/banks/md-imrphenom-s42.h5

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

from astrogwb_paper.cli.run_mcmc import resolve_run_proposal
from astrogwb_paper.config.mcmc import ProposalConfig, RunConfig, build_run_config
from astrogwb_paper.config.runs import add_config_arguments, load_merged_config
from astrogwb_paper.runtime import add_runtime_arguments, configure_runtime

if TYPE_CHECKING:
    from astrogwb_paper.catalogs import CatalogSource

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
        "--bank",
        dest="banks",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "One waveform bank file, as NAME=PATH; repeat once per distinct "
            "bank the run config's [catalog.injection] / [catalog.proposal] "
            "names."
        ),
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
    injection_source: CatalogSource,
    proposal_source: CatalogSource,
    proposal: ProposalConfig,
    jax,
):
    """Build the production model inputs and return (potential_fn, init_params).

    Shares the inference-input pipeline with ``astrogwb-run-mcmc`` and stops
    just short of the NUTS/MCMC step, extracting the potential-energy function
    via NumPyro's public ``initialize_model``.
    """
    from numpyro.infer.initialization import init_to_value
    from numpyro.infer.util import initialize_model

    from astrogwb_paper.inference import (
        build_model,
        initial_values,
        prepare_inference_inputs,
    )

    inputs = prepare_inference_inputs(
        injection_source,
        proposal_source,
        fiducials=config.fiducials,
        proposal_config=proposal,
        grid=config.analysis_grid,
        detectors=config.analysis.detectors,
    )
    model, _ = build_model(
        config,
        merger_rate_and_log_weights_fn=inputs.merger_rate_and_log_weights_fn,
    )

    init_strategy = init_to_value(values=initial_values(config))
    info = initialize_model(
        jax.random.PRNGKey(config.seed),
        model,
        init_strategy=init_strategy,
        model_kwargs=inputs.masked_model_kwargs(),
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


def _parse_bank_args(values: list[str]) -> dict[str, Path]:
    """Parse repeated ``NAME=PATH`` flags into a bank-name -> path mapping."""
    banks: dict[str, Path] = {}
    for item in values:
        name, sep, raw_path = item.partition("=")
        if not sep or not name:
            raise ValueError(f"--bank must be NAME=PATH, got {item!r}")
        banks[name] = Path(raw_path).resolve()
    return banks


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    bank_paths = _parse_bank_args(args.banks)
    outdir = args.outdir.resolve()
    config = build_run_config(load_merged_config(args), seed=args.seed)

    # Resolve the proposal density from bank provenance before JAX starts, the
    # same way astrogwb-run-mcmc does -- what is profiled must be the
    # production model on production inputs.
    from astrogwb_paper.catalogs import CatalogSource

    injection_source = CatalogSource.resolve(
        config.catalog.injection, bank_paths, role="injection"
    )
    proposal_source = CatalogSource.resolve(
        config.catalog.proposal, bank_paths, role="proposal"
    )
    proposal = resolve_run_proposal(config, proposal_source)

    jax, _ = configure_runtime(
        num_chains=config.sampler.num_chains,
        platform=args.platform,
        host_device_count=args.host_device_count,
        cpu_threads=args.cpu_threads,
        chain_method=args.chain_method,
    )

    potential_fn, init_params = build_potential(
        config, injection_source, proposal_source, proposal, jax
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
