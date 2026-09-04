"""Pre-JAX runtime configuration for NumPyro MCMC entrypoints.

``OMP_NUM_THREADS`` / ``XLA_FLAGS``, ``JAX_PLATFORMS``, and
``numpyro.set_host_device_count(...)`` must be set *before* JAX initializes its
backend. This module therefore imports only the stdlib at load time; jax and
numpyro are imported inside :func:`configure_runtime`.
"""

from __future__ import annotations

import argparse
import logging
import os
from types import ModuleType

logger = logging.getLogger(__name__)

# Every env var that can fan out a BLAS/vector thread pool per chain worker.
# This matches the set Snakemake injects per job (snakemake/shell.py), which is
# why a partial override under a Snakemake launch leaks job ``threads`` into
# pools this flag is meant to pin (see --cpu-threads).
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for runtime CLI controls."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add deployment/runtime controls shared by runner entrypoints."""
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda", "tpu"),
        default="auto",
        help="JAX platform (default: auto; written to JAX_PLATFORMS when not auto).",
    )
    parser.add_argument(
        "--host-device-count",
        type=_positive_int,
        default=None,
        help="Logical host devices (default: sampler.num_chains).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=_positive_int,
        default=None,
        help="Cap OMP/BLAS/XLA CPU threads; omitted leaves inherited settings.",
    )
    parser.add_argument(
        "--chain-method",
        choices=("auto", "parallel", "sequential", "vectorized"),
        default="auto",
        help="NumPyro chain method (default: auto).",
    )


def configure_runtime(
    *,
    num_chains: int,
    platform: str = "auto",
    host_device_count: int | None = None,
    cpu_threads: int | None = None,
    chain_method: str = "auto",
) -> tuple[ModuleType, str]:
    """Configure CPU threads, device platform, and host device count, then import jax.

    Returns the imported ``jax`` module and the resolved chain method. This is the
    only place allowed to set the env vars / host device count, and it must run
    before anything else triggers JAX backend initialization.
    """
    # 1. CPU thread limits (no-op when omitted). --cpu-threads owns the
    #    thread policy: set BLAS/OMP caps and replace XLA_FLAGS outright.
    if cpu_threads is not None:
        n = str(cpu_threads)
        for var in CPU_THREAD_ENV_VARS:
            os.environ[var] = n
        os.environ["XLA_FLAGS"] = (
            f"--xla_cpu_multi_thread_eigen=true "
            f"intra_op_parallelism_threads={cpu_threads}"
        )

    # 2. Force a platform when requested; "auto" lets JAX pick (CUDA if present).
    if platform == "auto":
        os.environ.pop("JAX_PLATFORMS", None)
    else:
        os.environ["JAX_PLATFORMS"] = platform

    # 3. Host device count for CPU parallel chains -- must precede jax init.
    resolved_host_device_count = host_device_count or num_chains
    import numpyro

    numpyro.set_host_device_count(resolved_host_device_count)

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
        resolved_host_device_count,
    )

    # 5. Resolve chain_method. Use one device per chain whenever enough devices
    #    are visible. When an accelerator has fewer devices than chains, map
    #    vectorized chains onto it; insufficient CPU devices run sequentially.
    #    JAX reports the platform as "gpu" even when JAX_PLATFORMS=cuda.
    resolved_chain_method = chain_method
    if resolved_chain_method == "auto":
        if len(devices) >= num_chains:
            resolved_chain_method = "parallel"
        elif resolved_platform in ("gpu", "tpu"):
            resolved_chain_method = "vectorized"
        else:
            resolved_chain_method = "sequential"
    logger.info("chain_method=%s (num_chains=%d)", resolved_chain_method, num_chains)

    return jax, resolved_chain_method
