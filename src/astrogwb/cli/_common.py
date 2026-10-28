"""Shared command-line options and runtime configuration."""

import os

import typer

DETECTOR_CONFIG_OPTION = typer.Option(
    "--detector-config",
    exists=True,
    dir_okay=False,
    help="Detector registry TOML layer; repeat to merge.",
)


def _apply_runtime(platform: str, cpu_threads: int | None) -> None:
    """Apply the runtime settings; runs before anything touches JAX."""
    if platform not in {"auto", "cpu", "cuda", "tpu"}:
        raise typer.BadParameter(
            f"unknown platform {platform!r}", param_hint="--platform"
        )
    if cpu_threads is not None:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = str(cpu_threads)
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=true "
            f"intra_op_parallelism_threads={cpu_threads}"
        )
    if platform != "auto":
        os.environ["JAX_PLATFORMS"] = platform
    import jax

    jax.config.update("jax_enable_x64", True)
