"""Shared command-line options and runtime configuration."""

import typer

from astrogwb.runtime import configure_runtime

DETECTOR_CONFIG_OPTION = typer.Option(
    "--detector-config",
    exists=True,
    dir_okay=False,
    help="Detector registry TOML layer; repeat to merge.",
)

CONFIG_DIR_OPTION = typer.Option(
    "--config-dir",
    exists=True,
    file_okay=False,
    help="Directory of shared TOML layers, merged by astrogwb.config.merge_config.",
)


def _apply_runtime(platform: str, cpu_threads: int | None) -> None:
    """Apply the runtime settings; runs before anything touches JAX."""
    try:
        configure_runtime(platform=platform, cpu_threads=cpu_threads)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--platform") from error
