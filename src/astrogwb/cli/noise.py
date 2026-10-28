"""Generate detector noise."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ._common import DETECTOR_CONFIG_OPTION, _apply_runtime


def noise(
    realization: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=False, help="A realization with a background."
        ),
    ],
    output: Annotated[Path, typer.Argument(dir_okay=False, help="HDF5 file to write.")],
    detector_config: Annotated[list[Path], DETECTOR_CONFIG_OPTION],
    platform: Annotated[
        str, typer.Option(help="JAX platform: auto, cpu, cuda or tpu.")
    ] = "auto",
) -> None:
    """Write the detector noise of a realization's background networks.

    The networks are those recorded in the realization's background, and the
    registry is trusted to be the one it was simulated with.
    """
    _apply_runtime(platform, None)
    from astrogwb.detector.config import load_detector_config
    from astrogwb.simulators import Realization, detector_noise

    registry = load_detector_config(detector_config)
    recorded = Realization.open(realization)
    background = recorded.background()
    detector_noise(
        background.frequencies,
        registry,
        recorded.population_metadata.observation_time,
        network_names=tuple(background.spectra),
    ).write(output)
    typer.echo(f"noise of {', '.join(background.spectra)} -> {output}")
