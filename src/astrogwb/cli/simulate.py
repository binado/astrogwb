"""Simulate one background into one HDF5 file."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Any

import typer

from ._common import DETECTOR_CONFIG_OPTION, _apply_runtime


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def simulate(
    output: Annotated[
        Path, typer.Argument(dir_okay=False, help="Output HDF5 file, e.g. 41-0.h5.")
    ],
    population_metadata: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Population metadata TOML."),
    ],
    waveform_metadata: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Waveform metadata TOML.")
    ],
    detector_config: Annotated[list[Path], DETECTOR_CONFIG_OPTION],
    seed: Annotated[int, typer.Option(help="Seed of the realization.")],
    index: Annotated[
        int, typer.Option(min=0, help="Realization index for the given seed.")
    ] = 0,
    chunk_size: Annotated[int, typer.Option(min=1, help="Waveforms per chunk.")] = 1024,
    segment_duration: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="Segment length in seconds (e.g. 2048); assigns sources to "
            "segments and samples per-segment detector noise.",
        ),
    ] = None,
    platform: Annotated[
        str, typer.Option(help="JAX platform: auto, cpu, cuda or tpu.")
    ] = "auto",
    cpu_threads: Annotated[
        int | None, typer.Option(min=1, help="Cap OMP/BLAS/XLA CPU threads.")
    ] = None,
) -> None:
    """Simulate the population, waveforms and background in one pass into OUTPUT.

    Parameters
    ----------
    output
        Output file path.
    population_metadata
        Path to population metadata.
    waveform_metadata
        Path to waveform metadata.
    detector_config
        Detector configuration layer paths.
    seed
        Random seed.
    index
        Realization index.
    chunk_size
        Number of sources per chunk.
    segment_duration
        Noise segment duration in seconds.
    platform
        JAX execution platform.
    cpu_threads
        Number of CPU threads.
    """
    _apply_runtime(platform, cpu_threads)
    from astrogwb.detector.config import load_detector_config
    from astrogwb.simulators import (
        PopulationMetadata,
        SimulationConfig,
        realization_key,
    )
    from astrogwb.simulators import simulate as run
    from astrogwb.waveform import WaveformMetadata

    config = SimulationConfig(
        population=PopulationMetadata.model_validate(_read_toml(population_metadata)),
        waveform=WaveformMetadata.model_validate(_read_toml(waveform_metadata)),
        detectors=load_detector_config(detector_config),
        segment_duration=segment_duration,
    )
    artifact = run(config, realization_key(seed, index), chunk_size=chunk_size)
    artifact.write(output)
    typer.echo(f"{artifact.count} sources -> {output}")
