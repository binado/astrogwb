"""Simulate one background into one HDF5 file."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ._common import CONFIG_DIR_OPTION, DETECTOR_CONFIG_OPTION, _apply_runtime


def simulate(
    output: Annotated[
        Path, typer.Argument(dir_okay=False, help="Output HDF5 file, e.g. 41-0.h5.")
    ],
    ensemble: Annotated[
        str, typer.Option(help="Name of an [ensembles.<name>] table in the config.")
    ],
    detector_config: Annotated[list[Path], DETECTOR_CONFIG_OPTION],
    seed: Annotated[int, typer.Option(help="Seed of the realization.")],
    config_dir: Annotated[Path, CONFIG_DIR_OPTION] = Path("config"),
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
    ensemble
        Name of the ensemble table, holding ``realization`` and ``waveform``.
    detector_config
        Detector configuration layer paths.
    seed
        Random seed.
    config_dir
        Directory of shared TOML layers.
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
    from astrogwb.config import merge_config
    from astrogwb.detector.config import load_detector_config
    from astrogwb.simulators import (
        RealizationMetadata,
        SimulationConfig,
        realization_key,
    )
    from astrogwb.simulators import simulate as run
    from astrogwb.waveform import WaveformMetadata

    tables = merge_config(config_dir)["ensembles"][ensemble]
    config = SimulationConfig(
        population=RealizationMetadata.model_validate(tables["realization"]),
        waveform=WaveformMetadata.model_validate(tables["waveform"]),
        detectors=load_detector_config(detector_config),
        segment_duration=segment_duration,
    )
    artifact = run(config, realization_key(seed, index), chunk_size=chunk_size)
    artifact.write(output)
    typer.echo(f"{artifact.count} sources -> {output}")
