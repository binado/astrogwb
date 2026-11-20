"""Simulate one background from a physical event pool into one HDF5 file."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated

import typer

from astrogwb.runtime import configure_runtime


def simulate(
    output: Annotated[
        Path, typer.Argument(dir_okay=False, help="Output HDF5 file, e.g. 41-0.h5.")
    ],
    population: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Event pool: a CSV catalogue, or an HDF5 catalogue under the "
            "'source_parameters' dataset, with canonical gwmock-pop column names.",
        ),
    ],
    waveform_config: Annotated[
        Path,
        typer.Option(
            "--waveform-config",
            exists=True,
            dir_okay=False,
            help="TOML file of the waveform settings (WaveformMetadata fields).",
        ),
    ],
    detector_config: Annotated[
        Path,
        typer.Option(
            "--detector-config",
            exists=True,
            dir_okay=False,
            help="TOML file containing detector definitions and network memberships.",
        ),
    ],
    rate_per_year: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="Observer-frame event rate, per year, of the population the pool "
            "represents.",
        ),
    ],
    num_segments: Annotated[
        int, typer.Option(min=1, help="Number of observation segments.")
    ],
    segment_seconds: Annotated[
        float, typer.Option(min=0.0, help="Segment length in seconds, e.g. 2048.")
    ],
    seed: Annotated[int, typer.Option(help="Seed of the draw.")],
    index: Annotated[
        int, typer.Option(min=0, help="Draw index for the given seed.")
    ] = 0,
    incoherent: Annotated[
        bool,
        typer.Option(
            "--incoherent",
            help="Sum event powers without detector noise or segment interference: "
            "the noiseless signal of the same Poisson draw.",
        ),
    ] = False,
    chunk_size: Annotated[int, typer.Option(min=1, help="Waveforms per chunk.")] = 1024,
    segment_batch_size: Annotated[
        int, typer.Option(min=1, help="Segments per batch; cost only.")
    ] = 1024,
    platform: Annotated[
        str, typer.Option(help="JAX platform: auto, cpu, cuda or tpu.")
    ] = "auto",
    cpu_threads: Annotated[
        int | None, typer.Option(min=1, help="Cap OMP/BLAS/XLA CPU threads.")
    ] = None,
) -> None:
    """Simulate a background from the event pool POPULATION into OUTPUT.

    Events are drawn without replacement from the pool, with independent Poisson
    counts in each segment at the given rate. Each segment sums the coherent
    strain of its events and adds fresh detector noise, unless --incoherent.

    Parameters
    ----------
    output
        Output file path.
    population
        Event pool path.
    waveform_config
        Waveform settings TOML path.
    detector_config
        Complete detector configuration TOML path.
    rate_per_year
        Event rate of the pool's population, per year.
    num_segments
        Number of segments.
    segment_seconds
        Segment length in seconds.
    seed
        Random seed.
    index
        Draw index.
    incoherent
        Skip noise and coherent summation.
    chunk_size
        Number of sources per chunk.
    segment_batch_size
        Number of segments per batch.
    platform
        JAX execution platform.
    cpu_threads
        Number of CPU threads.
    """
    try:
        configure_runtime(platform=platform, cpu_threads=cpu_threads)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--platform") from error
    from astrogwb.background.simulation import (
        SimulationConfig,
        read_event_pool,
        realization_key,
        simulate_background,
    )
    from astrogwb.detector import load_networks_from_file
    from astrogwb.utils import years_to_seconds
    from astrogwb.waveform import WaveformMetadata

    try:
        networks = load_networks_from_file(detector_config)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--detector-config") from error
    try:
        with waveform_config.open("rb") as file:
            waveform = WaveformMetadata.model_validate(tomllib.load(file))
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--waveform-config") from error
    try:
        config = SimulationConfig(
            waveform=waveform,
            networks=networks,
            observer_rate_hz=rate_per_year / years_to_seconds(1.0),
            num_segments=num_segments,
            segment_seconds=segment_seconds,
            incoherent=incoherent,
            population=str(population),
            seed=seed,
            index=index,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    draw = simulate_background(
        realization_key(seed, index),
        read_event_pool(population),
        config.waveform,
        config.networks,
        observer_rate_hz=config.observer_rate_hz,
        num_segments=num_segments,
        segment_seconds=segment_seconds,
        chunk_size=chunk_size,
        segment_batch_size=segment_batch_size,
        incoherent=incoherent,
    )
    draw.write(output, config)
    typer.echo(f"{int(draw.segment_counts.sum())} events -> {output}")
