"""Simulate one background from a physical event pool into one HDF5 file."""

from __future__ import annotations

import tomllib
from pathlib import Path

from astrogwb.runtime import configure_runtime


def simulate(
    output: Path,
    population: Path,
    waveform_config: Path,
    detector_config: Path,
    rate_per_year: float,
    num_segments: int,
    segment_seconds: float,
    seed: int,
    population_root: str = "source_parameters",
    incoherent: bool = False,
    chunk_size: int = 1024,
    segment_batch_size: int = 1024,
    platform: str = "auto",
    cpu_threads: int | None = None,
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
    population_root
        HDF5 dataset or group containing the population.
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
    configure_runtime(platform=platform, cpu_threads=cpu_threads)
    import jax

    from astrogwb.background.simulation import (
        SimulationConfig,
        read_population_file,
        simulate_background,
    )
    from astrogwb.detector import load_networks_from_file
    from astrogwb.utils import years_to_seconds
    from astrogwb.waveform import WaveformConfig

    networks = load_networks_from_file(detector_config)
    with waveform_config.open("rb") as file:
        waveform = WaveformConfig.model_validate(tomllib.load(file))
    config = SimulationConfig(
        waveform=waveform,
        networks=networks,
        observer_rate_hz=rate_per_year / years_to_seconds(1.0),
        num_segments=num_segments,
        segment_seconds=segment_seconds,
        incoherent=incoherent,
        population=str(population),
        population_root=population_root,
        seed=seed,
    )
    draw = simulate_background(
        jax.random.key(seed),
        read_population_file(population, root=population_root),
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
    print(f"{int(draw.segment_counts.sum())} events -> {output}")
