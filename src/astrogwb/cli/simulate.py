"""Simulate one background from a physical event pool into one HDF5 file."""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Sequence
from pathlib import Path

from astrogwb.runtime import configure_runtime


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _detector_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(","))
    if not all(names):
        raise argparse.ArgumentTypeError("must be comma-separated detector names")
    return names


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def configure_parser(command: argparse.ArgumentParser) -> None:
    """Configure arguments for the simulate command."""
    command.add_argument("output", type=Path, help="Output HDF5 file.")
    command.add_argument("population", type=Path, help="CSV or HDF5 event catalogue.")
    command.add_argument(
        "--waveform-config",
        type=Path,
        required=True,
        help="Waveform settings TOML file.",
    )
    command.add_argument(
        "--detector-config",
        type=Path,
        required=True,
        help="Detector configuration TOML file.",
    )
    command.add_argument(
        "--network",
        type=_detector_names,
        required=True,
        metavar="NAMES",
        help="Comma-separated detectors of the network, such as H1,L1.",
    )
    command.add_argument(
        "--poisson-rate",
        type=_nonnegative_float,
        required=True,
        help="Observer-frame Poisson event rate in events per second.",
    )
    command.add_argument(
        "--num-segments",
        type=_positive_int,
        required=True,
        help="Number of observation segments.",
    )
    command.add_argument(
        "--segment-duration",
        type=_nonnegative_float,
        required=True,
        help="Segment length in seconds.",
    )
    command.add_argument("--seed", type=int, required=True, help="Seed of the draw.")
    command.add_argument(
        "--population-root",
        default="source_parameters",
        help="HDF5 population dataset or group path; ignored for CSV.",
    )
    command.add_argument(
        "--simulate-detector-noise",
        action="store_true",
        help="Add independent detector noise to every segment.",
    )
    command.add_argument(
        "--chunk-size", type=_positive_int, default=1024, help="Waveforms per chunk."
    )
    command.add_argument(
        "--segment-batch-size",
        type=_positive_int,
        default=1024,
        help="Segments per batch.",
    )
    command.add_argument(
        "--platform", default="auto", help="JAX platform: auto, cpu, cuda or tpu."
    )
    command.add_argument(
        "--cpu-threads", type=_positive_int, help="Cap OMP/BLAS/XLA CPU threads."
    )


def simulate(
    output: Path,
    population: Path,
    waveform_config: Path,
    detector_config: Path,
    network: Sequence[str],
    poisson_rate: float,
    num_segments: int,
    segment_duration: float,
    seed: int,
    population_root: str = "source_parameters",
    simulate_detector_noise: bool = False,
    chunk_size: int = 1024,
    segment_batch_size: int = 1024,
    platform: str = "auto",
    cpu_threads: int | None = None,
) -> None:
    """Simulate a background from the event pool POPULATION into OUTPUT.

    Events are drawn without replacement from the pool, with independent Poisson
    counts in each segment at the given rate. Each segment sums the coherent
    strain of its events and optionally adds fresh detector noise.

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
    network
        Names of the detectors in the network, defined in the detector configuration.
    poisson_rate
        Observer-frame event rate of the pool's population, in events per second.
    num_segments
        Number of segments.
    segment_duration
        Segment length in seconds.
    seed
        Random seed.
    population_root
        HDF5 dataset or group containing the population.
    simulate_detector_noise
        Add independent detector noise to every segment.
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
        simulate_background,
    )
    from astrogwb.detector import load_detectors_from_file, select_network
    from astrogwb.populations.io import read_population_file
    from astrogwb.waveform import WaveformConfig, waveform_generator

    selected = select_network(load_detectors_from_file(detector_config), network)
    with waveform_config.open("rb") as file:
        waveform = WaveformConfig.model_validate(tomllib.load(file))
    config = SimulationConfig(
        waveform=waveform,
        network=selected,
        poisson_rate=poisson_rate,
        num_segments=num_segments,
        segment_duration=segment_duration,
        simulate_detector_noise=simulate_detector_noise,
        population=str(population),
        population_root=population_root,
        seed=seed,
    )
    draw = simulate_background(
        jax.random.key(seed),
        read_population_file(population, root=population_root),
        waveform_generator(config.waveform),
        config.network,
        poisson_rate=config.poisson_rate,
        num_segments=config.num_segments,
        segment_duration=config.segment_duration,
        chunk_size=chunk_size,
        segment_batch_size=segment_batch_size,
        simulate_detector_noise=config.simulate_detector_noise,
    )
    draw.write(output, config)
    print(f"{int(draw.segment_counts.sum())} events -> {output}")
