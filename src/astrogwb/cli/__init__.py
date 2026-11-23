"""The astrogwb command line.

``astrogwb simulate OUTPUT.h5 POPULATION`` simulates the background of a
physical event pool, a gwmock-pop population catalogue (HDF5 or CSV), into one
HDF5 file. ``--waveform-config`` names a TOML file of waveform settings and
``--detector-config`` names a complete detector configuration TOML file. For example::

    astrogwb simulate ensemble/41-0.h5 events.h5 \\
        --waveform-config waveform.toml --detector-config detectors.toml --network H1,L1 \\
        --poisson-rate 0.00317 --num-segments 16 --segment-duration 2048 --seed 41

The detector configuration contains [detectors.<name>] definitions, and
``--network`` lists the comma-separated names of the detectors to simulate. The file carries the configuration as JSON in its ``metadata``
attribute, with the astrogwb ``version`` that wrote it. Each segment includes
source interference. ``--simulate-detector-noise`` also adds detector noise.

A reader checks the file's format version. Runtime settings
(--platform, --cpu-threads) are applied before the JAX backend initializes.
"""

import argparse
from pathlib import Path

from .simulate import simulate

__all__ = ["main"]


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


def main(argv: list[str] | None = None) -> None:
    """Run the astrogwb command-line interface."""
    parser = argparse.ArgumentParser(
        description="Bayesian inference for stochastic GW backgrounds."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "simulate", help="Simulate a background from a population catalogue."
    )
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
    args = vars(parser.parse_args(argv))
    args.pop("command")
    try:
        simulate(**args)
    except (ValueError, OSError) as error:
        command.error(str(error))
