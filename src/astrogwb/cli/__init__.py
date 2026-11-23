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

from .simulate import configure_parser, simulate

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    """Run the astrogwb command-line interface."""
    parser = argparse.ArgumentParser(
        description="Bayesian inference for stochastic GW backgrounds."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "simulate", help="Simulate a background from a population catalogue."
    )
    configure_parser(command)
    args = vars(parser.parse_args(argv))
    args.pop("command")
    try:
        simulate(**args)
    except (ValueError, OSError) as error:
        command.error(str(error))
