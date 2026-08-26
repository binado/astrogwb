"""Assemble canonical MCMC run configs from the ``config/analysis/`` tree.

This is the JAX-free gate between the three-layer TOML merge and everything
that samples. ``Snakefile`` calls it once per run::

    astrogwb-assemble-config --experiment cosmological-parameters \\
        --run ET-2L-aligned-CE-Hanford \\
        --output outputs/configs/cosmological-parameters/ET-2L-aligned-CE-Hanford.json

Writing ``save_config(RunConfig)`` rather than the raw merge keeps each config
self-contained and diff-able: the file on disk always carries every default
filled in, so a run that inherits ``target_accept`` and one that spells it out
produce identical configs.

``--all`` assembles every discovered run in one process. That mode exists for
the ``configs`` aggregate target and for pre-flight checking: it fails on the
first invalid run, so a config typo surfaces without building any bank.

Deliberately does *not* resolve the proposal density. That needs the proposal
bank's provenance attributes, and requiring the banks to exist before any
config can be validated would invert the cheap/expensive order. The density is
resolved at run time instead -- see :mod:`astrogwb_paper.cli.run_mcmc`.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path

from astrogwb_paper.config.banks import BankGenerationConfig, discover_banks
from astrogwb_paper.config.mcmc import build_run_config, save_config
from astrogwb_paper.config.runs import (
    assemble_run,
    check_bank_references,
    discover_runs,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge config/analysis/{base,runs} into canonical, defaults-filled "
            "JSON run configs."
        )
    )
    parser.add_argument("--experiment", help="Experiment directory name.")
    parser.add_argument("--run", help="Run name (the TOML stem) within it.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Destination JSON path for a single run.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Assemble every discovered run instead of one.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/configs"),
        help="Root directory for <experiment>/<run>.json under --all.",
    )
    args = parser.parse_args(argv)
    if args.all:
        if args.experiment or args.run or args.output:
            parser.error("--all takes --output-dir only")
    elif not (args.experiment and args.run and args.output):
        parser.error("pass --experiment, --run and --output, or --all")
    return args


def assemble(
    experiment: str,
    run: str,
    destination: Path,
    banks: Mapping[str, BankGenerationConfig] | None = None,
) -> None:
    """Merge, validate, and write one run config."""
    config = build_run_config(assemble_run(experiment, run))
    check_bank_references(config, label=f"{experiment}/{run}", banks=banks)
    save_config(config, destination)
    logger.info("Wrote %s", destination)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if not args.all:
        assemble(args.experiment, args.run, args.output)
        return

    # Load the bank configs once: --all is the fail-fast gate over all runs.
    banks = discover_banks()
    for experiment, runs in discover_runs().items():
        for run in runs:
            assemble(
                experiment,
                run,
                args.output_dir / experiment / f"{run}.json",
                banks=banks,
            )


if __name__ == "__main__":
    main()
