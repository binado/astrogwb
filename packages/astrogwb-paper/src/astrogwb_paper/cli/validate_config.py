"""Assemble canonical MCMC run configs from the shared YAML inventory.

This is the JAX-free gate after each run overlay is merged with the shared
``base`` mapping. ``Snakefile`` calls it as::

    astrogwb-validate-config inputs/experiments.yaml --output-dir outputs/configs

Writing ``save_config(RunConfig)`` rather than the raw merge keeps each run
config self-contained and diff-able: the file on disk always carries every
default filled in, so a run that inherits ``target_accept`` and one that
spells it out produce identical configs.

Like :mod:`astrogwb_paper.config.mcmc`, this module stays lightweight and
JAX-free, so a bad config fails before anything can initialize JAX.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from astrogwb_paper.config.experiments import (
    load_base,
    load_experiments,
    overlay_for,
)
from astrogwb_paper.config.mcmc import build_run_config, save_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge and validate every experiment run in the shared YAML inventory "
            "and write canonical, defaults-filled JSON configs."
        )
    )
    parser.add_argument(
        "config",
        type=Path,
        help="YAML inventory containing the shared base and all experiments.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for <experiment>/<run>.json outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base = load_base(args.config)
    for specification in load_experiments(args.config).values():
        for run in specification.runs:
            raw = overlay_for(specification, run, base=base)
            relative = Path(specification.name) / f"{run}.json"
            save_config(build_run_config(raw), args.output_dir / relative)


if __name__ == "__main__":
    main()
