"""Assemble a canonical MCMC run config from the shared base and one experiment run.

This is the JAX-free gate after a run overlay is merged with
``inputs/mcmc.base.toml``. ``workflow/mcmc.smk`` calls it as::

    astrogwb-validate-config --base inputs/mcmc.base.toml \\
        --run ET-triangular experiments/H0-all-detectors.toml -o run.json

Writing ``save_config(RunConfig)`` rather than the raw merge is what keeps
:func:`~astrogwb_paper.config.mcmc.config_sha256` a stable identity for "same
inference settings": the file on disk always carries every default filled in,
so a run that inherits ``target_accept`` and one that spells it out produce
identical configs and identical digests.

Like :mod:`astrogwb_paper.config.mcmc`, this module is stdlib + pydantic only,
so a bad config fails before anything can initialize JAX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from astrogwb_paper.config.experiments import load_experiment, overlay_for
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.config.mcmc import build_run_config, save_config

STDIN = "-"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge an experiment run with the shared MCMC base (or validate an "
            "already-merged mapping) and write canonical, defaults-filled JSON."
        )
    )
    parser.add_argument(
        "config",
        help=(
            "Experiment TOML when --base/--run are set, otherwise a merged "
            f"TOML or JSON config, or {STDIN!r} for JSON on stdin."
        ),
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Shared MCMC base TOML merged before the selected run overlay.",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="Run id inside the experiment TOML's [runs] table.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Destination for the canonical JSON config.",
    )
    args = parser.parse_args(argv)
    if (args.base is None) != (args.run is None):
        parser.error("--base and --run must be used together")
    return args


def load_raw(config: str) -> dict[str, Any]:
    """Read a merged config mapping, from a path or stdin."""
    if config == STDIN:
        return json.load(sys.stdin)
    return load_mapping(Path(config))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.base is not None:
        spec = load_experiment(Path(args.config))
        raw = overlay_for(spec, args.run, base=load_mapping(args.base))
    else:
        raw = load_raw(args.config)
    save_config(build_run_config(raw), args.output)


if __name__ == "__main__":
    main()
