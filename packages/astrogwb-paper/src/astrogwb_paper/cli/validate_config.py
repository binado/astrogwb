"""Validate a merged MCMC config and write it back in canonical form.

This is the gate after a run override is merged with the shared MCMC base.
``workflow/mcmc.smk`` pipes ``knf``'s merge straight into it::

    knf <base> <run> --strict -f json | astrogwb-validate-config - --output run.json

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

from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.config.mcmc import build_run_config, save_config

STDIN = "-"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a merged MCMC config as a RunConfig and write it back as "
            "canonical, defaults-filled JSON."
        )
    )
    parser.add_argument(
        "config",
        help=f"TOML or JSON config to validate, or {STDIN!r} for JSON on stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Destination for the canonical JSON config.",
    )
    return parser.parse_args(argv)


def load_raw(config: str) -> dict[str, Any]:
    """Read the config mapping to validate, from a path or stdin."""
    if config == STDIN:
        return json.load(sys.stdin)
    return load_mapping(Path(config))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    save_config(build_run_config(load_raw(args.config)), args.output)


if __name__ == "__main__":
    main()
