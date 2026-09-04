"""Validate every declared run config before anything expensive is built.

This is the pre-flight gate ``astrogwb-assemble-config --all`` used to provide.
There is no assembled-config artifact any more -- each entrypoint merges its
own layers -- but the gate is worth keeping on its own: it fails on the first
invalid run *before any bank is built*, and a bank is a GPU job.

Three checks per run, cheapest first:

1. the three layers merge (a malformed TOML fails here);
2. the merge validates into a :class:`RunConfig` (a typo'd key, an impossible
   prior, a missing detector list);
3. every bank the run names exists in ``config/banks/``, and a two-component
   mixture draws its parts with distinct seeds.

JAX-free: nothing here touches a device, so it is cheap enough to run before
every campaign.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from astrogwb_paper.config.banks import validate_all_runs

logger = logging.getLogger("validate_configs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge and validate every config/analysis/ run, and check the banks "
            "each one names, without building anything."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root holding config/ (default: the checkout).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write a stamp file listing the validated runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    try:
        labels = validate_all_runs(args.root)
    except (ValueError, TypeError) as error:
        # A config error is the expected failure here, and a traceback through
        # pydantic buries the message that says which run is wrong.
        logger.error("%s", error)
        raise SystemExit(1) from None

    logger.info("Validated %d runs", len(labels))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(labels) + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    sys.exit(main())
