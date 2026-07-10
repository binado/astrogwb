#!/usr/bin/env python3
"""Freeze an editable MCMC campaign inventory into an immutable lock file."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from astrogwb.sampling.campaign import materialize_campaign


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path, help="Versioned campaign TOML inventory")
    parser.add_argument("--lock", type=Path, help="Lock path (defaults beside inventory)")
    parser.add_argument("--frozen-dir", type=Path, help="Directory for generated configs")
    parser.add_argument("--manifest", type=Path, help="Deterministic SLURM array manifest")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    lock, configs, manifest = materialize_campaign(
        args.inventory,
        lock_path=args.lock,
        frozen_dir=args.frozen_dir,
        manifest_path=args.manifest,
    )
    logging.info("campaign %s: materialized %d configs", lock.campaign_id, len(configs))
    logging.info("manifest: %s", manifest)


if __name__ == "__main__":
    main()
