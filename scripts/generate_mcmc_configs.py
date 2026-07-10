#!/usr/bin/env python3
"""Generate MCMC JSON configs for the three sweep campaigns.

Run from the repository root::

    uv run --extra mcmc python scripts/generate_mcmc_configs.py --force

Writes into ``configs/mcmc/{cosmology,modified-propagation,astrophysical}/``.
Base settings (fiducials, analysis, cosmology, sampler, runtime, output) are taken
from the example TOML (default ``configs/mcmc.example.toml``). Campaign-specific
fields override detectors, ``sampled_params``, and prior tables. Requires the
``mcmc`` optional extra (pydantic) for ``RunConfig`` validation.
"""

from __future__ import annotations

import argparse
import logging
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

from astrogwb.config.mcmc import RunConfig, build_run_config, save_config
from astrogwb.sampling.sweeps import (
    CAMPAIGNS,
    DETECTOR_NETWORKS,
    FIDUCIAL_H0,
    FIDUCIAL_LOCAL_MERGER_RATE,
    PRIOR_TABLES,
    RELATIVE_GAUSSIAN_SIGMA,
    sweep_filenames,
)

logger = logging.getLogger("generate_mcmc_configs")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = "configs/mcmc"
DEFAULT_EXAMPLE_CONFIG = REPO_ROOT / "configs" / "mcmc.example.toml"

def _resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def _resolve_output_dir(path: str | Path) -> Path:
    return _resolve_repo_path(path)


def _load_base_config(example_config: Path) -> dict[str, Any]:
    with example_config.open("rb") as handle:
        return tomllib.load(handle)


def make_config(
    detectors: tuple[str, ...],
    sampled_params: tuple[str, ...],
    *,
    example_config: Path = DEFAULT_EXAMPLE_CONFIG,
    prior_overrides: dict[str, dict[str, Any]] | None = None,
) -> RunConfig:
    """Build a validated run config for one sweep point."""
    raw = deepcopy(_load_base_config(_resolve_repo_path(example_config)))
    raw["analysis"] = {**raw["analysis"], "detectors": list(detectors)}
    raw["sampled_params"] = list(sampled_params)
    priors = {name: deepcopy(PRIOR_TABLES[name]) for name in sampled_params}
    if prior_overrides:
        for name, override in prior_overrides.items():
            if name not in sampled_params:
                raise ValueError(
                    f"prior override for {name!r} but it is not in sampled_params"
                )
            priors[name] = deepcopy(override)
    raw["priors"] = priors
    return build_run_config(raw)


def generate_configs(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    example_config: str | Path = DEFAULT_EXAMPLE_CONFIG,
    skip_existing: bool = True,
) -> tuple[Path, list[Path], list[Path]]:
    """Write campaign JSON configs; return (output_dir, written, skipped) paths."""
    resolved_output_dir = _resolve_output_dir(output_dir)
    resolved_example_config = _resolve_repo_path(example_config)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    skipped: list[Path] = []

    for campaign, sample_sets in CAMPAIGNS.items():
        campaign_dir = resolved_output_dir / campaign
        campaign_dir.mkdir(parents=True, exist_ok=True)

        for network_label, detectors in DETECTOR_NETWORKS.items():
            for sample_label, (sampled_params, prior_overrides) in sample_sets.items():
                filename = f"{network_label}__{sample_label}.json"
                path = campaign_dir / filename
                if skip_existing and path.exists():
                    skipped.append(path)
                    logger.info("skipping existing config %s", path)
                    continue

                config = make_config(
                    detectors,
                    sampled_params,
                    example_config=resolved_example_config,
                    prior_overrides=prior_overrides,
                )
                save_config(config, path)
                written.append(path)
                logger.info(
                    "wrote config %s campaign=%s detectors=%s sampled_params=%s",
                    path,
                    campaign,
                    detectors,
                    sampled_params,
                )

    logger.info(
        "done output_dir=%s written=%d skipped=%d",
        resolved_output_dir,
        len(written),
        len(skipped),
    )
    return resolved_output_dir, written, skipped


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MCMC JSON configs for the cosmology, "
            "modified-propagation, and astrophysical sweep campaigns."
        )
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help=f"MCMC configs root directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--example",
        type=Path,
        default=DEFAULT_EXAMPLE_CONFIG,
        help=(
            "Base TOML template for fiducials, catalog, cosmology, sampler, "
            "runtime, and output."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config files instead of skipping them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    generate_configs(
        args.output_dir,
        example_config=args.example,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
