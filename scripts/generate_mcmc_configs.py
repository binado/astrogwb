#!/usr/bin/env python3
"""Generate MCMC JSON configs for the three sweep campaigns.

Run from the repository root::

    uv run --extra mcmc python scripts/generate_mcmc_configs.py --force

Writes into ``configs/mcmc/{cosmology,modified-propagation,astrophysical}/``.
Base settings (fiducials, catalog, cosmology, sampler, runtime, output) are taken
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

from astrogwb.sampling.config import RunConfig, build_run_config, save_config

logger = logging.getLogger("generate_mcmc_configs")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = "configs/mcmc"
DEFAULT_EXAMPLE_CONFIG = REPO_ROOT / "configs" / "mcmc.example.toml"

# Fiducials mirrored from configs/mcmc.example.toml (for 1% Gaussian scales).
FIDUCIAL_H0 = 67.66
FIDUCIAL_LOCAL_MERGER_RATE = 161.0
RELATIVE_GAUSSIAN_SIGMA = 0.01

DETECTOR_NETWORKS: dict[str, tuple[str, ...]] = {
    "ET-triangular": ("E1", "E2", "E3"),
    "ET-triangular-CE-Hanford": ("E1", "E2", "E3", "C1"),
    "ET-2L-aligned": ("S1", "R1"),
    "ET-2L-aligned-CE-Hanford": ("S1", "R1", "C1"),
    "ET-2L-misaligned": ("S2", "R2"),
    "ET-2L-misaligned-CE-Hanford": ("S2", "R2", "C1"),
}

# Uniform bounds from mcmc.jl hyperprior_dists (Python parameter names).
PRIOR_TABLES: dict[str, dict[str, Any]] = {
    "H0": {"type": "uniform", "low": 20.0, "high": 140.0},
    "Omega_m": {"type": "uniform", "low": 0.05, "high": 0.95},
    "xi_0": {"type": "uniform", "low": 0.5, "high": 5.0},
    "xi_n": {"type": "uniform", "low": 0.3, "high": 3.0},
    "gamma": {"type": "uniform", "low": 0.5, "high": 10.0},
    "kappa": {"type": "uniform", "low": 0.05, "high": 10.0},
    "z_peak": {"type": "uniform", "low": 0.05, "high": 10.0},
    "local_merger_rate": {"type": "uniform", "low": 7.6, "high": 250.0},
}

# campaign -> sample_label -> (sampled_params, prior_overrides)
CAMPAIGNS: dict[str, dict[str, tuple[tuple[str, ...], dict[str, dict[str, Any]]]]] = {
    "cosmology": {
        "H0": (("H0",), {}),
        "H0-Omega_m": (("H0", "Omega_m"), {}),
        "H0-merger-rate": (("H0", "local_merger_rate"), {}),
        "H0-merger-rate-gauss": (
            ("H0", "local_merger_rate"),
            {
                "local_merger_rate": {
                    "type": "normal",
                    "loc": FIDUCIAL_LOCAL_MERGER_RATE,
                    "scale": RELATIVE_GAUSSIAN_SIGMA * FIDUCIAL_LOCAL_MERGER_RATE,
                },
            },
        ),
    },
    "modified-propagation": {
        "Xi_0": (("xi_0",), {}),
        "Xi_0-n": (("xi_0", "xi_n"), {}),
        "Xi_0-H0-gauss": (
            ("xi_0", "H0"),
            {
                "H0": {
                    "type": "normal",
                    "loc": FIDUCIAL_H0,
                    "scale": RELATIVE_GAUSSIAN_SIGMA * FIDUCIAL_H0,
                },
            },
        ),
    },
    "astrophysical": {
        "H0-peak": (("H0", "z_peak"), {}),
        "H0-MD": (("H0", "gamma", "kappa", "z_peak"), {}),
        "Xi_0-MD": (("xi_0", "gamma", "kappa", "z_peak"), {}),
    },
}


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
    raw["catalog"] = {**raw["catalog"], "detectors": list(detectors)}
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


def sweep_filenames() -> list[str]:
    """Return ``{campaign}/{network}__{sample}.json`` paths relative to sweep root."""
    return [
        f"{campaign}/{network_label}__{sample_label}.json"
        for campaign, sample_sets in CAMPAIGNS.items()
        for network_label in DETECTOR_NETWORKS
        for sample_label in sample_sets
    ]


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
