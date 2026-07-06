#!/usr/bin/env python3
"""Generate one MCMC JSON config per detector / sample_only sweep point.

Run from the repository root::

    uv run python scripts/generate_mcmc_configs.py [output_dir]

Base settings (fiducials, catalog, cosmology, sampler, runtime, output) are taken
from ``configs/mcmc.example.toml``. Sweep-specific fields override detectors,
``sampled_params``, and the corresponding prior tables.
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
DEFAULT_OUTPUT_DIR = "configs/mcmc/sweep"
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "mcmc.example.toml"

DETECTOR_NETWORKS: dict[str, tuple[str, ...]] = {
    "ET-triangular": ("E1", "E2", "E3"),
    "ET-triangular-CE-Hanford": ("E1", "E2", "E3", "C1"),
    "ET-2L-aligned": ("S1", "R1"),
    "ET-2L-aligned-CE-Hanford": ("S1", "R1", "C1"),
    "ET-2L-misaligned": ("S2", "R2"),
    "ET-2L-misaligned-CE-Hanford": ("S2", "R2", "C1"),
}

# Julia generate_mcmc_configs.jl sets minus w0 (unsupported in the Python runner).
SAMPLE_ONLY_SETS: dict[str, tuple[str, ...]] = {
    "H0": ("H0", "local_merger_rate"),
    "Omega_m": ("H0", "Omega_m", "local_merger_rate"),
    "modified-propagation": ("xi_0", "xi_n", "local_merger_rate"),
    "H0-MD": ("H0", "gamma", "kappa", "z_peak", "local_merger_rate"),
    "H0-peak": ("H0", "z_peak", "local_merger_rate"),
    "Xi_0-MD": ("xi_0", "gamma", "kappa", "z_peak", "local_merger_rate"),
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


def _resolve_output_dir(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def _base_from_example() -> dict[str, Any]:
    with EXAMPLE_CONFIG.open("rb") as handle:
        return tomllib.load(handle)


def make_config(
    detectors: tuple[str, ...],
    sampled_params: tuple[str, ...],
) -> RunConfig:
    """Build a validated run config for one sweep point."""
    raw = deepcopy(_base_from_example())
    raw["catalog"] = {**raw["catalog"], "detectors": list(detectors)}
    raw["sampled_params"] = list(sampled_params)
    raw["priors"] = {name: PRIOR_TABLES[name] for name in sampled_params}
    return build_run_config(raw)


def generate_configs(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    skip_existing: bool = True,
) -> tuple[Path, list[Path], list[Path]]:
    """Write sweep JSON configs; return (output_dir, written, skipped) paths."""
    resolved_output_dir = _resolve_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    skipped: list[Path] = []

    for network_label, detectors in DETECTOR_NETWORKS.items():
        for sample_label, sampled_params in SAMPLE_ONLY_SETS.items():
            filename = f"{network_label}__{sample_label}.json"
            path = resolved_output_dir / filename
            if skip_existing and path.exists():
                skipped.append(path)
                logger.info("skipping existing config %s", path)
                continue

            config = make_config(detectors, sampled_params)
            save_config(config, path)
            written.append(path)
            logger.info(
                "wrote config %s detectors=%s sampled_params=%s",
                path,
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
            "Generate one MCMC JSON config per detector / sample_only sweep point."
        )
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
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
    generate_configs(args.output_dir, skip_existing=not args.force)


if __name__ == "__main__":
    main()
