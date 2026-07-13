#!/usr/bin/env python3
"""Generate MCMC JSON configs for the three sweep campaigns.

Run from the repository root::

    uv run --extra mcmc python scripts/generate_mcmc_configs.py --force

Writes into ``configs/mcmc/{cosmology,modified-propagation,astrophysical}/``.
Base settings (fiducials, analysis, cosmology, sampler, output) are taken
from the example TOML (default ``configs/mcmc.example.toml``). Campaign-specific
fields override detectors, ``sampled_params``, and prior tables. Requires the
``mcmc`` optional extra (pydantic) for ``RunConfig`` validation.

Pass ``--write-manifests`` to also (re)generate the Snakemake batch manifest
for each campaign (``configs/mcmc.batch.{campaign}.json``), listing every
config written for that campaign for use with ``workflow/mcmc.smk``::

    uv run --extra mcmc python scripts/generate_mcmc_configs.py --write-manifests --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrogwb.config.loading import load_mapping
from astrogwb.config.mcmc import RunConfig, build_run_config, save_config

logger = logging.getLogger("generate_mcmc_configs")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = "configs/mcmc"
DEFAULT_EXAMPLE_CONFIG = REPO_ROOT / "configs" / "mcmc.example.toml"
DEFAULT_SWEEP_SPEC = REPO_ROOT / "configs" / "mcmc.sweeps.toml"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "configs"
DEFAULT_CATALOG_ID = "bns-n16384-df1"
DEFAULT_CATALOG_PATH = "out/catalogs/bns-n16384-df1.h5"
DEFAULT_CHAINS_DIR = "chains"
DEFAULT_JAX_PLATFORMS = "cuda"


@dataclass(frozen=True)
class SweepSpec:
    """Networks, priors, and campaign runs used to generate MCMC configs."""

    networks: dict[str, tuple[str, ...]]
    priors: dict[str, dict[str, Any]]
    campaigns: dict[str, dict[str, tuple[tuple[str, ...], dict[str, dict[str, Any]]]]]


def load_sweep_spec(path: Path = DEFAULT_SWEEP_SPEC) -> SweepSpec:
    """Parse a sweep TOML file into the data required to generate its configs."""
    raw = load_mapping(path)
    networks = {name: tuple(detectors) for name, detectors in raw["networks"].items()}
    priors = deepcopy(raw["priors"])
    campaigns = {
        campaign: {
            label: (
                tuple(run["sampled_params"]),
                deepcopy(run.get("priors", {})),
            )
            for label, run in runs.items()
        }
        for campaign, runs in raw["campaigns"].items()
    }
    return SweepSpec(networks=networks, priors=priors, campaigns=campaigns)


def _resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def make_config(
    detectors: tuple[str, ...],
    sampled_params: tuple[str, ...],
    *,
    example_config: Path = DEFAULT_EXAMPLE_CONFIG,
    priors: dict[str, dict[str, Any]],
    prior_overrides: dict[str, dict[str, Any]] | None = None,
) -> RunConfig:
    """Build a validated run config for one sweep point."""
    raw = load_mapping(_resolve_repo_path(example_config))
    raw["analysis"] = {**raw["analysis"], "detectors": list(detectors)}
    raw["sampled_params"] = list(sampled_params)
    selected_priors = {name: deepcopy(priors[name]) for name in sampled_params}
    if prior_overrides:
        for name, override in prior_overrides.items():
            if name not in sampled_params:
                raise ValueError(
                    f"prior override for {name!r} but it is not in sampled_params"
                )
            selected_priors[name] = deepcopy(override)
    raw["priors"] = selected_priors
    return build_run_config(raw)


def render_manifest(
    campaign: str,
    run_configs: list[Path],
    *,
    catalog_id: str,
    catalog_path: str,
    chains_dir: str,
    jax_platforms: str,
) -> str:
    """Render a Snakemake batch manifest (``workflow/mcmc.smk`` schema) as JSON.

    Snakemake's ``--configfile`` loader (``snakemake.common.configfile``) tries
    JSON before YAML regardless of file extension, so JSON needs no hand-rolled
    escaping and no extra dependency (``pyyaml`` is not part of the ``mcmc``
    extra this script is documented to run under).
    """
    manifest = {
        "catalog": {"id": catalog_id, "path": catalog_path},
        "chains_dir": chains_dir,
        "jax_platforms": jax_platforms,
        "runs": [
            {"campaign": campaign, "config": path.as_posix()} for path in run_configs
        ],
    }
    return json.dumps(manifest, indent=2) + "\n"


def generate_configs(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    example_config: str | Path = DEFAULT_EXAMPLE_CONFIG,
    sweep_spec: SweepSpec | None = None,
    skip_existing: bool = True,
    write_manifests: bool = False,
    manifest_dir: str | Path = DEFAULT_MANIFEST_DIR,
    catalog_id: str = DEFAULT_CATALOG_ID,
    catalog_path: str = DEFAULT_CATALOG_PATH,
    chains_dir: str = DEFAULT_CHAINS_DIR,
    jax_platforms: str = DEFAULT_JAX_PLATFORMS,
) -> tuple[Path, list[Path], list[Path], list[Path], list[Path]]:
    """Write campaign JSON configs and, if requested, their batch manifests.

    Returns ``(output_dir, written, skipped, written_manifests, skipped_manifests)``.
    """
    resolved_output_dir = _resolve_repo_path(output_dir)
    resolved_example_config = _resolve_repo_path(example_config)
    resolved_manifest_dir = _resolve_repo_path(manifest_dir)
    sweep_spec = sweep_spec or load_sweep_spec()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    skipped: list[Path] = []
    written_manifests: list[Path] = []
    skipped_manifests: list[Path] = []

    for campaign, sample_sets in sweep_spec.campaigns.items():
        campaign_dir = resolved_output_dir / campaign
        campaign_dir.mkdir(parents=True, exist_ok=True)
        campaign_runs: list[Path] = []

        for network_label, detectors in sweep_spec.networks.items():
            for sample_label, (sampled_params, prior_overrides) in sample_sets.items():
                filename = f"{network_label}__{sample_label}.json"
                path = campaign_dir / filename
                campaign_runs.append(Path(os.path.relpath(path, REPO_ROOT)))
                if skip_existing and path.exists():
                    skipped.append(path)
                    logger.info("skipping existing config %s", path)
                    continue

                config = make_config(
                    detectors,
                    sampled_params,
                    example_config=resolved_example_config,
                    priors=sweep_spec.priors,
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

        if write_manifests:
            resolved_manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = resolved_manifest_dir / f"mcmc.batch.{campaign}.json"
            if skip_existing and manifest_path.exists():
                skipped_manifests.append(manifest_path)
                logger.info("skipping existing manifest %s", manifest_path)
                continue
            manifest_text = render_manifest(
                campaign,
                campaign_runs,
                catalog_id=catalog_id,
                catalog_path=catalog_path,
                chains_dir=chains_dir,
                jax_platforms=jax_platforms,
            )
            manifest_path.write_text(manifest_text, encoding="utf-8")
            written_manifests.append(manifest_path)
            logger.info(
                "wrote manifest %s campaign=%s runs=%d",
                manifest_path,
                campaign,
                len(campaign_runs),
            )

    logger.info(
        "done output_dir=%s written=%d skipped=%d written_manifests=%d skipped_manifests=%d",
        resolved_output_dir,
        len(written),
        len(skipped),
        len(written_manifests),
        len(skipped_manifests),
    )
    return resolved_output_dir, written, skipped, written_manifests, skipped_manifests


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
        "--sweep-spec",
        type=Path,
        default=DEFAULT_SWEEP_SPEC,
        help=f"Sweep campaign TOML (default: {DEFAULT_SWEEP_SPEC})",
    )
    parser.add_argument(
        "--example",
        type=Path,
        default=DEFAULT_EXAMPLE_CONFIG,
        help=("Base TOML template for fiducials, cosmology, sampler, and output."),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config files instead of skipping them.",
    )
    parser.add_argument(
        "--write-manifests",
        action="store_true",
        help=(
            "Also (re)generate a Snakemake batch manifest per campaign "
            "(configs/mcmc.batch.{campaign}.json)."
        ),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help=(
            "Directory for generated batch manifests "
            f"(default: {DEFAULT_MANIFEST_DIR}). Only used with --write-manifests."
        ),
    )
    parser.add_argument(
        "--catalog-id",
        default=DEFAULT_CATALOG_ID,
        help=f"Catalog id for generated manifests (default: {DEFAULT_CATALOG_ID}).",
    )
    parser.add_argument(
        "--catalog-path",
        default=DEFAULT_CATALOG_PATH,
        help=f"Catalog path for generated manifests (default: {DEFAULT_CATALOG_PATH}).",
    )
    parser.add_argument(
        "--chains-dir",
        default=DEFAULT_CHAINS_DIR,
        help=f"chains_dir for generated manifests (default: {DEFAULT_CHAINS_DIR}).",
    )
    parser.add_argument(
        "--jax-platforms",
        default=DEFAULT_JAX_PLATFORMS,
        help=f"jax_platforms for generated manifests (default: {DEFAULT_JAX_PLATFORMS}).",
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
        sweep_spec=load_sweep_spec(args.sweep_spec),
        skip_existing=not args.force,
        write_manifests=args.write_manifests,
        manifest_dir=args.manifest_dir,
        catalog_id=args.catalog_id,
        catalog_path=args.catalog_path,
        chains_dir=args.chains_dir,
        jax_platforms=args.jax_platforms,
    )


if __name__ == "__main__":
    main()
