"""Validation for explicit, manually selected MCMC batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONFIG_SUFFIXES = {".json", ".toml"}


@dataclass(frozen=True)
class BatchRun:
    """One MCMC configuration selected for a campaign."""

    campaign: str
    config_path: Path
    run: str


@dataclass(frozen=True)
class McmcBatch:
    """Resolved inputs and output routing for one submitted MCMC batch."""

    catalog_id: str
    catalog_path: Path
    chains_dir: Path
    runs: tuple[BatchRun, ...]
    jax_platforms: str


def parse_mcmc_batch(raw: Mapping[str, Any]) -> McmcBatch:
    """Validate a Snakemake config mapping as an explicit MCMC batch."""
    try:
        catalog = raw["catalog"]
        catalog_id = str(catalog["id"])
        catalog_path = Path(catalog["path"])
        chains_dir = Path(raw["chains_dir"])
        raw_runs = raw["runs"]
        jax_platforms = str(raw.get("jax_platforms", "cuda"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "MCMC batch must define catalog.id, catalog.path, chains_dir, and runs"
        ) from exc

    _validate_identifier("catalog id", catalog_id)
    if not catalog_path.name:
        raise ValueError("catalog path must name a file")
    if not chains_dir.name:
        raise ValueError("chains_dir must not be empty")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("MCMC batch runs must be a non-empty list")
    if not jax_platforms:
        raise ValueError("jax_platforms must not be empty")

    runs: list[BatchRun] = []
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(raw_runs):
        try:
            campaign = str(entry["campaign"])
            config_path = Path(entry["config"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"MCMC batch run {index} must define campaign and config"
            ) from exc

        _validate_identifier("campaign", campaign)
        if config_path.suffix.lower() not in _CONFIG_SUFFIXES:
            raise ValueError(
                f"MCMC config {config_path} must have a .json or .toml extension"
            )
        run = config_path.stem
        _validate_identifier("run", run)
        key = (campaign, run)
        if key in seen:
            raise ValueError(f"duplicate MCMC run {campaign}/{run}")
        seen.add(key)
        runs.append(BatchRun(campaign, config_path, run))

    return McmcBatch(
        catalog_id=catalog_id,
        catalog_path=catalog_path,
        chains_dir=chains_dir,
        runs=tuple(runs),
        jax_platforms=jax_platforms,
    )


def _validate_identifier(field: str, value: str) -> None:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"invalid {field} {value!r}")
