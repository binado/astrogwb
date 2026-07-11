"""Waveform-catalog recipes and their deterministic workflow paths."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


_CATALOG_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class CatalogRecipe:
    """Inputs required to build one reusable waveform catalog."""

    catalog_id: str
    population_config: Path
    n_samples: int
    seed: int
    approximant: str
    sampling_frequency: float
    minimum_frequency: float
    maximum_frequency: float
    reference_frequency: float
    frequency_resolution: float
    chunk_size: int


def population_path(catalog_id: str) -> Path:
    """Workflow output path for a named population realization."""
    return Path("out/populations") / f"{catalog_id}.h5"


def catalog_path(catalog_id: str) -> Path:
    """Workflow output path for a named waveform catalog."""
    return Path("out/catalogs") / f"{catalog_id}.h5"


def load_catalog_recipe(path: Path) -> CatalogRecipe:
    """Load one catalog recipe, using the TOML filename as its catalog ID."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    catalog_id = path.stem
    if not _CATALOG_ID_PATTERN.fullmatch(catalog_id):
        raise ValueError(f"invalid catalog id {catalog_id!r}")
    if not isinstance(raw, dict):
        raise ValueError(f"catalog recipe {path} must be a TOML table")
    return _recipe_from_mapping(catalog_id, raw)


def _recipe_from_mapping(catalog_id: str, config: dict[str, Any]) -> CatalogRecipe:
    try:
        population = config["population"]
        waveform = config["waveform"]
        return CatalogRecipe(
            catalog_id=catalog_id,
            population_config=Path(population["config"]),
            n_samples=int(population["n_samples"]),
            seed=int(population["seed"]),
            approximant=str(waveform["approximant"]),
            sampling_frequency=float(waveform["sampling_frequency"]),
            minimum_frequency=float(waveform["minimum_frequency"]),
            maximum_frequency=float(waveform["maximum_frequency"]),
            reference_frequency=float(waveform["reference_frequency"]),
            frequency_resolution=float(waveform["frequency_resolution"]),
            chunk_size=int(waveform["chunk_size"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid catalog recipe {catalog_id!r}") from exc
