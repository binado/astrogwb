"""Named waveform-catalog recipes and their deterministic workflow paths."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def load_catalog_recipes(path: Path) -> dict[str, CatalogRecipe]:
    """Load a committed TOML registry of named catalog recipes."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    catalogs = raw.get("catalogs")
    if not isinstance(catalogs, dict) or not catalogs:
        raise ValueError(f"catalog registry {path} must define [catalogs.<id>] tables")

    recipes: dict[str, CatalogRecipe] = {}
    for catalog_id, config in catalogs.items():
        if "/" in catalog_id or not catalog_id:
            raise ValueError(f"invalid catalog id {catalog_id!r}")
        if not isinstance(config, dict):
            raise ValueError(f"catalog {catalog_id!r} must be a table")
        recipes[catalog_id] = _recipe_from_mapping(catalog_id, config)
    return recipes


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
