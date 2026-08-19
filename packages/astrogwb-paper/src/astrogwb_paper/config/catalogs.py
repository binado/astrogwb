"""Typed waveform-catalog recipe loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrogwb_paper.config.loading import deep_merge, load_mapping
from astrogwb_paper.paths import paper_project_root

CATALOGS_PATH = Path("inputs/catalogs.yaml")
_INVENTORY_KEYS = frozenset({"base", "catalogs"})


@dataclass(frozen=True)
class CatalogRecipe:
    """Inputs required to build one reusable waveform catalog."""

    population_config: Path
    num_samples: int
    seed: int
    approximant: str
    sampling_frequency: float
    minimum_frequency: float
    maximum_frequency: float
    reference_frequency: float
    frequency_resolution: float
    chunk_size: int


def inventory_path(root: Path | None = None) -> Path:
    """Return the committed catalog recipe inventory path."""
    return (root or paper_project_root()) / CATALOGS_PATH


def load_inventory(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the top-level catalog YAML inventory."""
    resolved = path or inventory_path()
    raw = load_mapping(resolved)
    unknown = sorted(set(raw) - _INVENTORY_KEYS)
    if unknown:
        raise ValueError(f"{resolved} has unknown top-level keys: {', '.join(unknown)}")
    base = raw.get("base")
    catalogs = raw.get("catalogs")
    if not isinstance(base, Mapping) or not base:
        raise ValueError(f"{resolved} must define a non-empty base mapping")
    if not isinstance(catalogs, Mapping) or not catalogs:
        raise ValueError(f"{resolved} must define non-empty catalogs")
    return raw


def load_catalogs(path: Path | None = None) -> dict[str, CatalogRecipe]:
    """Load every committed catalog recipe from the YAML inventory."""
    resolved = path or inventory_path()
    raw = load_inventory(resolved)
    base = raw["base"]
    catalogs_raw = raw["catalogs"]
    assert isinstance(base, Mapping)
    assert isinstance(catalogs_raw, Mapping)
    catalogs: dict[str, CatalogRecipe] = {}
    for name, overlay in catalogs_raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{resolved} has an invalid catalog name")
        if overlay is not None and not isinstance(overlay, Mapping):
            raise TypeError(f"{resolved} catalog {name!r} must be a mapping")
        catalogs[name] = _recipe_from_mapping(name, deep_merge(base, overlay or {}))
    return catalogs


def catalog_recipe(name: str) -> CatalogRecipe:
    """Return a named catalog recipe or raise a user-facing error."""
    catalogs = load_catalogs()
    try:
        return catalogs[name]
    except KeyError:
        choices = ", ".join(catalogs)
        raise ValueError(f"unknown catalog {name!r}; choose from {choices}") from None


def _recipe_from_mapping(name: str, raw: Mapping[str, Any]) -> CatalogRecipe:
    try:
        population = raw["population"]
        waveform = raw["waveform"]
        return CatalogRecipe(
            population_config=Path(population["config"]),
            num_samples=int(population["num_samples"]),
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
        raise ValueError(f"invalid catalog recipe {name!r}") from exc
