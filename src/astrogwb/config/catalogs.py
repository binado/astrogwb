"""Typed waveform-catalog recipe loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class CatalogRecipe:
    """Inputs required to build one reusable waveform catalog."""

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


def load_catalog_recipe(path: Path) -> CatalogRecipe:
    """Load one catalog recipe from TOML."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    return _recipe_from_mapping(path.stem, raw)


def _recipe_from_mapping(recipe_name: str, config: dict[str, Any]) -> CatalogRecipe:
    try:
        population = config["population"]
        waveform = config["waveform"]
        return CatalogRecipe(
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
        raise ValueError(f"invalid catalog recipe {recipe_name!r}") from exc
