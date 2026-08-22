"""Typed loading for the catalog-centric generation inventory."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from astrogwb_paper.config.loading import deep_merge, load_inventory, load_mapping
from astrogwb_paper.paths import paper_project_root

CATALOGS_PATH = Path("inputs/catalogs.yaml")
INJECTION_CATALOG_NAME = "injection-bns-n32768-eps=0-df1"
_INVENTORY_SECTIONS = ("base", "catalogs")
_STRICT = ConfigDict(frozen=True, extra="forbid")


class PopulationRecipe(BaseModel):
    """Population graph inputs and sampling settings for one catalog."""

    model_config = _STRICT

    base_config: Path
    md_redshift_config: Path
    uniform_redshift_config: Path
    num_samples: Annotated[int, Field(gt=0)]
    seed: int
    uniform_mixing_fraction: Annotated[
        float, Field(ge=0.0, le=1.0, allow_inf_nan=False)
    ]


class WaveformRecipe(BaseModel):
    """Waveform settings shared by generated catalogs."""

    model_config = _STRICT

    approximant: str
    sampling_frequency: Annotated[float, Field(gt=0.0)]
    minimum_frequency: Annotated[float, Field(ge=0.0)]
    maximum_frequency: Annotated[float, Field(gt=0.0)]
    reference_frequency: Annotated[float, Field(gt=0.0)]
    frequency_resolution: Annotated[float, Field(gt=0.0)]
    chunk_size: Annotated[int, Field(gt=0)]


class CatalogRecipe(BaseModel):
    """Everything needed to generate one reusable waveform catalog."""

    model_config = _STRICT

    name: str
    population: PopulationRecipe
    waveform: WaveformRecipe


def inventory_path(root: Path | None = None) -> Path:
    """Return the committed catalog inventory path."""
    return (root or paper_project_root()) / CATALOGS_PATH


def load_catalogs(path: Path | None = None) -> dict[str, CatalogRecipe]:
    """Load and validate every committed catalog recipe."""
    resolved = path or inventory_path()
    raw = load_inventory(resolved, required=_INVENTORY_SECTIONS)
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
        catalogs[name] = CatalogRecipe.model_validate(
            {"name": name, **deep_merge(base, overlay or {})}
        )
    return catalogs


def catalog_recipe(name: str, path: Path | None = None) -> CatalogRecipe:
    """Return a named catalog recipe or raise a user-facing error."""
    catalogs = load_catalogs(path)
    try:
        return catalogs[name]
    except KeyError:
        choices = ", ".join(catalogs)
        raise ValueError(f"unknown catalog {name!r}; choose from {choices}") from None


def proposal_config(
    recipe: CatalogRecipe, *, root: Path | None = None
) -> dict[str, float | int]:
    """Expand the fixed redshift proposal represented by ``recipe``."""
    project_root = root or paper_project_root()
    population = recipe.population
    md = load_mapping(project_root / population.md_redshift_config)
    md_parameters = _mapping(md.get("parameters"), label="MD parameters")
    redshift = _mapping(md_parameters.get("redshift"), label="MD redshift parameter")
    sampler = _mapping(redshift.get("sampler"), label="MD redshift sampler")
    md_sampler = _mapping(
        sampler.get("arguments"), label="MD redshift sampler arguments"
    )

    minimum_redshift = float(md_sampler["z_min"])
    maximum_redshift = float(md_sampler["z_max"])
    h0 = float(md_sampler["hubble_constant"])
    omega_m = float(md_sampler["omega_m"])

    return {
        "uniform_mixing_fraction": population.uniform_mixing_fraction,
        "minimum_redshift": minimum_redshift,
        "maximum_redshift": maximum_redshift,
        "n_grid": int(md_sampler.get("n_grid", 4096)),
        "H0": h0,
        "Omega_m": omega_m,
        "gamma": float(md_sampler["gamma"]),
        "kappa": float(md_sampler["kappa"]),
        "z_peak": float(md_sampler["z_peak"]),
    }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Narrow an arbitrary loaded YAML value to a mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value
