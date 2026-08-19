"""Typed source, production-population, and waveform-catalog recipes.

Like :mod:`astrogwb_paper.config.loading`, this module imports neither JAX nor
``astrogwb``: only stdlib, PyYAML, and pydantic. That is load-bearing rather
than tidy. ``Snakefile`` imports it and calls :func:`load_catalog_inventory` at
*workflow-parse* time -- on every dry run and every job dispatch -- and
:mod:`astrogwb_paper.config.experiments` reaches it from the JAX-free
``astrogwb-validate-config`` path, so a heavy import here would be paid many
times over and would defeat the runtime-before-JAX ordering the CLIs rely on.

Recipes are validated declaratively (``extra="forbid"``): a mistyped key fails
at parse time instead of silently falling back to a default and generating a
catalog nobody asked for.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb_paper.config.loading import deep_merge, load_inventory
from astrogwb_paper.paths import paper_project_root

CATALOGS_PATH = Path("inputs/catalogs.yaml")
INJECTION_CATALOG_NAME = "injection-bns-n32768"
_INVENTORY_SECTIONS = ("base", "sources", "injection", "catalogs")
AssemblyOperation = Literal["identity", "subsample", "mixture"]
ASSEMBLY_OPERATIONS: tuple[AssemblyOperation, ...] = get_args(AssemblyOperation)
"""Runtime view of :data:`AssemblyOperation`, for argparse choices and checks.

Derived from the type rather than restated, so the CLI, the recipe validator,
and the assembly dispatch can never disagree about the supported operations.
"""

_STRICT = ConfigDict(frozen=True, extra="forbid")
_ALIASED: ConfigDict = {**_STRICT, "validate_by_name": True}


class SourcePopulationRecipe(BaseModel):
    """One finite population pool generated directly by ``gwmock-pop``."""

    model_config = _ALIASED

    name: str
    population_config: Path = Field(alias="config")
    num_samples: Annotated[int, Field(gt=0)]
    seed: int


class AssemblyComponent(BaseModel):
    """One weighted source pool used to assemble a production population."""

    model_config = _STRICT

    source: str
    weight: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 1.0


class ProductionPopulationRecipe(BaseModel):
    """A production population assembled from one or more source pools."""

    model_config = _STRICT

    operation: AssemblyOperation
    num_samples: Annotated[int, Field(gt=0)]
    seed: int
    components: Annotated[tuple[AssemblyComponent, ...], Field(min_length=1)]
    uniform_redshift_fraction: Annotated[float, Field(gt=0.0, lt=1.0)] | None = None


class WaveformRecipe(BaseModel):
    """Waveform settings shared by injection and proposal production populations."""

    model_config = _STRICT

    approximant: str
    sampling_frequency: float
    minimum_frequency: float
    maximum_frequency: float
    reference_frequency: float
    frequency_resolution: float
    chunk_size: int


class CatalogRecipe(BaseModel):
    """One production population and the waveform catalog generated from it."""

    model_config = _STRICT

    name: str
    production: ProductionPopulationRecipe
    waveform: WaveformRecipe

    @model_validator(mode="after")
    def _validate_components(self) -> CatalogRecipe:
        production = self.production
        if production.operation != "mixture" and len(production.components) != 1:
            raise ValueError(
                f"{self.name} {production.operation} production requires "
                "exactly one component"
            )
        # Pydantic collects validators in MRO order, so this runs before the
        # subclass ones: ProposalCatalogRecipe can normalize by the weight
        # total without guarding against a zero divisor.
        if sum(component.weight for component in production.components) <= 0.0:
            raise ValueError(
                f"{self.name} production component weights must sum to > 0"
            )
        return self


class InjectionCatalogRecipe(CatalogRecipe):
    """The independent injection catalog, drawn from the true population."""

    @model_validator(mode="after")
    def _reject_proposal_density(self) -> InjectionCatalogRecipe:
        if self.production.uniform_redshift_fraction is not None:
            raise ValueError(
                f"{self.name} injection production cannot define "
                "uniform_redshift_fraction"
            )
        return self


class ProposalCatalogRecipe(CatalogRecipe):
    """A proposal catalog: a guarded mixture with a known proposal density."""

    @model_validator(mode="after")
    def _require_proposal_density(self) -> ProposalCatalogRecipe:
        epsilon = self.production.uniform_redshift_fraction
        if epsilon is None:
            raise ValueError(
                f"{self.name} production must define uniform_redshift_fraction"
            )
        weights = [component.weight for component in self.production.components]
        total = sum(weights)
        normalized = [weight / total for weight in weights]
        if len(normalized) != 2 or not math.isclose(
            normalized[1], epsilon, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"{self.name} component weights must match uniform_redshift_fraction"
            )
        return self


class CatalogInventory(BaseModel):
    """Complete catalog DAG declared by the committed inventory."""

    model_config = _STRICT

    sources: dict[str, SourcePopulationRecipe]
    injection: InjectionCatalogRecipe
    catalogs: dict[str, ProposalCatalogRecipe]


def inventory_path(root: Path | None = None) -> Path:
    """Return the committed catalog recipe inventory path."""
    return (root or paper_project_root()) / CATALOGS_PATH


def load_catalog_inventory(path: Path | None = None) -> CatalogInventory:
    """Load and cross-validate the complete committed catalog DAG."""
    resolved = path or inventory_path()
    raw = load_inventory(resolved, required=_INVENTORY_SECTIONS)
    base = raw["base"]
    sources_raw = raw["sources"]
    injection_raw = raw["injection"]
    catalogs_raw = raw["catalogs"]
    assert isinstance(base, Mapping)
    assert isinstance(sources_raw, Mapping)
    assert isinstance(injection_raw, Mapping)
    assert isinstance(catalogs_raw, Mapping)

    sources: dict[str, SourcePopulationRecipe] = {}
    for name, value in sources_raw.items():
        source = SourcePopulationRecipe.model_validate(_named(name, value))
        sources[source.name] = source

    # `base` is an overlay template, not a recipe of its own (it declares no
    # `num_samples`), so merging happens on raw mappings and validation runs
    # once, afterwards -- mirroring `build_run_config` in config.mcmc.
    injection = InjectionCatalogRecipe.model_validate(
        _named("injection", deep_merge(base, injection_raw))
    )
    _require_known_sources(injection, sources)

    catalogs: dict[str, ProposalCatalogRecipe] = {}
    for name, overlay in catalogs_raw.items():
        # Guard before deep_merge, which would AttributeError on a scalar.
        if overlay is not None and not isinstance(overlay, Mapping):
            raise TypeError(f"{resolved} catalog {name!r} must be a mapping")
        catalog = ProposalCatalogRecipe.model_validate(
            _named(name, deep_merge(base, overlay or {}))
        )
        _require_known_sources(catalog, sources)
        catalogs[catalog.name] = catalog
    return CatalogInventory(sources=sources, injection=injection, catalogs=catalogs)


def load_sources(path: Path | None = None) -> dict[str, SourcePopulationRecipe]:
    """Load the named source pools generated by ``gwmock-pop``."""
    return load_catalog_inventory(path).sources


def injection_recipe(path: Path | None = None) -> InjectionCatalogRecipe:
    """Load the shared independent injection recipe."""
    return load_catalog_inventory(path).injection


def load_catalogs(path: Path | None = None) -> dict[str, ProposalCatalogRecipe]:
    """Load every named proposal waveform-catalog recipe."""
    return load_catalog_inventory(path).catalogs


def catalog_recipe(name: str) -> ProposalCatalogRecipe:
    """Return a named proposal recipe or raise a user-facing error."""
    catalogs = load_catalogs()
    try:
        return catalogs[name]
    except KeyError:
        choices = ", ".join(catalogs)
        raise ValueError(f"unknown catalog {name!r}; choose from {choices}") from None


def _named(name: Any, raw: Any) -> Any:
    """Prepend the inventory key as the recipe's ``name`` field.

    Non-mappings pass through untouched so pydantic reports the type error
    against the recipe itself rather than raising a different one here.
    """
    if isinstance(raw, Mapping):
        return {"name": name, **raw}
    return raw


def _require_known_sources(
    recipe: CatalogRecipe, sources: Mapping[str, SourcePopulationRecipe]
) -> None:
    """Check every component against the declared source pools.

    Genuinely inter-model -- a recipe cannot see its siblings -- so this stays
    in the loader rather than becoming a field validator.
    """
    for component in recipe.production.components:
        if component.source not in sources:
            choices = ", ".join(sources)
            raise ValueError(
                f"{recipe.name} production names unknown source "
                f"{component.source!r}; choose from {choices}"
            )
