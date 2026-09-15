"""PolarizationPowerCatalog configuration: what a catalog is made of, before it is made.

A *catalog* is one persisted waveform draw: expensive to build (population
draw + ripple waveform generation) and reused by every run that names it. Two
layers under ``config/catalogs/`` -- a shared ``base/`` and one
``defs/<name>.toml`` per catalog, whose stem is the name and which produces
``outputs/catalogs/<stem>.h5``. No registry file translates between the two.

This file describes a catalog only until it exists. Afterwards the *file* is
authoritative: it records its own registered population model, that model's
construction settings, the hyperparameters it was drawn at, and the density
factors included in importance weighting. Nothing here is re-read at analysis
time, and no run config restates any of it, so there is nothing for the two to
disagree about.

That replaced a genuinely fragile arrangement. Three partial records used to
describe one run -- the merged run TOML, a catalog attribute naming only the
*shape* of the redshift proposal, and a config object derived from those two --
reconciled by exact float equality over five hard-coded parameter names. Three
more parameters that change the answer (``xi_0``, ``xi_n``,
``local_merger_rate``) were checked by nothing at all.

Deliberately JAX-free at import: nothing here needs it. The registry lookup
that validates a model name imports :mod:`astrogwb.populations` inside its own
body. The load-bearing constraint is not this one, though -- it is that
:func:`astrogwb.paper.runtime.configure_runtime` runs before the XLA backend is
initialized, which ``tests/paper/test_cli.py`` guards directly.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb.paper.config.mcmc import RunConfig
from astrogwb.paper.config.runs import (
    CATALOG_DEFS_DIR,
    catalog_config_paths,
    discover_catalog_names,
    merge_config_layers,
)

logger = logging.getLogger(__name__)

_STRICT = ConfigDict(frozen=True, extra="forbid")


class WaveformConfig(BaseModel):
    """Waveform-generation settings for one catalog.

    Owned by ``config/waveform.json``; a catalog def may overlay fields, and
    only ``md-taylorf2-s41-n32768`` does (the approximant). The stored band
    matches ``config/analysis/base/model.toml``'s ``[analysis]`` ``f_min`` /
    ``f_max``. ``sampling_frequency`` is the backend Nyquist, not the stored
    grid. ``approximant="analytical"`` selects the closed-form inspiral
    through :func:`astrogwb.paper.config.waveform_generator`.
    """

    model_config = _STRICT

    approximant: str
    sampling_frequency: Annotated[float, Field(gt=0.0)]
    minimum_frequency: Annotated[float, Field(ge=0.0)]
    maximum_frequency: Annotated[float, Field(gt=0.0)]
    reference_frequency: Annotated[float, Field(gt=0.0)]
    frequency_resolution: Annotated[float, Field(gt=0.0)]


class PopulationConfig(BaseModel):
    """The population declaration a catalog is drawn from, and drawn at.

    ``source_model`` and ``rate_model`` are keys in the
    :mod:`astrogwb.populations` source and merger-rate registries, never
    import paths: registry keys change only on purpose, while module paths
    move as collateral whenever a module is reorganized. Composing the two
    independently is what lets a catalog def pair a guard-mixture redshift
    law with the same rate a physical population uses.

    ``kwargs`` are construction settings routed to *both* models -- the
    redshift window and grid, the one correctness-relevant overlap between
    them (see ``astrogwb.populations.SHARED_MODEL_KWARGS``). ``source_kwargs``
    are settings the source model alone takes, such as a guard mixture's
    ``uniform_mixing_fraction``. Both must be JSON-serializable scalars,
    because that is how they travel in the catalog's HDF5 attributes, merged
    into one flat mapping. ``params`` are the hyperparameters the draw is made
    at, and they are what a target evaluation is compared against;
    ``local_merger_rate`` belongs here even though it does not affect the
    normalized source draws, because the observation's total rate is
    reconstructed from it.

    Density factors and source outputs are declared by the registered source
    model class. Generation records its effective density selection.
    """

    model_config = _STRICT

    source_model: str
    rate_model: str
    kwargs: dict[str, float | int] = Field(default_factory=dict)
    source_kwargs: dict[str, float | int] = Field(default_factory=dict)
    params: dict[str, float]


class CatalogDefinition(BaseModel):
    """Everything needed to generate one reusable catalog.

    ``seed`` and ``num_samples`` live here rather than beside the population
    declaration on purpose: ``md-imrphenom-s41-n32768`` and
    ``md-imrphenom-s42-n16384`` are the *same* population drawn at different
    seeds and sizes.
    """

    model_config = _STRICT

    name: str
    seed: int
    num_samples: Annotated[int, Field(gt=0)]
    population: PopulationConfig
    waveform: WaveformConfig

    @model_validator(mode="after")
    def _validate_redshift_window(self) -> CatalogDefinition:
        kwargs = self.population.kwargs
        window = ("z_min", "z_max")
        if all(name in kwargs for name in window) and not float(
            kwargs["z_min"]
        ) < float(kwargs["z_max"]):
            raise ValueError(
                "population.kwargs.z_min must be less than population.kwargs.z_max"
            )
        return self


def load_catalog_layers(paths: Sequence[Path]) -> CatalogDefinition:
    """Merge and validate one catalog from its ordered layer files.

    The last layer is the catalog's own definition, and its filename stem is
    the catalog name -- the same convention that maps it to
    ``outputs/catalogs/<stem>.h5``. Taking layers on argv rather than a name
    keeps the generator symmetric with every other entrypoint: the workflow
    rule declares exactly these files as its ``input:``.
    """
    if not paths:
        raise ValueError("no catalog config layers given")
    merged = merge_config_layers(paths)
    return CatalogDefinition.model_validate({"name": Path(paths[-1]).stem, **merged})


def load_catalog_definition(name: str, root: Path | None = None) -> CatalogDefinition:
    """Merge and validate one catalog's layers, addressing it by name."""
    return load_catalog_layers(catalog_config_paths(name, root=root))


def discover_catalogs(root: Path | None = None) -> dict[str, CatalogDefinition]:
    """Load every committed catalog config, keyed by name, in sorted order."""
    return {
        name: load_catalog_definition(name, root)
        for name in discover_catalog_names(root)
    }


def check_source_model(name: str, *, label: str) -> None:
    """Reject a source-model name that is not registered.

    Imports :mod:`astrogwb.populations` in its own body: the registry is
    populated by importing the models, which pulls in JAX, and this module is
    otherwise free of it.
    """
    from astrogwb.populations import known_source_models

    known = known_source_models()
    if name not in known:
        raise ValueError(
            f"{label}: unknown source model {name!r}; registered models are: "
            f"{', '.join(known)}"
        )


def check_rate_model(name: str, *, label: str) -> None:
    """Reject a merger-rate-model name that is not registered."""
    from astrogwb.populations import known_merger_rate_models

    known = known_merger_rate_models()
    if name not in known:
        raise ValueError(
            f"{label}: unknown merger-rate model {name!r}; registered models are: "
            f"{', '.join(known)}"
        )


def check_catalog_references(
    config: RunConfig,
    *,
    label: str,
    catalogs: Mapping[str, CatalogDefinition] | None = None,
) -> None:
    """Reject a run naming an unknown catalog.

    Runs against the committed catalog configs rather than the built files, so
    a typo fails without building anything expensive. Without it the typo would
    only surface as a Snakemake wildcard that matches no rule.

    Lives here rather than in :mod:`astrogwb.paper.config.runs` because it
    needs both a validated ``RunConfig`` and the catalog registry, and that
    module must stay importable by the ``Snakefile`` without pydantic.
    """
    known = catalogs if catalogs is not None else discover_catalogs()
    for role, name in (
        ("injection", config.catalog.injection),
        ("proposal", config.catalog.proposal),
    ):
        if name not in known:
            choices = ", ".join(known)
            raise ValueError(
                f"{label} catalog.{role} names unknown catalog {name!r}; "
                f"choose from {choices}"
            )


def validate_all_runs(root: Path | None = None) -> list[str]:
    """Merge, validate, and catalog-check every declared run; return their labels.

    The pre-flight gate ``astrogwb-assemble-config --all`` used to provide,
    kept because its real value was never the JSON it wrote: it fails on the
    first invalid run *before any catalog is built*, and a catalog is a GPU job.
    Source and rate models are resolved here too, so an unregistered name is caught
    by the same pre-flight rather than at the top of a queued generation job.

    It lives here rather than in :mod:`astrogwb.paper.config.runs` for the same
    reason :func:`check_catalog_references` does -- that module must stay
    importable by the ``Snakefile`` without pydantic. Catalog configs are loaded
    once: this is one gate over all runs, not a per-run check.
    """
    from astrogwb.paper.config.mcmc import build_run_config
    from astrogwb.paper.config.runs import assemble_run, discover_runs

    catalogs = discover_catalogs(root)
    for name, definition in catalogs.items():
        check_source_model(
            definition.population.source_model,
            label=f"catalog {name!r} population.source_model",
        )
        check_rate_model(
            definition.population.rate_model,
            label=f"catalog {name!r} population.rate_model",
        )
    labels: list[str] = []
    for experiment, runs in discover_runs(root).items():
        for run in runs:
            label = f"{experiment}/{run}"
            config = build_run_config(assemble_run(experiment, run, root=root))
            check_catalog_references(config, label=label, catalogs=catalogs)
            check_source_model(
                config.analysis.source_model,
                label=f"{label} analysis.source_model",
            )
            check_rate_model(
                config.analysis.rate_model,
                label=f"{label} analysis.rate_model",
            )
            logger.info("ok %s", label)
            labels.append(label)
    return labels


__all__ = [
    "CATALOG_DEFS_DIR",
    "CatalogDefinition",
    "PopulationConfig",
    "WaveformConfig",
    "check_catalog_references",
    "check_rate_model",
    "check_source_model",
    "discover_catalogs",
    "load_catalog_definition",
    "load_catalog_layers",
    "validate_all_runs",
]
