"""PolarizationPowerCatalog configuration: what a catalog is made of, before it is made.

A *catalog* is one persisted waveform draw: expensive to build (population
draw + ripple waveform generation) and reused by every run that names it. Two
layers under ``config/catalogs/`` -- a shared ``base/`` and one
``defs/<name>.toml`` per catalog, whose stem is the name and which produces
``outputs/catalogs/<stem>.h5``. No registry file translates between the two.

This file describes a catalog only until it exists. Afterwards the *file* is
authoritative: it records its own registered population model, that model's
construction kwargs, the hyperparameters it was drawn at, and the density
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
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb.paper.config.mcmc import RunConfig
from astrogwb.paper.config.runs import (
    CATALOG_DEFS_DIR,
    catalog_config_paths,
    discover_catalog_names,
    merge_config_layers,
)

if TYPE_CHECKING:
    from astrogwb.populations import PopulationRecord
    from astrogwb.waveform import PolarizationPowerGenerator

logger = logging.getLogger(__name__)

_STRICT = ConfigDict(frozen=True, extra="forbid")

#: The one ``approximant`` that is not a Ripple name: it selects the
#: closed-form inspiral, which is the only generator taking an ``alpha``.
_ANALYTICAL_APPROXIMANT = "AnalyticInspiral"

#: Spellings close enough to :data:`_ANALYTICAL_APPROXIMANT` to be meant as it.
#: They are rejected by name rather than passed through, because the failure
#: they would otherwise cause is silent: anything that is not the canonical
#: spelling is treated as a Ripple approximant, so a near miss selects the
#: wrong backend instead of the wrong-looking one. Checked as a deny list
#: rather than against Ripple's own catalogue, which cannot be consulted
#: without reaching JAX.
_CONFUSABLE_ANALYTICAL_APPROXIMANTS = frozenset(
    {"analytical", "analytic", "Analytic", "AnalyticalInspiral", "analytic_inspiral"}
)


class WaveformConfig(BaseModel):
    """Waveform-generation settings for one catalog, and the generator they build.

    Owned by ``config/waveform.json``; a catalog def may overlay fields, and
    only ``md-taylorf2-s41-n32768`` does (the approximant). The stored band
    matches ``config/analysis/base/model.toml``'s ``[analysis]`` ``f_min`` /
    ``f_max``. ``sampling_frequency`` is the backend Nyquist, not the stored
    grid. ``approximant="AnalyticInspiral"`` selects the closed-form inspiral.

    This is the wire format for
    :class:`~astrogwb.waveform.PolarizationPowerGenerator`, and :meth:`build`
    is the one edge between them. The core generator stays a frozen dataclass
    -- it is closed over by ``jax.jit`` and its concrete subclasses hold a
    compiled kernel -- so validation of the *settings* lives here and the
    domain invariants stay in the dataclass.
    """

    model_config = _STRICT

    approximant: str
    sampling_frequency: Annotated[float, Field(gt=0.0)]
    minimum_frequency: Annotated[float, Field(ge=0.0)]
    maximum_frequency: Annotated[float, Field(gt=0.0)]
    reference_frequency: Annotated[float, Field(gt=0.0)]
    frequency_resolution: Annotated[float, Field(gt=0.0)]
    #: The inspiral termination constant, valid only for the closed-form
    #: approximant. Unset means
    #: :data:`~astrogwb.constants.ISCO_ALPHA`; nothing else defaults to it.
    #:
    #: It is a field of ``AnalyticInspiralGenerator`` and not of the base
    #: descriptor, so it is *not* among the attributes a catalog persists. An
    #: analytical catalog records the band it was drawn on but not the alpha
    #: that terminated it, because a loaded catalog is rebuilt as the base
    #: descriptor either way.
    alpha: Annotated[float, Field(gt=0.0)] | None = None

    @model_validator(mode="after")
    def _validate_approximant(self) -> WaveformConfig:
        if self.approximant in _CONFUSABLE_ANALYTICAL_APPROXIMANTS:
            raise ValueError(
                f"waveform.approximant {self.approximant!r} is not a Ripple "
                f"approximant; the closed-form inspiral is spelled "
                f"{_ANALYTICAL_APPROXIMANT!r}"
            )
        if self.alpha is not None and self.approximant != _ANALYTICAL_APPROXIMANT:
            raise ValueError(
                f"waveform.alpha is only valid when "
                f"approximant == {_ANALYTICAL_APPROXIMANT!r}; "
                f"this catalog names {self.approximant!r}"
            )
        return self

    def build(self) -> PolarizationPowerGenerator:
        """Construct the generator these settings describe.

        Imports the generators in its own body, not at module scope: building
        one reaches JAX, and Ripple construction initializes the XLA backend,
        so this is not safe to call before
        :func:`astrogwb.paper.runtime.configure_runtime`. Keeping the import
        here is what lets the ``Snakefile`` import this module to build its DAG.
        """
        from astrogwb.constants import ISCO_ALPHA
        from astrogwb.waveform import AnalyticInspiralGenerator, RippleGenerator

        settings = self.model_dump(exclude={"alpha"})
        if self.approximant == _ANALYTICAL_APPROXIMANT:
            return AnalyticInspiralGenerator(
                **settings, alpha=ISCO_ALPHA if self.alpha is None else self.alpha
            )
        return RippleGenerator(**settings)


class PopulationConfig(BaseModel):
    """The population declaration a catalog is drawn from, and drawn at.

    ``model`` is a key in the :mod:`astrogwb.populations` registry, never an
    import path: registry keys change only on purpose, while module paths move
    as collateral whenever a module is reorganized. It names the population --
    the source model and its merger rate together -- so a def cannot pair a
    redshift law with a rate that is not its own normalization.

    ``kwargs`` are the population's construction kwargs, passed whole to
    :func:`~astrogwb.populations.build_population`: the redshift window and
    grid every population takes, plus whatever else the named one takes, such
    as a guard mixture's ``uniform_mixing_fraction``. They must be
    JSON-serializable scalars, because that is how they travel in the
    catalog's HDF5 attributes. A key the named population does not accept
    fails pre-flight rather than being filtered away.

    ``params`` are the hyperparameters the draw is made at, and they are what a
    target evaluation is compared against; ``local_merger_rate`` belongs here
    even though it does not affect the normalized source draws, because the
    observation's total rate is reconstructed from it.

    Density factors and source outputs are declared by the registered
    population. Generation records its effective density selection.
    """

    model_config = _STRICT

    model: str
    kwargs: dict[str, float | int] = Field(default_factory=dict)
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

    def population_record(self) -> PopulationRecord:
        """The population declaration this def hands to a generated catalog.

        The record is what the ``.h5`` persists, so assembling it here -- next
        to the fields it is assembled from -- is what keeps the declaration in
        the config layer rather than in the script that runs the draw.
        :data:`~astrogwb.populations.DEFAULT_DENSITY_SITES` is supplied because
        no def declares density sites: they follow from the registered
        population, not from configuration.

        Imports :mod:`astrogwb.populations` in its own body; the registry is
        populated by importing the models, which pulls in JAX, and this module
        is otherwise free of it.
        """
        from astrogwb.populations import DEFAULT_DENSITY_SITES, PopulationRecord

        return PopulationRecord(
            model_name=self.population.model,
            model_kwargs=self.population.kwargs,
            density_sites=DEFAULT_DENSITY_SITES,
            seed=self.seed,
        )

    @model_validator(mode="after")
    def _validate_redshift_window(self) -> CatalogDefinition:
        kwargs = self.population.kwargs
        window = ("minimum_redshift", "maximum_redshift")
        if all(name in kwargs for name in window) and not float(
            kwargs["minimum_redshift"]
        ) < float(kwargs["maximum_redshift"]):
            raise ValueError(
                "population.kwargs.minimum_redshift must be less than "
                "population.kwargs.maximum_redshift"
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


def check_population_model(
    name: str,
    *,
    label: str,
    kwargs: Mapping[str, float | int] | None = None,
    requires_merger_rate: bool = False,
) -> None:
    """Reject a population a run or catalog def cannot actually be built from.

    Checks as much as the caller supplies: the name is registered, ``kwargs``
    are kwargs that population takes, and -- for an analysis target, which
    reconstructs an observed total rate -- that it declares a merger rate at
    all. A guard mixture does not, so naming one as a target is a
    configuration error rather than a silently meaningless spectrum.

    Imports :mod:`astrogwb.populations` in its own body: the registry is
    populated by importing the models, which pulls in JAX, and this module is
    otherwise free of it.
    """
    from astrogwb.populations import build_population, known_populations

    known = known_populations()
    if name not in known:
        raise ValueError(
            f"{label}: unknown population {name!r}; registered populations are: "
            f"{', '.join(known)}"
        )
    if kwargs is None:
        return
    try:
        population = build_population(name, **kwargs)
    except TypeError as error:
        raise ValueError(f"{label}: {error}") from None
    if requires_merger_rate and population.merger_rate_fn is None:
        raise ValueError(
            f"{label}: population {name!r} declares no merger rate, so it "
            "cannot be an analysis target; it is a proposal density"
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
    Populations are built here too, so an unregistered name, a construction
    setting the named population does not take, or a proposal density named as
    an analysis target is caught by the same pre-flight rather than at the top
    of a queued generation job.

    It lives here rather than in :mod:`astrogwb.paper.config.runs` for the same
    reason :func:`check_catalog_references` does -- that module must stay
    importable by the ``Snakefile`` without pydantic. Catalog configs are loaded
    once: this is one gate over all runs, not a per-run check.
    """
    from astrogwb.paper.config.mcmc import build_run_config
    from astrogwb.paper.config.runs import assemble_run, discover_runs

    catalogs = discover_catalogs(root)
    for name, definition in catalogs.items():
        check_population_model(
            definition.population.model,
            label=f"catalog {name!r} population.model",
            kwargs=definition.population.kwargs,
        )
    labels: list[str] = []
    for experiment, runs in discover_runs(root).items():
        for run in runs:
            label = f"{experiment}/{run}"
            config = build_run_config(assemble_run(experiment, run, root=root))
            check_catalog_references(config, label=label, catalogs=catalogs)
            grid = config.analysis_grid
            check_population_model(
                config.analysis.population_model,
                label=f"{label} analysis.population_model",
                kwargs={
                    "minimum_redshift": grid.minimum_redshift,
                    "maximum_redshift": grid.maximum_redshift,
                    "n_grid": grid.n_grid,
                },
                requires_merger_rate=True,
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
    "check_population_model",
    "discover_catalogs",
    "load_catalog_definition",
    "load_catalog_layers",
    "validate_all_runs",
]
