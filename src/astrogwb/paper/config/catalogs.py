"""PolarizationPowerCatalog configuration: what a catalog is made of, before it is made.

A *catalog* is one persisted waveform draw: expensive to build (population
draw + ripple waveform generation) and reused by every run that names it. Two
layers: the shared ``config/waveform.json``, ``config/population.json`` and
``config/fiducials.json``, then one ``config/catalogs/<name>.json`` per catalog,
whose stem is the name and which produces ``outputs/catalogs/<stem>.h5``. No
registry file translates between the two.

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
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb.metadata import PopulationMetadata
from astrogwb.paper.config.mcmc import RunConfig
from astrogwb.paper.config.runs import (
    CATALOGS_DIR,
    catalog_config_paths,
    discover_catalog_names,
    merge_config_layers,
)

if TYPE_CHECKING:
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


class CatalogDefinition(BaseModel):
    """Everything needed to generate one reusable catalog.

    ``population`` is the :class:`~astrogwb.metadata.PopulationMetadata` the
    generated ``.h5`` persists verbatim, assembled here rather than bridged
    from a second config-layer model: the declaration and the record were the
    same four facts stated twice, and a bridge between them is one more place
    for them to disagree. ``model_name`` is a key in the
    :mod:`astrogwb.populations` registry, never an import path -- registry keys
    change only on purpose, while module paths move as collateral whenever a
    module is reorganized. It names the population, the source model and its
    merger rate together, so a def cannot pair a redshift law with a rate that
    is not its own normalization.

    ``seed`` and ``num_samples`` live here rather than inside ``population`` on
    purpose: ``md-imrphenom-s41-n32768`` and ``md-imrphenom-s42-n16384`` are the
    *same* population drawn at different seeds and sizes, so the shared
    ``config/population.json`` declares neither. The seed is folded into the
    record by :meth:`_assemble_population_record`, because that is where it
    belongs once a particular draw exists.

    ``fiducials`` are the hyperparameters the draw is made at, inherited from
    ``config/fiducials.json`` -- the same table the runs initialize at, so the
    two cannot drift. They stay outside ``population`` because they are not part
    of the record: the record describes the density that produced the samples,
    while these describe the samples. A def that wants an injection away from
    the fiducials overrides the ``[fiducials]`` block like any other layer.

    ``description`` is the def's own prose, carried as data because JSON has no
    comments.
    """

    model_config = _STRICT

    name: str
    description: str = ""
    seed: int
    num_samples: Annotated[int, Field(gt=0)]
    population: PopulationMetadata
    fiducials: dict[str, float]
    waveform: WaveformConfig

    @model_validator(mode="before")
    @classmethod
    def _assemble_population_record(cls, data: Any) -> Any:
        """Complete the ``[population]`` block into a full record.

        The config layer declares the two facts that are configuration --
        ``model_name`` and ``model_kwargs``. The other two are not: ``seed``
        belongs to this particular draw and is stated once, at the top level,
        and :data:`~astrogwb.populations.DEFAULT_DENSITY_SITES` follows from the
        registered population rather than from a file, so no def declares it.

        Both are rejected rather than ignored when a layer does declare one.
        A ``[population]`` ``seed`` would otherwise win here and leave
        ``definition.seed`` disagreeing with the seed the ``.h5`` records, which
        is the one thing this assembly exists to make impossible.

        Imports the registry in its own body: populating it means importing the
        population models, which reaches JAX, and this module is otherwise free
        of it.
        """
        if not isinstance(data, Mapping):
            return data
        population = data.get("population")
        if not isinstance(population, Mapping):
            # Already a built record, or absent -- either way pydantic reports
            # it better than a KeyError here would.
            return data

        supplied = [name for name in ("seed", "density_sites") if name in population]
        if supplied:
            raise ValueError(
                f"population may not declare {', '.join(supplied)}: the seed is "
                "the def's own, and the density sites follow from the "
                "registered population"
            )

        from astrogwb.populations import DEFAULT_DENSITY_SITES

        completed = {
            **population,
            "density_sites": DEFAULT_DENSITY_SITES,
        }
        if "seed" in data:
            completed["seed"] = data["seed"]
        return {**data, "population": completed}

    @model_validator(mode="after")
    def _validate_redshift_window(self) -> CatalogDefinition:
        kwargs = self.population.model_kwargs
        window = ("minimum_redshift", "maximum_redshift")
        if all(name in kwargs for name in window) and not float(
            kwargs["minimum_redshift"]
        ) < float(kwargs["maximum_redshift"]):
            raise ValueError(
                "population.model_kwargs.minimum_redshift must be less than "
                "population.model_kwargs.maximum_redshift"
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
            definition.population.model_name,
            label=f"catalog {name!r} population.model_name",
            kwargs=definition.population.model_kwargs,
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
    "CATALOGS_DIR",
    "CatalogDefinition",
    "WaveformConfig",
    "check_catalog_references",
    "check_population_model",
    "discover_catalogs",
    "load_catalog_definition",
    "load_catalog_layers",
    "validate_all_runs",
]
