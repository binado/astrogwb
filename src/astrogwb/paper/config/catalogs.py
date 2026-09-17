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
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb.metadata import PopulationMetadata, WaveformMetadata
from astrogwb.paper.config.mcmc import RunConfig, check_redshift_grid
from astrogwb.paper.config.runs import (
    CATALOGS_DIR,
    catalog_config_paths,
    discover_catalog_names,
    merge_config_layers,
)

logger = logging.getLogger(__name__)

_STRICT = ConfigDict(frozen=True, extra="forbid")


class CatalogDefinition(BaseModel):
    """Everything needed to generate one reusable catalog.

    ``population`` is the :class:`~astrogwb.metadata.PopulationMetadata` the
    generated ``.h5`` persists verbatim, assembled here rather than bridged
    from a second config-layer model: the declaration and the record were the
    same facts stated twice, and a bridge between them is one more place for
    them to disagree. ``model_name`` is a key in the
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

    A def carries no prose. JSON has no comments, and a ``description`` field
    would be a second place for one to rot: what each committed catalog is for
    is documented once, in ``config/catalogs/README.md``, next to the files it
    describes. ``extra="forbid"`` is what keeps it there.
    """

    model_config = _STRICT

    name: str
    seed: int
    num_samples: Annotated[int, Field(gt=0)]
    population: PopulationMetadata
    fiducials: dict[str, float]
    waveform: WaveformMetadata

    @model_validator(mode="before")
    @classmethod
    def _assemble_population_record(cls, data: Any) -> Any:
        """Complete the ``[population]`` block into a full record.

        The config layer declares the two facts that are configuration --
        ``model_name`` and ``model_kwargs``. ``seed`` is not: it belongs to this
        particular draw and is stated once, at the top level, so it is folded in
        here.

        A ``[population]`` ``seed`` is rejected rather than ignored. It would
        otherwise win here and leave ``definition.seed`` disagreeing with the
        seed the ``.h5`` records, which is the one thing this assembly exists to
        make impossible.
        """
        if not isinstance(data, Mapping):
            return data
        population = data.get("population")
        if not isinstance(population, Mapping):
            # Already a built record, or absent -- either way pydantic reports
            # it better than a KeyError here would.
            return data

        if "seed" in population:
            raise ValueError(
                "population may not declare seed: the seed is the def's own"
            )

        completed = dict(population)
        if "seed" in data:
            completed["seed"] = data["seed"]
        return {**data, "population": completed}

    @model_validator(mode="after")
    def _validate_redshift_window(self) -> CatalogDefinition:
        check_redshift_grid(
            self.population.model_kwargs, label="population.model_kwargs"
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
            target = config.analysis.population
            check_population_model(
                target.model_name,
                label=f"{label} analysis.population.model_name",
                kwargs=target.model_kwargs,
                requires_merger_rate=True,
            )
            logger.info("ok %s", label)
            labels.append(label)
    return labels


__all__ = [
    "CATALOGS_DIR",
    "CatalogDefinition",
    "check_catalog_references",
    "check_population_model",
    "discover_catalogs",
    "load_catalog_definition",
    "load_catalog_layers",
    "validate_all_runs",
]
