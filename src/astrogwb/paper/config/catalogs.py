"""Validating the catalogs a run asks for, before any is built.

A *catalog* is one persisted waveform draw: expensive to build (population
draw + ripple waveform generation) and shared by every run that asks for the
same one. A run declares what it needs in ``[analysis.injection]`` and
``[analysis.proposal]`` -- each a :class:`~astrogwb.simulators.polarization_power.CatalogMetadata`
once the merge has resolved its references -- and the file lives at
``outputs/catalogs/<stem>.h5``, where ``<stem>`` is
``polarization_power-<key>-<digest>``: the request's content hash and the
hash of the seed it is drawn at. No name translates between the two.

Once built, the *file* is authoritative about what it holds, and
``scripts/run_mcmc.py`` checks it against the request its run resolves.

The registry lookup that validates a model name imports
:mod:`astrogwb.populations` inside its own body. Importing this module loads
the run models, and with them the detector registry and JAX. The load-bearing
constraint is that :func:`astrogwb.paper.runtime.configure_runtime` runs before
the XLA backend is initialized, which ``tests/paper/test_cli.py`` guards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from astrogwb.paper.config.mcmc import (
    RunConfig,
    build_run_config,
    check_redshift_grid,
)
from astrogwb.paper.config.runs import CATALOG_ROLES, assemble_run, discover_runs
from astrogwb.populations.metadata import PopulationMetadata
from astrogwb.simulators.polarization_power import CatalogMetadata, polarization_power

logger = logging.getLogger(__name__)


def check_population_model(
    population: PopulationMetadata,
    *,
    label: str,
    requires_merger_rate: bool = False,
) -> None:
    """Reject a population a run or catalog cannot actually be built from.

    Builds the population the record names: the population and each sub-model
    are registered, their kwargs are kwargs they take, and -- for an analysis
    target, which reconstructs an observed total rate -- that the population
    declares a merger rate at all. A guard-mixture redshift model does not, so
    naming one as a target is a configuration error rather than a silently
    meaningless spectrum.

    Imports :mod:`astrogwb.populations` through ``build``, in its own body: the
    registry is populated by importing the models, which pulls in JAX, and this
    module is otherwise free of it.
    """
    try:
        built = population.build()
    except KeyError as error:
        raise ValueError(f"{label}: {error.args[0]}") from None
    except TypeError as error:
        raise ValueError(f"{label}: {error}") from None
    if requires_merger_rate and built.merger_rate_fn is None:
        raise ValueError(
            f"{label}: population {population.model_name!r} with redshift model "
            f"{population.redshift.model!r} declares no merger rate, so it "
            "cannot be an analysis target or an injection; it is a proposal density"
        )


def check_catalog_requests(config: RunConfig, *, label: str) -> None:
    """Reject a run whose catalogs could not be drawn.

    Each role is already a validated record, so what is left is to check its
    redshift window and build the population it names: an unregistered name or
    a construction setting it does not take fails here rather than at the top
    of a queued generation job.
    """
    for role in CATALOG_ROLES:
        role_label = f"{label} analysis.{role}"
        population = config.catalog_request(role)[0].population
        check_redshift_grid(
            population.redshift.kwargs, label=f"{role_label}.population.redshift.kwargs"
        )
        # The injection is the "observed" data, whose spectrum is scaled by a
        # physical rate; a guard mixture declares none, so it can only ever be
        # a proposal.
        check_population_model(
            population,
            label=f"{role_label} population",
            requires_merger_rate=role == "injection",
        )


@dataclass(frozen=True)
class RunCatalogs:
    """Every catalog the committed runs ask for, and which run asks for which.

    ``requests`` maps each distinct file stem to its ``(metadata, seed)``, so a
    draw several runs share appears once; ``by_run`` maps ``(experiment, run)``
    to its ``{role: stem}``. This is the whole inventory the workflow builds
    from -- there is no catalog config to glob.
    """

    requests: dict[str, tuple[CatalogMetadata, np.uint64]]
    by_run: dict[tuple[str, str], dict[str, str]]

    def users(self, stem: str) -> list[str]:
        """Every ``experiment/run:role`` that samples against ``stem``."""
        return [
            f"{experiment}/{run}:{role}"
            for (experiment, run), roles in self.by_run.items()
            for role, used in roles.items()
            if used == stem
        ]


def resolve_run_catalogs(root: Path | None = None) -> RunCatalogs:
    """Resolve both catalogs of every committed run into requests named by file stem.

    Works off the raw merge rather than a validated :class:`RunConfig`, so the
    ``Snakefile`` can build its DAG without validating all 27 runs; the request
    itself is still validated, because the key is taken over its canonical
    form. ``RunConfig`` validates the same merged table, and a test pins the
    two agreeing.
    """
    requests: dict[str, tuple[CatalogMetadata, np.uint64]] = {}
    by_run: dict[tuple[str, str], dict[str, str]] = {}
    for experiment, names in discover_runs(root).items():
        for run in names:
            analysis = assemble_run(experiment, run, root=root).get("analysis") or {}
            seeds = analysis.get("seeds") or {}
            roles: dict[str, str] = {}
            for role in CATALOG_ROLES:
                try:
                    request = CatalogMetadata.model_validate(analysis.get(role))
                    seed = np.uint64(seeds[role])
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        f"{experiment}/{run} analysis.{role}: {error!r}"
                    ) from None
                stem = polarization_power.path({"seed": seed}, request, "").stem
                requests.setdefault(stem, (request, seed))
                roles[role] = stem
            by_run[(experiment, run)] = roles
    return RunCatalogs(requests=requests, by_run=by_run)


def validate_all_runs(root: Path | None = None) -> list[str]:
    """Merge, validate, and catalog-check every declared run; return their labels.

    The pre-flight gate ``astrogwb-assemble-config --all`` used to provide,
    kept because its real value was never the JSON it wrote: it fails on the
    first invalid run *before any catalog is built*, and a catalog is a GPU job.
    Populations are built here too, so an unregistered name, a construction
    setting the named population does not take, or a proposal density named as
    an analysis target is caught by the same pre-flight rather than at the top
    of a queued generation job.

    It lives here rather than in :mod:`astrogwb.paper.config.runs` because
    that module must stay stdlib-only.
    """
    labels: list[str] = []
    for experiment, runs in discover_runs(root).items():
        for run in runs:
            label = f"{experiment}/{run}"
            config = build_run_config(assemble_run(experiment, run, root=root))
            check_catalog_requests(config, label=label)
            target = config.analysis.population
            check_population_model(
                target,
                label=f"{label} analysis.population",
                requires_merger_rate=True,
            )
            logger.info("ok %s", label)
            labels.append(label)
    return labels


__all__ = [
    "RunCatalogs",
    "check_catalog_requests",
    "check_population_model",
    "resolve_run_catalogs",
    "validate_all_runs",
]
