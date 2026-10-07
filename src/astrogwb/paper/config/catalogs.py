"""Validating the catalogs a run asks for, before any is built.

A *catalog* is one persisted waveform draw: expensive to build (population
draw + ripple waveform generation) and shared by every run that asks for the
same one. A run declares what it needs in ``[analysis.injection]`` and
``[analysis.proposal]`` -- each a :class:`~astrogwb.simulators.polarization_power.CatalogMetadata`
once the merge has resolved its references -- and the file lives at
``outputs/catalogs/<stem>.h5``, where ``<stem>`` is
``polarization_power-<key>-<seed>``: the request's content hash and the
seed it is drawn at. No name translates between the two.

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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from astrogwb.paper.config.mcmc import (
    RunConfig,
    check_redshift_grid,
)
from astrogwb.paper.config.runs import CATALOG_ROLES, assemble_run, discover_runs
from astrogwb.simulators.polarization_power import CatalogMetadata, catalog_stem

logger = logging.getLogger(__name__)


def check_population_model(
    name: str,
    *,
    label: str,
    kwargs: Mapping[str, float | int | bool | str] | None = None,
) -> None:
    """Reject a population a run or catalog cannot actually be built from.

    Checks as much as the caller supplies: the name is registered and
    ``kwargs`` are kwargs that population takes.

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
        build_population(name, **kwargs)
    except TypeError as error:
        raise ValueError(f"{label}: {error}") from None


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
            population.model_kwargs, label=f"{role_label}.population.model_kwargs"
        )
        check_population_model(
            population.model_name,
            label=f"{role_label} population.model_name",
            kwargs=population.model_kwargs,
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
                stem = catalog_stem(request, seed)
                requests.setdefault(stem, (request, seed))
                roles[role] = stem
            by_run[(experiment, run)] = roles
    return RunCatalogs(requests=requests, by_run=by_run)


__all__ = [
    "RunCatalogs",
    "check_catalog_requests",
    "check_population_model",
    "resolve_run_catalogs",
]
