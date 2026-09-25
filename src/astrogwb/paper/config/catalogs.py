"""Validating the catalogs a run asks for, before any is built.

A *catalog* is one persisted waveform draw: expensive to build (population
draw + ripple waveform generation) and shared by every run that asks for the
same one. A run declares what it needs in ``[analysis.catalog]`` -- a partial
spec per role, resolved over the run's own ``[waveform]``, ``[population]``
and ``[fiducials]`` into a :class:`~astrogwb.metadata.CatalogRequest` -- and
the file lives at ``outputs/catalogs/<key>.h5``, where ``<key>`` is that
request's content hash. No name translates between the two.

Once built, the *file* is authoritative about what it holds, and
``scripts/run_mcmc.py`` checks it against the request its run resolves.

Deliberately JAX-free at import: nothing here needs it. The registry lookup
that validates a model name imports :mod:`astrogwb.populations` inside its own
body. The load-bearing constraint is not this one, though -- it is that
:func:`astrogwb.paper.runtime.configure_runtime` runs before the XLA backend is
initialized, which ``tests/paper/test_cli.py`` guards directly.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from astrogwb.metadata import CatalogRequest
from astrogwb.paper.config.mcmc import (
    RunConfig,
    build_run_config,
    check_redshift_grid,
)
from astrogwb.paper.config.runs import (
    CATALOG_ROLES,
    assemble_run,
    discover_runs,
    resolve_catalog_blocks,
)

logger = logging.getLogger(__name__)


def check_population_model(
    name: str,
    *,
    label: str,
    kwargs: Mapping[str, float | int] | None = None,
    requires_merger_rate: bool = False,
    amplitude_parameter: str | None = None,
) -> None:
    """Reject a population a run or catalog cannot actually be built from.

    Checks as much as the caller supplies: the name is registered, ``kwargs``
    are kwargs that population takes, and -- for an analysis target, which
    reconstructs an observed total rate -- that it declares a merger rate at
    all. A guard mixture does not, so naming one as a target is a
    configuration error rather than a silently meaningless spectrum.
    ``amplitude_parameter``, when given, must be one the population declares
    it can marginalize analytically; otherwise the marginalized likelihood
    would be a silently wrong posterior.

    Imports :mod:`astrogwb.populations` in its own body: the registry is
    populated by importing the models, which pulls in JAX, and this module is
    otherwise free of it.
    """
    from astrogwb.populations import (
        amplitude_parameters,
        build_population,
        known_populations,
    )

    known = known_populations()
    if name not in known:
        raise ValueError(
            f"{label}: unknown population {name!r}; registered populations are: "
            f"{', '.join(known)}"
        )
    if amplitude_parameter is not None:
        supported = amplitude_parameters(name)
        if amplitude_parameter not in supported:
            raise ValueError(
                f"{label}: population {name!r} cannot marginalize "
                f"{amplitude_parameter!r} analytically; its amplitude parameters "
                f"are: {', '.join(supported) or 'none'}"
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
            "cannot be an analysis target or an injection; it is a proposal density"
        )


def check_catalog_requests(config: RunConfig, *, label: str) -> None:
    """Reject a run whose catalogs could not be drawn.

    Resolves each role into its request -- which validates the waveform
    settings and the population record -- and builds the population it names,
    so an unregistered name or a construction setting it does not take fails
    here rather than at the top of a queued generation job.
    """
    for role in CATALOG_ROLES:
        role_label = f"{label} analysis.catalog.{role}"
        try:
            request = config.catalog_request(role)
        except ValueError as error:
            raise ValueError(f"{role_label}: {error}") from None
        population = request.metadata.population
        check_redshift_grid(
            population.model_kwargs, label=f"{role_label}.population.model_kwargs"
        )
        # The injection is the "observed" data, whose spectrum is scaled by a
        # physical rate; a guard mixture declares none, so it can only ever be
        # a proposal.
        check_population_model(
            population.model_name,
            label=f"{role_label} population.model_name",
            kwargs=population.model_kwargs,
            requires_merger_rate=role == "injection",
        )


@dataclass(frozen=True)
class RunCatalogs:
    """Every catalog the committed runs ask for, and which run asks for which.

    ``requests`` maps each distinct key to its request, so a draw several runs
    share appears once; ``by_run`` maps ``(experiment, run)`` to its
    ``{role: key}``. This is the whole inventory the workflow builds from --
    there is no catalog config to glob.
    """

    requests: dict[str, CatalogRequest]
    by_run: dict[tuple[str, str], dict[str, str]]

    def users(self, key: str) -> list[str]:
        """Every ``experiment/run:role`` that samples against ``key``."""
        return [
            f"{experiment}/{run}:{role}"
            for (experiment, run), roles in self.by_run.items()
            for role, used in roles.items()
            if used == key
        ]


def resolve_run_catalogs(root: Path | None = None) -> RunCatalogs:
    """Resolve both catalogs of every committed run into keyed requests.

    Works off the raw merge rather than a validated :class:`RunConfig`, so the
    ``Snakefile`` can build its DAG without validating all 27 runs; the request
    itself is still validated, because the key is taken over its canonical
    form. ``RunConfig.catalog_request`` goes through the same
    :func:`~astrogwb.paper.config.runs.resolve_catalog_blocks`, and a test pins
    the two agreeing.
    """
    requests: dict[str, CatalogRequest] = {}
    by_run: dict[tuple[str, str], dict[str, str]] = {}
    for experiment, names in discover_runs(root).items():
        for run in names:
            raw = assemble_run(experiment, run, root=root)
            roles: dict[str, str] = {}
            for role in CATALOG_ROLES:
                try:
                    request = CatalogRequest.from_blocks(
                        **resolve_catalog_blocks(raw, role)
                    )
                except ValueError as error:
                    raise ValueError(
                        f"{experiment}/{run} analysis.catalog.{role}: {error}"
                    ) from None
                key = request.key()
                requests.setdefault(key, request)
                roles[role] = key
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
                target.model_name,
                label=f"{label} analysis.population.model_name",
                kwargs=target.model_kwargs,
                requires_merger_rate=True,
                amplitude_parameter=config.analysis.amplitude_parameter,
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
