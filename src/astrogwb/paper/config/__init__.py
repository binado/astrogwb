"""Configuration support modules, and the shared scientific values themselves.

The leaf modules (:mod:`~astrogwb.paper.config.runs`,
:mod:`~astrogwb.paper.config.mcmc`, :mod:`~astrogwb.paper.config.catalogs`) are
not re-exported; import them explicitly.

What this package *does* expose is the tables the shared layers declare --
``[fiducials]``, ``[priors]`` and ``[networks]``, the same bytes the workflow
merges into every run -- plus the accessors that build something from the
default draw every run inherits, ``[catalog]``: :func:`waveform_generator`
from its waveform, and :func:`population_model` / :func:`population_metadata`
from its population. Before the tables lived here, the notebook and the figure
scripts each kept a hand-written copy, and those copies drifted: the notebook
sampled ``local_merger_rate`` under a prior that excluded its own fiducial.
Consume them from here instead::

    from astrogwb.paper.config import fiducials, networks, priors, waveform_generator

    fid = fiducials()
    detectors = networks()["ET-2L-aligned-CE-Hanford"]
    generator = waveform_generator()

There is deliberately no accessor for the hyperparameters a catalog is drawn
at: that is :func:`fiducials`, which a run's catalogs inherit from the run
itself. One table, one place.

Every accessor also takes keyword overrides, merged over the file, so a
notebook can vary one entry without editing TOML or retyping the table::

    fiducials(H0=70.0)
    priors(xi_0={"dist": "Uniform", "kwargs": {"low": 0.1, "high": 5.0}})
    networks(**{"ET-2L-aligned": ("S1", "R1", "C1")})

**These are functions, not module-level dicts, and that is load-bearing.** The
``Snakefile`` imports :mod:`astrogwb.paper.config.runs` to build the DAG, which
executes this module; eager dicts would mean file I/O at import (failing from
any working directory but the repository root) and, for the priors, a numpyro
import on every ``--dry-run``. :func:`priors` therefore imports
:func:`~astrogwb.paper.config.mcmc.materialize_prior` inside its own body;
:func:`waveform_generator` imports
:class:`~astrogwb.waveform.WaveformMetadata` the same way. A
subprocess test in ``tests/paper/test_cli.py`` pins both halves.

Paths are relative to the working directory, which for the workflow and every
script is the repository root -- the same contract as
:mod:`astrogwb.paper.config.runs`. Tests, which pytest may invoke from
anywhere, pass ``root=`` explicitly.

Every accessor reads one cached merge of the shared layers, with its
references resolved, so a table here is exactly what a run inherits. Each
hands back a fresh copy, so a caller that mutates what it got does not poison
the cache for everyone else -- overrides are merged *after* the cached parse,
so they cannot either. A long-lived Jupyter session will not see an edit to a
shared layer until ``_shared.cache_clear()``.
"""

from __future__ import annotations

import copy
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrogwb.paper.config.runs import load_base

if TYPE_CHECKING:
    from numpyro.distributions import Distribution

    from astrogwb.paper.config.detectors import DetectorRegistry
    from astrogwb.populations import PopulationMetadata
    from astrogwb.populations.registry import Population
    from astrogwb.waveform import PolarizationPowerGenerator, WaveformMetadata

__all__ = [
    "detector_registry",
    "fiducials",
    "networks",
    "population_metadata",
    "population_model",
    "priors",
    "waveform_generator",
    "waveform_metadata",
]


@cache
def _shared(root: Path) -> dict[str, Any]:
    """Merge the shared layers once, references resolved."""
    return load_base(root)


def _table(root: Path | None, *path: str) -> dict[str, Any]:
    """A fresh copy of one table of the shared merge, addressed by key path."""
    table: Any = _shared(root or Path())
    for depth, key in enumerate(path):
        table = table.get(key) if isinstance(table, dict) else None
        if not isinstance(table, dict):
            raise TypeError(
                f"the shared config layers must declare a "
                f"[{'.'.join(path[: depth + 1])}] table"
            )
    return copy.deepcopy(table)


def fiducials(root: Path | None = None, **kwargs: float) -> dict[str, float]:
    """The fiducial hyperparameter values: the shared ``[fiducials]`` table.

    These are **not** the injection: what was injected is recorded in the
    injection catalog file, which is where the observed spectrum's rate and
    density come from. These are where NUTS initializes each sampled parameter,
    what the non-sampled sites are conditioned at, and the reference point an
    amplitude-marginalized run forms its ratio against. Nothing cross-checks
    them against a catalog, because nothing needs to.

    Every fiducial carries a prior in :func:`priors`; ``RunConfig`` retains the
    complete table and ``analysis.sampled_params`` selects the NUTS latents,
    leaving the remaining sites to be fixed by NumPyro effect handlers.

    Keyword arguments override the file, and may name a fiducial the file does
    not declare. An added fiducial is the caller's to keep consistent with
    :func:`priors` -- only ``RunConfig`` cross-checks the two tables.
    """
    table = {**_table(root, "fiducials"), **kwargs}
    return {name: float(value) for name, value in table.items()}


def priors(root: Path | None = None, **kwargs: Any) -> dict[str, Distribution]:
    """The prior for each parameter, materialized from the shared ``[priors]``.

    One entry per fiducial. The returned distributions hold plain Python
    floats and evaluating none of them touches JAX, so calling this does not
    initialize the XLA backend -- ``configure_runtime`` may still run after it.

    Note this is the *inference* prior. A diagnostic that scans a parameter is
    free to scan wider (see ``GRID_SCAN_RANGES`` in
    ``scripts/importance_weights_grid.py``); it just has to say so.

    Keyword arguments override the file, and may name a parameter the file does
    not declare. A value may be a wire-format spec
    (``{"dist": ..., "kwargs": {...}}``) or an already-built ``numpyro``
    distribution, which ``materialize_prior`` passes through unchanged.
    """
    # Imported here, not at module scope: `mcmc` reaches pydantic, and the
    # Snakefile's DAG construction imports this package via `config.runs`.
    from astrogwb.paper.config.mcmc import materialize_prior

    table = {**_table(root, "priors"), **kwargs}
    return {name: materialize_prior(spec) for name, spec in table.items()}


def networks(root: Path | None = None, **kwargs: Any) -> dict[str, tuple[str, ...]]:
    """Detector networks by name: the shared ``[networks]`` table.

    Each run names one of these keys as ``analysis.network``; ``RunConfig``
    resolves it to the detector list recorded in the chain's own config. The
    :func:`detector_registry` resolves the geometry and sensitivity settings.

    Membership and content live here; *order* does not. The ordered legend of
    the network-comparison figures is
    :data:`astrogwb.paper.plotting.DETECTOR_NETWORKS`, because order is
    presentation -- and because a TOML table is not an ordered thing.

    Keyword arguments override the file, and may name a network the file does
    not declare. Use :func:`detector_registry` with matching overrides when
    building runtime detectors. Every committed name is hyphenated
    (``ET-2L-aligned``), so an override has to be unpacked from a mapping
    rather than written as a literal keyword::

        networks(**{"ET-2L-aligned": ("S1", "R1", "C1")})
    """
    table = {**_table(root, "networks"), **kwargs}
    return {name: tuple(detectors) for name, detectors in table.items()}


def detector_registry(root: Path | None = None, **overrides: Any) -> DetectorRegistry:
    """Resolve shared detector settings lazily and return an independent registry.

    Overrides may contain ``detectors`` and ``networks`` tables, deep-merged
    over the shared file. Runtime objects are built only by its build methods.
    """
    from astrogwb.paper.config.detectors import DetectorRegistry
    from astrogwb.paper.utils import deep_merge

    shared = {name: _table(root, name) for name in ("detectors", "networks")}
    settings = deep_merge(shared, overrides)
    unknown = settings.keys() - {"detectors", "networks"}
    if unknown:
        raise ValueError(f"unknown detector registry fields: {sorted(unknown)}")
    return DetectorRegistry.model_validate(settings)


def waveform_generator(
    root: Path | None = None, **kwargs: Any
) -> PolarizationPowerGenerator:
    """Build the polarization-power generator the default draw uses.

    Keyword arguments override the file. ``approximant="AnalyticInspiral"`` selects
    the closed-form inspiral, and is the only approximant taking an ``alpha``;
    any other name is a Ripple approximant.

    The settings are validated through
    :class:`~astrogwb.waveform.WaveformMetadata` rather than coerced
    field by field here, so this accessor and a catalog request reach a generator
    down the same path and an override is checked instead of trusted.

    Imported here, not at module scope: ``catalogs`` reaches pydantic and
    ``build`` reaches JAX, while the Snakefile's DAG construction imports this
    package via ``config.runs``. Ripple construction initializes the XLA
    backend, so this is not safe to call before ``configure_runtime``.
    """
    return waveform_metadata(root, **kwargs).build()


def waveform_metadata(root: Path | None = None, **kwargs: Any) -> WaveformMetadata:
    """The waveform the default draw, ``[catalog]``, uses, as a record.

    What :func:`waveform_generator` builds, before it is built: the form a
    :class:`~astrogwb.simulators.spectra.SpectraMetadata` or a catalog request carries.
    Keyword arguments override the file and are validated, not trusted.
    Touches no JAX, so it is safe before ``configure_runtime``.
    """
    from astrogwb.waveform import WaveformMetadata

    settings = {**_table(root, "catalog", "waveform"), **kwargs}
    return WaveformMetadata.model_validate(settings)


def population_model(root: Path | None = None, **kwargs: float | bool) -> Population:
    """Build the population the default draw, ``[catalog]``, is drawn from.

    Returns the registered :class:`~astrogwb.populations.registry.Population` --
    source model and merger rate together -- with its construction settings
    bound. Keyword arguments override ``model_kwargs``, which is how a notebook
    studies the committed population on a coarser grid or a narrower redshift
    window without editing the file::

        population_model(n_grid=256, minimum_redshift=0.3)

    ``sample_inclination=False`` selects explicit analytic quadrupole
    averaging instead of the default isotropic inclination draw.

    A key the named population does not take raises here rather than being
    filtered away, which is the same contract a catalog request gets.

    Imports the registry in its own body: populating it means importing the
    population models, which reaches JAX, so this is not safe to call before
    ``configure_runtime``. Keeping the import here is what lets the ``Snakefile``
    import this package to build its DAG.
    """
    return population_metadata(root).with_model_kwargs(**kwargs).build()


def population_metadata(
    root: Path | None = None, **kwargs: float | bool
) -> PopulationMetadata:
    """The record the default draw, ``[catalog]``, carries for its population.

    Keyword arguments override ``model_kwargs``, validated rather than trusted, so this
    accessor and :class:`~astrogwb.simulators.polarization_power.CatalogMetadata` reach a record down
    the same path. An already-built record is re-derived with
    :meth:`~astrogwb.populations.PopulationMetadata.with_model_kwargs`.

    Touches no JAX, so it is safe before ``configure_runtime``; building the
    record's population is not, for the reason :func:`population_model` gives.
    """
    from astrogwb.populations import PopulationMetadata

    table = _table(root, "catalog", "population")
    table["model_kwargs"] = {**table.get("model_kwargs", {}), **kwargs}
    return PopulationMetadata.model_validate(table)
