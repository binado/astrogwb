"""Configuration support modules, and the shared scientific values themselves.

The leaf modules (:mod:`~astrogwb.paper.config.runs`,
:mod:`~astrogwb.paper.config.mcmc`, :mod:`~astrogwb.paper.config.catalogs`) are
not re-exported; import them explicitly.

What this package *does* expose is the three shared tables that
``config/fiducials.json``, ``config/priors.json`` and ``config/networks.json``
own -- the same bytes the workflow merges into every run -- plus the three
accessors that build something from a shared catalog layer:
:func:`waveform_generator` from ``config/waveform.json``, and
:func:`population_model` / :func:`population_metadata` from
``config/population.json``. Those two files are catalog layers, not run layers;
``config/fiducials.json`` is both.
Before the tables lived here, the notebook and the figure scripts each kept a
hand-written copy, and those copies drifted: the notebook sampled
``local_merger_rate`` under a prior that excluded its own fiducial. Consume
them from here instead::

    from astrogwb.paper.config import fiducials, networks, priors, waveform_generator

    fid = fiducials()
    detectors = networks()["ET-2L-aligned-CE-Hanford"]
    generator = waveform_generator()

There is deliberately no accessor for the hyperparameters a catalog is drawn
at: that is :func:`fiducials`, which ``config/catalogs/<name>.json`` inherits
as a layer like any other. One table, one file.

Every accessor also takes keyword overrides, merged over the file, so a
notebook can vary one entry without editing JSON or retyping the table::

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
:class:`~astrogwb.paper.config.catalogs.WaveformMetadata` the same way. A
subprocess test in ``tests/paper/test_cli.py`` pins both halves.

Paths are relative to the working directory, which for the workflow and every
script is the repository root -- the same contract as
:mod:`astrogwb.paper.config.runs`. Tests, which pytest may invoke from
anywhere, pass ``root=`` explicitly.

Each accessor caches its parse and hands back a fresh copy, so a caller that
mutates what it got does not poison the cache for everyone else -- overrides are
merged *after* the cached parse, so they cannot either. A long-lived Jupyter
session will not see an edit to the JSON until ``fiducials.cache_clear()``.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrogwb.paper.config.runs import (
    FIDUCIALS_PATH,
    NETWORKS_PATH,
    POPULATION_PATH,
    PRIORS_PATH,
    WAVEFORM_PATH,
)
from astrogwb.paper.utils import load_mapping

if TYPE_CHECKING:
    from numpyro.distributions import Distribution

    from astrogwb.metadata import PopulationMetadata
    from astrogwb.populations.registry import Population
    from astrogwb.waveform import PolarizationPowerGenerator

__all__ = [
    "fiducials",
    "networks",
    "population_metadata",
    "population_model",
    "priors",
    "waveform_generator",
]


@cache
def _load(path: Path, key: str) -> dict[str, Any]:
    """Parse one single-key layer file and return the table it declares."""
    raw = load_mapping(path)
    table = raw.get(key)
    if not isinstance(table, dict):
        raise TypeError(f"{path} must declare a [{key}] table")
    return table


def fiducials(root: Path | None = None, **kwargs: float) -> dict[str, float]:
    """The fiducial hyperparameter values, from ``config/fiducials.json``.

    These are **not** the injection: what was injected is recorded in the
    injection catalog file, which is where the observed spectrum's rate and
    density come from. These are where NUTS initializes each sampled parameter,
    what the non-sampled sites are conditioned at, and the reference point an
    amplitude-marginalized run forms its ratio against. Nothing cross-checks
    them against a catalog, because nothing needs to.

    Every fiducial carries a prior in :func:`priors`; ``RunConfig`` retains the
    complete table and ``sampled_params`` selects the NUTS latents, leaving the
    remaining sites to be fixed by NumPyro effect handlers.

    Keyword arguments override the file, and may name a fiducial the file does
    not declare. An added fiducial is the caller's to keep consistent with
    :func:`priors` -- only ``RunConfig`` cross-checks the two tables.
    """
    table = {
        **_load((root or Path()) / FIDUCIALS_PATH, "fiducials"),
        **kwargs,
    }
    return {name: float(value) for name, value in table.items()}


def priors(root: Path | None = None, **kwargs: Any) -> dict[str, Distribution]:
    """The prior for each parameter, materialized from ``config/priors.json``.

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

    table = {
        **_load((root or Path()) / PRIORS_PATH, "priors"),
        **kwargs,
    }
    return {name: materialize_prior(spec) for name, spec in table.items()}


def networks(root: Path | None = None, **kwargs: Any) -> dict[str, tuple[str, ...]]:
    """Detector networks by name, from ``config/networks.json``.

    Each run names one of these keys as ``analysis.network``; ``RunConfig``
    resolves it to the detector list recorded in the chain's own config. The
    names resolve further, to geometry and sensitivity, through
    ``astrogwb.detector``.

    Membership and content live here; *order* does not. The ordered legend of
    the network-comparison figures is
    :data:`astrogwb.paper.plotting.DETECTOR_NETWORKS`, because order is
    presentation -- and because a JSON object is not an ordered thing.

    Keyword arguments override the file, and may name a network the file does
    not declare -- whose detectors then have to resolve through
    ``astrogwb.detector`` on their own. Every committed name is hyphenated
    (``ET-2L-aligned``), so an override has to be unpacked from a mapping
    rather than written as a literal keyword::

        networks(**{"ET-2L-aligned": ("S1", "R1", "C1")})
    """
    table = {
        **_load((root or Path()) / NETWORKS_PATH, "networks"),
        **kwargs,
    }
    return {name: tuple(detectors) for name, detectors in table.items()}


def waveform_generator(
    root: Path | None = None, **kwargs: Any
) -> PolarizationPowerGenerator:
    """Build the polarization-power generator from ``config/waveform.json``.

    Keyword arguments override the file. ``approximant="AnalyticInspiral"`` selects
    the closed-form inspiral, and is the only approximant taking an ``alpha``;
    any other name is a Ripple approximant.

    The settings are validated through
    :class:`~astrogwb.paper.config.catalogs.WaveformMetadata` rather than coerced
    field by field here, so this accessor and a catalog def reach a generator
    down the same path and an override is checked instead of trusted.

    Imported here, not at module scope: ``catalogs`` reaches pydantic and
    ``build`` reaches JAX, while the Snakefile's DAG construction imports this
    package via ``config.runs``. Ripple construction initializes the XLA
    backend, so this is not safe to call before ``configure_runtime``.
    """
    from astrogwb.metadata import WaveformMetadata

    settings = {**_load((root or Path()) / WAVEFORM_PATH, "waveform"), **kwargs}
    return WaveformMetadata.model_validate(settings).build()


def population_model(root: Path | None = None, **kwargs: float) -> Population:
    """Build the generating population from ``config/population.json``.

    Returns the registered :class:`~astrogwb.populations.registry.Population` --
    source model and merger rate together -- with its construction settings
    bound. Keyword arguments override ``model_kwargs``, which is how a notebook
    studies the committed population on a coarser grid or a narrower redshift
    window without editing the file::

        population_model(n_grid=256, minimum_redshift=0.3)

    A key the named population does not take raises here rather than being
    filtered away, which is the same contract a catalog def gets.

    Imports the registry in its own body: populating it means importing the
    population models, which reaches JAX, so this is not safe to call before
    ``configure_runtime``. Keeping the import here is what lets the ``Snakefile``
    import this package to build its DAG.
    """
    from astrogwb.populations import build_population

    table = _load((root or Path()) / POPULATION_PATH, "population")
    settings = {**table.get("model_kwargs", {}), **kwargs}
    return build_population(table["model_name"], **settings)


def population_metadata(
    root: Path | None = None, *, seed: int, **kwargs: float
) -> PopulationMetadata:
    """The population record a catalog drawn from ``config/population.json`` carries.

    ``seed`` is required and has no entry in the file: the shared layer declares
    the population, while a particular draw of it declares the seed -- which is
    why ``config/catalogs/<name>.json`` carries one and
    ``config/population.json`` does not.

    Keyword arguments override ``model_kwargs``, validated rather than trusted,
    so this accessor and
    :class:`~astrogwb.paper.config.catalogs.CatalogDefinition` reach a record
    down the same path. An already-built record is re-derived with
    :meth:`~astrogwb.metadata.PopulationMetadata.with_model_kwargs`.

    Imports the registry in its own body, and is not safe to call before
    ``configure_runtime``, for the reason :func:`population_model` gives.
    """
    from astrogwb.metadata import PopulationMetadata

    table = _load((root or Path()) / POPULATION_PATH, "population")
    return PopulationMetadata(
        model_name=table["model_name"],
        model_kwargs={**table.get("model_kwargs", {}), **kwargs},
        seed=seed,
    )
