"""Accessors for the tables the shared config layers declare.

``[fiducials]``, ``[priors]`` and ``[networks]``, plus builders for the default
draw ``[catalog]``: :func:`waveform_generator`, :func:`population_model` and
:func:`population_metadata`::

    from astrogwb.paper.config import fiducials, networks, priors, waveform_generator

    fid = fiducials()
    detectors = networks()["ET-2L-aligned-CE-Hanford"]
    generator = waveform_generator()

Every accessor takes keyword overrides, merged over the file::

    fiducials(H0=70.0)
    priors(xi_0={"dist": "Uniform", "kwargs": {"low": 0.1, "high": 5.0}})
    networks(**{"ET-2L-aligned": ("S1", "R1", "C1")})

They are functions rather than module-level dicts, so importing does no file
I/O. Paths are relative to the working directory (the repository root); tests
pass ``root=``. Each accessor returns a fresh copy of one cached merge; a
long-lived session sees layer edits only after ``_shared.cache_clear()``.
"""

from __future__ import annotations

import copy
from functools import cache
from pathlib import Path
from typing import Any

import numpyro.distributions as dist

from astrogwb.detector.config import DetectorRegistry
from astrogwb.distributions.config import DistributionConfig
from astrogwb.paper.config.runs import load_base
from astrogwb.paper.utils import deep_merge
from astrogwb.populations import PopulationMetadata
from astrogwb.populations.registry import Population
from astrogwb.waveform import (
    WaveformGenerator,
    WaveformMetadata,
)

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

    The point catalogs are drawn at and spectra evaluated at, not the
    injection. Every fiducial carries a prior in :func:`priors`.

    Keyword arguments override the file, and may name a fiducial the file does
    not declare; keeping it consistent with :func:`priors` is the caller's job.
    """
    table = {**_table(root, "fiducials"), **kwargs}
    return {name: float(value) for name, value in table.items()}


def priors(root: Path | None = None, **kwargs: Any) -> dict[str, dist.Distribution]:
    """The prior for each parameter, materialized from the shared ``[priors]``.

    One entry per fiducial. Calling this does not initialize the XLA backend.

    Keyword arguments override the file, and may name a parameter the file does
    not declare. A value may be a wire-format spec
    (``{"dist": ..., "kwargs": {...}}``) or an already-built ``numpyro``
    distribution, which is passed through unchanged.
    """
    table = {**_table(root, "priors"), **kwargs}
    return {
        name: spec
        if isinstance(spec, dist.Distribution)
        else DistributionConfig.model_validate(spec).build()
        for name, spec in table.items()
    }


def networks(root: Path | None = None, **kwargs: Any) -> dict[str, tuple[str, ...]]:
    """Detector networks by name: the shared ``[networks]`` table.

    Membership only; the figure order is
    :data:`astrogwb.paper.plotting.DETECTOR_NETWORKS`. Keyword arguments
    override the file; names are hyphenated, so unpack them from a mapping::

        networks(**{"ET-2L-aligned": ("S1", "R1", "C1")})
    """
    table = {**_table(root, "networks"), **kwargs}
    return {name: tuple(detectors) for name, detectors in table.items()}


def detector_registry(root: Path | None = None, **overrides: Any) -> DetectorRegistry:
    """Resolve shared detector settings lazily and return an independent registry.

    Overrides may contain ``detectors`` and ``networks`` tables, deep-merged
    over the shared file. Runtime objects are built only by its build methods.
    """
    shared = {name: _table(root, name) for name in ("detectors", "networks")}
    settings = deep_merge(shared, overrides)
    unknown = settings.keys() - {"detectors", "networks"}
    if unknown:
        raise ValueError(f"unknown detector registry fields: {sorted(unknown)}")
    return DetectorRegistry.model_validate(settings)


def waveform_generator(root: Path | None = None, **kwargs: Any) -> WaveformGenerator:
    """Build the waveform generator the default draw uses.

    Keyword arguments override the file and are validated through
    :class:`~astrogwb.waveform.WaveformMetadata`. Not safe before
    ``configure_runtime``.
    """
    return waveform_metadata(root, **kwargs).build()


def waveform_metadata(root: Path | None = None, **kwargs: Any) -> WaveformMetadata:
    """The waveform the default draw, ``[catalog]``, uses, as a record.

    Keyword arguments override the file. Safe before ``configure_runtime``.
    """
    settings = {**_table(root, "catalog", "waveform"), **kwargs}
    return WaveformMetadata.model_validate(settings)


def population_model(
    root: Path | None = None, **kwargs: float | bool | str
) -> Population:
    """Build the population the default draw, ``[catalog]``, is drawn from.

    Keyword arguments override ``model_kwargs``; an unknown key raises::

        population_model(n_grid=256, minimum_redshift=0.3)

    Not safe before ``configure_runtime``.
    """
    return population_metadata(root).with_model_kwargs(**kwargs).build()


def population_metadata(
    root: Path | None = None, **kwargs: float | bool
) -> PopulationMetadata:
    """The record the default draw, ``[catalog]``, carries for its population.

    Keyword arguments override ``model_kwargs``. Safe before
    ``configure_runtime``; building the record's population is not.
    """
    table = _table(root, "catalog", "population")
    table["model_kwargs"] = {**table.get("model_kwargs", {}), **kwargs}
    return PopulationMetadata.model_validate(table)
