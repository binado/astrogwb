"""Configuration support modules, and the shared scientific values themselves.

The leaf modules (:mod:`~astrogwb.paper.config.runs`,
:mod:`~astrogwb.paper.config.mcmc`, :mod:`~astrogwb.paper.config.catalogs`) are
not re-exported; import them explicitly.

What this package *does* expose is the three shared tables that
``config/fiducials.json``, ``config/priors.json`` and ``config/networks.json``
own -- the same bytes the workflow merges into every run. Before they lived
here, the notebook and the figure scripts each kept a hand-written copy, and
those copies drifted: the notebook sampled ``local_merger_rate`` under a prior
that excluded its own fiducial. Consume them from here instead::

    from astrogwb.paper.config import fiducials, networks, priors

    fid = fiducials()
    detectors = networks()["ET-2L-aligned-CE-Hanford"]

**These are functions, not module-level dicts, and that is load-bearing.** The
``Snakefile`` imports :mod:`astrogwb.paper.config.runs` to build the DAG, which
executes this module; eager dicts would mean file I/O at import (failing from
any working directory but the repository root) and, for the priors, a numpyro
import on every ``--dry-run``. :func:`priors` therefore imports
:func:`~astrogwb.paper.config.mcmc.materialize_prior` inside its own body. A
subprocess test in ``tests/paper/test_cli.py`` pins both halves.

Paths are relative to the working directory, which for the workflow and every
script is the repository root -- the same contract as
:mod:`astrogwb.paper.config.runs`. Tests, which pytest may invoke from
anywhere, pass ``root=`` explicitly.

Each accessor caches its parse and hands back a fresh copy, so a caller that
mutates what it got does not poison the cache for everyone else. A long-lived
Jupyter session will not see an edit to the JSON until ``fiducials.cache_clear()``.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrogwb.paper.config.runs import FIDUCIALS_PATH, NETWORKS_PATH, PRIORS_PATH
from astrogwb.paper.utils import load_mapping

if TYPE_CHECKING:
    from numpyro.distributions import Distribution

__all__ = ["fiducials", "networks", "priors"]


@cache
def _load(path: Path, key: str) -> dict[str, Any]:
    """Parse one single-key layer file and return the table it declares."""
    raw = load_mapping(path)
    table = raw.get(key)
    if not isinstance(table, dict):
        raise TypeError(f"{path} must declare a [{key}] table")
    return table


def fiducials(root: Path | None = None) -> dict[str, float]:
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
    """
    return {
        name: float(value)
        for name, value in _load((root or Path()) / FIDUCIALS_PATH, "fiducials").items()
    }


def priors(root: Path | None = None) -> dict[str, Distribution]:
    """The prior for each parameter, materialized from ``config/priors.json``.

    One entry per fiducial. The returned distributions hold plain Python
    floats and evaluating none of them touches JAX, so calling this does not
    initialize the XLA backend -- ``configure_runtime`` may still run after it.

    Note this is the *inference* prior. A diagnostic that scans a parameter is
    free to scan wider (see ``GRID_SCAN_RANGES`` in
    ``scripts/importance_weights_grid.py``); it just has to say so.
    """
    # Imported here, not at module scope: `mcmc` reaches pydantic, and the
    # Snakefile's DAG construction imports this package via `config.runs`.
    from astrogwb.paper.config.mcmc import materialize_prior

    return {
        name: materialize_prior(spec)
        for name, spec in _load((root or Path()) / PRIORS_PATH, "priors").items()
    }


def networks(root: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Detector networks by name, from ``config/networks.json``.

    Each run names one of these keys as ``analysis.network``; ``RunConfig``
    resolves it to the detector list recorded in the chain's own config. The
    names resolve further, to geometry and sensitivity, through
    ``astrogwb.detector``.

    Membership and content live here; *order* does not. The ordered legend of
    the network-comparison figures is
    :data:`astrogwb.paper.plotting.DETECTOR_NETWORKS`, because order is
    presentation -- and because a JSON object is not an ordered thing.
    """
    return {
        name: tuple(detectors)
        for name, detectors in _load(
            (root or Path()) / NETWORKS_PATH, "networks"
        ).items()
    }
