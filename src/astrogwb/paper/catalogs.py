"""Loading the two catalogs a run samples against, and the checks between them.

A catalog is a file: ``outputs/catalogs/<key>.h5``, where ``<key>`` is the
content hash of the :class:`~astrogwb.metadata.CatalogRequest` a run's role
resolves to. The workflow builds them with ``scripts/generate_catalog.py``;
:func:`run_catalog` reaches the same files from a notebook, generating one on
a miss. Either way the density its samples follow comes back off the file's
own population record rather than being reassembled from the run config.

What used to live here has mostly moved to where it belongs:
:meth:`~astrogwb.catalog.PolarizationPowerCatalog.restrict_redshift`
narrows the samples and the recorded population together, and the proposal density is evaluated by
:func:`~astrogwb.importance.spectral.build_importance_spectrum` directly from
the catalog's own source model. Fiducial GW propagation is gone
entirely: which propagation law applies is now part of the population
declaration, so a catalog's power and its recorded distances are consistent by
construction instead of being patched up at the call site.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from astrogwb.catalog import (
    PolarizationPowerCatalog,
    check_catalog_answers,
    load_or_generate,
)
from astrogwb.metadata import CatalogRequest
from astrogwb.paper.config.mcmc import build_run_config
from astrogwb.paper.config.runs import CATALOGS_ROOT, assemble_run


def load_run_catalog(
    path: Path | str, *, label: str, request: CatalogRequest | None = None
) -> PolarizationPowerCatalog:
    """Load one catalog file, validating its format and its population record.

    ``label`` is the role -- ``"injection"`` or ``"proposal"`` -- and is what
    identifies the catalog in error messages. Given a ``request``, the file
    must also record exactly that request, which is how a run refuses a file
    handed to the wrong role or built from a draw it no longer asks for.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} catalog not found: {path}")
    try:
        catalog = PolarizationPowerCatalog.load(path)
        if request is not None:
            check_catalog_answers(catalog, request, label=str(path))
    except ValueError as error:
        raise ValueError(f"{label} catalog {path}: {error}") from error
    return catalog


def run_catalog(
    experiment: str,
    run: str,
    role: str,
    *,
    cache_dir: Path | str = CATALOGS_ROOT,
    root: Path | None = None,
) -> PolarizationPowerCatalog:
    """The catalog one role of a committed run samples against.

    Resolves the run's config into its request and returns the cached file
    the workflow would have built, generating it on a miss. This is the
    notebook's way in: no path to hard-code, and no way to pair a run with a
    catalog it does not ask for. Generation reaches JAX, so call this after
    :func:`~astrogwb.paper.runtime.configure_runtime`.
    """
    config = build_run_config(assemble_run(experiment, run, root=root))
    return load_or_generate(config.catalog_request(role), cache_dir)


def validate_matching_frequency_grids(
    injection_frequencies: ArrayLike,
    proposal_frequencies: ArrayLike,
    *,
    label: str = "injection and proposal",
) -> None:
    """Require two waveform catalogs to share the exact frequency grid."""
    injection_frequencies = np.asarray(injection_frequencies)
    proposal_frequencies = np.asarray(proposal_frequencies)
    if not np.array_equal(injection_frequencies, proposal_frequencies):
        raise ValueError(f"{label} catalogs must have identical frequency grids")


__all__ = ["load_run_catalog", "run_catalog", "validate_matching_frequency_grids"]
