"""Loading the two catalog files a run names, and the checks between them.

A catalog is a file: ``outputs/catalogs/<name>.h5``, built once by
``scripts/generate_catalog.py`` from ``config/catalogs/<name>.json``. A
run names one for each role, so there is nothing to compose here -- loading is
just reading the file, and the density its samples follow comes back off the
file's own population record rather than being reassembled from the run config.

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

from astrogwb.catalog import PolarizationPowerCatalog


def load_run_catalog(path: Path | str, *, label: str) -> PolarizationPowerCatalog:
    """Load one catalog file, validating its format and its population record.

    ``label`` is the role -- ``"injection"`` or ``"proposal"`` -- and is what
    identifies the catalog in error messages.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} catalog not found: {path}")
    try:
        return PolarizationPowerCatalog.load(path)
    except ValueError as error:
        raise ValueError(f"{label} catalog {path}: {error}") from error


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


__all__ = ["load_run_catalog", "validate_matching_frequency_grids"]
