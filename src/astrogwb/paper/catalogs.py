"""Reading and checking waveform catalogs."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from astrogwb.gwb.catalog import CatalogMetadata, PolarizationPowerData
from astrogwb.utils.io import load


def read_reference_catalog(
    path: str | Path,
) -> tuple[PolarizationPowerData, CatalogMetadata, int]:
    """The reference catalog at ``path``, its record and the seed it was drawn at.

    ``astrogwb model-catalog`` writes it; nothing here draws.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"no reference catalog at {path}")
    data, metadata, attrs = load(path, CatalogMetadata)
    return cast(PolarizationPowerData, data), metadata, int(attrs["seed"])


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


__all__ = ["read_reference_catalog", "validate_matching_frequency_grids"]
