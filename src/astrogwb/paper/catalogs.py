"""Checks between waveform catalogs."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


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


__all__ = ["validate_matching_frequency_grids"]
