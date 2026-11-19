"""Compare tensor overlap reduction functions for idealized detector geometry."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pytest
from numpy.typing import NDArray
from tolerances import assert_close

from astrogwb.detector import (
    Detector,
    isotropic_overlap_reduction_function,
)


@pytest.mark.parametrize("pair", range(3), ids=["A-B", "A-C", "B-C"])
def test_orf_matches_pygwb_where_its_formula_is_exact(
    pair: int,
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    frequencies: NDArray[np.float64],
    exact_detectors: tuple[Detector, ...],
) -> None:
    first, second = list(combinations(exact_detectors, 2))[pair]

    orf = isotropic_overlap_reduction_function(frequencies, first, second)

    assert_close(orf, pygwb_reference["orf_exact"][pair])


def test_colocated_aligned_orf_matches_shared_normalization(
    frequencies: NDArray[np.float64],
    exact_detectors: tuple[Detector, ...],
) -> None:
    # Both conventions normalize co-located, co-aligned right-angle detectors
    # to unity. For separated detectors pygwb uses a spherical-Earth formula
    # with geocentric tangent planes. astrogwb uses the full response tensors,
    # retaining arm tilts and the local geodetic vertical at real sites.
    detector = exact_detectors[0]
    orf = isotropic_overlap_reduction_function(frequencies, detector, detector)

    assert_close(orf, np.ones_like(frequencies))
