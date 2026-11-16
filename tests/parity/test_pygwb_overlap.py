"""The isotropic overlap reduction function against ``pygwb.orfs.calc_orf``.

pygwb evaluates the spherical-Earth closed form of arXiv:0903.0528: it projects
each arm bisector on the plane normal to the geocentric radius. On the
archive's exact sites (equator and pole) that plane is the local horizontal,
so both packages compute the same quantity. On real sites the geodetic and
geocentric verticals differ by up to 0.2 degrees and the arms are tilted,
which pygwb drops and this library keeps.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pytest
from numpy.typing import NDArray

from astrogwb.detector import (
    Detector,
    isotropic_overlap_reduction_function,
    load_detector,
)

REAL_SITES = ["H1", "K1", "L1", "V1"]
REAL_PAIRS = list(combinations(REAL_SITES, 2))


@pytest.mark.parametrize("pair", range(3), ids=["A-B", "A-C", "B-C"])
def test_orf_matches_pygwb_where_its_formula_is_exact(
    pair: int,
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    frequencies: NDArray[np.float64],
    exact_detectors: tuple[Detector, ...],
) -> None:
    first, second = list(combinations(exact_detectors, 2))[pair]

    orf = isotropic_overlap_reduction_function(frequencies, first, second)

    # pygwb's closed form cancels to ~1e-6 at small arguments; the ORF is O(0.1).
    np.testing.assert_allclose(
        orf, pygwb_reference["orf_exact"][pair], rtol=0.0, atol=1e-5
    )


@pytest.mark.parametrize(
    ("index", "names"),
    list(enumerate(REAL_PAIRS)),
    ids=["-".join(p) for p in REAL_PAIRS],
)
def test_orf_follows_pygwb_on_real_sites(
    index: int,
    names: tuple[str, str],
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    frequencies: NDArray[np.float64],
) -> None:
    """Convention check: sign, normalization and azimuth reference agree.

    The residual, up to 6e-3 for H1-V1, is pygwb's spherical-Earth
    approximation, not a tolerance on this library.
    """
    assert [str(name) for name in pygwb_reference["site_real_names"]] == REAL_SITES
    first, second = (load_detector(name) for name in names)

    orf = isotropic_overlap_reduction_function(frequencies, first, second)

    np.testing.assert_allclose(
        orf, pygwb_reference["orf_real"][index], rtol=0.0, atol=1e-2
    )
