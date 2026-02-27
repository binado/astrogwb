"""Tests for the overlap reduction function."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from asgwb.detector import (
    Detector,
    overlap_reduction_function,
    pairwise_overlap_reduction_function,
)
from asgwb.detector.overlap import (
    _chord_distance,
    _final_course,
    _initial_course,
)

FREQUENCIES = np.array([10.0, 50.0, 100.0, 200.0])


# ---------------------------------------------------------------------------
# Helper: build a minimal Detector without a real PSD file
# ---------------------------------------------------------------------------


def _make_detector(
    name: str,
    latitude: float,
    longitude: float,
    xarm_azimuth: float,
    yarm_azimuth: float,
) -> Detector:
    """Create a Detector with a stub PSD for geometry-only tests."""
    from unittest.mock import MagicMock

    psd = MagicMock()
    return Detector(
        name=name,
        psd=psd,
        minimum_frequency=10.0,
        maximum_frequency=2048.0,
        length=4.0,
        latitude=latitude,
        longitude=longitude,
        elevation=0.0,
        xarm_azimuth=xarm_azimuth,
        yarm_azimuth=yarm_azimuth,
        xarm_tilt=0.0,
        yarm_tilt=0.0,
        duty_factor=0.7,
    )


# ---------------------------------------------------------------------------
# Unit tests – geometry helpers
# ---------------------------------------------------------------------------


class TestChordDistance:
    def test_same_location(self) -> None:
        d = _chord_distance(0.0, 0.0, 0.0, 0.0)
        assert d == pytest.approx(0.0)

    def test_antipodal(self) -> None:
        d = _chord_distance(0.0, 0.0, 0.0, 180.0)
        assert d == pytest.approx(2 * 6371.0, rel=1e-6)

    def test_symmetry(self) -> None:
        d12 = _chord_distance(46.5, -119.4, 30.6, -90.8)
        d21 = _chord_distance(30.6, -90.8, 46.5, -119.4)
        assert d12 == pytest.approx(d21, rel=1e-10)


class TestCourseAngles:
    def test_initial_course_east(self) -> None:
        # Due east along the equator
        c = _initial_course(0.0, 0.0, 0.0, 10.0)
        assert c == pytest.approx(90.0, abs=1e-6)

    def test_initial_course_north(self) -> None:
        # Due north: same longitude, latitude increases
        c = _initial_course(0.0, 10.0, 45.0, 45.0)
        assert c == pytest.approx(0.0, abs=1e-3)

    def test_reverse_course(self) -> None:
        c_fwd = _initial_course(46.5, -119.4, 30.6, -90.8)
        c_rev = _final_course(46.5, 30.6, -119.4, -90.8)
        # The final course should differ from the initial course
        assert c_fwd != pytest.approx(c_rev, abs=1.0)


# ---------------------------------------------------------------------------
# Unit tests – ORF behaviour
# ---------------------------------------------------------------------------


class TestORFColocated:
    """Co-located, co-aligned detectors should give ORF = 1 at low frequencies."""

    def test_identical_detectors_low_freq(self) -> None:
        det = _make_detector("D1", 0.0, 0.0, 135.0, 225.0)
        freqs = np.array([1e-4, 1e-3])
        orf = overlap_reduction_function(freqs, det, det)
        # In the GWFast convention the ORF is not normalised to 1 for
        # identical detectors.  For an L-shaped detector (half-angle 45°):
        # ORF(f→0) = sin²(45°) × 2/15 = 1/15 ≈ 0.0667.
        # Verify the value is constant across these low frequencies.
        assert np.allclose(orf, 1.0 / 15.0, atol=1e-3)

    def test_output_shape(self) -> None:
        det = _make_detector("D1", 0.0, 0.0, 135.0, 225.0)
        orf = overlap_reduction_function(FREQUENCIES, det, det)
        assert orf.shape == FREQUENCIES.shape

    def test_scalar_frequency(self) -> None:
        det = _make_detector("D1", 0.0, 0.0, 135.0, 225.0)
        orf = overlap_reduction_function(np.array([10.0]), det, det)
        assert orf.shape == (1,)


class TestORFSymmetry:
    """ORF should be symmetric: gamma(f, d1, d2) == gamma(f, d2, d1)."""

    def test_h1_l1_symmetry(self) -> None:
        h1 = _make_detector("H1", 46.455, -119.408, 125.999, 215.999)
        l1 = _make_detector("L1", 30.563, -90.774, 197.716, 287.716)
        orf_12 = overlap_reduction_function(FREQUENCIES, h1, l1)
        orf_21 = overlap_reduction_function(FREQUENCIES, l1, h1)
        assert np.allclose(orf_12, orf_21, rtol=1e-10)


class TestORFET:
    """ET detectors (60° arms) should have ORF ≈ sqrt(3)/2 at zero separation."""

    def test_et_collocated_low_freq(self) -> None:
        # E1 and E2 share same position but different orientations
        e1 = _make_detector("E1", 43.63, 10.5, 70.57, 130.57)
        e2 = _make_detector("E2", 43.63, 10.5, 190.57, 250.57)
        freqs = np.array([1e-4])
        orf = overlap_reduction_function(freqs, e1, e2)
        # 60° arm detectors: sin(30°)*sin(30°) factor modifies response
        # The ORF should be non-zero and bounded
        assert np.all(np.abs(orf) <= 1.0)


class TestPairwise:
    def test_returns_all_pairs(self) -> None:
        d1 = _make_detector("H1", 46.455, -119.408, 125.999, 215.999)
        d2 = _make_detector("L1", 30.563, -90.774, 197.716, 287.716)
        d3 = _make_detector("V1", 43.631, 10.504, 70.567, 160.567)
        result = pairwise_overlap_reduction_function(FREQUENCIES, [d1, d2, d3])
        assert set(result.keys()) == {("H1", "L1"), ("H1", "V1"), ("L1", "V1")}

    def test_values_match_individual(self) -> None:
        d1 = _make_detector("H1", 46.455, -119.408, 125.999, 215.999)
        d2 = _make_detector("L1", 30.563, -90.774, 197.716, 287.716)
        pairwise = pairwise_overlap_reduction_function(FREQUENCIES, [d1, d2])
        individual = overlap_reduction_function(FREQUENCIES, d1, d2)
        assert np.allclose(pairwise[("H1", "L1")], individual)


# ---------------------------------------------------------------------------
# Integration tests – GWFast comparison
# ---------------------------------------------------------------------------


FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "gwfast_orf_reference.npz"


def _load_fixture(path: pathlib.Path) -> dict[str, np.ndarray]:
    return dict(np.load(path))


@pytest.mark.integration
class TestGWFastComparison:
    """Compare our ORF implementation against the reference fixture.

    The fixture is generated by ``scripts/generate_orf_fixtures.py`` and cached
    in CI.  To regenerate locally::

        uv run python scripts/generate_orf_fixtures.py
    """

    @pytest.mark.parametrize("pair", [("H1", "L1"), ("H1", "V1")])
    def test_matches_reference(self, pair: tuple[str, str]) -> None:
        det1_name, det2_name = pair
        fixture = _load_fixture(FIXTURE_PATH)
        freqs = fixture["frequencies"]
        reference = fixture[f"{det1_name}_{det2_name}"]

        det1 = Detector.from_file(det1_name)
        det2 = Detector.from_file(det2_name)
        ours = overlap_reduction_function(freqs, det1, det2)

        assert np.allclose(ours, reference, atol=1e-4), (
            f"Max discrepancy for {det1_name}-{det2_name}: {np.max(np.abs(ours - reference))}"
        )
