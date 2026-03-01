"""Tests for the overlap reduction function."""

from __future__ import annotations

import math
import pathlib
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from asgwb.detector import (
    Detector,
    overlap_reduction_function,
    pairwise_overlap_reduction_function,
)
from asgwb.detector.overlap import (
    R_EARTH,
    _azimuth_bisector,
    _chord_distance,
    _final_course,
    _initial_course,
)


@pytest.fixture
def frequencies() -> np.ndarray:
    """Standard frequency array for tests."""
    return np.geomspace(20, 2048, 128)


@pytest.fixture
def detector_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def detector(detector_name: str) -> Detector:
    return Detector.from_file(detector_name)


@pytest.fixture
def detector_pair_names(request: pytest.FixtureRequest) -> tuple[str, str]:
    return request.param


@pytest.fixture
def detector_pair(detector_pair_names: tuple[str, str]) -> tuple[Detector, Detector]:
    det1_name, det2_name = detector_pair_names
    return Detector.from_file(det1_name), Detector.from_file(det2_name)


@pytest.fixture
def detector_network_names(request: pytest.FixtureRequest) -> tuple[str, ...]:
    return request.param


@pytest.fixture
def detector_network(detector_network_names: tuple[str, ...]) -> list[Detector]:
    return [Detector.from_file(name) for name in detector_network_names]


@pytest.fixture
def fixture_path() -> pathlib.Path:
    """Path to the GWFast reference fixture."""
    return pathlib.Path(__file__).parent / "fixtures" / "gwfast_orf_reference.npz"


@pytest.fixture
def et_fixture_path() -> pathlib.Path:
    """Path to the GWFast ET triangular reference fixture."""
    return (
        pathlib.Path(__file__).parent
        / "fixtures"
        / "gwfast_orf_reference_et_triangle.npz"
    )


@pytest.fixture
def load_fixture() -> Callable[[pathlib.Path], dict[str, np.ndarray]]:
    """Factory fixture to load NPZ fixture files."""

    def _loader(path: pathlib.Path) -> dict[str, np.ndarray]:
        if not path.exists():
            pytest.skip(
                f"Missing ORF fixture: {path.name}. "
                "Regenerate with `uv run --script scripts/generate_orf_fixtures.py`."
            )
        return dict(np.load(path))

    return _loader


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
        c_fwd = _initial_course(46.5, 30.6, -119.4, -90.8)
        c_rev = _final_course(46.5, 30.6, -119.4, -90.8)
        # The final course should differ from the initial course
        assert c_fwd != pytest.approx(c_rev, abs=1.0)


class TestAzimuthBisector:
    def test_wraparound(self) -> None:
        # Bisector of 10° and 350° should be 0° (not 180°).
        b = np.rad2deg(_azimuth_bisector(10.0, 350.0))
        assert b == pytest.approx(0.0, abs=1e-6)


class TestORFColocated:
    """Co-located, co-aligned detectors should give ORF = 1."""

    @pytest.mark.parametrize("detector_name", ["H1"], indirect=True)
    def test_identical_detectors(
        self, detector: Detector, frequencies: np.ndarray
    ) -> None:
        actual = overlap_reduction_function(frequencies, detector, detector)
        expected = np.ones_like(frequencies)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected)

    @pytest.mark.parametrize("detector_name", ["H1"], indirect=True)
    def test_scalar_frequency(self, detector: Detector) -> None:
        orf = overlap_reduction_function(np.array([10.0]), detector, detector)
        assert orf.shape == (1,)


class TestORFSymmetry:
    """ORF should be symmetric: gamma(f, d1, d2) == gamma(f, d2, d1)."""

    @pytest.mark.parametrize("detector_pair_names", [("H1", "L1")], indirect=True)
    def test_h1_l1_symmetry(
        self, frequencies: np.ndarray, detector_pair: tuple[Detector, Detector]
    ) -> None:
        det1, det2 = detector_pair
        orf_12 = overlap_reduction_function(frequencies, det1, det2)
        orf_21 = overlap_reduction_function(frequencies, det2, det1)
        np.testing.assert_allclose(orf_12, orf_21)


class TestPairwise:
    @pytest.mark.parametrize(
        "detector_network_names", [("H1", "L1", "V1")], indirect=True
    )
    def test_returns_correct_shape(
        self, frequencies: np.ndarray, detector_network: list[Detector]
    ) -> None:
        result = pairwise_overlap_reduction_function(frequencies, detector_network)
        assert result.shape == (3, 3, len(frequencies))

    @pytest.mark.parametrize("detector_pair_names", [("H1", "L1")], indirect=True)
    def test_values_match_individual(
        self, frequencies: np.ndarray, detector_pair: tuple[Detector, Detector]
    ) -> None:
        d1, d2 = detector_pair
        pairwise = pairwise_overlap_reduction_function(frequencies, [d1, d2])
        individual = overlap_reduction_function(frequencies, d1, d2)
        np.testing.assert_allclose(pairwise[0, 1, :], individual)

    @pytest.mark.parametrize(
        "detector_network_names", [("H1", "L1", "V1")], indirect=True
    )
    def test_symmetric(
        self, frequencies: np.ndarray, detector_network: list[Detector]
    ) -> None:
        result = pairwise_overlap_reduction_function(frequencies, detector_network)
        np.testing.assert_allclose(result, np.transpose(result, (1, 0, 2)))

    @pytest.mark.parametrize("detector_pair_names", [("H1", "L1")], indirect=True)
    def test_diagonal_is_normalized(
        self, frequencies: np.ndarray, detector_pair: tuple[Detector, Detector]
    ) -> None:
        d1, d2 = detector_pair
        result = pairwise_overlap_reduction_function(frequencies, [d1, d2])
        np.testing.assert_allclose(result[0, 0, :], 1.0)
        np.testing.assert_allclose(result[1, 1, :], 1.0)


class TestGWFastComparison:
    """Compare our ORF implementation against the reference fixture.

    The fixture is generated by ``scripts/generate_orf_fixtures.py`` and cached
    in CI.  To regenerate locally::

        uv run --script scripts/generate_orf_fixtures.py
    """

    @staticmethod
    def _pairs_from_fixture(fixture: dict[str, np.ndarray]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for key in fixture:
            if key == "frequencies":
                continue
            parts = key.split("_", maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Unexpected fixture key format: {key}")
            pairs.append((parts[0], parts[1]))
        return pairs

    def test_matches_reference(
        self,
        fixture_path: pathlib.Path,
        load_fixture: Callable[[pathlib.Path], dict[str, np.ndarray]],
    ) -> None:
        fixture = load_fixture(fixture_path)
        freqs = fixture["frequencies"]
        pairs = self._pairs_from_fixture(fixture)
        assert pairs, "No detector pairs found in ORF fixture"

        for det1_name, det2_name in pairs:
            reference = fixture[f"{det1_name}_{det2_name}"]
            det1 = Detector.from_file(det1_name)
            det2 = Detector.from_file(det2_name)
            ours = overlap_reduction_function(freqs, det1, det2)
            np.testing.assert_allclose(ours, reference, atol=1e-4)

    def test_et_triangle_sum_upper_pairs_matches_reference(
        self,
        et_fixture_path: pathlib.Path,
        load_fixture: Callable[[pathlib.Path], dict[str, np.ndarray]],
    ) -> None:
        fixture = load_fixture(et_fixture_path)
        freqs = fixture["frequencies"]
        reference_sum = fixture["sum_upper_pairs"]

        # Build a gwfast-aligned ET triangle from the ETS definition:
        # lat=40+31/60, long=9+25/60, xax=0, arm length=10 km.
        base = Detector.from_file("E1")
        ets_lat = 40.0 + 31.0 / 60.0
        ets_lon = 9.0 + 25.0 / 60.0
        arm_length_km = 10.0

        # Equilateral triangle vertices in local EN coordinates (km), centered.
        height = math.sqrt(3.0) * arm_length_km / 2.0
        en_offsets_km = [
            (-arm_length_km / 2.0, -height / 3.0),
            (arm_length_km / 2.0, -height / 3.0),
            (0.0, 2.0 * height / 3.0),
        ]

        def _to_lat_lon(east_km: float, north_km: float) -> tuple[float, float]:
            dlat = math.degrees(north_km / R_EARTH)
            dlon = math.degrees(east_km / (R_EARTH * math.cos(math.radians(ets_lat))))
            return ets_lat + dlat, ets_lon + dlon

        # Effective bisector orientations matching gwfast ETS branch indexing.
        xax_values = [-90.0, -30.0, 30.0]
        et_dets: list[Detector] = []
        for idx, ((east_km, north_km), xax) in enumerate(
            zip(en_offsets_km, xax_values, strict=True)
        ):
            lat, lon = _to_lat_lon(east_km, north_km)
            et_dets.append(
                replace(
                    base,
                    name=f"ETS_like_{idx}",
                    latitude=lat,
                    longitude=lon,
                    xarm_azimuth=(xax - 30.0) % 360.0,
                    yarm_azimuth=(xax + 30.0) % 360.0,
                )
            )

        pairwise = pairwise_overlap_reduction_function(freqs, et_dets)
        ours_sum = pairwise[0, 1, :] + pairwise[0, 2, :] + pairwise[1, 2, :]

        np.testing.assert_allclose(ours_sum, reference_sum, atol=1e-4)
