from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from gwmock_signal.detector import CustomDetector

from astrogwb.detector import (
    Detector,
    PowerSpectralDensity,
    effective_psd,
    overlap_reduction_function,
    pairwise_overlap_reduction_function,
)
from astrogwb.detector.overlap import (
    R_EARTH,
    _LOW_ALPHA_THRESHOLD,
    _azimuth_bisector,
    _chord_distance,
    _final_course,
    _get_orf,
    _initial_course,
)


@pytest.fixture
def frequencies() -> np.ndarray:
    return np.geomspace(20, 2048, 128)


@pytest.fixture
def load_fixture() -> Callable[[Path], dict[str, np.ndarray]]:
    def _loader(path: Path) -> dict[str, np.ndarray]:
        if not path.exists():
            pytest.skip(f"Missing ORF fixture: {path.name}.")
        return dict(np.load(path))

    return _loader


def test_psd_loads_and_interpolates_noise_curve() -> None:
    psd = PowerSpectralDensity.from_noise_curve_dir("AplusDesign_psd.txt")

    values = psd.evaluate(np.array([20.0, 100.0]))

    assert psd.file.name == "AplusDesign_psd.txt"
    assert values.shape == (2,)
    assert np.all(np.isfinite(values))


def test_detector_from_file() -> None:
    detector = Detector.from_file("H1")

    assert detector.name == "H1"
    assert detector.length == 4.0
    assert detector.minimum_frequency == 20.0
    assert isinstance(detector.psd, PowerSpectralDensity)


def test_chord_distance_antipodal() -> None:
    assert _chord_distance(0.0, 0.0, 0.0, 180.0) == pytest.approx(2.0 * R_EARTH)


def test_course_angles() -> None:
    assert _initial_course(0.0, 0.0, 0.0, 10.0) == pytest.approx(90.0)
    assert _initial_course(0.0, 10.0, 45.0, 45.0) == pytest.approx(0.0)
    assert _final_course(46.5, 30.6, -119.4, -90.8) != pytest.approx(
        _initial_course(46.5, 30.6, -119.4, -90.8),
        abs=1.0,
    )


def test_azimuth_bisector_wraparound() -> None:
    assert np.rad2deg(_azimuth_bisector(10.0, 350.0)) == pytest.approx(0.0)


def test_get_orf_low_alpha_uses_both_opening_angles() -> None:
    alpha = np.array([_LOW_ALPHA_THRESHOLD * 0.5])
    actual = _get_orf(alpha, 0.9, 0.37, 1.1, math.pi / 2.0, math.pi / 3.0)
    expected = np.cos(4.0 * 0.37) * np.sin(math.pi / 2.0) * np.sin(math.pi / 3.0)

    np.testing.assert_allclose(actual, np.array([expected]))


def test_orf_colocated_is_normalized(frequencies: np.ndarray) -> None:
    detector = Detector.from_file("H1")

    actual = overlap_reduction_function(frequencies, detector, detector)

    np.testing.assert_allclose(actual, np.ones_like(frequencies))


def test_orf_accepts_gwmock_custom_detector(frequencies: np.ndarray) -> None:
    detector = CustomDetector(
        name="T1",
        latitude_rad=0.0,
        longitude_rad=0.0,
        elevation_m=0.0,
        xarm_azimuth_rad=0.0,
        yarm_azimuth_rad=math.pi / 2.0,
    )

    actual = overlap_reduction_function(frequencies, detector, detector)

    np.testing.assert_allclose(actual, np.ones_like(frequencies))


def test_pairwise_overlap_shape_and_symmetry(frequencies: np.ndarray) -> None:
    detectors = [
        Detector.from_file("H1"),
        Detector.from_file("L1"),
        Detector.from_file("V1"),
    ]

    actual = pairwise_overlap_reduction_function(frequencies, detectors)

    assert actual.shape == (3, 3, frequencies.shape[0])
    np.testing.assert_allclose(actual, np.transpose(actual, (1, 0, 2)))
    np.testing.assert_allclose(actual[0, 0, :], 1.0)


def test_effective_psd_inf_for_insufficient_network(frequencies: np.ndarray) -> None:
    actual = effective_psd(frequencies, [Detector.from_file("H1")])

    assert actual.shape == frequencies.shape
    assert np.all(np.isinf(actual))


def test_effective_psd_finite_for_detector_pair(frequencies: np.ndarray) -> None:
    detectors = [Detector.from_file("H1"), Detector.from_file("L1")]

    actual = effective_psd(frequencies, detectors)

    assert actual.shape == frequencies.shape
    assert np.any(np.isfinite(actual))


def test_matches_gwfast_reference(
    load_fixture: Callable[[Path], dict[str, np.ndarray]],
) -> None:
    fixture = load_fixture(
        Path(__file__).parent / "fixtures" / "gwfast_orf_reference.npz"
    )
    freqs = fixture["frequencies"]

    for key, reference in fixture.items():
        if key == "frequencies":
            continue
        det1_name, det2_name = key.split("_", maxsplit=1)
        ours = overlap_reduction_function(
            freqs,
            Detector.from_file(det1_name),
            Detector.from_file(det2_name),
        )
        np.testing.assert_allclose(ours, reference, atol=1e-4)


def test_et_triangle_sum_upper_pairs_matches_reference(
    load_fixture: Callable[[Path], dict[str, np.ndarray]],
) -> None:
    fixture = load_fixture(
        Path(__file__).parent / "fixtures" / "gwfast_orf_reference_et_triangle.npz"
    )
    freqs = fixture["frequencies"]
    ets_lat = 40.0 + 31.0 / 60.0
    ets_lon = 9.0 + 25.0 / 60.0
    arm_length_km = 10.0
    height = math.sqrt(3.0) * arm_length_km / 2.0
    en_offsets_km = [
        (-arm_length_km / 2.0, -height / 3.0),
        (arm_length_km / 2.0, -height / 3.0),
        (0.0, 2.0 * height / 3.0),
    ]
    xax_values = [-90.0, -30.0, 30.0]

    def _to_lat_lon(east_km: float, north_km: float) -> tuple[float, float]:
        dlat = math.degrees(north_km / R_EARTH)
        dlon = math.degrees(east_km / (R_EARTH * math.cos(math.radians(ets_lat))))
        return ets_lat + dlat, ets_lon + dlon

    detectors = []
    for idx, ((east_km, north_km), xax) in enumerate(
        zip(en_offsets_km, xax_values, strict=True)
    ):
        lat, lon = _to_lat_lon(east_km, north_km)
        detectors.append(
            CustomDetector(
                name=f"ETS_like_{idx}",
                latitude_rad=math.radians(lat),
                longitude_rad=math.radians(lon),
                elevation_m=0.0,
                xarm_azimuth_rad=math.radians((xax - 30.0) % 360.0),
                yarm_azimuth_rad=math.radians((xax + 30.0) % 360.0),
            )
        )

    pairwise = pairwise_overlap_reduction_function(freqs, detectors)
    ours_sum = pairwise[0, 1, :] + pairwise[0, 2, :] + pairwise[1, 2, :]

    np.testing.assert_allclose(ours_sum, fixture["sum_upper_pairs"], atol=1e-4)
