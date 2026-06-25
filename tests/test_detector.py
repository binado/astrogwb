from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from gwmock_signal.detector import CustomDetector
from gwmock_signal.network import Network

from astrogwb.detector import (
    load_detector,
    overlap_reduction_function,
    pairwise_overlap_reduction_function,
    resolve_detector,
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


def test_resolve_detector_str_lookup() -> None:
    det = resolve_detector("H1")

    assert isinstance(det, CustomDetector)
    assert det.name == "H1"
    assert math.degrees(det.latitude_rad) == pytest.approx(46.45514666666667)


def test_load_detector_returns_custom_detector() -> None:
    det = load_detector("V1")

    assert isinstance(det, CustomDetector)
    assert det.name == "V1"
    assert math.degrees(det.latitude_rad) == pytest.approx(43.631414472222225)


def test_load_detector_cosmic_explorer_is_not_lal_prototype() -> None:
    # LAL's "C1" code is the Caltech 40m prototype (CIT_40, lat ~34.17), but
    # astrogwb's C1 is Cosmic Explorer at Hanford. load_detector must source
    # the latter so CE networks are built from the right site.
    det = load_detector("C1")

    assert math.degrees(det.latitude_rad) == pytest.approx(46.45514666666667)


def test_load_detector_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        load_detector("NOPE")


def test_resolve_detector_passthrough_custom_detector() -> None:
    custom = CustomDetector(
        name="T1",
        latitude_rad=0.1,
        longitude_rad=0.2,
        elevation_m=0.0,
        xarm_azimuth_rad=0.3,
        yarm_azimuth_rad=0.4,
    )

    assert resolve_detector(custom) is custom


def test_resolve_detector_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        resolve_detector("NOPE")


def test_chord_distance_antipodal() -> None:
    assert _chord_distance(0.0, 0.0, 0.0, math.pi) == pytest.approx(2.0 * R_EARTH)


def test_course_angles() -> None:
    assert _initial_course(0.0, 0.0, 0.0, math.radians(10.0)) == pytest.approx(
        math.pi / 2.0
    )
    assert _initial_course(
        0.0, math.radians(10.0), math.radians(45.0), math.radians(45.0)
    ) == pytest.approx(0.0)
    lat1, lat2 = math.radians(46.5), math.radians(30.6)
    lon1, lon2 = math.radians(-119.4), math.radians(-90.8)
    assert _final_course(lat1, lat2, lon1, lon2) != pytest.approx(
        _initial_course(lat1, lat2, lon1, lon2), abs=math.radians(1.0)
    )


def test_azimuth_bisector_wraparound() -> None:
    bisector = _azimuth_bisector(math.radians(10.0), math.radians(350.0))
    assert math.degrees(bisector) == pytest.approx(0.0)


def test_get_orf_low_alpha_uses_both_opening_angles() -> None:
    alpha = np.array([_LOW_ALPHA_THRESHOLD * 0.5])
    actual = _get_orf(alpha, 0.9, 0.37, 1.1, math.pi / 2.0, math.pi / 3.0)
    expected = np.cos(4.0 * 0.37) * np.sin(math.pi / 2.0) * np.sin(math.pi / 3.0)

    np.testing.assert_allclose(actual, np.array([expected]))


def test_orf_colocated_is_normalized(frequencies: np.ndarray) -> None:
    actual = overlap_reduction_function(frequencies, "H1", "H1")

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


def test_orf_accepts_mixed_str_and_custom_detector(frequencies: np.ndarray) -> None:
    custom_h1 = resolve_detector("H1")

    from_str = overlap_reduction_function(frequencies, "H1", "L1")
    mixed = overlap_reduction_function(frequencies, custom_h1, "L1")

    np.testing.assert_allclose(mixed, from_str)


def test_pairwise_overlap_shape_and_symmetry(frequencies: np.ndarray) -> None:
    actual = pairwise_overlap_reduction_function(frequencies, ["H1", "L1", "V1"])

    assert actual.shape == (3, 3, frequencies.shape[0])
    np.testing.assert_allclose(actual, np.transpose(actual, (1, 0, 2)))
    np.testing.assert_allclose(actual[0, 0, :], 1.0)


@pytest.mark.integration
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
        ours = overlap_reduction_function(freqs, det1_name, det2_name)
        np.testing.assert_allclose(ours, reference, atol=1e-4)


@pytest.mark.integration
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


def _opening_angle_deg(det: CustomDetector) -> float:
    diff = (
        (det.xarm_azimuth_rad - det.yarm_azimuth_rad + math.pi) % (2.0 * math.pi)
    ) - math.pi
    return math.degrees(abs(diff))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("preset", "expected_opening"),
    [
        ("ET-Sardinia", 60.0),
        ("ET-Triangle-Sardinia", 60.0),
        ("ET-2L-Aligned", 90.0),
        ("ET-2L-Misaligned", 90.0),
    ],
)
def test_gwmock_et_preset_opening_angles(preset: str, expected_opening: float) -> None:
    """gwmock ET presets carry the arm opening angles astrogwb's ORF assumes."""
    network = Network.from_name(preset)

    for detector in network.detector_names:
        assert isinstance(detector, CustomDetector)
        assert _opening_angle_deg(detector) == pytest.approx(expected_opening, abs=0.1)


@pytest.mark.integration
def test_gwmock_et_triangle_orf_matches_geometry_table(
    frequencies: np.ndarray,
) -> None:
    """astrogwb's ORF agrees on the co-located ET triangle across geometry sources.

    The gwmock ``ET-Sardinia`` preset and astrogwb's ``geometry.toml``
    E1/E2/E3 rows describe 10 km, 60-degree triangles at nearby (but not
    identical) sites, so their ORF triangle sums coincide to within a small
    geometric tolerance. This cross-validates the gwmock-preset geometry
    path against the validated angle table.
    """
    preset = list(Network.from_name("ET-Sardinia").detector_names)
    preset_pw = pairwise_overlap_reduction_function(frequencies, preset)
    preset_sum = preset_pw[0, 1, :] + preset_pw[0, 2, :] + preset_pw[1, 2, :]

    table_pw = pairwise_overlap_reduction_function(frequencies, ["E1", "E2", "E3"])
    table_sum = table_pw[0, 1, :] + table_pw[0, 2, :] + table_pw[1, 2, :]

    np.testing.assert_allclose(preset_sum, table_sum, atol=5e-3)
