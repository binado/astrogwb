from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from astrogwb.detector import (
    load_detector,
    overlap_reduction_function,
    pairwise_overlap_reduction_function,
    resolve_detector,
)
from astrogwb.detector.overlap import R_EARTH
from gwmock_signal.detector import CustomDetector
from gwmock_signal.network import Network


@pytest.mark.parametrize("loader", [resolve_detector, load_detector])
@pytest.mark.parametrize("name", ["H1", "V1"])
def test_detector_loaders_return_custom_detector(
    loader: Callable[[str], CustomDetector], name: str
) -> None:
    det = loader(name)

    assert isinstance(det, CustomDetector)
    assert det.name == name
    assert -math.pi / 2.0 < det.latitude_rad < math.pi / 2.0


def test_load_detector_cosmic_explorer_is_not_lal_prototype() -> None:
    # LAL's "C1" code is the Caltech 40m prototype (CIT_40, lat ~34.17), but
    # astrogwb's C1 is Cosmic Explorer at Hanford. load_detector must source
    # the latter so CE networks are built from the right site.
    det = load_detector("C1")

    assert math.degrees(det.latitude_rad) == pytest.approx(46.455, abs=0.01)


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


def test_orf_colocated_is_normalized(frequencies: np.ndarray) -> None:
    actual = overlap_reduction_function(frequencies, "H1", "H1")

    np.testing.assert_allclose(actual, np.ones_like(frequencies))


def test_orf_continuous_across_low_alpha_threshold() -> None:
    # For two detectors ~1 km apart, alpha = 2*pi*f*d/c crosses the internal
    # low-alpha series-expansion threshold near f ~= 95 Hz. The ORF must be
    # continuous across that branch switch: a bad expansion or threshold would
    # show up as a jump on a fine frequency grid straddling it.
    dlat = 1.0 / R_EARTH  # ~1 km separation along a meridian
    det_a = CustomDetector(
        name="A",
        latitude_rad=0.0,
        longitude_rad=0.0,
        elevation_m=0.0,
        xarm_azimuth_rad=0.0,
        yarm_azimuth_rad=math.pi / 2.0,
    )
    det_b = CustomDetector(
        name="B",
        latitude_rad=dlat,
        longitude_rad=0.0,
        elevation_m=0.0,
        xarm_azimuth_rad=0.0,
        yarm_azimuth_rad=math.pi / 2.0,
    )
    freqs = np.linspace(50.0, 150.0, 2001)

    orf = overlap_reduction_function(freqs, det_a, det_b)

    assert np.all(np.isfinite(orf))
    assert np.max(np.abs(np.diff(orf))) < 1e-3


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
    load_orf_fixture: Callable[[str], dict[str, np.ndarray]],
) -> None:
    fixture = load_orf_fixture("gwfast_orf_reference")
    freqs = fixture["frequencies"]

    for key, reference in fixture.items():
        if key == "frequencies":
            continue
        det1_name, det2_name = key.split("_", maxsplit=1)
        ours = overlap_reduction_function(freqs, det1_name, det2_name)
        np.testing.assert_allclose(ours, reference, atol=1e-4)


@pytest.mark.integration
def test_et_triangle_sum_upper_pairs_matches_reference(
    load_orf_fixture: Callable[[str], dict[str, np.ndarray]],
) -> None:
    fixture = load_orf_fixture("gwfast_orf_reference_et_triangle")
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
