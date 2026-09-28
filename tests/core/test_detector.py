from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

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
from astrogwb.detector.overlap import C_LIGHT, R_EARTH


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


def _arm_direction(
    det: CustomDetector, azimuth: float, tilt: float = 0.0
) -> np.ndarray:
    # gwmock azimuths are measured clockwise from local North.
    east = np.array([-math.sin(det.longitude_rad), math.cos(det.longitude_rad), 0.0])
    north = np.array(
        [
            -math.sin(det.latitude_rad) * math.cos(det.longitude_rad),
            -math.sin(det.latitude_rad) * math.sin(det.longitude_rad),
            math.cos(det.latitude_rad),
        ]
    )
    up = np.cross(east, north)
    return (
        math.cos(tilt) * (math.sin(azimuth) * east + math.cos(azimuth) * north)
        + math.sin(tilt) * up
    )


def _position_and_tensor(det: CustomDetector) -> tuple[np.ndarray, np.ndarray]:
    lat, lon = det.latitude_rad, det.longitude_rad
    position = R_EARTH * np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
    )
    # The analytic ORF deliberately uses a spherical Earth and zero arm tilts.
    x = _arm_direction(det, det.xarm_azimuth_rad)
    y = _arm_direction(det, det.yarm_azimuth_rad)
    return position, 0.5 * (np.outer(x, x) - np.outer(y, y))


def _sky_quadrature_orf(
    frequencies: np.ndarray, spec_1: str, spec_2: str
) -> np.ndarray:
    """ORF by direct sky integration, (5/8pi) sum_A int dOmega F1^A F2^A e^(ik.dx).

    The sky is parametrized about the separation vector, so the phase only
    oscillates in cos(theta), where Gauss-Legendre converges exponentially.
    """
    position_1, tensor_1 = _position_and_tensor(resolve_detector(spec_1))
    position_2, tensor_2 = _position_and_tensor(resolve_detector(spec_2))
    separation = position_2 - position_1
    distance = float(np.linalg.norm(separation))
    z_axis = separation / distance
    x_axis = np.cross(z_axis, [1.0, 0.0, 0.0] if abs(z_axis[0]) < 0.9 else [0, 1, 0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    alpha = 2.0 * math.pi * frequencies * distance / C_LIGHT
    mu, mu_weights = np.polynomial.legendre.leggauss(int(alpha.max()) + 60)
    # Half-step offset: no direction is parallel to x_axis, which seeds the
    # polarization basis below.
    phi = 2.0 * math.pi * (np.arange(64) + 0.5) / 64
    mu_grid, phi_grid = np.meshgrid(mu, phi, indexing="ij")
    sin_theta = np.sqrt(1.0 - mu_grid**2)[..., None]
    direction = (
        sin_theta * np.cos(phi_grid)[..., None] * x_axis
        + sin_theta * np.sin(phi_grid)[..., None] * y_axis
        + mu_grid[..., None] * z_axis
    )
    m = np.cross(direction, x_axis)
    m /= np.linalg.norm(m, axis=-1, keepdims=True)
    n = np.cross(direction, m)
    e_plus = np.einsum("...i,...j->...ij", m, m) - np.einsum("...i,...j->...ij", n, n)
    e_cross = np.einsum("...i,...j->...ij", m, n) + np.einsum("...i,...j->...ij", n, m)
    response_product = sum(
        np.einsum("ij,...ij->...", tensor_1, e)
        * np.einsum("ij,...ij->...", tensor_2, e)
        for e in (e_plus, e_cross)
    )
    weights = mu_weights[:, None] * (2.0 * math.pi / phi.size) * response_product
    phase = np.cos(alpha[:, None, None] * mu_grid)
    return 5.0 / (8.0 * math.pi) * np.sum(weights * phase, axis=(1, 2))


@pytest.mark.parametrize(
    ("detector_1", "detector_2", "max_frequency"),
    [
        # ~10 km apart: alpha spans the small-alpha regime where the closed-form
        # g1/g2/g3 cancel catastrophically, and crosses the series threshold.
        ("E1", "E2", 1e4),
        ("E2", "E3", 1e4),
        # ~3000 km apart: alpha reaches ~125, deep in the oscillatory regime.
        ("H1", "L1", 2e3),
        ("V1", "K1", 2e3),
    ],
)
def test_orf_matches_direct_sky_quadrature(
    detector_1: str, detector_2: str, max_frequency: float
) -> None:
    freqs = np.concatenate([[0.0], np.geomspace(0.01, max_frequency, 400)])

    actual = overlap_reduction_function(freqs, detector_1, detector_2)

    expected = _sky_quadrature_orf(freqs, detector_1, detector_2)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


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
                xarm_azimuth_rad=(math.pi / 2.0 - math.radians(xax - 30.0))
                % (2.0 * math.pi),
                yarm_azimuth_rad=(math.pi / 2.0 - math.radians(xax + 30.0))
                % (2.0 * math.pi),
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


@pytest.mark.parametrize(
    "name",
    ["H1", "L1", "V1", "K1", "E1", "E2", "E3", "S1", "S2", "R1", "R2", "C1", "C2"],
)
def test_native_geometry_response_matches_lal(name: str) -> None:
    det = load_detector(name)
    x = _arm_direction(det, det.xarm_azimuth_rad, det.xarm_tilt_rad)
    y = _arm_direction(det, det.yarm_azimuth_rad, det.yarm_tilt_rad)
    tensor = 0.5 * (np.outer(x, x) - np.outer(y, y))
    # LAL stores the response in float32 and arm angles in FrDetector as floats.
    np.testing.assert_allclose(det.to_lal().response, tensor, rtol=0.0, atol=2e-7)


def test_hanford_translation_preserves_original_east_based_tensor() -> None:
    det = load_detector("H1")
    x = _arm_direction(
        det, math.pi / 2.0 - math.radians(125.9994), math.radians(-0.0006195)
    )
    y = _arm_direction(
        det, math.pi / 2.0 - math.radians(215.9994), math.radians(1.25e-05)
    )
    expected = 0.5 * (np.outer(x, x) - np.outer(y, y))
    np.testing.assert_allclose(det.to_lal().response, expected, rtol=0.0, atol=2e-7)


@pytest.mark.parametrize(
    "xarm,yarm", [(0.0, math.pi / 2), (0.3, 1.3), (6.1, 0.2), (1.3, 0.3)]
)
def test_colocated_native_orf_matches_lal_contraction(xarm: float, yarm: float) -> None:
    first = replace(load_detector("H1"), xarm_tilt_rad=0.0, yarm_tilt_rad=0.0)
    second = replace(first, name="other", xarm_azimuth_rad=xarm, yarm_azimuth_rad=yarm)
    expected = 2.0 * np.sum(first.to_lal().response * second.to_lal().response)
    np.testing.assert_allclose(
        overlap_reduction_function([0.0, 10.0, 1e4], first, second),
        expected,
        rtol=0.0,
        atol=5e-7,
    )


def test_orf_azimuth_wraparound() -> None:
    first = load_detector("E3")
    wrapped = replace(
        first,
        xarm_azimuth_rad=first.xarm_azimuth_rad + 4.0 * math.pi,
        yarm_azimuth_rad=first.yarm_azimuth_rad - 2.0 * math.pi,
    )
    frequencies = np.geomspace(0.01, 1e4, 100)
    np.testing.assert_allclose(
        overlap_reduction_function(frequencies, first, "E1"),
        overlap_reduction_function(frequencies, wrapped, "E1"),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("ce,site", [("C1", "H1"), ("C2", "L1")])
def test_ce_preserves_explicit_site_geometry(ce: str, site: str) -> None:
    assert replace(load_detector(ce), name=site) == load_detector(site)
