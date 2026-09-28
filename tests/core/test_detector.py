from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from gwmock_signal.detector import CustomDetector
from gwmock_signal.network import Network
from gwmock_signal.stochastic.overlap import (
    detector_names,
    long_wavelength_overlap_reduction,
)

from astrogwb.detector import (
    DetectorSpec,
    load_detector,
    overlap_reduction_function,
    pairwise_overlap_reduction_function,
    resolve_detector,
)
from astrogwb.detector.overlap import C_LIGHT, R_EARTH

# Earth-baseline phase is still << 1 at 1 Hz for HLV and the ET 2L baseline,
# so astrogwb's frequency-dependent ORF is in the same regime as gwmock's
# tensor contraction. Higher frequencies (the ``frequencies`` fixture's
# 2-2048 Hz band) are where the two are *supposed* to diverge.
_LONG_WAVELENGTH_HZ = np.array([1.0])


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


def _arm_direction(det: CustomDetector, azimuth: float) -> np.ndarray:
    # geometry.toml azimuths are measured counter-clockwise from local East.
    east = np.array([-math.sin(det.longitude_rad), math.cos(det.longitude_rad), 0.0])
    north = np.array(
        [
            -math.sin(det.latitude_rad) * math.cos(det.longitude_rad),
            -math.sin(det.latitude_rad) * math.sin(det.longitude_rad),
            math.cos(det.latitude_rad),
        ]
    )
    return math.cos(azimuth) * east + math.sin(azimuth) * north


def _position_and_tensor(det: CustomDetector) -> tuple[np.ndarray, np.ndarray]:
    lat, lon = det.latitude_rad, det.longitude_rad
    position = R_EARTH * np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
    )
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


def _gwmock_pair_orf(
    detector_1: DetectorSpec,
    detector_2: DetectorSpec,
    frequencies: np.ndarray,
) -> np.ndarray:
    """gwmock's long-wavelength ORF for one unordered detector pair."""
    pair = [detector_1, detector_2]
    first, second = sorted(detector_names(pair))
    result = long_wavelength_overlap_reduction(pair, frequencies)
    return result[(first, second)]


def _colocated_detector(
    name: str, *, xarm_azimuth_rad: float, opening_rad: float
) -> CustomDetector:
    return CustomDetector(
        name=name,
        latitude_rad=0.4,
        longitude_rad=0.2,
        elevation_m=0.0,
        xarm_azimuth_rad=xarm_azimuth_rad,
        yarm_azimuth_rad=xarm_azimuth_rad + opening_rad,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("opening", "relative_rotation"),
    [
        pytest.param(math.pi / 2.0, 0.0, id="identical-90"),
        pytest.param(math.pi / 2.0, math.pi / 6.0, id="rotated-30"),
        pytest.param(math.pi / 2.0, math.pi / 4.0, id="rotated-45"),
        pytest.param(math.pi / 3.0, 2.0 * math.pi / 3.0, id="et-like-60-120"),
    ],
)
def test_gwmock_long_wavelength_orf_matches_colocated_detectors(
    frequencies: np.ndarray, opening: float, relative_rotation: float
) -> None:
    """Co-located detectors: astrogwb is frequency-independent and equals 2 D_i:D_j.

    Both implementations see the same ``CustomDetector`` objects, so this is a
    pure algorithm comparison (no geometry-table vs LAL mismatch).
    """
    det_a = _colocated_detector("P", xarm_azimuth_rad=0.0, opening_rad=opening)
    det_b = _colocated_detector(
        "Q", xarm_azimuth_rad=relative_rotation, opening_rad=opening
    )

    ours = overlap_reduction_function(frequencies, det_a, det_b)
    theirs = _gwmock_pair_orf(det_a, det_b, frequencies)

    np.testing.assert_allclose(ours, theirs, atol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize(
    "pair",
    [
        pytest.param(("H1", "L1"), id="H1-L1"),
        pytest.param(("H1", "V1"), id="H1-V1"),
        pytest.param(("L1", "V1"), id="L1-V1"),
        pytest.param(("H1", "K1"), id="H1-K1"),
    ],
)
def test_gwmock_long_wavelength_orf_matches_hlv_at_low_frequency(
    pair: tuple[str, str],
) -> None:
    """HLV site codes: analytic ORF at 1 Hz vs gwmock's LAL tensor contraction.

    astrogwb resolves strings through ``geometry.toml``; gwmock uses LAL's
    built-in detectors. The residual is that geometry difference plus a small
    finite-frequency correction, not a disagreement about the ORF itself.
    """
    det_a, det_b = pair
    ours = overlap_reduction_function(_LONG_WAVELENGTH_HZ, det_a, det_b)
    theirs = _gwmock_pair_orf(det_a, det_b, _LONG_WAVELENGTH_HZ)

    np.testing.assert_allclose(ours, theirs, atol=5e-3)


@pytest.mark.integration
@pytest.mark.parametrize("preset", ["ET-Sardinia", "ET-2L-Aligned"])
def test_gwmock_long_wavelength_orf_matches_et_preset_at_low_frequency(
    preset: str,
) -> None:
    """gwmock ET presets: same ``CustomDetector`` objects into both ORF paths.

    The triangle vertices sit ~10 km apart, so the tensor contraction (exact
    co-located limit) and the analytic formula (small but finite baseline)
    differ at a few parts in a thousand.
    """
    detectors = list(Network.from_name(preset).detector_names)
    for i, det_a in enumerate(detectors):
        for det_b in detectors[i + 1 :]:
            ours = overlap_reduction_function(_LONG_WAVELENGTH_HZ, det_a, det_b)
            theirs = _gwmock_pair_orf(det_a, det_b, _LONG_WAVELENGTH_HZ)
            np.testing.assert_allclose(ours, theirs, atol=5e-3)


@pytest.mark.integration
def test_gwmock_et_2l_misaligned_orf_matches_in_magnitude() -> None:
    """ET-2L-Misaligned agrees in magnitude; the analytic bisector can flip sign.

    The detector tensor is invariant under reversing an arm (``n`` vs ``-n``),
    but the analytic bisector follows the 90-degree wedge implied by the stored
    azimuths. For this preset that wedge is the opposite diagonal from the
    tensor's principal axis, so ``gamma`` picks up a minus sign. The isotropic
    search contracts ``gamma^2``, so the sign is unused; the magnitude is the
    quantity that must not drift.
    """
    det_a, det_b = list(Network.from_name("ET-2L-Misaligned").detector_names)
    ours = overlap_reduction_function(_LONG_WAVELENGTH_HZ, det_a, det_b)
    theirs = _gwmock_pair_orf(det_a, det_b, _LONG_WAVELENGTH_HZ)

    np.testing.assert_allclose(np.abs(ours), np.abs(theirs), atol=1e-3)
