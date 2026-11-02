from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from astrogwb.constants import SPEED_OF_LIGHT
from astrogwb.detector import (
    Detector,
    DetectorGeometry,
    OrientationReference,
    Sensitivity,
    align_detector,
    isotropic_overlap_reduction_function,
    load_detector,
    pairwise_overlap_reduction_function,
    relative_orientation,
)

# The sky-quadrature reference works in kilometres.
C_LIGHT = SPEED_OF_LIGHT / 1000.0


def _owned_detector(name: str, **fields: float) -> Detector:
    return Detector(name, DetectorGeometry(**fields), Sensitivity("ET_D_psd"))


def test_load_detector_cosmic_explorer_is_not_lal_prototype() -> None:
    # LAL's "C1" code is the Caltech 40m prototype (CIT_40, lat ~34.17), but
    # astrogwb's C1 is Cosmic Explorer at Hanford. load_detector must source
    # the latter so CE networks are built from the right site.
    det = load_detector("C1")

    assert math.degrees(det.geometry.latitude_rad) == pytest.approx(46.455, abs=0.01)


def _sky_quadrature_orf(
    frequencies: np.ndarray, first: Detector, second: Detector
) -> np.ndarray:
    """ORF by direct sky integration, (5/8pi) sum_A int dOmega F1^A F2^A e^(ik.dx).

    The sky is parametrized about the separation vector, so the phase only
    oscillates in cos(theta), where Gauss-Legendre converges exponentially.
    """
    position_1, tensor_1 = (
        first.geometry.location / 1000,
        first.geometry.response_tensor,
    )
    position_2, tensor_2 = (
        second.geometry.location / 1000,
        second.geometry.response_tensor,
    )
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
    first, second = load_detector(detector_1), load_detector(detector_2)

    actual = isotropic_overlap_reduction_function(freqs, first, second)

    expected = _sky_quadrature_orf(freqs, first, second)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-11)


@pytest.mark.parametrize(
    ("fields_1", "fields_2"),
    [
        pytest.param(
            {
                "latitude_rad": 0.8,
                "longitude_rad": -1.9,
                "elevation_m": 500.0,
                "xarm_azimuth_rad": 0.3,
                "yarm_azimuth_rad": 1.9,
                "xarm_tilt_rad": 0.02,
                "yarm_tilt_rad": -0.05,
            },
            {
                "latitude_rad": 0.5,
                "longitude_rad": -1.4,
                "elevation_m": 2000.0,
                "xarm_azimuth_rad": 2.5,
                "yarm_azimuth_rad": 4.0,
                "xarm_tilt_rad": 0.1,
                "yarm_tilt_rad": 0.03,
            },
            id="continental-tilted",
        ),
        pytest.param(
            {
                "latitude_rad": 0.7,
                "longitude_rad": 0.15,
                "elevation_m": -200.0,
                "xarm_azimuth_rad": 1.0,
                "yarm_azimuth_rad": 2.0,
                "xarm_tilt_rad": 0.3,
                "yarm_tilt_rad": 0.2,
            },
            {
                "latitude_rad": 0.7,
                "longitude_rad": 0.16,
                "elevation_m": 300.0,
                "xarm_azimuth_rad": 2.2,
                "yarm_azimuth_rad": 3.1,
                "xarm_tilt_rad": -0.1,
                "yarm_tilt_rad": 0.4,
            },
            id="nearby-steeply-tilted",
        ),
    ],
)
def test_orf_matches_native_sky_quadrature_for_tilted_elevated_arms(
    fields_1: dict[str, float], fields_2: dict[str, float]
) -> None:
    freqs = np.concatenate([[0.0], np.geomspace(0.01, 2e3, 300)])
    first = _owned_detector("T1", **fields_1)
    second = _owned_detector("T2", **fields_2)

    actual = isotropic_overlap_reduction_function(freqs, first, second)

    expected = _sky_quadrature_orf(freqs, first, second)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-11)


def test_pairwise_overlap_shape_and_symmetry(frequencies: np.ndarray) -> None:
    actual = pairwise_overlap_reduction_function(
        frequencies, [load_detector(name) for name in ("H1", "L1", "V1")]
    )

    assert actual.shape == (3, 3, frequencies.shape[0])
    np.testing.assert_allclose(actual, np.transpose(actual, (1, 0, 2)))
    np.testing.assert_allclose(actual[0, 0, :], 1.0)


@pytest.mark.parametrize(
    ("pair", "north_deg", "great_circle_deg"),
    [(("S1", "R1"), 0.0, -2.51), (("S2", "R2"), 45.0, 42.49)],
)
def test_et_2l_orientations_match_branchesi(
    pair: tuple[str, str], north_deg: float, great_circle_deg: float
) -> None:
    # Branchesi et al. (2023), Sec. 2: alpha = 0 and 45 deg from local North,
    # with alpha ~= beta + 2.51 deg for the great-circle angle beta.
    first, second = (load_detector(name) for name in pair)
    north = math.degrees(relative_orientation(first, second, reference="north"))
    great_circle = math.degrees(relative_orientation(first, second))

    assert north == pytest.approx(north_deg, abs=1e-9)
    assert great_circle == pytest.approx(great_circle_deg, abs=0.01)


@pytest.mark.parametrize("reference", ["great_circle", "north"])
@pytest.mark.parametrize("angle_deg", [-60.0, 45.0])
def test_align_detector_reaches_target_orientation(
    reference: OrientationReference, angle_deg: float
) -> None:
    reference_detector = load_detector("S1")
    site = load_detector("R2")
    rotated = align_detector(
        reference_detector, site, math.radians(angle_deg), reference
    )

    assert math.degrees(
        relative_orientation(reference_detector, rotated, reference)
    ) == pytest.approx(angle_deg, abs=1e-9)
    assert (
        rotated.geometry.latitude_rad,
        rotated.geometry.longitude_rad,
        rotated.geometry.elevation_m,
    ) == (
        site.geometry.latitude_rad,
        site.geometry.longitude_rad,
        site.geometry.elevation_m,
    )


def test_orf_long_wavelength_scales_as_cos_two_orientation() -> None:
    def gamma(angle_deg: float) -> float:
        reference = load_detector("S1")
        rotated = align_detector(
            reference, load_detector("R1"), math.radians(angle_deg)
        )
        return float(isotropic_overlap_reduction_function([1.0], reference, rotated)[0])

    assert gamma(30.0) / gamma(0.0) == pytest.approx(0.5, abs=1e-3)
    assert abs(gamma(45.0)) < 1e-3 * abs(gamma(0.0))


def _gwmock_reference(geometry, gmst, ra, dec, psi, frequencies, plus, cross):
    """Reconstruct gwmock geometry independently, then call its projection kernels."""
    from gwmock_signal.detector import CustomDetector
    from gwmock_signal.projection import jax_projection as reference
    from gwmock_signal.projection.geometry import reconstructed_geometry

    custom = CustomDetector("reference", **dataclasses.asdict(geometry))
    tensor, location = reconstructed_geometry(custom.to_lal().frDetector.prefix)
    gmst, ra, dec, psi = np.broadcast_arrays(gmst, ra, dec, psi)
    patterns, delays, strains = [], [], []
    for t, a, d, p in zip(gmst.flat, ra.flat, dec.flat, psi.flat, strict=True):
        coordinates = {"right_ascension": a, "declination": d}
        fp, fc = reference.antenna_pattern(
            tensor, t, **coordinates, polarization_angle=p
        )
        delay = reference.time_delay_from_geocenter(location, t, **coordinates)
        strain = reference.project_polarizations_fd(
            frequencies,
            plus,
            cross,
            f_plus=fp,
            f_cross=fc,
            time_delay=delay,
            n_samples=8,
            sampling_frequency=1024.0,
        )
        patterns.append((fp, fc))
        delays.append(delay)
        strains.append(np.fft.rfft(strain)[:4] / 1024.0)
    return (
        tensor,
        location,
        np.asarray(patterns).T.reshape((2, *gmst.shape)),
        np.asarray(delays).reshape(gmst.shape),
        np.asarray(strains).reshape((*gmst.shape, len(frequencies))),
    )


PACKAGED_NAMES = [
    "H1",
    "L1",
    "V1",
    "K1",
    "E1",
    "E2",
    "E3",
    "S1",
    "S2",
    "R1",
    "R2",
    "C1",
    "C2",
]


@pytest.mark.integration
@pytest.mark.parametrize("name", ["H1", "E1", "tilted", "nonorthogonal"])
@pytest.mark.parametrize("batched", [False, True])
def test_native_projection_matches_gwmock_polarization_convention(
    name: str, batched: bool
) -> None:
    """Precision bounds include LAL's REAL4 arm fields and elevation conversion."""
    if name in PACKAGED_NAMES:
        detector = load_detector(name)
    else:
        detector = _owned_detector(
            name,
            latitude_rad=0.4,
            longitude_rad=-2.1,
            elevation_m=1234.567,
            xarm_azimuth_rad=0.7,
            yarm_azimuth_rad=2.2 if name == "nonorthogonal" else 0.7 + np.pi / 2,
            xarm_tilt_rad=0.04,
            yarm_tilt_rad=-0.03,
        )
    geometry = detector.geometry
    if batched:
        ra = np.array([0.0, 2 * np.pi, -np.pi, 0.7])[None, :]
        dec = np.array([-np.pi / 2, 0.0, np.pi / 2])[:, None]
        gmst = np.array([0.0, 0.8, -2.0, 2 * np.pi])[None, :]
        psi = np.array([0.0, np.pi, np.pi / 2, 0.3])[None, :]
    else:
        ra, dec, gmst, psi = 0.3, -0.2, 1.2, 0.8
    frequencies = np.array([0.0, 128.0, 256.0, 384.0])
    plus = np.array([0j, -0.2j, -0.4 + 1j, 0.6 - 0.7j])
    cross = np.array([0j, 0.3 + 1j, -1j, 0.8 + 0.6j])
    tensor, location, patterns, delays, strains = _gwmock_reference(
        geometry, gmst, ra, dec, psi, frequencies, plus, cross
    )
    # Each REAL4 field rounds by at most half an ulp. An arm's angular
    # displacement bounds its vector error; the half outer-product has
    # Frobenius error <= sqrt(2) times that displacement. A polarization
    # tensor has norm sqrt(2), hence pattern error <= 2 * sum(angle errors).
    angle_error = sum(
        abs(float(np.spacing(np.float32(getattr(geometry, field))))) / 2
        for field in (
            "xarm_azimuth_rad",
            "yarm_azimuth_rad",
            "xarm_tilt_rad",
            "yarm_tilt_rad",
        )
    )
    eps = np.finfo(np.float64).eps
    tensor_bound = np.sqrt(2) * angle_error + 64 * eps
    position_bound = (
        abs(float(np.spacing(np.float32(geometry.elevation_m)))) / 2
        + 32 * eps * 6378137
    )
    pattern_bound = 2 * angle_error + 128 * eps
    delay_bound = position_bound / 299792458.0
    np.testing.assert_allclose(
        geometry.response_tensor, tensor, rtol=0, atol=tensor_bound
    )
    np.testing.assert_allclose(geometry.location, location, rtol=0, atol=position_bound)
    coordinates = {"right_ascension": ra, "declination": dec, "polarization_angle": psi}
    actual_patterns = detector.antenna_pattern(gmst, **coordinates)
    np.testing.assert_allclose(actual_patterns, patterns, rtol=0, atol=pattern_bound)
    np.testing.assert_allclose(
        detector.time_delay_from_geocenter(gmst, right_ascension=ra, declination=dec),
        delays,
        rtol=0,
        atol=delay_bound,
    )
    amplitude_bound = np.abs(plus) + np.abs(cross)
    strain_bound = (
        amplitude_bound * (pattern_bound + 2 * np.pi * frequencies * delay_bound)
        + 128 * eps * amplitude_bound.sum()
    )
    actual = detector.project_polarizations_fd(
        frequencies, plus, cross, gmst, **coordinates
    )
    assert np.all(np.abs(actual - strains) <= strain_bound)


def test_native_projection_broadcasts_and_jits() -> None:
    import jax

    from astrogwb.detector.projection import project_polarizations_fd

    detector = load_detector("H1")
    args = (
        detector.geometry.response_tensor,
        detector.geometry.location,
        np.array([10.0, 100.0]),
        np.array([1j, 1.0]),
        np.array([1.0, -1j]),
        np.array([0.0, 1.0, 2.0])[None, :],
    )
    coordinates = {
        "right_ascension": np.array([0.3, 1.0])[:, None],
        "declination": 0.2,
        "polarization_angle": 0.7,
    }
    eager = project_polarizations_fd(*args, **coordinates)
    compiled = jax.jit(project_polarizations_fd)(*args, **coordinates)
    assert eager.shape == (2, 3, 2)
    np.testing.assert_allclose(eager, compiled)
    for i in range(2):
        for j in range(3):
            scalar = detector.project_polarizations_fd(
                args[2],
                args[3],
                args[4],
                args[5][0, j],
                right_ascension=coordinates["right_ascension"][i, 0],
                declination=0.2,
                polarization_angle=0.7,
            )
            np.testing.assert_allclose(eager[i, j], scalar)
