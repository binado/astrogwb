"""Frequency-dependent overlap reduction functions.

The frequency-dependent overlap reduction function (ORF) is astrogwb's own
contribution: gwmock ships only the long-wavelength, co-located limit
(``gamma = 2 D_i:D_j``). This module keeps the validated analytic isotropic
ORF (angle-based ``g1/g2/g3`` expansion) for resolved detector geometry,
:func:`isotropic_overlap_reduction_function`, and the response of a pair to
one source, :func:`overlap_reduction_function`, whose isotropic average it approximates.
Both single-source responses go through :func:`stokes_cross_power`, which
takes any source's polarization Stokes parameters.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike as JaxArrayLike
from numpy.typing import ArrayLike, NDArray

from astrogwb.constants import EARTH_MEAN_RADIUS_IN_METERS, SPEED_OF_LIGHT
from astrogwb.waveform.polarization_power import quadrupolar_stokes_factors

from ._types import DetectorSpec
from .geometry import Detector, DetectorGeometry, resolve_detector

# The analytic ORF works in kilometres: detector chord distances are
# O(1e4) km and the bundled overlap-reduction fixtures were generated in
# those units. Both conversions are exact in float64, so these are
# bit-identical to the literals they replace.
R_EARTH: float = EARTH_MEAN_RADIUS_IN_METERS / 1000.0  # km
C_LIGHT: float = SPEED_OF_LIGHT / 1000.0  # km / s

# Below this alpha the closed-form g1/g2/g3 lose ~eps/alpha^4 to cancellation
# (their O(alpha) numerator terms cancel to O(alpha^5)), so they switch to
# their Taylor series in alpha^2. Truncated after alpha^18, the series' error
# at the threshold is ~1e-20; the closed form's there is ~1e-15.
_SERIES_ALPHA_THRESHOLD = 1.0

# Taylor coefficients of g1, g2, g3 in powers of alpha^2, lowest first.
_G1_SERIES: tuple[float, ...] = (
    1.0,
    -5.0 / 42.0,
    5.0 / 1008.0,
    -1.0 / 9504.0,
    1.0 / 741312.0,
    -1.0 / 86486400.0,
    1.0 / 14114580480.0,
    -1.0 / 3071845969920.0,
    1.0 / 860116871577600.0,
    -1.0 / 301305556397260800.0,
)
_G2_SERIES: tuple[float, ...] = (
    0.0,
    -1.0 / 14.0,
    5.0 / 1008.0,
    -1.0 / 7392.0,
    1.0 / 494208.0,
    -1.0 / 51891840.0,
    1.0 / 7841433600.0,
    -1.0 / 1609062174720.0,
    1.0 / 430058435788800.0,
    -1.0 / 145073045672755200.0,
)
_G3_SERIES: tuple[float, ...] = (
    0.0,
    1.0 / 14.0,
    -1.0 / 189.0,
    5.0 / 33264.0,
    -1.0 / 432432.0,
    1.0 / 44478720.0,
    -1.0 / 6616209600.0,
    1.0 / 1340885145600.0,
    -1.0 / 354798209525760.0,
    1.0 / 118696128277708800.0,
)


def _chord_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line (chord) distance in km between two points on Earth.

    All angles are in radians.
    """
    x1 = math.cos(lat1) * math.cos(lon1)
    y1 = math.cos(lat1) * math.sin(lon1)
    z1 = math.sin(lat1)

    x2 = math.cos(lat2) * math.cos(lon2)
    y2 = math.cos(lat2) * math.sin(lon2)
    z2 = math.sin(lat2)

    return R_EARTH * math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def _initial_course(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2 (radians)."""
    dlon = lon2 - lon1
    x = math.cos(lat2) * math.sin(dlon)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        dlon
    )
    return math.atan2(x, y) % (2.0 * math.pi)


def _final_course(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    """Final great-circle bearing arriving at point 2 (radians)."""
    return (_initial_course(lat2, lat1, lon2, lon1) + math.pi) % (2.0 * math.pi)


def _opening_angle(az1: float, az2: float) -> float:
    """Signed opening from y to x in gwmock's clockwise-from-North convention.

    Keeping the sign accounts for reversed arm ordering in external detectors;
    all packaged paper detectors retain their positive opening angles.
    """
    return ((az1 - az2 + math.pi) % (2.0 * math.pi)) - math.pi


def _azimuth_bisector(az1: float, az2: float) -> float:
    """Circular-mean bisector of two azimuths (radians in, radians out)."""
    return math.atan2(math.sin(az1) + math.sin(az2), math.cos(az1) + math.cos(az2))


def _bisector_from_east(det: DetectorGeometry) -> float:
    """Arm bisector of a detector, counter-clockwise from local East (radians).

    The analytic ORF expansion uses counter-clockwise-from-East bisectors,
    while runtime detectors carry gwmock's clockwise-from-North arm azimuths.
    """
    return _azimuth_bisector(
        math.pi / 2.0 - det.xarm_azimuth_rad,
        math.pi / 2.0 - det.yarm_azimuth_rad,
    )


def _great_circle_offsets(
    det1: DetectorGeometry, det2: DetectorGeometry
) -> tuple[float, float]:
    """Great-circle bearings at both sites, shifted by a quarter turn (radians).

    Adding them to the counter-clockwise-from-East bisectors measures each
    detector's orientation with respect to the great circle joining the sites.
    """
    ang_1 = (
        _initial_course(
            det1.latitude_rad,
            det2.latitude_rad,
            det1.longitude_rad,
            det2.longitude_rad,
        )
        - 0.5 * math.pi
    )
    ang_2 = (
        _final_course(
            det1.latitude_rad,
            det2.latitude_rad,
            det1.longitude_rad,
            det2.longitude_rad,
        )
        - 0.5 * math.pi
    )
    return ang_1, ang_2


def _series_or_closed_form(
    alpha: NDArray[np.float64],
    series: tuple[float, ...],
    closed_form: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    small = alpha < _SERIES_ALPHA_THRESHOLD
    # Evaluate the closed form only where it is stable; the dummy 1.0 keeps
    # the discarded lanes finite.
    large_alpha = np.where(small, 1.0, alpha)
    return np.where(
        small,
        np.polynomial.polynomial.polyval(alpha**2, series),
        closed_form(large_alpha),
    )


def _g1_closed_form(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    return (
        (5.0 / 16.0)
        * (
            -9.0 * alpha * np.cos(alpha)
            - 6.0 * alpha**3 * np.cos(alpha)
            + 9.0 * np.sin(alpha)
            + 3.0 * alpha**2 * np.sin(alpha)
            + alpha**4 * np.sin(alpha)
        )
        / alpha**5
    )


def _g2_closed_form(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    return (
        (5.0 / 16.0)
        * (
            45.0 * alpha * np.cos(alpha)
            + 6.0 * alpha**3 * np.cos(alpha)
            - 45.0 * np.sin(alpha)
            + 9.0 * alpha**2 * np.sin(alpha)
            + 3.0 * alpha**4 * np.sin(alpha)
        )
        / alpha**5
    )


def _g3_closed_form(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    return (
        (5.0 / 4.0)
        * (
            15.0 * alpha * np.cos(alpha)
            - 4.0 * alpha**3 * np.cos(alpha)
            - 15.0 * np.sin(alpha)
            + 9.0 * alpha**2 * np.sin(alpha)
            - alpha**4 * np.sin(alpha)
        )
        / alpha**5
    )


def _g1(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    return _series_or_closed_form(alpha, _G1_SERIES, _g1_closed_form)


def _g2(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    return _series_or_closed_form(alpha, _G2_SERIES, _g2_closed_form)


def _g3(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    return _series_or_closed_form(alpha, _G3_SERIES, _g3_closed_form)


def _get_orf(
    alpha: NDArray[np.float64],
    beta: float,
    delta: float,
    big_delta: float,
    ang_btw_arms_1: float,
    ang_btw_arms_2: float,
) -> NDArray[np.float64]:
    sin1 = math.sin(ang_btw_arms_1)
    sin2 = math.sin(ang_btw_arms_2)

    g1 = _g1(alpha)
    g2 = _g2(alpha)
    g3 = _g3(alpha)

    theta_1 = (math.cos(0.5 * beta) ** 4) * g1
    theta_2 = (
        (math.cos(0.5 * beta) ** 4) * g2 + g3 - (math.sin(0.5 * beta) ** 4) * (g2 + g1)
    )
    return (
        (math.cos(4.0 * delta) * theta_1 + math.cos(4.0 * big_delta) * theta_2)
        * sin1
        * sin2
    )


def isotropic_overlap_reduction_function(
    frequencies: ArrayLike,
    detector_1: DetectorSpec,
    detector_2: DetectorSpec,
) -> NDArray[np.float64]:
    """Frequency-dependent ORF between two detectors for an isotropic background.

    A spherical-Earth, level-arm approximation to the native projection
    average. Elevation, WGS-84 flattening and arm tilts are omitted. Each
    detector is a packaged name or an owned :class:`Detector`.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    det1 = resolve_detector(detector_1).geometry
    det2 = resolve_detector(detector_2).geometry

    d = _chord_distance(
        det1.latitude_rad,
        det1.longitude_rad,
        det2.latitude_rad,
        det2.longitude_rad,
    )
    alpha = 2.0 * math.pi * frequencies * d / C_LIGHT

    xax_1 = _bisector_from_east(det1)
    xax_2 = _bisector_from_east(det2)
    ang_1, ang_2 = _great_circle_offsets(det1, det2)

    delta = 0.5 * ((xax_1 + ang_1) - (xax_2 + ang_2))
    big_delta = 0.5 * ((xax_1 + ang_1) + (xax_2 + ang_2))

    asin_arg = max(-1.0, min(1.0, 0.5 * d / R_EARTH))
    beta = 2.0 * math.asin(asin_arg)

    ang_btw_arms_1 = _opening_angle(det1.xarm_azimuth_rad, det1.yarm_azimuth_rad)
    ang_btw_arms_2 = _opening_angle(det2.xarm_azimuth_rad, det2.yarm_azimuth_rad)

    return _get_orf(alpha, beta, delta, big_delta, ang_btw_arms_1, ang_btw_arms_2)


def pairwise_overlap_reduction_function(
    frequencies: ArrayLike,
    detectors: Sequence[DetectorSpec],
) -> NDArray[np.float64]:
    """Symmetric ``(n, n, nfreq)`` isotropic ORF matrix with unit diagonal."""
    frequencies = np.asarray(frequencies, dtype=float)
    det_list = list(detectors)
    n = len(det_list)
    out = np.zeros((n, n, frequencies.shape[0]), dtype=float)

    for i, det1 in enumerate(det_list):
        out[i, i, :] = 1.0
        for j in range(i + 1, n):
            orf = isotropic_overlap_reduction_function(frequencies, det1, det_list[j])
            out[i, j, :] = orf
            out[j, i, :] = orf
    return out


#: Reference direction for the relative orientation of two detectors:
#: ``"great_circle"`` measures each arm bisector from the great circle joining
#: the sites, ``"north"`` from local North at each site.
OrientationReference = Literal["great_circle", "north"]


def _wrap_half_turn(angle: float) -> float:
    """Wrap an angle to ``(-pi/2, pi/2]``; a half turn leaves an L unchanged."""
    wrapped = math.remainder(angle, math.pi)
    return math.pi / 2.0 if wrapped == -math.pi / 2.0 else wrapped


def relative_orientation(
    detector_1: DetectorSpec,
    detector_2: DetectorSpec,
    reference: OrientationReference = "great_circle",
) -> float:
    r"""Rotation of the second detector's arm bisector relative to the first.

    With bisectors :math:`\sigma_i` measured counter-clockwise from the chosen
    reference direction at each site, this returns
    :math:`\sigma_2 - \sigma_1`, wrapped to :math:`(-\pi/2, \pi/2]`. In the
    long-wavelength limit the overlap reduction function of two L-shaped
    detectors is proportional to :math:`\cos 2(\sigma_2 - \sigma_1)` for the
    great-circle reference, so :math:`\pm\pi/4` nearly cancels it. The
    ``"north"`` reference is the angle :math:`\alpha` of Branchesi et al.
    (2023), which differs from the great-circle angle by the convergence of
    the meridians along the baseline.

    Parameters
    ----------
    detector_1, detector_2
        Site codes resolved through ``geometry.toml``, or owned ``Detector``
        instances.
    reference
        ``"great_circle"`` or ``"north"``.

    Returns
    -------
    float
        Relative orientation in radians, in :math:`(-\pi/2, \pi/2]`.
    """
    det1 = resolve_detector(detector_1).geometry
    det2 = resolve_detector(detector_2).geometry
    angle = _bisector_from_east(det2) - _bisector_from_east(det1)
    if reference == "great_circle":
        ang_1, ang_2 = _great_circle_offsets(det1, det2)
        angle += ang_2 - ang_1
    elif reference != "north":
        raise ValueError(f"Unknown orientation reference {reference!r}.")
    return _wrap_half_turn(angle)


def align_detector(
    reference_detector: DetectorSpec,
    detector: DetectorSpec,
    angle: float,
    reference: OrientationReference = "great_circle",
) -> Detector:
    """Rotate a detector about its vertex to a target relative orientation.

    Only the arm azimuths change: the site, elevation, tilts and opening
    angle are kept, so the great-circle bearings are unaffected and the
    result satisfies ``relative_orientation(reference_detector, result,
    reference) == angle`` up to the half-turn wrap.

    Parameters
    ----------
    reference_detector
        Detector whose orientation is held fixed.
    detector
        Detector to rotate.
    angle
        Target relative orientation in radians.
    reference
        ``"great_circle"`` or ``"north"``; see :func:`relative_orientation`.

    Returns
    -------
    Detector
        Copy of ``detector`` with rotated arm azimuths.
    """
    det = resolve_detector(detector)
    current = relative_orientation(reference_detector, det, reference)
    # A counter-clockwise rotation lowers gwmock's clockwise-from-North azimuths.
    rotation = _wrap_half_turn(angle - current)
    geometry = det.geometry
    return dataclasses.replace(
        det,
        geometry=dataclasses.replace(
            geometry,
            xarm_azimuth_rad=(geometry.xarm_azimuth_rad - rotation) % (2.0 * math.pi),
            yarm_azimuth_rad=(geometry.yarm_azimuth_rad - rotation) % (2.0 * math.pi),
        ),
    )


# The isotropic mean of Re[H_a^* H_b] for a unit source: <g> = 4/5 from the
# inclination times 1/5 from sky and polarization, for co-located aligned
# L-shaped detectors, whose isotropic ORF is one.
_ISOTROPIC_PAIR_RESPONSE = 4.0 / 25.0


def stokes_cross_power(
    frequencies: JaxArrayLike,
    detector_1: DetectorSpec,
    detector_2: DetectorSpec,
    right_ascension: JaxArrayLike,
    declination: JaxArrayLike,
    polarization_angle: JaxArrayLike,
    stokes: JaxArrayLike,
    *,
    gmst: JaxArrayLike = 0.0,
) -> jax.Array:
    r"""Cross-power of radiation-frame Stokes parameters through native projection.

    Uses the gwmock/LAL polarization basis and delay sign. This equals
    :math:`\mathrm{Re}(\tilde h_a^* \tilde h_b)` for directly projected
    polarizations whose Stokes parameters are ``stokes``.

    Parameters
    ----------
    frequencies
        Frequencies in Hz, ``(F,)``.
    detector_1, detector_2
        Packaged names or owned detectors.
    right_ascension, declination, polarization_angle, gmst
        Celestial coordinates, polarization angle and compatible sidereal angle
        in radians, broadcasting to a common source shape. No epoch conversion.
    stokes
        Radiation-frame ``(I, Q, U, V)``, broadcasting to ``(4, ..., F)``.

    Returns
    -------
    jax.Array
        Cross-power with shape ``(..., F)``.

    Notes
    -----
    The isotropic mean is approximately ``I * gamma / 5`` when ``gamma`` is
    the analytic spherical-Earth, level-arm ORF. Native geometry includes
    WGS-84 flattening, elevations and arm tilts, so equality is approximate.
    """
    det1, det2 = resolve_detector(detector_1), resolve_detector(detector_2)
    coordinates = {"right_ascension": right_ascension, "declination": declination}
    plus_1, cross_1 = det1.antenna_pattern(
        gmst, **coordinates, polarization_angle=polarization_angle
    )
    plus_2, cross_2 = det2.antenna_pattern(
        gmst, **coordinates, polarization_angle=polarization_angle
    )
    response_i = plus_1 * plus_2 + cross_1 * cross_2
    response_q = plus_1 * plus_2 - cross_1 * cross_2
    response_u = plus_1 * cross_2 + cross_1 * plus_2
    response_v = plus_1 * cross_2 - cross_1 * plus_2

    delay = det1.time_delay_from_geocenter(
        gmst, **coordinates
    ) - det2.time_delay_from_geocenter(gmst, **coordinates)
    phase = 2.0 * jnp.pi * delay[..., None] * jnp.asarray(frequencies, dtype=float)
    intensity, linear_q, linear_u, circular = jnp.asarray(stokes)
    linear = (
        intensity * response_i[..., None]
        + linear_q * response_q[..., None]
        + linear_u * response_u[..., None]
    )
    return 0.5 * (
        jnp.cos(phase) * linear - jnp.sin(phase) * circular * response_v[..., None]
    )


def overlap_reduction_function(
    frequencies: JaxArrayLike,
    detector_1: DetectorSpec,
    detector_2: DetectorSpec,
    right_ascension: JaxArrayLike,
    declination: JaxArrayLike,
    polarization_angle: JaxArrayLike,
    cos_inclination: JaxArrayLike,
    *,
    gmst: JaxArrayLike = 0.0,
) -> jax.Array:
    r"""Return the native per-source pair response, normalized by 4/25.

    This is :func:`stokes_cross_power` of the unit quadrupolar polarizations
    :math:`h_+ = -(1+c^2)/2`, :math:`h_\times = i c`, with
    :math:`c = \cos\iota`. Coordinates and ``gmst`` are radians in compatible
    celestial frames. Inputs broadcast to a source shape and the output has
    a final frequency axis.

    The isotropic mean equals one for co-located, aligned, orthogonal, level
    arms. For general sites it approximates the analytic isotropic ORF,
    which omits WGS-84 flattening, elevation and arm tilts.
    """
    declination, right_ascension, polarization_angle, cos_inclination = (
        jnp.broadcast_arrays(
            declination, right_ascension, polarization_angle, cos_inclination
        )
    )
    stokes = quadrupolar_stokes_factors(cos_inclination)[..., None]
    return (
        stokes_cross_power(
            frequencies,
            detector_1,
            detector_2,
            right_ascension,
            declination,
            polarization_angle,
            stokes,
            gmst=gmst,
        )
        / _ISOTROPIC_PAIR_RESPONSE
    )
