"""Frequency-dependent overlap reduction functions.

The frequency-dependent overlap reduction function (ORF) is astrogwb's own
contribution: gwmock ships only the long-wavelength, co-located limit
(``gamma = 2 D_i:D_j``). This module provides the analytic isotropic ORF for
the native detector geometry, :func:`isotropic_overlap_reduction_function`, and
the response of a pair to one source, :func:`overlap_reduction_function`, whose
isotropic average it equals.
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

from astrogwb.constants import SPEED_OF_LIGHT
from astrogwb.waveform.polarization_power import quadrupolar_stokes_factors

from .geometry import Detector, DetectorGeometry

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


def _isotropic_orf_from_tensors(
    frequencies: ArrayLike,
    tensor_1: ArrayLike,
    tensor_2: ArrayLike,
    separation_m: ArrayLike,
) -> NDArray[np.float64]:
    r"""Isotropic ORF for arbitrary traceless symmetric response tensors.

    The polarization sum is a quartic polynomial in the propagation direction
    and the phase moments are derivatives of :math:`j_0`, so for traceless
    :math:`D_a` the sky integral factors into three geometric invariants and
    three functions of :math:`\alpha = 2 \pi f |\Delta x| / c`:

    .. math::

        \gamma = \rho_1 \, \mathrm{tr}(D_1 D_2)
        + \rho_2 \, \hat d^T \{D_1, D_2\} \hat d
        + \rho_3 \, (\hat d^T D_1 \hat d)(\hat d^T D_2 \hat d),

    with :math:`\rho_1 = 2(g_1 + g_2 - g_3)`, :math:`\rho_2 = 4 g_3` and
    :math:`\rho_3 = -8 (g_2 + g_3)`. Nothing here assumes a spherical Earth,
    level arms or perpendicular arms.

    Parameters
    ----------
    frequencies
        Frequencies in Hz.
    tensor_1, tensor_2
        Traceless symmetric ``(3, 3)`` response tensors in the Earth-fixed frame.
    separation_m
        Earth-fixed separation vector between the two vertices in metres.

    Returns
    -------
    numpy.ndarray
        The ORF on ``frequencies``.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    d1 = np.asarray(tensor_1, dtype=float)
    d2 = np.asarray(tensor_2, dtype=float)
    separation = np.asarray(separation_m, dtype=float)
    distance = float(np.linalg.norm(separation))
    # Co-located detectors have alpha = 0, where rho_2 and rho_3 vanish, so the
    # direction is arbitrary.
    direction = separation / distance if distance > 0.0 else np.array([0.0, 0.0, 1.0])
    alpha = 2.0 * math.pi * frequencies * distance / SPEED_OF_LIGHT

    g1, g2, g3 = _g1(alpha), _g2(alpha), _g3(alpha)
    trace = float(np.sum(d1 * d2))
    anticommutator = float(direction @ (d1 @ d2 + d2 @ d1) @ direction)
    aligned = float((direction @ d1 @ direction) * (direction @ d2 @ direction))
    return (
        2.0 * (g1 + g2 - g3) * trace
        + 4.0 * g3 * anticommutator
        - 8.0 * (g2 + g3) * aligned
    )


def isotropic_overlap_reduction_function(
    frequencies: ArrayLike,
    detector_1: Detector,
    detector_2: Detector,
) -> NDArray[np.float64]:
    """Frequency-dependent ORF between two detectors for an isotropic background.

    Exact for the native geometry: WGS-84 positions, elevation and arm tilts
    enter through the response tensors and the vertex separation (see
    :func:`_isotropic_orf_from_tensors`).
    """
    first, second = detector_1.geometry, detector_2.geometry
    return _isotropic_orf_from_tensors(
        frequencies,
        first.response_tensor,
        second.response_tensor,
        second.location - first.location,
    )


def pairwise_overlap_reduction_function(
    frequencies: ArrayLike,
    detectors: Sequence[Detector],
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
    detector_1: Detector,
    detector_2: Detector,
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
    det1 = detector_1.geometry
    det2 = detector_2.geometry
    angle = _bisector_from_east(det2) - _bisector_from_east(det1)
    if reference == "great_circle":
        ang_1, ang_2 = _great_circle_offsets(det1, det2)
        angle += ang_2 - ang_1
    elif reference != "north":
        raise ValueError(f"Unknown orientation reference {reference!r}.")
    return _wrap_half_turn(angle)


def align_detector(
    reference_detector: Detector,
    detector: Detector,
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
    det = detector
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
    detector_1: Detector,
    detector_2: Detector,
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
        Owned detectors.
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
    The isotropic mean is ``I * gamma / 5`` with ``gamma`` the
    :func:`isotropic_overlap_reduction_function` of the same native geometry.
    """
    det1, det2 = detector_1, detector_2
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
    detector_1: Detector,
    detector_2: Detector,
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

    The isotropic mean is the :func:`isotropic_overlap_reduction_function` of
    the pair, equal to one for co-located, aligned, orthogonal arms.
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
