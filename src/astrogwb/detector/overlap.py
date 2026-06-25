"""Frequency-dependent overlap reduction and effective PSD utilities.

The frequency-dependent overlap reduction function (ORF) is astrogwb's own
contribution: gwmock ships only the long-wavelength, co-located limit
(``gamma = 2 D_i:D_j``). This module keeps the validated analytic ORF
(angle-based ``g1/g2/g3`` expansion) for resolved detector geometry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from gwmock_signal.stochastic.overlap import detector_names
from numpy.typing import ArrayLike, NDArray

from ._types import DetectorSpec
from .geometry import resolve_geometry
from .sensitivity import Sensitivity, evaluate_psd

R_EARTH = 6371.0  # km
C_LIGHT = 299792.458  # km/s

_LOW_ALPHA_THRESHOLD = 2e-3
_NETWORK_SENSITIVITY_FACTOR = 0.16


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
    """Smallest angle between two arm azimuths (radians in, radians out)."""
    diff = ((az1 - az2 + math.pi) % (2.0 * math.pi)) - math.pi
    return abs(diff)


def _azimuth_bisector(az1: float, az2: float) -> float:
    """Circular-mean bisector of two azimuths (radians in, radians out)."""
    return math.atan2(math.sin(az1) + math.sin(az2), math.cos(az1) + math.cos(az2))


def _g1(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(invalid="ignore", divide="ignore"):
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


def _g2(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(invalid="ignore", divide="ignore"):
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


def _g3(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(invalid="ignore", divide="ignore"):
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

    with np.errstate(invalid="ignore", divide="ignore"):
        g1 = _g1(alpha)
        g2 = _g2(alpha)
        g3 = _g3(alpha)

        theta_1 = (math.cos(0.5 * beta) ** 4) * g1
        theta_2 = (
            (math.cos(0.5 * beta) ** 4) * g2
            + g3
            - (math.sin(0.5 * beta) ** 4) * (g2 + g1)
        )
        high_alpha_orf = (
            (math.cos(4.0 * delta) * theta_1 + math.cos(4.0 * big_delta) * theta_2)
            * sin1
            * sin2
        )

    low_alpha_orf = math.cos(4.0 * delta) * sin1 * sin2
    return np.where(alpha > _LOW_ALPHA_THRESHOLD, high_alpha_orf, low_alpha_orf)


def overlap_reduction_function(
    frequencies: ArrayLike,
    detector_1: DetectorSpec,
    detector_2: DetectorSpec,
) -> NDArray[np.float64]:
    """Frequency-dependent ORF between two detectors.

    Each detector is a ``str`` site code (resolved via ``geometry.toml``)
    or a gwmock ``CustomDetector``.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    det1 = resolve_geometry(detector_1)
    det2 = resolve_geometry(detector_2)

    d = _chord_distance(
        det1.latitude_rad,
        det1.longitude_rad,
        det2.latitude_rad,
        det2.longitude_rad,
    )
    alpha = 2.0 * math.pi * frequencies * d / C_LIGHT

    xax_1 = _azimuth_bisector(det1.xarm_azimuth_rad, det1.yarm_azimuth_rad)
    xax_2 = _azimuth_bisector(det2.xarm_azimuth_rad, det2.yarm_azimuth_rad)

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
    """Symmetric ``(n, n, nfreq)`` ORF matrix with unit diagonal."""
    frequencies = np.asarray(frequencies, dtype=float)
    det_list = list(detectors)
    n = len(det_list)
    out = np.zeros((n, n, frequencies.shape[0]), dtype=float)

    for i, det1 in enumerate(det_list):
        out[i, i, :] = 1.0
        for j in range(i + 1, n):
            orf = overlap_reduction_function(frequencies, det1, det_list[j])
            out[i, j, :] = orf
            out[j, i, :] = orf
    return out


def effective_psd(
    frequencies: ArrayLike,
    detectors: Sequence[DetectorSpec],
    sensitivities: Mapping[str, Sensitivity],
) -> NDArray[np.float64]:
    """Network effective PSD from an inverse-variance cross-correlation sum.

    ``detectors`` are matched to ``sensitivities`` by their public name
    (gwmock ``detector_names``): plain str codes match directly, while
    preset ``CustomDetector`` objects match on their ``name`` (e.g.
    ``ET1_SARD``). The contraction and ``_NETWORK_SENSITIVITY_FACTOR``
    are unchanged from the pre-refactor implementation.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    det_list = list(detectors)
    if len(det_list) < 2:
        return np.full(frequencies.shape, np.inf, dtype=float)

    names = detector_names(det_list)
    psds = [
        evaluate_psd(sensitivities[name].psd_reference, frequencies) for name in names
    ]

    inverse_variance = np.zeros(frequencies.shape, dtype=float)
    for i, det1 in enumerate(det_list):
        for j in range(i + 1, len(det_list)):
            gamma = overlap_reduction_function(frequencies, det1, det_list[j])
            with np.errstate(invalid="ignore", divide="ignore"):
                contribution = gamma**2 / (psds[i] * psds[j])
            inverse_variance += np.where(np.isfinite(contribution), contribution, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = 1.0 / (_NETWORK_SENSITIVITY_FACTOR * np.sqrt(inverse_variance))
    return np.where(inverse_variance > 0.0, result, np.inf)
