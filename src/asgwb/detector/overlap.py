"""Overlap reduction function (ORF) for detector pairs.

Implements the ORF following the GWFast algorithm (arXiv:astro-ph/9305029),
adapted to the Detector dataclass convention where xarm_azimuth and yarm_azimuth
are stored in degrees.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from .detector import Detector

R_EARTH = 6371.0  # km
C_LIGHT = 299792.458  # km/s

_LOW_ALPHA_THRESHOLD = 2e-3


def _chord_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance through Earth between two surface points (km).

    Parameters
    ----------
    lat1 : float
        Latitude of detector 1 in degrees.
    lon1 : float
        Longitude of detector 1 in degrees.
    lat2 : float
        Latitude of detector 2 in degrees.
    lon2 : float
        Longitude of detector 2 in degrees.

    Returns
    -------
    float
        Chord distance in km.
    """
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)

    x1 = math.cos(lat1_r) * math.cos(lon1_r)
    y1 = math.cos(lat1_r) * math.sin(lon1_r)
    z1 = math.sin(lat1_r)

    x2 = math.cos(lat2_r) * math.cos(lon2_r)
    y2 = math.cos(lat2_r) * math.sin(lon2_r)
    z2 = math.sin(lat2_r)

    return R_EARTH * math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def _initial_course(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    """Bearing (azimuth) from detector 1 toward detector 2 (degrees, N-clockwise).

    Parameters
    ----------
    lat1 : float
        Latitude of detector 1 in degrees.
    lat2 : float
        Latitude of detector 2 in degrees.
    lon1 : float
        Longitude of detector 1 in degrees.
    lon2 : float
        Longitude of detector 2 in degrees.

    Returns
    -------
    float
        Initial bearing in degrees.
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.cos(lat2_r) * math.sin(dlon)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(
        lat2_r
    ) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360


def _final_course(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    """Bearing arriving at detector 2 coming from detector 1 (degrees, N-clockwise).

    Computed as the reverse bearing at detector 2.

    Parameters
    ----------
    lat1 : float
        Latitude of detector 1 in degrees.
    lat2 : float
        Latitude of detector 2 in degrees.
    lon1 : float
        Longitude of detector 1 in degrees.
    lon2 : float
        Longitude of detector 2 in degrees.

    Returns
    -------
    float
        Final bearing in degrees.
    """
    return (_initial_course(lat2, lat1, lon2, lon1) + 180.0) % 360


def _g1(alpha: np.ndarray) -> np.ndarray:
    """First angular response function.

    Parameters
    ----------
    alpha : np.ndarray
        Angular frequency parameter.

    Returns
    -------
    np.ndarray
        First angular response values.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (
            (1.0 / (alpha**2))
            - (9.0 / (2.0 * alpha**4))
            + (9.0 / (2.0 * alpha**3)) * np.sinc(2.0 * alpha / math.pi)
            + (9.0 / (4.0 * alpha**4)) * np.cos(2.0 * alpha)
        )
    return result


def _g2(alpha: np.ndarray) -> np.ndarray:
    """Second angular response function.

    Parameters
    ----------
    alpha : np.ndarray
        Angular frequency parameter.

    Returns
    -------
    np.ndarray
        Second angular response values.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (
            -(1.0 / (3.0 * alpha**2))
            + (3.0 / (2.0 * alpha**4))
            - (3.0 / (2.0 * alpha**3)) * np.sinc(2.0 * alpha / math.pi)
            - (3.0 / (4.0 * alpha**4)) * np.cos(2.0 * alpha)
        )
    return result


def _g3(alpha: np.ndarray) -> np.ndarray:
    """Third angular response function.

    Parameters
    ----------
    alpha : np.ndarray
        Angular frequency parameter.

    Returns
    -------
    np.ndarray
        Third angular response values.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (
            -(1.0 / (6.0 * alpha**2))
            + (1.0 / (2.0 * alpha**3)) * np.sinc(2.0 * alpha / math.pi)
            + (1.0 / (4.0 * alpha**4)) * np.cos(2.0 * alpha)
            - (1.0 / (4.0 * alpha**4))
        )
    return result


def _get_orf(
    alpha: np.ndarray,
    beta: float,
    delta: float,
    big_delta: float,
    ang_btw_arms_1: float,
    ang_btw_arms_2: float,
) -> np.ndarray:
    """Core ORF computation.

    Parameters
    ----------
    alpha : np.ndarray
        2*pi*f*d/c, frequency-distance parameter (array).
    beta : float
        Angle between the two detector bisectors (radians).
    delta : float
        Orientation angle of detector 1 relative to baseline (radians).
    big_delta : float
        Orientation angle of detector 2 relative to baseline (radians).
    ang_btw_arms_1 : float
        Half opening angle of detector 1 arms (radians).
    ang_btw_arms_2 : float
        Half opening angle of detector 2 arms (radians).

    Returns
    -------
    np.ndarray
        ORF values as a numpy array.
    """
    sin1 = math.sin(ang_btw_arms_1)
    sin2 = math.sin(ang_btw_arms_2)

    cos_beta = math.cos(beta)
    cos2_delta = math.cos(2.0 * delta)
    cos2_big_delta = math.cos(2.0 * big_delta)

    low_alpha = alpha < _LOW_ALPHA_THRESHOLD

    g1 = np.where(low_alpha, 2.0 / 15.0, _g1(alpha))
    g2 = np.where(low_alpha, -1.0 / 15.0, _g2(alpha))
    g3 = np.where(low_alpha, -1.0 / 30.0, _g3(alpha))

    orf = (
        sin1
        * sin2
        * (
            g1 * cos_beta
            + g2 * cos_beta * cos2_delta * cos2_big_delta
            - g3
            * (
                math.cos(2.0 * (delta - big_delta))
                + math.cos(2.0 * (delta + big_delta))
            )
        )
    )
    return orf


def overlap_reduction_function(
    frequencies: np.ndarray,
    detector_1: Detector,
    detector_2: Detector,
) -> np.ndarray:
    """Compute the ORF for a detector pair over a frequency array.

    The overlap reduction function encodes the geometric sensitivity of a
    detector pair to an isotropic stochastic GW background.

    Parameters
    ----------
    frequencies : np.ndarray
        1-D frequency array in Hz.
    detector_1 : Detector
        First detector.
    detector_2 : Detector
        Second detector.

    Returns
    -------
    np.ndarray
        1-D array of ORF values, same length as ``frequencies``.
    """
    frequencies = np.asarray(frequencies, dtype=float)

    lat1 = detector_1.latitude
    lon1 = detector_1.longitude
    lat2 = detector_2.latitude
    lon2 = detector_2.longitude

    d = _chord_distance(lat1, lon1, lat2, lon2)
    alpha = 2.0 * math.pi * frequencies * d / C_LIGHT

    # Bisector angle of each detector (orientation angle)
    bisector_1 = math.radians((detector_1.xarm_azimuth + detector_1.yarm_azimuth) / 2.0)
    bisector_2 = math.radians((detector_2.xarm_azimuth + detector_2.yarm_azimuth) / 2.0)

    # Bearing angles along the baseline
    course_1 = math.radians(_initial_course(lat1, lat2, lon1, lon2))
    course_2 = math.radians(_final_course(lat1, lat2, lon1, lon2))

    # delta: detector orientation relative to direction toward the other detector
    delta = bisector_1 - course_1
    big_delta = bisector_2 - course_2

    # beta: angle between the bisectors (in the baseline frame)
    beta = course_2 - course_1 - math.pi

    # Half opening angle of each detector (radians)
    ang_btw_arms_1 = math.radians(
        abs(detector_1.xarm_azimuth - detector_1.yarm_azimuth) / 2.0
    )
    ang_btw_arms_2 = math.radians(
        abs(detector_2.xarm_azimuth - detector_2.yarm_azimuth) / 2.0
    )

    return _get_orf(alpha, beta, delta, big_delta, ang_btw_arms_1, ang_btw_arms_2)


def pairwise_overlap_reduction_function(
    frequencies: np.ndarray,
    detectors: Sequence[Detector],
) -> np.ndarray:
    """Compute ORFs for all unique pairs in a detector network.

    Parameters
    ----------
    frequencies : np.ndarray
        1-D frequency array in Hz.
    detectors : Sequence[Detector]
        Sequence of Detector objects.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_detector, n_detector, n_frequencies)`` containing
        ORF values. The array is symmetric in the first two indices, and the
        diagonal entries are zero.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    det_list = list(detectors)
    n = len(det_list)
    n_freq = frequencies.shape[0]

    out = np.zeros((n, n, n_freq), dtype=float)

    for i, d1 in enumerate(det_list):
        for j in range(i + 1, n):
            d2 = det_list[j]
            orf = overlap_reduction_function(frequencies, d1, d2)
            out[i, j, :] = orf
            out[j, i, :] = orf

    return out
