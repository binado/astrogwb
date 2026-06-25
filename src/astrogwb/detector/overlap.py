"""Frequency-dependent overlap reduction and effective PSD utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from gwmock_signal.detector import CustomDetector
from numpy.typing import ArrayLike, NDArray

from .detector import Detector

R_EARTH = 6371.0  # km
C_LIGHT = 299792.458  # km/s

_LOW_ALPHA_THRESHOLD = 2e-3
_NETWORK_SENSITIVITY_FACTOR = 0.16

DetectorSpec = Detector | CustomDetector


@dataclass(frozen=True)
class _Geometry:
    name: str
    latitude: float
    longitude: float
    elevation: float
    xarm_azimuth: float
    yarm_azimuth: float


def _geometry(detector: DetectorSpec) -> _Geometry:
    if isinstance(detector, Detector):
        return _Geometry(
            name=detector.name,
            latitude=detector.latitude,
            longitude=detector.longitude,
            elevation=detector.elevation,
            xarm_azimuth=detector.xarm_azimuth,
            yarm_azimuth=detector.yarm_azimuth,
        )
    if isinstance(detector, CustomDetector):
        return _Geometry(
            name=detector.name,
            latitude=math.degrees(detector.latitude_rad),
            longitude=math.degrees(detector.longitude_rad),
            elevation=detector.elevation_m,
            xarm_azimuth=math.degrees(detector.xarm_azimuth_rad),
            yarm_azimuth=math.degrees(detector.yarm_azimuth_rad),
        )
    raise TypeError(
        "detector must be an astrogwb Detector or gwmock_signal CustomDetector"
    )


def _chord_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.cos(lat2_r) * math.sin(dlon)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(
        lat2_r
    ) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360


def _final_course(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    return (_initial_course(lat2, lat1, lon2, lon1) + 180.0) % 360


def _opening_angle(az1: float, az2: float) -> float:
    diff = ((az1 - az2 + 180.0) % 360.0) - 180.0
    return math.radians(abs(diff))


def _azimuth_bisector(az1: float, az2: float) -> float:
    a1 = math.radians(az1)
    a2 = math.radians(az2)
    return math.atan2(math.sin(a1) + math.sin(a2), math.cos(a1) + math.cos(a2))


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
    frequencies = np.asarray(frequencies, dtype=float)
    det1 = _geometry(detector_1)
    det2 = _geometry(detector_2)

    d = _chord_distance(det1.latitude, det1.longitude, det2.latitude, det2.longitude)
    alpha = 2.0 * math.pi * frequencies * d / C_LIGHT

    xax_1 = _azimuth_bisector(det1.xarm_azimuth, det1.yarm_azimuth)
    xax_2 = _azimuth_bisector(det2.xarm_azimuth, det2.yarm_azimuth)

    ang_1 = math.radians(
        _initial_course(det1.latitude, det2.latitude, det1.longitude, det2.longitude)
        - 90.0
    )
    ang_2 = math.radians(
        _final_course(det1.latitude, det2.latitude, det1.longitude, det2.longitude)
        - 90.0
    )

    delta = 0.5 * ((xax_1 + ang_1) - (xax_2 + ang_2))
    big_delta = 0.5 * ((xax_1 + ang_1) + (xax_2 + ang_2))

    asin_arg = max(-1.0, min(1.0, 0.5 * d / R_EARTH))
    beta = 2.0 * math.asin(asin_arg)

    ang_btw_arms_1 = _opening_angle(det1.xarm_azimuth, det1.yarm_azimuth)
    ang_btw_arms_2 = _opening_angle(det2.xarm_azimuth, det2.yarm_azimuth)

    return _get_orf(alpha, beta, delta, big_delta, ang_btw_arms_1, ang_btw_arms_2)


def pairwise_overlap_reduction_function(
    frequencies: ArrayLike,
    detectors: Sequence[DetectorSpec],
) -> NDArray[np.float64]:
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
    detectors: Sequence[Detector],
) -> NDArray[np.float64]:
    frequencies = np.asarray(frequencies, dtype=float)
    det_list = list(detectors)
    if len(det_list) < 2:
        return np.full(frequencies.shape, np.inf, dtype=float)

    inverse_variance = np.zeros(frequencies.shape, dtype=float)
    for i, det1 in enumerate(det_list):
        psd1 = det1.psd.evaluate(frequencies)
        for det2 in det_list[i + 1 :]:
            psd2 = det2.psd.evaluate(frequencies)
            gamma = overlap_reduction_function(frequencies, det1, det2)
            with np.errstate(invalid="ignore", divide="ignore"):
                contribution = gamma**2 / (psd1 * psd2)
            inverse_variance += np.where(np.isfinite(contribution), contribution, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = 1.0 / (_NETWORK_SENSITIVITY_FACTOR * np.sqrt(inverse_variance))
    return np.where(inverse_variance > 0.0, result, np.inf)
