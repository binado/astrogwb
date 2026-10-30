"""Validated geodetic geometry and owned detector objects.

Geometry uses WGS-84, metres, and radians with azimuths clockwise from North.
Construction and cached NumPy geometry never initialize the XLA backend.
"""

from __future__ import annotations

import math
from functools import cache, cached_property
from typing import Annotated

import jax
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from . import projection
from .sensitivity import Sensitivity


@dataclass(frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False))
class DetectorGeometry:
    """Defining geodetic fields; derived arrays are excluded from serialization."""

    latitude_rad: Annotated[float, Field(ge=-math.pi / 2, le=math.pi / 2)]
    longitude_rad: Annotated[float, Field(ge=-math.pi, le=math.pi)]
    elevation_m: Annotated[float, Field(ge=-1e4, le=1e5)]
    xarm_azimuth_rad: float
    yarm_azimuth_rad: float
    xarm_tilt_rad: float = 0.0
    yarm_tilt_rad: float = 0.0

    @cached_property
    def location(self) -> NDArray[np.float64]:
        """Earth-fixed WGS-84 vertex location in metres, read-only."""
        lat, lon = self.latitude_rad, self.longitude_rad
        eccentricity_squared = (1 / 298.257223563) * (2 - 1 / 298.257223563)
        radius = 6378137.0 / math.sqrt(1 - eccentricity_squared * math.sin(lat) ** 2)
        result = np.array(
            [
                (radius + self.elevation_m) * math.cos(lat) * math.cos(lon),
                (radius + self.elevation_m) * math.cos(lat) * math.sin(lon),
                (radius * (1 - eccentricity_squared) + self.elevation_m)
                * math.sin(lat),
            ],
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result

    @cached_property
    def response_tensor(self) -> NDArray[np.float64]:
        """Half the difference of arm outer products, read-only float64."""
        lat, lon = self.latitude_rad, self.longitude_rad
        north = np.array(
            [
                -math.sin(lat) * math.cos(lon),
                -math.sin(lat) * math.sin(lon),
                math.cos(lat),
            ]
        )
        east = np.array([-math.sin(lon), math.cos(lon), 0.0])
        up = np.array(
            [
                math.cos(lat) * math.cos(lon),
                math.cos(lat) * math.sin(lon),
                math.sin(lat),
            ]
        )

        def arm(azimuth: float, tilt: float) -> NDArray[np.float64]:
            return (
                math.cos(tilt) * (math.cos(azimuth) * north + math.sin(azimuth) * east)
                + math.sin(tilt) * up
            )

        x = arm(self.xarm_azimuth_rad, self.xarm_tilt_rad)
        y = arm(self.yarm_azimuth_rad, self.yarm_tilt_rad)
        result = (np.outer(x, x) - np.outer(y, y)) / 2
        result.setflags(write=False)
        return result


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Detector:
    """A named detector with geometry and a lazily evaluated sensitivity."""

    name: Annotated[str, Field(min_length=1, pattern=r"\S")]
    geometry: DetectorGeometry
    sensitivity: Sensitivity

    def antenna_pattern(
        self,
        gmst: ArrayLike,
        *,
        right_ascension: ArrayLike,
        declination: ArrayLike,
        polarization_angle: ArrayLike,
    ) -> tuple[jax.Array, jax.Array]:
        """Return the gwmock/LAL plus and cross factors."""
        return projection.antenna_pattern(
            self.geometry.response_tensor,
            gmst,
            right_ascension=right_ascension,
            declination=declination,
            polarization_angle=polarization_angle,
        )

    def time_delay_from_geocenter(
        self, gmst: ArrayLike, *, right_ascension: ArrayLike, declination: ArrayLike
    ) -> jax.Array:
        """Return the geocenter-to-detector delay in seconds."""
        return projection.time_delay_from_geocenter(
            self.geometry.location,
            gmst,
            right_ascension=right_ascension,
            declination=declination,
        )

    def project_polarizations_fd(
        self,
        frequencies: ArrayLike,
        plus: ArrayLike,
        cross: ArrayLike,
        gmst: ArrayLike,
        *,
        right_ascension: ArrayLike,
        declination: ArrayLike,
        polarization_angle: ArrayLike,
    ) -> jax.Array:
        """Project radiation-frame polarizations, with a final frequency axis."""
        return projection.project_polarizations_fd(
            self.geometry.response_tensor,
            self.geometry.location,
            frequencies,
            plus,
            cross,
            gmst,
            right_ascension=right_ascension,
            declination=declination,
            polarization_angle=polarization_angle,
        )


@cache
def load_detector(name: str) -> Detector:
    """Load packaged geometry and sensitivity without reading the PSD curve."""
    from .config import load_detector_config

    return load_detector_config([]).detectors[name].to_detector(name)
