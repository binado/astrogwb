from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from gwmock_signal.detector import CustomDetector

from .psd import PowerSpectralDensity

DETECTOR_FILE = Path(__file__).parent / "detectors.toml"


@dataclass(frozen=True)
class Detector:
    """Detector geometry plus a sensitivity curve for SGWB contractions."""

    geometry: CustomDetector
    psd: PowerSpectralDensity
    minimum_frequency: float
    maximum_frequency: float
    length: float
    duty_factor: float

    @property
    def name(self) -> str:
        return self.geometry.name

    @property
    def latitude(self) -> float:
        return math.degrees(self.geometry.latitude_rad)

    @property
    def longitude(self) -> float:
        return math.degrees(self.geometry.longitude_rad)

    @property
    def elevation(self) -> float:
        return self.geometry.elevation_m

    @property
    def xarm_azimuth(self) -> float:
        return math.degrees(self.geometry.xarm_azimuth_rad)

    @property
    def yarm_azimuth(self) -> float:
        return math.degrees(self.geometry.yarm_azimuth_rad)

    @classmethod
    def from_dict(
        cls,
        data: dict,
        psd: PowerSpectralDensity | str | Path | None = None,
    ) -> Self:
        psd = psd or data["default_noise_curve"]
        if isinstance(psd, (str, Path)):
            psd = PowerSpectralDensity.from_noise_curve_dir(psd)
        if psd is None:
            raise ValueError("Noise curve is required")

        name = data["name"]
        geometry = CustomDetector(
            name=name,
            latitude_rad=math.radians(float(data["latitude"])),
            longitude_rad=math.radians(float(data["longitude"])),
            elevation_m=float(data["elevation"]),
            xarm_azimuth_rad=math.radians(float(data["xarm_azimuth"])),
            yarm_azimuth_rad=math.radians(float(data["yarm_azimuth"])),
            xarm_tilt_rad=math.radians(float(data.get("xarm_tilt", 0.0))),
            yarm_tilt_rad=math.radians(float(data.get("yarm_tilt", 0.0))),
        )
        return cls(
            geometry=geometry,
            psd=psd,
            minimum_frequency=float(data["minimum_frequency"]),
            maximum_frequency=float(data["maximum_frequency"]),
            length=float(data["length"]),
            duty_factor=float(data["duty_factor"]),
        )

    @classmethod
    def from_file(
        cls,
        name: str,
        path: Path | None = None,
        table: str | None = None,
        psd: PowerSpectralDensity | str | Path | None = None,
    ) -> Self:
        if path is None:
            path = DETECTOR_FILE
            table = "detectors"
        with open(path, "rb") as f:
            data = tomllib.load(f)

        detectors_data = data[table] if table else data
        detector_data = detectors_data[name]
        detector_data.setdefault("name", name)
        return cls.from_dict(detector_data, psd=psd)
