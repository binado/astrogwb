from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from bilby.gw.detector import Interferometer

from .psd import PowerSpectralDensity

DETECTOR_FILE = Path(__file__).parent / "detectors.toml"


@dataclass
class Detector:
    name: str
    psd: PowerSpectralDensity
    minimum_frequency: float
    maximum_frequency: float
    length: float
    latitude: float
    longitude: float
    elevation: float
    xarm_azimuth: float
    yarm_azimuth: float
    xarm_tilt: float
    yarm_tilt: float
    duty_factor: float

    @classmethod
    def from_dict(
        cls, data: dict, psd: PowerSpectralDensity | str | Path | None = None
    ) -> Self:
        psd = psd or data["default_noise_curve"]
        if isinstance(psd, (str, Path)):
            psd = PowerSpectralDensity.from_noise_curve_dir(psd)
        elif psd is None:
            raise ValueError("Noise curve is required")

        return cls(
            name=data["name"],
            psd=psd,
            minimum_frequency=data["minimum_frequency"],
            maximum_frequency=data["maximum_frequency"],
            length=data["length"],
            latitude=data["latitude"],
            longitude=data["longitude"],
            elevation=data["elevation"],
            xarm_azimuth=data["xarm_azimuth"],
            yarm_azimuth=data["yarm_azimuth"],
            xarm_tilt=data.get("xarm_tilt", 0.0),
            yarm_tilt=data.get("yarm_tilt", 0.0),
            duty_factor=data["duty_factor"],
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

    def to_bilby_detector(self) -> Interferometer:
        return Interferometer(
            name=self.name,
            power_spectral_density=self.psd.to_bilby_psd(),
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            length=self.length,
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            xarm_azimuth=self.xarm_azimuth,
            yarm_azimuth=self.yarm_azimuth,
            xarm_tilt=self.xarm_tilt,
            yarm_tilt=self.yarm_tilt,
        )
