from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

if TYPE_CHECKING:
    from bilby.gw.detector import PowerSpectralDensity as BilbyPowerSpectralDensity

NOISE_CURVES_BASE_DIR = Path(__file__).parent / "noise_curves"


@dataclass
class PowerSpectralDensity:
    file: Path
    curve_type: Literal["psd", "asd"]

    def to_bilby_psd(self) -> BilbyPowerSpectralDensity:
        if self.curve_type == "psd":
            from bilby.gw.detector import (
                PowerSpectralDensity as BilbyPowerSpectralDensity,
            )

            return BilbyPowerSpectralDensity.from_power_spectral_density_file(self.file)
        elif self.curve_type == "asd":
            from bilby.gw.detector import (
                PowerSpectralDensity as BilbyPowerSpectralDensity,
            )

            return BilbyPowerSpectralDensity.from_amplitude_spectral_density_file(
                self.file
            )
        else:
            raise ValueError(f"Unknown curve type: {self.curve_type}")

    @classmethod
    def from_noise_curve_dir(cls, noise_curve: str | Path) -> Self:
        filepath = NOISE_CURVES_BASE_DIR / noise_curve
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return cls(filepath, "psd")
