from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import numpy.typing as npt
from bilby.gw.detector import PowerSpectralDensity as BilbyPowerSpectralDensity

if TYPE_CHECKING:
    from scipy.interpolate import CubicSpline


NOISE_CURVES_BASE_DIR = Path(__file__).parent / "noise_curves"


@dataclass
class PowerSpectralDensity:
    file: Path
    curve_type: Literal["psd", "asd"]

    def __post_init__(self):
        self._psd_interpolator: CubicSpline | None = None

    def _build_interpolator(self) -> CubicSpline:
        from scipy.interpolate import CubicSpline

        f, psd = np.loadtxt(self.file, unpack=True)
        if self.curve_type == "asd":
            psd = psd**2
        return CubicSpline(f, psd)

    def __call__(self, frequency: npt.NDArray) -> npt.NDArray:
        if self._psd_interpolator is None:
            self._psd_interpolator = self._build_interpolator()
        return self._psd_interpolator(frequency)

    def to_bilby_psd(self) -> BilbyPowerSpectralDensity:
        if self.curve_type not in {"psd", "asd"}:
            raise ValueError(f"Unknown curve type: {self.curve_type}")

        if self.curve_type == "psd":
            return BilbyPowerSpectralDensity.from_power_spectral_density_file(self.file)

        return BilbyPowerSpectralDensity.from_amplitude_spectral_density_file(self.file)

    @classmethod
    def from_noise_curve_dir(cls, noise_curve: str | Path) -> Self:
        filepath = NOISE_CURVES_BASE_DIR / noise_curve
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return cls(filepath, "psd")
