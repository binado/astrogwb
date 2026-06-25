from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

NOISE_CURVES_BASE_DIR = Path(__file__).parent / "noise_curves"


@dataclass(frozen=True)
class PowerSpectralDensity:
    """One-sided detector noise PSD loaded from a two-column text file."""

    file: Path
    _frequency_grid: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _psd_values: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        curve = np.loadtxt(self.file)
        object.__setattr__(
            self, "_frequency_grid", np.asarray(curve[:, 0], dtype=np.float64)
        )
        object.__setattr__(self, "_psd_values", np.asarray(curve[:, 1], dtype=np.float64))

    @classmethod
    def from_noise_curve_dir(cls, noise_curve: str | Path) -> Self:
        filepath = NOISE_CURVES_BASE_DIR / noise_curve
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return cls(filepath)

    def evaluate(self, frequencies: ArrayLike) -> NDArray[np.float64]:
        return np.interp(
            np.asarray(frequencies, dtype=float),
            self._frequency_grid,
            self._psd_values,
            left=np.inf,
            right=np.inf,
        )
