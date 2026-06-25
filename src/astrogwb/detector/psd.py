from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

NOISE_CURVES_BASE_DIR = Path(__file__).parent / "noise_curves"


@dataclass(frozen=True)
class PowerSpectralDensity:
    """One-sided detector noise PSD loaded from a two-column text file."""

    file: Path

    @classmethod
    def from_noise_curve_dir(cls, noise_curve: str | Path) -> Self:
        filepath = NOISE_CURVES_BASE_DIR / noise_curve
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return cls(filepath)

    def evaluate(self, frequencies: ArrayLike) -> NDArray[np.float64]:
        curve = np.loadtxt(self.file)
        return np.interp(
            np.asarray(frequencies, dtype=float),
            curve[:, 0],
            curve[:, 1],
            left=np.inf,
            right=np.inf,
        )
