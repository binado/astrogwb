from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .waveform import FrequencyGrid


@dataclass
class SpectralDensity:
    """Spectral density values defined on a frequency grid."""

    grid: FrequencyGrid
    spectral_density: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        self.spectral_density = np.asarray(self.spectral_density, dtype=np.float64)
        if self.spectral_density.ndim != 1:
            raise ValueError(
                "spectral_density must be a 1D array, "
                f"got {self.spectral_density.ndim} dimensions"
            )
        if self.spectral_density.shape != self.grid.frequencies.shape:
            raise ValueError(
                "spectral_density shape does not match grid frequencies: "
                f"{self.spectral_density.shape} vs {self.grid.frequencies.shape}"
            )

    def dump(self, path: Path) -> None:
        from .io import dump_spectral_density_hdf5

        dump_spectral_density_hdf5(path=path, spectral_density=self)

    @classmethod
    def load(cls, path: Path) -> SpectralDensity:
        from .io import load_spectral_density_hdf5

        return load_spectral_density_hdf5(path)
