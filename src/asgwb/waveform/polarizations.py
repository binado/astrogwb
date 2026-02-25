from __future__ import annotations

from dataclasses import dataclass

import numpy.typing as npt
import numpy as np

from .grid import FrequencyGrid


@dataclass
class WaveformPolarizations:
    """Plus and cross polarization arrays on a frequency grid."""

    grid: FrequencyGrid
    hp: npt.NDArray[np.complex128]
    hc: npt.NDArray[np.complex128]
