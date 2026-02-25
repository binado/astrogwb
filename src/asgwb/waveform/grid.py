from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class FrequencyGrid:
    """Immutable frequency grid parameterising a waveform computation."""

    duration: float
    sampling_frequency: float
    minimum_frequency: float
    maximum_frequency: float
    reference_frequency: float

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if self.minimum_frequency >= self.maximum_frequency:
            raise ValueError(
                f"minimum_frequency ({self.minimum_frequency}) must be less than "
                f"maximum_frequency ({self.maximum_frequency})"
            )
        if self.minimum_frequency < 0:
            raise ValueError(
                f"minimum_frequency must be non-negative, got {self.minimum_frequency}"
            )

    @cached_property
    def frequencies(self) -> npt.NDArray[np.float64]:
        delta_f = 1.0 / self.duration
        return np.arange(
            self.minimum_frequency,
            self.maximum_frequency + delta_f,
            delta_f,
            dtype=np.float64,
        )
