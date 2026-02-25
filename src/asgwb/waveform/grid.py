from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt


def frequency_array(
    duration: float, sampling_frequency: float
) -> npt.NDArray[np.float64]:
    nsamples = np.rint(duration * sampling_frequency).astype(np.int32).item()
    nfrequencies = nsamples // 2 + 1
    nyquist_frequency = sampling_frequency / 2
    return np.linspace(0, nyquist_frequency, nfrequencies, dtype=np.float64)


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
        if self.sampling_frequency <= 0:
            raise ValueError(
                f"sampling_frequency must be positive, got {self.sampling_frequency}"
            )
        if self.minimum_frequency >= self.maximum_frequency:
            raise ValueError(
                f"minimum_frequency ({self.minimum_frequency}) must be less than "
                f"maximum_frequency ({self.maximum_frequency})"
            )
        if self.minimum_frequency < 0:
            raise ValueError(
                f"minimum_frequency must be non-negative, got {self.minimum_frequency}"
            )

        nyquist_frequency = self.sampling_frequency / 2
        if self.maximum_frequency > nyquist_frequency:
            raise ValueError(
                f"maximum_frequency ({self.maximum_frequency}) must be less than or "
                f"equal to the Nyquist frequency ({nyquist_frequency})"
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

    @cached_property
    def in_band_mask(self) -> npt.NDArray[bool]:
        return (self.frequencies >= self.minimum_frequency) & (
            self.frequencies <= self.maximum_frequency
        )
