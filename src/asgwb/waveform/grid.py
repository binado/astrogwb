from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt


def frequency_array(
    duration: float, sampling_frequency: float
) -> npt.NDArray[np.float64]:
    """
    Generate a frequency array for a given duration and sampling frequency.

    Adapted from https://github.com/bilby-dev/bilby/0985f75c664786e21cc4f662d4f12fe181b1a536/bilby/core/utils/series.py#L107

    Parameters
    ----------
    duration : float
        The duration of the segment in seconds.
    sampling_frequency : float
        The sampling frequency in Hertz.

    Returns
    -------
    ndarray
        An array of frequencies in Hertz.
    """
    nsamples = np.rint(duration * sampling_frequency).astype(np.int32).item()
    nfrequencies = nsamples // 2 + 1
    nyquist_frequency = sampling_frequency / 2
    return np.linspace(0, nyquist_frequency, nfrequencies, dtype=np.float64)


@dataclass(frozen=True)
class FrequencyGrid:
    """
    Immutable frequency grid parameterising a waveform computation.

    Parameters
    ----------
    duration : float
        The duration of the segment in seconds.
    sampling_frequency : float
        The sampling frequency in Hertz.
    minimum_frequency : float
        The minimum frequency to include in the grid in Hertz.
    maximum_frequency : float
        The maximum frequency to include in the grid in Hertz.
    reference_frequency : float
        The reference frequency in Hertz.
    """

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
        """
        The full frequency array.

        Returns
        -------
        ndarray
            An array of frequencies in Hertz.
        """
        return frequency_array(self.duration, self.sampling_frequency)

    @cached_property
    def in_band_mask(self) -> npt.NDArray[np.bool_]:
        """
        A boolean mask for frequencies within the band [minimum_frequency, maximum_frequency].

        Returns
        -------
        ndarray
            A boolean array of the same shape as `frequencies`.
        """
        return (self.frequencies >= self.minimum_frequency) & (
            self.frequencies <= self.maximum_frequency
        )

    @cached_property
    def in_band_frequencies(self) -> npt.NDArray[np.float64]:
        """
        The in-band frequency array.

        Returns
        -------
        ndarray
            An array of frequencies in Hertz from `minimum_frequency` to `maximum_frequency`.
        """
        return self.frequencies[self.in_band_mask]
