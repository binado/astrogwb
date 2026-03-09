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


def resolve_frequency_bounds(
    sampling_frequency: float,
    minimum_frequency: float = 10.0,
    maximum_frequency: float | None = None,
) -> tuple[float, float]:
    if sampling_frequency <= 0:
        raise ValueError(
            f"sampling_frequency must be positive, got {sampling_frequency}"
        )

    nyquist_frequency = sampling_frequency / 2.0
    resolved_maximum_frequency = (
        nyquist_frequency if maximum_frequency is None else maximum_frequency
    )

    if minimum_frequency < 0:
        raise ValueError(
            f"minimum_frequency must be non-negative, got {minimum_frequency}"
        )
    if minimum_frequency >= resolved_maximum_frequency:
        raise ValueError(
            f"minimum_frequency ({minimum_frequency}) must be less than "
            f"maximum_frequency ({resolved_maximum_frequency})"
        )
    if resolved_maximum_frequency > nyquist_frequency:
        raise ValueError(
            f"maximum_frequency ({resolved_maximum_frequency}) must be less than or "
            f"equal to the Nyquist frequency ({nyquist_frequency})"
        )

    return minimum_frequency, resolved_maximum_frequency


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
    minimum_frequency : float, optional
        The minimum frequency to include in the grid in Hertz.
    reference_frequency : float
        The reference frequency in Hertz.
    maximum_frequency : float | None, optional
        The maximum frequency to include in the grid in Hertz. If None, defaults
        to the Nyquist frequency.
    """

    duration: float
    sampling_frequency: float
    reference_frequency: float
    minimum_frequency: float = 10.0
    maximum_frequency: float | None = None

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if self.sampling_frequency <= 0:
            raise ValueError(
                f"sampling_frequency must be positive, got {self.sampling_frequency}"
            )

        resolved_minimum, resolved_maximum = resolve_frequency_bounds(
            sampling_frequency=self.sampling_frequency,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
        )
        object.__setattr__(self, "minimum_frequency", resolved_minimum)
        object.__setattr__(self, "maximum_frequency", resolved_maximum)

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
        if self.maximum_frequency is None:
            raise RuntimeError("maximum_frequency should be resolved in __post_init__")
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

    def resample(
        self,
        x: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
        grid: FrequencyGrid,
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Resample values from ``grid`` onto this frequency grid.

        Values are linearly interpolated over the overlap of the source and target
        in-band frequency ranges. Samples outside that overlap are masked with
        ``NaN``.
        """
        values = np.asarray(x)
        if values.ndim != 1:
            raise ValueError(f"x must be a 1D array, got {values.ndim} dimensions")
        if values.shape != grid.frequencies.shape:
            raise ValueError(
                "x shape does not match source grid frequencies: "
                f"{values.shape} vs {grid.frequencies.shape}"
            )
        if self == grid:
            return values.copy()

        source_frequencies = grid.in_band_frequencies
        source_values = values[grid.in_band_mask]

        if np.iscomplexobj(source_values):
            out = np.full(
                self.frequencies.shape, np.nan + 1j * np.nan, dtype=np.complex128
            )
        else:
            out = np.full(self.frequencies.shape, np.nan, dtype=np.float64)

        if source_frequencies.size == 0:
            return out

        if source_frequencies.size == 1:
            target_mask = self.in_band_mask & np.isclose(
                self.frequencies, source_frequencies[0]
            )
            out[target_mask] = source_values[0]
            return out

        target_mask = (
            self.in_band_mask
            & (self.frequencies >= source_frequencies[0])
            & (self.frequencies <= source_frequencies[-1])
        )
        target_frequencies = self.frequencies[target_mask]

        if np.iscomplexobj(source_values):
            out[target_mask] = np.interp(
                target_frequencies, source_frequencies, source_values.real
            ) + 1j * np.interp(
                target_frequencies, source_frequencies, source_values.imag
            )
            return out

        out[target_mask] = np.interp(
            target_frequencies, source_frequencies, source_values
        )
        return out
