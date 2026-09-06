"""Ripple-backed frequency-domain polarization-power generator."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping

import numpy as np
from gwmock_signal.waveform import RippleBackend
from numpy.typing import ArrayLike, NDArray

from astrogwb.waveform.generator.base import PolarizationPowerGenerator
from astrogwb.waveform.polarization_power import polarization_power

__all__ = ["RippleGenerator"]

logger = logging.getLogger(__name__)


class RippleGenerator(PolarizationPowerGenerator):
    """Generate chunked polarization power with one fixed Ripple frequency grid."""

    __slots__ = ("_backend", "chunk_size", "frequency_resolution")
    _backend: RippleBackend
    chunk_size: int
    frequency_resolution: float

    def __init__(
        self,
        *,
        approximant: str,
        sampling_frequency: float,
        minimum_frequency: float,
        maximum_frequency: float,
        reference_frequency: float,
        frequency_resolution: float,
        chunk_size: int,
    ) -> None:
        resolution = float(frequency_resolution)
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("frequency_resolution must be a finite positive scalar")
        if isinstance(chunk_size, bool) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer")

        segment_duration = _resolve_segment_duration(resolution)
        effective_df = _effective_resolution(segment_duration, sampling_frequency)
        frequencies = _ripple_frequency_grid(
            sampling_frequency, effective_df, maximum_frequency
        )
        super().__init__(
            frequencies=frequencies,
            approximant=approximant,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            reference_frequency=reference_frequency,
            sampling_frequency=sampling_frequency,
            df=effective_df,
        )
        object.__setattr__(self, "frequency_resolution", resolution)
        object.__setattr__(self, "chunk_size", chunk_size)
        object.__setattr__(
            self,
            "_backend",
            RippleBackend(
                f_ref=self.reference_frequency,
                segment_duration=segment_duration,
            ),
        )

    def __call__(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> NDArray[np.float64]:
        """Generate power in ``(frequency, sample)`` layout, chunk by chunk."""
        n_events = np.asarray(source_parameters["detector_frame_mass_1"]).shape[0]
        if n_events == 0:
            raise ValueError("source_parameters must contain at least one event")
        step = min(n_events, self.chunk_size)

        power_chunks: list[NDArray[np.float64]] = []
        for start in range(0, n_events, step):
            stop = min(start + step, n_events)
            chunk_samples = {
                name: np.asarray(values)[start:stop]
                for name, values in source_parameters.items()
            }
            polarizations = self._backend.generate_fd_polarizations_batch(
                self.approximant,
                sampling_frequency=self.sampling_frequency,
                minimum_frequency=self.minimum_frequency,
                parameters=chunk_samples,
            )
            chunk_frequencies = np.asarray(polarizations.frequencies)
            chunk_plus = np.asarray(polarizations.plus)
            chunk_cross = np.asarray(polarizations.cross)
            mask = chunk_frequencies <= self.maximum_frequency
            if not np.array_equal(chunk_frequencies[mask], self.frequencies):
                raise ValueError(
                    "Ripple returned a frequency grid different from its descriptor"
                )
            power_chunks.append(
                polarization_power(chunk_plus[:, mask], chunk_cross[:, mask])
            )
            logger.info("Generated chunk %d:%d of %d events", start, stop, n_events)

        return np.concatenate(power_chunks, axis=1)


def _resolve_segment_duration(frequency_resolution: float) -> float:
    """Map a target frequency resolution to a segment duration in seconds."""
    return 1.0 / frequency_resolution


def _effective_resolution(segment_duration: float, sampling_frequency: float) -> float:
    """Return the achieved df after Ripple rounds duration to a power of two."""
    rounded_seconds = float(2.0 ** math.ceil(math.log2(segment_duration)))
    return sampling_frequency / round(rounded_seconds * sampling_frequency)


def _ripple_frequency_grid(
    sampling_frequency: float, df: float, maximum_frequency: float
) -> NDArray[np.float64]:
    """Build Ripple's nonnegative FFT grid and apply the upper-frequency cut."""
    n_samples = round(sampling_frequency / df)
    frequencies = np.arange(n_samples // 2 + 1, dtype=np.float64) * df
    return frequencies[frequencies <= maximum_frequency]
