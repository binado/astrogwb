"""Ripple-backed frequency-domain polarization-power generator."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
from gwmock_signal.waveform import RippleBackend
from numpy.typing import ArrayLike, NDArray

from astrogwb.utils import array_dict_shape
from astrogwb.waveform.generator.base import (
    GRID_SPACING_TOLERANCE_ULP,
    PolarizationPowerGenerator,
)
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
        resolved_sampling_frequency = float(sampling_frequency)
        if (
            not np.isfinite(resolved_sampling_frequency)
            or resolved_sampling_frequency <= 0.0
        ):
            raise ValueError("sampling_frequency must be finite and positive")
        if isinstance(chunk_size, bool) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer")

        segment_duration = 1.0 / resolution
        segment_duration = float(2.0 ** math.ceil(math.log2(segment_duration)))
        n_samples = round(segment_duration * resolved_sampling_frequency)
        if n_samples <= 0:
            raise ValueError("sampling_frequency produces no Ripple samples")
        effective_df = resolved_sampling_frequency / n_samples

        minimum = float(minimum_frequency)
        alignment = minimum / effective_df
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(alignment))
        if not np.isclose(alignment, round(alignment), rtol=0.0, atol=tolerance):
            raise ValueError(
                "minimum_frequency must align with Ripple's effective frequency "
                f"resolution ({effective_df} Hz)"
            )
        super().__init__(
            approximant=approximant,
            minimum_frequency=minimum,
            maximum_frequency=maximum_frequency,
            reference_frequency=reference_frequency,
            sampling_frequency=resolved_sampling_frequency,
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
        parameters = {
            name: jnp.asarray(values) for name, values in source_parameters.items()
        }
        parameter_shape = array_dict_shape(parameters)
        if len(parameter_shape) != 1:
            raise ValueError(
                "Ripple source parameters must be one-dimensional; "
                f"received shape {parameter_shape}"
            )
        n_events = parameter_shape[0]
        if n_events == 0:
            raise ValueError("source_parameters must contain at least one event")
        step = min(n_events, self.chunk_size)

        power_chunks = []
        for start in range(0, n_events, step):
            stop = min(start + step, n_events)
            chunk_samples = {
                name: values[start:stop] for name, values in parameters.items()
            }
            polarizations = self._backend.generate_fd_polarizations_batch(
                self.approximant,
                sampling_frequency=self.sampling_frequency,
                minimum_frequency=self.minimum_frequency,
                parameters=chunk_samples,
            )
            chunk_frequencies = jnp.asarray(polarizations.frequencies)
            chunk_plus = jnp.asarray(polarizations.plus)
            chunk_cross = jnp.asarray(polarizations.cross)
            mask = (chunk_frequencies >= self.minimum_frequency) & (
                chunk_frequencies <= self.maximum_frequency
            )
            descriptor_frequencies = self.frequencies
            candidate = chunk_frequencies[mask]
            grid_tolerance = (
                GRID_SPACING_TOLERANCE_ULP
                * jnp.finfo(jnp.float64).eps
                * jnp.maximum(1.0, jnp.max(jnp.abs(descriptor_frequencies)))
            )
            grids_match = candidate.shape == descriptor_frequencies.shape and bool(
                jnp.allclose(
                    candidate, descriptor_frequencies, rtol=0.0, atol=grid_tolerance
                )
            )
            if not grids_match:
                raise ValueError(
                    "Ripple returned a frequency grid different from its descriptor"
                )
            power_chunks.append(
                polarization_power(chunk_plus[:, mask], chunk_cross[:, mask])
            )
            logger.info("Generated chunk %d:%d of %d events", start, stop, n_events)

        return np.asarray(jnp.concatenate(power_chunks, axis=1), dtype=np.float64)
