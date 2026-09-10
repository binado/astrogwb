"""Ripple-backed frequency-domain polarization-power generator."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from gwmock_signal.waveform import RippleBackend
from numpy.typing import ArrayLike

from astrogwb.utils import array_dict_shape
from astrogwb.waveform.generator.base import PolarizationPowerGenerator
from astrogwb.waveform.polarization_power import polarization_power

__all__ = ["RippleGenerator"]

logger = logging.getLogger(__name__)


class RippleGenerator(PolarizationPowerGenerator):
    """Generate chunked polarization power with one fixed Ripple frequency grid."""

    __slots__ = ("_backend", "chunk_size")
    _backend: RippleBackend
    chunk_size: int

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

        # astrogwb's own power-of-two rounding policy for the segment duration
        # -- deliberate, and only ever makes the grid finer than asked. It no
        # longer produces a spacing: Ripple's own rounding
        # (`_next_smooth_even`, 5-smooth, not power-of-two) decides the actual
        # frequency grid, so `n_samples` survives only as a feasibility guard.
        segment_duration = float(2.0 ** np.ceil(np.log2(1.0 / resolution)))
        n_samples = round(segment_duration * resolved_sampling_frequency)
        if n_samples <= 0:
            raise ValueError("sampling_frequency produces no Ripple samples")

        super().__init__(
            approximant=approximant,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            reference_frequency=reference_frequency,
            sampling_frequency=resolved_sampling_frequency,
            frequency_resolution=resolution,
        )
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
    ) -> tuple[jax.Array, jax.Array]:
        """Return the Ripple frequency axis and power, chunk by chunk.

        Ripple sizes its FFT segment from ``segment_duration`` with its own
        rounding rule (5-smooth, not power-of-two -- see
        ``gwmock_signal.RippleBackend._segment_samples``), so the frequency
        axis is taken from the generated polarizations rather than recomputed
        here. ``segment_duration`` is pinned on ``_backend``, so every chunk
        lands on the same grid; a mismatch raises.

        ``minimum_frequency`` alignment is checked here, against the grid
        Ripple actually built, rather than at construction time: no
        backend-independent check on ``minimum_frequency`` alone can predict
        Ripple's 5-smooth ``n_samples`` without replicating the rule this
        generator exists to delete.
        """
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

        frequencies: jax.Array | None = None
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
            masked_frequencies = chunk_frequencies[mask]
            if frequencies is None:
                if masked_frequencies.size < 2:
                    raise ValueError(
                        "Ripple's in-band frequency grid has fewer than two "
                        f"bins in [{self.minimum_frequency}, "
                        f"{self.maximum_frequency}] Hz"
                    )
                # Not `==`: when `n_samples` is 5-smooth but not a power of
                # two, `delta_f = fs / n_samples` is inexact in binary and
                # `k * delta_f` need not reproduce `minimum_frequency` bit for
                # bit even when the configuration is valid.
                first_frequency = float(masked_frequencies[0])
                tolerance = (
                    64.0
                    * np.finfo(np.float64).eps
                    * max(1.0, abs(self.minimum_frequency), abs(first_frequency))
                )
                if not np.isclose(
                    first_frequency, self.minimum_frequency, rtol=0.0, atol=tolerance
                ):
                    raise ValueError(
                        f"minimum_frequency ({self.minimum_frequency} Hz) is not "
                        "on Ripple's frequency grid; the nearest in-band bin is "
                        f"{first_frequency} Hz"
                    )
                frequencies = masked_frequencies
            elif not bool(jnp.array_equal(frequencies, masked_frequencies)):
                raise ValueError("Ripple chunks produced different frequency grids")
            power_chunks.append(
                polarization_power(chunk_plus[:, mask], chunk_cross[:, mask])
            )
            logger.info("Generated chunk %d:%d of %d events", start, stop, n_events)

        assert frequencies is not None
        return frequencies, jnp.concatenate(power_chunks, axis=1)
