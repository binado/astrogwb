"""Ripple-backed frequency-domain waveform generator."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from astrogwb.frequency import frequency_grid
from astrogwb.utils import require_x64
from astrogwb.waveform._ripple import (
    build_polarization_kernel,
    build_power_kernel,
    check_sources,
    next_smooth_even,
    ripple_parameters,
)
from astrogwb.waveform.base import WaveformGenerator
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = ["RippleGenerator"]

#: Alignment tolerance, in ulps, of the first in-band bin: ``delta_f`` is
#: inexact in binary when ``n_samples`` is not a power of two.
_ALIGNMENT_TOLERANCE_EPS = 64.0


class RippleGenerator(WaveformGenerator):
    """Generate waveforms and polarization power with one fixed Ripple frequency grid.

    The frequency axis and kernel are fixed at construction. Generation is
    trace-safe and does not validate physical values; see
    :meth:`check_sources`.
    """

    __slots__ = (
        "_band",
        "_frequencies",
        "_kernel",
        "_n_samples",
        "_polarization_kernel",
        "_segment_duration",
    )
    _band: slice
    _frequencies: np.ndarray
    _kernel: Callable[[jax.Array, Mapping[str, jax.Array]], jax.Array]
    _polarization_kernel: Callable[
        [jax.Array, Mapping[str, jax.Array]], tuple[jax.Array, jax.Array]
    ]
    _n_samples: int | None
    _segment_duration: float | None

    def __init__(
        self,
        metadata: WaveformMetadata,
    ) -> None:
        super().__init__(metadata)
        segment_duration: float | None
        n_samples: int | None
        if metadata.frequency_spacing == "linear":
            segment_duration, n_samples, grid = self._fft_grid(metadata)
            band = self._resolve_band(grid)
        else:
            # The models used leave |h+|^2 + |hx|^2 independent of the grid
            # shape (see ``_ripple``), so the kernel sees the band itself.
            segment_duration = None
            n_samples = None
            grid = frequency_grid(
                metadata.frequency_spacing,
                metadata.minimum_frequency,
                metadata.maximum_frequency,
                metadata.frequency_resolution,
                metadata.turnover_frequency,
                dlogf=metadata.dlogf,
            )
            band = slice(None)

        object.__setattr__(self, "_segment_duration", segment_duration)
        object.__setattr__(self, "_n_samples", n_samples)
        object.__setattr__(self, "_frequencies", grid)
        object.__setattr__(self, "_band", band)
        kernel_options = {
            "use_taper_in_tidal_corrections": metadata.use_taper_in_tidal_corrections
        }
        object.__setattr__(
            self,
            "_kernel",
            build_power_kernel(
                metadata.approximant, metadata.reference_frequency, **kernel_options
            ),
        )
        object.__setattr__(
            self,
            "_polarization_kernel",
            build_polarization_kernel(
                metadata.approximant, metadata.reference_frequency, **kernel_options
            ),
        )

    @staticmethod
    def _fft_grid(metadata: WaveformMetadata) -> tuple[float, int, np.ndarray]:
        """Return the segment duration, sample count and one-sided FFT grid.

        The ``"linear"`` spacing: Ripple's own uniform grid from ``delta_f`` to
        Nyquist, of which the band is a slice.
        """
        resolution = metadata.frequency_resolution
        resolved_sampling_frequency = metadata.sampling_frequency
        # Rounded up to a power of two: only ever finer than asked.
        segment_duration = float(2.0 ** np.ceil(np.log2(1.0 / resolution)))
        n_samples = next_smooth_even(
            int(np.ceil(segment_duration * resolved_sampling_frequency))
        )

        # NumPy, not JAX: construction must not initialize the XLA backend.
        delta_f = resolved_sampling_frequency / n_samples
        # From bin 1: Ripple's DC bin is NaN, and masking its output still
        # leaks NaN into reverse-mode gradients (0 * NaN). Dropping the bin
        # keeps the spacing f[1] - f[0] that the models read; substituting a
        # value would not.
        grid = np.arange(1, n_samples // 2 + 1, dtype=np.float64) * delta_f
        return segment_duration, int(n_samples), grid

    def _resolve_band(self, grid: np.ndarray) -> slice:
        """Return the contiguous in-band slice of ``grid``, validating alignment.

        ``grid`` starts at ``delta_f``, so ``minimum_frequency = 0`` is
        reported as a misalignment.
        """
        in_band = np.flatnonzero(
            (grid >= self.metadata.minimum_frequency)
            & (grid <= self.metadata.maximum_frequency)
        )
        if in_band.size < 2:
            raise ValueError(
                "Ripple's in-band frequency grid has fewer than two bins in "
                f"[{self.metadata.minimum_frequency}, {self.metadata.maximum_frequency}] Hz"
            )
        first_frequency = float(grid[in_band[0]])
        tolerance = (
            _ALIGNMENT_TOLERANCE_EPS
            * np.finfo(np.float64).eps
            * max(1.0, abs(self.metadata.minimum_frequency), abs(first_frequency))
        )
        if not np.isclose(
            first_frequency, self.metadata.minimum_frequency, rtol=0.0, atol=tolerance
        ):
            raise ValueError(
                f"minimum_frequency ({self.metadata.minimum_frequency} Hz) is not on "
                "Ripple's frequency grid; the nearest in-band bin is "
                f"{first_frequency} Hz"
            )
        return slice(int(in_band[0]), int(in_band[-1]) + 1)

    @property
    def segment_duration(self) -> float | None:
        """The analysis-segment length in seconds the grid was sized from.

        ``None`` unless ``frequency_spacing`` is ``"linear"``: the other
        spacings are built directly and are not tied to an FFT segment.
        """
        return self._segment_duration

    @property
    def n_samples(self) -> int | None:
        """Ripple's segment length in samples: even, and 5-smooth.

        ``None`` unless ``frequency_spacing`` is ``"linear"``.
        """
        return self._n_samples

    @property
    def frequencies(self) -> jax.Array:
        """The frequency grid Ripple is evaluated on, restricted to the band.

        For ``"linear"`` spacing the underlying grid is Ripple's FFT grid, which
        runs from ``delta_f`` rather than zero; see :meth:`_fft_grid`. The other
        spacings are the band's own grid and are returned whole.
        """
        return jnp.asarray(self._frequencies[self._band])

    def check_sources(self, source_parameters: Mapping[str, ArrayLike]) -> None:
        """Check source values against what this approximant can represent.

        Eager only; see
        :meth:`~astrogwb.waveform.WaveformGenerator.check_sources`.

        Parameters
        ----------
        source_parameters
            Source parameter values by name.
        """
        check_sources(self.metadata.approximant, source_parameters)

    @require_x64
    def _generate_batch(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Generate power in ``(frequency, sample)`` layout via Ripple.

        With ``"linear"`` spacing the band is sliced from the output, not the
        input: the NRTidal and precessing models read the grid's spacing and
        top. Other spacings evaluate the band itself.
        """
        events = ripple_parameters(self.metadata.approximant, source_parameters)
        power = self._kernel(jnp.asarray(self._frequencies), events)
        return power[self._band, :]

    @require_x64
    def _generate_polarizations_batch(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        """Generate complex polarizations ``(sample, frequency)`` via Ripple.

        The same grid and band as :meth:`_generate_batch`.
        """
        events = ripple_parameters(self.metadata.approximant, source_parameters)
        plus, cross = self._polarization_kernel(jnp.asarray(self._frequencies), events)
        return plus[:, self._band], cross[:, self._band]

    def __call__(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        """Return the Ripple frequency axis and power."""
        return self.frequencies, self.generate_batch(source_parameters)
