"""Ripple-backed frequency-domain polarization-power generator."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from astrogwb.frequency import frequency_grid
from astrogwb.metadata import WaveformMetadata
from astrogwb.utils import require_x64
from astrogwb.waveform.generator._ripple import (
    build_power_kernel,
    check_sources,
    next_smooth_even,
    ripple_parameters,
)
from astrogwb.waveform.generator.base import PolarizationPowerGenerator

__all__ = ["RippleGenerator"]

#: How far the first in-band bin may sit from ``minimum_frequency`` and still
#: count as aligned. Not ``==``: when ``n_samples`` is 5-smooth but not a power
#: of two, ``delta_f = f_s / n_samples`` is inexact in binary and ``k * delta_f``
#: need not reproduce ``minimum_frequency`` bit for bit even when the
#: configuration is valid.
_ALIGNMENT_TOLERANCE_EPS = 64.0


class RippleGenerator(PolarizationPowerGenerator):
    """Generate polarization power with one fixed Ripple frequency grid.

    A run has one approximant, one reference frequency, and one sampling grid.
    Those fix the frequency axis and the waveform kernel at construction, so
    :attr:`frequencies` is available before the first generation (including for
    an empty catalog) and the source parameters are the only thing that varies
    between calls.

    Generation is trace-safe: it performs no host synchronization and no
    data-dependent branching, so it composes with :func:`jax.jit`,
    :func:`jax.lax.scan` and NumPyro inference. The price is that physical
    values are not validated here -- see :meth:`check_sources`.
    """

    __slots__ = ("_band", "_frequencies", "_kernel", "_n_samples", "_segment_duration")
    _band: slice
    _frequencies: np.ndarray
    _kernel: Callable[[jax.Array, Mapping[str, jax.Array]], jax.Array]
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
            # The grid is ours to choose: Ripple takes any strictly increasing
            # array, and the models astrogwb uses read its shape only in ways
            # that leave |h+|^2 + |hx|^2 alone (see the module docstring of
            # ``_ripple``). So the kernel sees the band's own grid, built
            # straight to ``maximum_frequency``, and nothing is sliced. It
            # starts at ``minimum_frequency > 0`` (the metadata guarantees it),
            # so there is no DC bin to drop.
            segment_duration = None
            n_samples = None
            grid = frequency_grid(
                metadata.frequency_spacing,
                metadata.minimum_frequency,
                metadata.maximum_frequency,
                metadata.frequency_resolution,
                metadata.turnover_frequency,
            )
            band = slice(None)

        object.__setattr__(self, "_segment_duration", segment_duration)
        object.__setattr__(self, "_n_samples", n_samples)
        object.__setattr__(self, "_frequencies", grid)
        object.__setattr__(self, "_band", band)
        object.__setattr__(
            self,
            "_kernel",
            build_power_kernel(
                metadata.approximant,
                metadata.reference_frequency,
                use_taper_in_tidal_corrections=metadata.use_taper_in_tidal_corrections,
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
        # astrogwb's own power-of-two rounding policy for the segment duration
        # -- deliberate, and only ever makes the grid finer than asked.
        segment_duration = float(2.0 ** np.ceil(np.log2(1.0 / resolution)))
        # next_smooth_even floors at 2, so this cannot be empty. A sampling
        # frequency too low to reach the band is caught by _resolve_band
        # instead, which can say which band it failed to cover.
        n_samples = next_smooth_even(
            int(np.ceil(segment_duration * resolved_sampling_frequency))
        )

        # NumPy, not JAX: constructing a generator must not touch the XLA
        # backend, so runtime configuration stays free to run after it.
        delta_f = resolved_sampling_frequency / n_samples
        # From bin 1, not bin 0. Ripple evaluates the DC bin to NaN -- the
        # f^(-7/6) amplitude divergence -- and while nan_to_num keeps that out
        # of the forward sum, reverse mode cannot be rescued downstream: the
        # local derivative at f = 0 is NaN, a slice or mask only zeros that
        # bin's cotangent, and 0 * NaN is NaN. Source parameters broadcast
        # across frequency, so their VJP sums every bin and the NaN comes back.
        # Excluding it from the input is the only fix, and it is free: Ripple
        # reads the bottom of the grid only through the spacing f[1] - f[0],
        # which dropping one bin leaves at delta_f. Dropping rather than
        # substituting is what preserves it -- a value patched into the DC bin
        # would change f[1] - f[0] and so move IMRPhenomXAS_NRTidalv3's
        # alignment frequency f[-1] + df.
        grid = np.arange(1, n_samples // 2 + 1, dtype=np.float64) * delta_f
        return segment_duration, int(n_samples), grid

    def _resolve_band(self, grid: np.ndarray) -> slice:
        """Return the contiguous in-band slice of ``grid``, validating alignment.

        Checked here rather than after a first generation: the grid follows
        from the configuration alone, so a misaligned ``minimum_frequency`` is
        a constructor error, not something to discover once a catalog has
        already been drawn.

        ``grid`` starts at ``delta_f``, so a ``minimum_frequency`` of zero is
        reported here as a misalignment. That band was never usable: its first
        bin was Ripple's NaN at DC, zeroed on the way out.
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
        :meth:`~astrogwb.waveform.PolarizationPowerGenerator.check_sources`.
        """
        check_sources(self.metadata.approximant, source_parameters)

    @require_x64
    def generate_batch(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Generate power in ``(frequency, sample)`` layout via Ripple.

        With ``"linear"`` spacing Ripple sees the whole one-sided grid above DC
        and the band is taken from its output, never from its input: several
        supported models -- among them the NRTidal and precessing families --
        read the grid's spacing and top, so restricting the input could change
        the in-band values. The DC bin is the one exception, excluded at
        construction: no model reads it, and the approximants astrogwb uses
        evaluate it to NaN. With ``"log"`` and ``"loglinear"`` spacing the grid
        is the band itself and the output is returned whole.
        """
        events = ripple_parameters(self.metadata.approximant, source_parameters)
        power = self._kernel(jnp.asarray(self._frequencies), events)
        return power[self._band, :]

    def __call__(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        """Return the Ripple frequency axis and power."""
        return self.frequencies, self.generate_batch(source_parameters)
