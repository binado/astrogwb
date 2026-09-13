"""Ripple-backed frequency-domain polarization-power generator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from gwmock_signal.waveform import RippleBackend
from numpy.typing import ArrayLike

from astrogwb.waveform.generator.base import PolarizationPowerGenerator
from astrogwb.waveform.polarization_power import polarization_power

__all__ = ["RippleGenerator"]


class _Polarizations(Protocol):
    """Structural view of the polarizations ``RippleBackend`` returns.

    gwmock's concrete type, ``FrequencyDomainPolarizations``, is not exported
    from its public surface, and this helper reads only these three fields --
    so the contract is typed structurally rather than through a private
    import.
    """

    @property
    def frequencies(self) -> jax.Array: ...

    @property
    def plus(self) -> jax.Array: ...

    @property
    def cross(self) -> jax.Array: ...


class RippleGenerator(PolarizationPowerGenerator):
    """Generate polarization power with one fixed Ripple frequency grid."""

    __slots__ = ("_backend", "_frequencies_cache")
    _backend: RippleBackend
    _frequencies_cache: jax.Array | None

    def __init__(
        self,
        *,
        approximant: str,
        sampling_frequency: float,
        minimum_frequency: float,
        maximum_frequency: float,
        reference_frequency: float,
        frequency_resolution: float,
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
        object.__setattr__(self, "_frequencies_cache", None)
        object.__setattr__(
            self,
            "_backend",
            RippleBackend(
                f_ref=self.reference_frequency,
                segment_duration=segment_duration,
            ),
        )

    @property
    def frequencies(self) -> jax.Array:
        """Return the frequency grid Ripple generated on, masked to the descriptor band.

        Ripple sizes its FFT segment from ``segment_duration`` with its own
        rounding rule (5-smooth, not power-of-two -- see
        ``gwmock_signal.RippleBackend._segment_samples``), so this grid is
        only known by asking Ripple for it, not by recomputing it here. It is
        cached from the first call to :meth:`generate_batch`.
        """
        if self._frequencies_cache is None:
            raise ValueError(
                "RippleGenerator has not generated yet; frequencies are only "
                "known after the first call"
            )
        return self._frequencies_cache

    def _power_from_polarizations(self, polarizations: _Polarizations) -> jax.Array:
        frequencies = jnp.asarray(polarizations.frequencies)
        plus = jnp.asarray(polarizations.plus)
        cross = jnp.asarray(polarizations.cross)
        if plus.ndim == 1:
            plus = plus[jnp.newaxis, :]
            cross = cross[jnp.newaxis, :]
        mask = (frequencies >= self.minimum_frequency) & (
            frequencies <= self.maximum_frequency
        )
        masked_frequencies = frequencies[mask]
        if self._frequencies_cache is None:
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
            # segment_duration is pinned on `_backend` (see __init__), so
            # Ripple's grid is a pure function of that fixed config, not
            # of the source parameters.
            object.__setattr__(self, "_frequencies_cache", masked_frequencies)
        elif not bool(jnp.array_equal(self._frequencies_cache, masked_frequencies)):
            raise ValueError("Ripple calls produced different frequency grids")
        return polarization_power(plus[:, mask], cross[:, mask])

    def generate_batch(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Generate power in ``(frequency, sample)`` layout via Ripple."""
        parameters = {
            name: jnp.asarray(values) for name, values in source_parameters.items()
        }
        polarizations = self._backend.generate_fd_polarizations_batch(
            self.approximant,
            sampling_frequency=self.sampling_frequency,
            minimum_frequency=self.minimum_frequency,
            parameters=parameters,
        )
        return self._power_from_polarizations(polarizations)

    def generate(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Generate power for a single source, shape ``(F,)``.

        Routes to the backend's per-event entry point instead of wrapping a
        length-one catalog through :meth:`generate_batch`: the pinned
        ``segment_duration`` keeps both paths on the same frequency grid, and
        the per-event path avoids compiling a one-event batch kernel.
        """
        parameters: dict[str, float] = {}
        for name, values in source_parameters.items():
            scalar = jnp.asarray(values)
            if scalar.size != 1:
                raise ValueError(
                    f"generate expects a single source; {name!r} carries "
                    f"{scalar.size} values"
                )
            parameters[name] = scalar.item()
        polarizations = self._backend.generate_fd_polarizations(
            self.approximant,
            sampling_frequency=self.sampling_frequency,
            minimum_frequency=self.minimum_frequency,
            **parameters,
        )
        return self._power_from_polarizations(polarizations)[:, 0]

    def __call__(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        """Return the Ripple frequency axis and power.

        Ripple sizes its FFT segment from ``segment_duration`` with its own
        rounding rule (5-smooth, not power-of-two -- see
        ``gwmock_signal.RippleBackend._segment_samples``), so the frequency
        axis is taken from the generated polarizations rather than recomputed
        here. ``segment_duration`` is pinned on ``_backend``, so repeated calls
        land on the same grid; a mismatch raises.

        ``minimum_frequency`` alignment is checked here, against the grid
        Ripple actually built, rather than at construction time: no
        backend-independent check on ``minimum_frequency`` alone can predict
        Ripple's 5-smooth ``n_samples`` without replicating the rule this
        generator exists to delete.
        """
        power = self.generate_batch(source_parameters)
        return self.frequencies, power
