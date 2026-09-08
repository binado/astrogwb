"""Common descriptor and interface for polarization-power generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from astrogwb.utils import require_x64

__all__ = ["PolarizationPowerGenerator"]

GRID_SPACING_TOLERANCE_ULP = 64.0


@dataclass(frozen=True, slots=True)
class PolarizationPowerGenerator:
    """Frequency-domain waveform descriptor and power-generation interface.

    Concrete subclasses turn source parameters into frequency-first
    polarization power. The base class is also used as a metadata-only
    descriptor when a persisted catalog is loaded.
    """

    approximant: str
    minimum_frequency: float
    maximum_frequency: float
    reference_frequency: float
    sampling_frequency: float
    df: float
    _frequencies_cache: jax.Array | None = field(
        default=None, init=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum_frequency", float(self.minimum_frequency))
        object.__setattr__(self, "maximum_frequency", float(self.maximum_frequency))
        object.__setattr__(self, "reference_frequency", float(self.reference_frequency))
        object.__setattr__(self, "sampling_frequency", float(self.sampling_frequency))
        object.__setattr__(self, "df", float(self.df))

        settings = (
            self.minimum_frequency,
            self.maximum_frequency,
            self.reference_frequency,
            self.sampling_frequency,
        )
        if not all(np.isfinite(settings)):
            raise ValueError("waveform frequency settings must be finite")
        if self.maximum_frequency < self.minimum_frequency:
            raise ValueError(
                "maximum_frequency must be greater than or equal to minimum_frequency"
            )
        if self.sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be positive")
        if not np.isfinite(self.df) or self.df <= 0.0:
            raise ValueError("df must be a finite positive scalar")

    @property
    @require_x64
    def frequencies(self) -> jax.Array:
        """Return the inclusive uniform grid ``minimum + k*df <= maximum``.

        Computed once per instance and cached.
        """
        if self._frequencies_cache is not None:
            return self._frequencies_cache

        span_in_bins = (self.maximum_frequency - self.minimum_frequency) / self.df
        num_bins = int(np.floor(span_in_bins)) + 1
        next_frequency = self.minimum_frequency + self.df * num_bins
        tolerance = (
            GRID_SPACING_TOLERANCE_ULP
            * np.finfo(np.float64).eps
            * max(
                1.0,
                abs(self.minimum_frequency),
                abs(self.maximum_frequency),
                abs(next_frequency),
            )
        )
        if next_frequency <= self.maximum_frequency + tolerance:
            num_bins += 1

        frequencies = self.minimum_frequency + self.df * jnp.arange(
            num_bins, dtype=jnp.float64
        )
        if bool(
            jnp.isclose(
                frequencies[-1], self.maximum_frequency, rtol=0.0, atol=tolerance
            )
        ):
            frequencies = frequencies.at[-1].set(self.maximum_frequency)
        object.__setattr__(self, "_frequencies_cache", frequencies)
        return frequencies

    def __call__(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Generate power for ``source_parameters``.

        The base implementation exists so it can describe a loaded catalog;
        only concrete generator subclasses are intended to generate power.
        """
        del source_parameters
        raise NotImplementedError(
            "PolarizationPowerGenerator is a metadata-only descriptor; "
            "use a concrete generator subclass"
        )
