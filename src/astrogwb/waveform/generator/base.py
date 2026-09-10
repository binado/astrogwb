"""Common descriptor and interface for polarization-power generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import numpy as np
from numpy.typing import ArrayLike

__all__ = ["PolarizationPowerGenerator"]


@dataclass(frozen=True, slots=True)
class PolarizationPowerGenerator:
    """Frequency-domain waveform descriptor and power-generation interface.

    Concrete subclasses turn source parameters into a frequency axis and
    frequency-first polarization power. The base class is also used as a
    metadata-only descriptor when a persisted catalog is loaded.
    """

    approximant: str
    minimum_frequency: float
    maximum_frequency: float
    reference_frequency: float
    sampling_frequency: float
    df: float

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

    def __call__(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        """Return ``(frequencies, polarization_power)`` for ``source_parameters``.

        Polarization power is frequency-first, shape ``(F, N)``. The base
        implementation exists so it can describe a loaded catalog; only
        concrete generator subclasses are intended to generate.
        """
        del source_parameters
        raise NotImplementedError(
            "PolarizationPowerGenerator is a metadata-only descriptor; "
            "use a concrete generator subclass"
        )
