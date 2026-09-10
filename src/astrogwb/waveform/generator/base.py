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
    frequency-first polarization power. :meth:`generate` is one source;
    :meth:`generate_batch` is a 1-D catalog. ``__call__`` returns
    ``(frequencies, polarization_power)`` so catalog construction records the
    axis the backend actually produced. The base class is also used as a
    metadata-only descriptor when a persisted catalog is loaded.

    ``frequency_resolution`` records what was *requested*; it is not
    necessarily the realized bin width. The generating backend chooses the
    actual grid, so the realized spacing belongs to the catalog it produces
    (see ``Catalog.df``), not to this descriptor.
    """

    approximant: str
    minimum_frequency: float
    maximum_frequency: float
    reference_frequency: float
    sampling_frequency: float
    frequency_resolution: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum_frequency", float(self.minimum_frequency))
        object.__setattr__(self, "maximum_frequency", float(self.maximum_frequency))
        object.__setattr__(self, "reference_frequency", float(self.reference_frequency))
        object.__setattr__(self, "sampling_frequency", float(self.sampling_frequency))
        object.__setattr__(
            self, "frequency_resolution", float(self.frequency_resolution)
        )

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
        if (
            not np.isfinite(self.frequency_resolution)
            or self.frequency_resolution <= 0.0
        ):
            raise ValueError("frequency_resolution must be a finite positive scalar")

    def generate(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Generate power for a single source, shape ``(F,)``."""
        del source_parameters
        raise NotImplementedError(
            "PolarizationPowerGenerator is a metadata-only descriptor; "
            "use a concrete generator subclass"
        )

    def generate_batch(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Generate power for a 1-D catalog, shape ``(F, N)``."""
        del source_parameters
        raise NotImplementedError(
            "PolarizationPowerGenerator is a metadata-only descriptor; "
            "use a concrete generator subclass"
        )

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
