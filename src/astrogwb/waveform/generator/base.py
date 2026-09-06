"""Common descriptor and interface for polarization-power generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["PolarizationPowerGenerator"]

GRID_SPACING_TOLERANCE_ULP = 64.0


@dataclass(frozen=True, slots=True)
class PolarizationPowerGenerator:
    """Frequency-domain waveform descriptor and power-generation interface.

    Concrete subclasses turn source parameters into frequency-first
    polarization power. The base class is also used as a metadata-only
    descriptor when a persisted catalog is loaded.
    """

    frequencies: NDArray[np.float64]
    approximant: str
    minimum_frequency: float
    maximum_frequency: float
    reference_frequency: float
    sampling_frequency: float
    df: float

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies, dtype=np.float64)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "minimum_frequency", float(self.minimum_frequency))
        object.__setattr__(self, "maximum_frequency", float(self.maximum_frequency))
        object.__setattr__(self, "reference_frequency", float(self.reference_frequency))
        object.__setattr__(self, "sampling_frequency", float(self.sampling_frequency))
        object.__setattr__(self, "df", float(self.df))
        _validate_frequency_grid(frequencies, self.df)

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

    @classmethod
    def from_bounds(
        cls,
        *,
        approximant: str,
        minimum_frequency: float,
        maximum_frequency: float,
        reference_frequency: float,
        sampling_frequency: float,
        df: float,
        **kwargs: Any,
    ) -> Self:
        """Construct the inclusive uniform grid ``minimum + k*df <= maximum``."""
        minimum = float(minimum_frequency)
        maximum = float(maximum_frequency)
        spacing = float(df)
        if not all(np.isfinite((minimum, maximum, spacing))):
            raise ValueError("frequency-grid settings must be finite")
        if spacing <= 0.0:
            raise ValueError("df must be positive")
        if maximum < minimum:
            raise ValueError(
                "maximum_frequency must be greater than or equal to minimum_frequency"
            )

        span_in_bins = (maximum - minimum) / spacing
        if not np.isfinite(span_in_bins):
            raise ValueError("frequency grid is too large to construct")
        num_bins = int(np.floor(span_in_bins)) + 1

        # Division can round an exact final bin below the next integer. Build
        # from integer indices and explicitly test the next computed value.
        next_frequency = minimum + spacing * num_bins
        if next_frequency <= maximum:
            num_bins += 1
        frequencies = np.asarray(
            minimum + spacing * np.arange(num_bins), dtype=np.float64
        )
        return cls(
            frequencies=frequencies,
            approximant=approximant,
            minimum_frequency=minimum,
            maximum_frequency=maximum,
            reference_frequency=reference_frequency,
            sampling_frequency=sampling_frequency,
            df=spacing,
            **kwargs,
        )

    def __call__(self, source_parameters: Mapping[str, ArrayLike]) -> NDArray[Any]:
        """Generate power for ``source_parameters``.

        The base implementation exists so it can describe a loaded catalog;
        only concrete generator subclasses are intended to generate power.
        """
        del source_parameters
        raise NotImplementedError(
            "PolarizationPowerGenerator is a metadata-only descriptor; "
            "use a concrete generator subclass"
        )


def _validate_frequency_grid(frequencies: NDArray[np.float64], df: float) -> None:
    if frequencies.ndim != 1:
        raise ValueError("frequencies must be one-dimensional")
    if frequencies.size == 0:
        raise ValueError("frequencies must contain at least one bin")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies must be finite")
    if not np.isfinite(df) or df <= 0.0:
        raise ValueError("df must be a finite positive scalar")
    if frequencies.size == 1:
        return

    differences = np.diff(frequencies)
    if not np.all(differences > 0.0):
        raise ValueError("frequencies must be strictly increasing")
    tolerance = (
        GRID_SPACING_TOLERANCE_ULP
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(frequencies))))
    )
    if not np.all(np.abs(differences - df) <= tolerance):
        raise ValueError(f"frequencies must be uniformly spaced by df={df} Hz")
