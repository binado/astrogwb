"""Array-native catalog container, population simulation, and power generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.catalog.generator import (
    AnalyticInspiralGenerator,
    PolarizationPowerGenerator,
)
from astrogwb.catalog.importance import ImportanceCatalog
from astrogwb.catalog.population import (
    PopulationMetadata,
    simulate_population,
    simulate_population_mixture,
)
from astrogwb.waveform.metadata import FrequencyDomainWaveformMetadata

__all__ = [
    "AnalyticInspiralGenerator",
    "Catalog",
    "FrequencyDomainWaveformMetadata",
    "ImportanceCatalog",
    "PolarizationPowerGenerator",
    "PopulationMetadata",
    "simulate_population",
    "simulate_population_mixture",
]


@dataclass(frozen=True, slots=True)
class Catalog:
    """Source parameters and frequency-first polarization power with metadata."""

    source_parameters: Mapping[str, NDArray[Any]]
    polarization_power: NDArray[Any]
    waveform_metadata: FrequencyDomainWaveformMetadata
    population_metadata: PopulationMetadata

    def __post_init__(self) -> None:
        power = np.asarray(self.polarization_power)
        if power.ndim != 2 or not np.issubdtype(power.dtype, np.number):
            raise ValueError(
                "polarization_power must be a real-valued two-dimensional array"
            )
        if np.issubdtype(power.dtype, np.complexfloating):
            raise ValueError(
                "polarization_power must be real-valued; pass |h+|^2 + |hx|^2, "
                "not raw complex polarizations"
            )
        num_frequencies, num_samples = power.shape
        if num_frequencies != self.waveform_metadata.frequencies.size:
            raise ValueError(
                "polarization_power frequency axis does not match waveform frequencies"
            )
        if num_samples != self.population_metadata.num_samples:
            raise ValueError(
                "population num_samples does not match polarization_power sample axis"
            )

        parameters: dict[str, NDArray[Any]] = {}
        for name, values in self.source_parameters.items():
            if not isinstance(name, str):
                raise TypeError("source parameter names must be strings")
            array = np.asarray(values)
            if array.ndim != 1:
                raise ValueError(
                    f"source parameter {name!r} must be one-dimensional, got "
                    f"shape {array.shape}"
                )
            if array.shape[0] != num_samples:
                raise ValueError(
                    f"source parameter {name!r} has {array.shape[0]} samples; "
                    f"expected {num_samples}"
                )
            parameters[name] = array

        object.__setattr__(self, "polarization_power", power)
        object.__setattr__(self, "source_parameters", parameters)

    @classmethod
    def from_generator(
        cls,
        source_parameters: Mapping[str, ArrayLike],
        *,
        generator: PolarizationPowerGenerator,
        waveform_metadata: FrequencyDomainWaveformMetadata,
        population_metadata: PopulationMetadata,
    ) -> Self:
        """Generate polarization power and return a validated array-native catalog."""
        parameters = {
            name: np.asarray(values) for name, values in source_parameters.items()
        }
        power = np.asarray(generator(source_parameters, waveform_metadata))
        return cls(
            source_parameters=parameters,
            polarization_power=power,
            waveform_metadata=waveform_metadata,
            population_metadata=population_metadata,
        )
