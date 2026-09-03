"""Array-native catalogs, population simulation, and power generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "AnalyticInspiralGenerator",
    "Catalog",
    "FrequencyDomainWaveformMetadata",
    "PolarizationPowerGenerator",
    "PopulationMetadata",
    "generate_catalog",
    "simulate_population",
]

ScalarProvenance = str | int | float
GRID_SPACING_TOLERANCE_ULP = 64.0


@dataclass(frozen=True, slots=True)
class FrequencyDomainWaveformMetadata:
    """The exact frequency grid and settings used to generate waveform power."""

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
    ) -> FrequencyDomainWaveformMetadata:
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
        )


@dataclass(frozen=True, slots=True)
class PopulationMetadata:
    """Population identity, simulation settings, and scalar provenance."""

    name: str
    seed: int
    num_samples: int
    source_type: str | None = None
    provenance: Mapping[str, ScalarProvenance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        if isinstance(self.num_samples, bool) or not isinstance(self.num_samples, int):
            raise TypeError("num_samples must be an int")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        for key, value in self.provenance.items():
            if not isinstance(key, str):
                raise TypeError("provenance names must be strings")
            if isinstance(value, bool) or not isinstance(value, str | int | float):
                raise TypeError(
                    f"provenance[{key!r}] must be a str, non-boolean int, or "
                    f"float scalar, got {type(value).__name__}"
                )


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


class PolarizationPowerGenerator(Protocol):
    """Adapter from prepared sources and an exact grid to polarization power."""

    def __call__(
        self,
        source_parameters: Mapping[str, ArrayLike],
        waveform_metadata: FrequencyDomainWaveformMetadata,
    ) -> ArrayLike: ...


@dataclass(frozen=True, slots=True)
class AnalyticInspiralGenerator:
    """Generate closed-form inspiral polarization power on the supplied grid."""

    alpha: float

    def __call__(
        self,
        source_parameters: Mapping[str, ArrayLike],
        waveform_metadata: FrequencyDomainWaveformMetadata,
    ) -> NDArray[Any]:
        from astrogwb.waveform.analytical import inspiral_polarization_power

        prepared_parameters = {
            name: np.asarray(values) for name, values in source_parameters.items()
        }
        return np.asarray(
            inspiral_polarization_power(
                waveform_metadata.frequencies,
                prepared_parameters,
                alpha=self.alpha,
            )
        ).T


def generate_catalog(
    source_parameters: Mapping[str, ArrayLike],
    *,
    generator: PolarizationPowerGenerator,
    waveform_metadata: FrequencyDomainWaveformMetadata,
    population_metadata: PopulationMetadata,
) -> Catalog:
    """Generate polarization power and return a validated array-native catalog."""
    parameters = {
        name: np.asarray(values) for name, values in source_parameters.items()
    }
    power = np.asarray(generator(source_parameters, waveform_metadata))
    return Catalog(
        source_parameters=parameters,
        polarization_power=power,
        waveform_metadata=waveform_metadata,
        population_metadata=population_metadata,
    )


def simulate_population(
    config: Mapping[str, Any] | str | Path,
    *,
    metadata: PopulationMetadata,
) -> dict[str, NDArray[Any]]:
    """Draw source arrays from a mapping or graph configuration file."""
    try:
        from gwmock_pop import GraphSimulator
    except ImportError as error:
        raise ImportError(
            "Population simulation requires the optional 'gwmock-pop' dependency; "
            "install it with `pip install astrogwb[simulation]`."
        ) from error

    if isinstance(config, Mapping):
        simulator = GraphSimulator(
            cast("dict[str, Any]", config),
            source_type=metadata.source_type,
            seed=metadata.seed,
        )
    else:
        simulator = GraphSimulator.from_config_file(
            config, source_type=metadata.source_type, seed=metadata.seed
        )
    return {
        name: np.asarray(values)
        for name, values in simulator.simulate(metadata.num_samples).items()
    }


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
