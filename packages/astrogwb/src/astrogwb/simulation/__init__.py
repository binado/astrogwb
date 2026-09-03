"""Population simulation and waveform-catalog generation interfaces.

Population graphs and any cosmology-dependent preparation of their draws stay
with callers.  This module only draws a configured graph and adapts prepared
source parameters into the shared :class:`~astrogwb.waveform.WaveformCatalog`
format.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.waveform import (
    WaveformCatalog,
    inspiral_polarization_power,
    make_catalog,
)

__all__ = [
    "AnalyticInspiralGenerator",
    "GeneratedPolarizationPower",
    "PolarizationPowerGenerator",
    "generate_catalog",
    "simulate_population",
]


@dataclass(frozen=True, slots=True)
class GeneratedPolarizationPower:
    """Frequency-first polarization power and its waveform metadata."""

    frequencies: NDArray[np.float64]
    polarization_power: NDArray[np.float64]
    approximant: str
    minimum_frequency: float
    maximum_frequency: float
    reference_frequency: float
    sampling_frequency: float
    df: float


class PolarizationPowerGenerator(Protocol):
    """Adapter from prepared source parameters to polarization power."""

    def __call__(
        self,
        source_parameters: Mapping[str, ArrayLike],
    ) -> GeneratedPolarizationPower: ...


@dataclass(frozen=True, slots=True)
class AnalyticInspiralGenerator:
    """Generate closed-form inspiral polarization power on a uniform grid."""

    minimum_frequency: float
    maximum_frequency: float
    df: float
    alpha: float

    def __call__(
        self,
        source_parameters: Mapping[str, ArrayLike],
    ) -> GeneratedPolarizationPower:
        minimum_frequency = float(self.minimum_frequency)
        maximum_frequency = float(self.maximum_frequency)
        df = float(self.df)
        if not all(np.isfinite((minimum_frequency, maximum_frequency, df))):
            raise ValueError("frequency-grid settings must be finite")
        if df <= 0.0:
            raise ValueError("df must be positive")
        if maximum_frequency <= minimum_frequency:
            raise ValueError("maximum_frequency must be greater than minimum_frequency")

        span_in_bins = (maximum_frequency - minimum_frequency) / df
        if not np.isfinite(span_in_bins):
            raise ValueError("frequency grid is too large to construct")
        num_bins = int(np.floor(span_in_bins)) + 1

        # Floating-point division can put an exact final bin just below the
        # next integer. Construct from integer indices, then include that bin
        # precisely when its computed frequency does not exceed the maximum.
        next_frequency = minimum_frequency + df * num_bins
        if next_frequency <= maximum_frequency:
            num_bins += 1
        frequencies = np.asarray(
            minimum_frequency + df * np.arange(num_bins), dtype=np.float64
        )
        prepared_parameters = {
            name: np.asarray(values) for name, values in source_parameters.items()
        }

        polarization_power = np.asarray(
            inspiral_polarization_power(
                frequencies,
                prepared_parameters,
                alpha=self.alpha,
            )
        ).T
        return GeneratedPolarizationPower(
            frequencies=frequencies,
            polarization_power=polarization_power,
            approximant="AnalyticInspiral",
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            reference_frequency=minimum_frequency,
            sampling_frequency=2.0 * maximum_frequency,
            df=df,
        )


def generate_catalog(
    source_parameters: Mapping[str, ArrayLike],
    *,
    generator: PolarizationPowerGenerator,
    extra_attrs: Mapping[str, str | float | int] | None = None,
) -> WaveformCatalog:
    """Generate polarization power and build a validated waveform catalog."""
    generated = generator(source_parameters)
    parameters = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in source_parameters.items()
    }
    return make_catalog(
        frequencies=generated.frequencies,
        polarization_power=generated.polarization_power,
        source_parameters=parameters,
        approximant=generated.approximant,
        minimum_frequency=generated.minimum_frequency,
        maximum_frequency=generated.maximum_frequency,
        reference_frequency=generated.reference_frequency,
        sampling_frequency=generated.sampling_frequency,
        df=generated.df,
        extra_attrs=extra_attrs,
    )


def simulate_population(
    config: Mapping[str, Any] | str | Path,
    *,
    num_samples: int,
    seed: int,
    source_type: str | None = None,
) -> dict[str, ArrayLike]:
    """Draw a population from a mapping or a graph configuration file.

    ``gwmock-pop`` is an optional dependency so analytic generation remains
    usable in a plain ``astrogwb`` installation.
    """
    try:
        from gwmock_pop import GraphSimulator
    except ImportError as error:
        raise ImportError(
            "Population simulation requires the optional 'gwmock-pop' dependency; "
            "install it with `pip install astrogwb[simulation]`."
        ) from error

    if isinstance(config, Mapping):
        simulator = GraphSimulator(
            cast("dict[str, Any]", config), source_type=source_type, seed=seed
        )
    else:
        simulator = GraphSimulator.from_config_file(
            config, source_type=source_type, seed=seed
        )
    return dict(simulator.simulate(num_samples))
