"""Array-native catalogs, population simulation, and power generation."""

from astrogwb.catalog.catalog import (
    AnalyticInspiralGenerator,
    Catalog,
    PolarizationPowerGenerator,
    generate_catalog,
)
from astrogwb.catalog.population import PopulationMetadata, simulate_population
from astrogwb.waveform.metadata import FrequencyDomainWaveformMetadata

__all__ = [
    "AnalyticInspiralGenerator",
    "Catalog",
    "FrequencyDomainWaveformMetadata",
    "PolarizationPowerGenerator",
    "PopulationMetadata",
    "generate_catalog",
    "simulate_population",
]
