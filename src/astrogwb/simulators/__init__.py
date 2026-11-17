"""Simulation and projection of physical event pools."""

from astrogwb.simulators._keys import realization_key
from astrogwb.simulators.background import (
    BackgroundArtifact,
    BackgroundDraw,
    NetworkNoise,
    combine_pair_spectra,
    project_background,
    simulate_background,
)
from astrogwb.simulators.config import SimulationConfig
from astrogwb.simulators.ensemble import NetworkSpectra, network_spectra_ensemble
from astrogwb.simulators.population import RealizationMetadata
from astrogwb.simulators.sources import simulate_sources

__all__ = [
    "BackgroundArtifact",
    "BackgroundDraw",
    "NetworkNoise",
    "NetworkSpectra",
    "RealizationMetadata",
    "SimulationConfig",
    "combine_pair_spectra",
    "network_spectra_ensemble",
    "project_background",
    "realization_key",
    "simulate_background",
    "simulate_sources",
]
