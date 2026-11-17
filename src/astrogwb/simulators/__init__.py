"""Simulation and projection of physical event pools."""

from astrogwb.simulators._keys import realization_key
from astrogwb.simulators.background import (
    BackgroundDraw,
    NetworkNoise,
    combine_pair_spectra,
    network_noise,
    project_background,
    read_background,
    simulate_background,
    write_background,
)
from astrogwb.simulators.config import SimulationConfig
from astrogwb.simulators.ensemble import NetworkSpectra, network_spectra_ensemble
from astrogwb.simulators.pool import read_event_pool
from astrogwb.simulators.sources import simulate_sources

__all__ = [
    "BackgroundDraw",
    "NetworkNoise",
    "NetworkSpectra",
    "SimulationConfig",
    "combine_pair_spectra",
    "network_noise",
    "network_spectra_ensemble",
    "project_background",
    "read_background",
    "read_event_pool",
    "realization_key",
    "simulate_background",
    "simulate_sources",
    "write_background",
]
