"""Simulation of detector-pair background spectra from physical event pools."""

from astrogwb.background.simulation._keys import realization_key
from astrogwb.background.simulation.config import SimulationConfig
from astrogwb.background.simulation.draw import (
    BackgroundDraw,
    NetworkNoise,
    network_noise,
    project_background,
    simulate_background,
)
from astrogwb.background.simulation.ensemble import (
    NetworkSpectra,
    network_spectra_ensemble,
)
from astrogwb.background.simulation.pool import read_event_pool

__all__ = [
    "BackgroundDraw",
    "NetworkNoise",
    "NetworkSpectra",
    "SimulationConfig",
    "network_noise",
    "network_spectra_ensemble",
    "project_background",
    "read_event_pool",
    "realization_key",
    "simulate_background",
]
