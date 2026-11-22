"""Simulation of detector-pair background spectra from physical event pools."""

from astrogwb.background.simulation.config import SimulationConfig
from astrogwb.background.simulation.draw import (
    BackgroundDraw,
    project_background,
    simulate_background,
)
from astrogwb.background.simulation.ensemble import (
    read_background_ensemble,
)
from astrogwb.background.simulation.pool import read_event_pool

__all__ = [
    "BackgroundDraw",
    "SimulationConfig",
    "project_background",
    "read_background_ensemble",
    "read_event_pool",
    "simulate_background",
]
