"""Simulation of detector-pair background spectra from physical event pools."""

from astrogwb.background.simulation.config import SimulationConfig
from astrogwb.background.simulation.draw import (
    BackgroundDraw,
    project_background,
    simulate_background,
)

__all__ = [
    "BackgroundDraw",
    "SimulationConfig",
    "project_background",
    "simulate_background",
]
