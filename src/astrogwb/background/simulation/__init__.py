"""Simulation of detector-pair background spectra from physical event pools."""

from astrogwb.background.simulation.config import SimulationConfig
from astrogwb.background.simulation.draw import (
    project_background,
    simulate_background,
)
from astrogwb.background.simulation.result import BackgroundDraw

__all__ = [
    "BackgroundDraw",
    "SimulationConfig",
    "project_background",
    "simulate_background",
]
