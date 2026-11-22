"""Simulation of detector-pair background spectra from physical event pools."""

from astrogwb.background.simulation.config import SimulationConfig
from astrogwb.background.simulation.draw import (
    BackgroundDraw,
    project_background,
    simulate_background,
)
from astrogwb.background.simulation.pool import read_population_file

__all__ = [
    "BackgroundDraw",
    "SimulationConfig",
    "project_background",
    "read_population_file",
    "simulate_background",
]
