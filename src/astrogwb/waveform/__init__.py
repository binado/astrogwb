"""Waveform generation, power reduction, and inspiral models."""

from astrogwb.waveform.analytical import (
    chirp_mass,
    inclination_factor,
    inspiral_intensity,
    termination_frequency,
)
from astrogwb.waveform.base import WaveformGenerator
from astrogwb.waveform.factory import waveform_generator
from astrogwb.waveform.intensity import (
    intensity,
    quadrupolar_polarizations,
    quadrupolar_stokes_factors,
)
from astrogwb.waveform.metadata import WaveformConfig

__all__ = [
    "WaveformConfig",
    "WaveformGenerator",
    "chirp_mass",
    "inclination_factor",
    "inspiral_intensity",
    "intensity",
    "quadrupolar_polarizations",
    "quadrupolar_stokes_factors",
    "termination_frequency",
    "waveform_generator",
]
