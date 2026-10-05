"""Waveform generation, power reduction, and inspiral models."""

from astrogwb.waveform.generator import (
    AnalyticInspiralGenerator,
    PolarizationPowerGenerator,
    RippleGenerator,
)
from astrogwb.waveform.generator.analytical import (
    chirp_mass,
    inclination_factor,
    inspiral_polarization_power,
    termination_frequency,
)
from astrogwb.waveform.metadata import WaveformMetadata
from astrogwb.waveform.polarization_power import polarization_power

__all__ = [
    "AnalyticInspiralGenerator",
    "PolarizationPowerGenerator",
    "RippleGenerator",
    "WaveformMetadata",
    "chirp_mass",
    "inclination_factor",
    "inspiral_polarization_power",
    "polarization_power",
    "termination_frequency",
]
