"""Waveform generation, power reduction, propagation, and inspiral models."""

from astrogwb.waveform.distance import apply_gw_distance_to_power
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
from astrogwb.waveform.polarization_power import polarization_power

__all__ = [
    "AnalyticInspiralGenerator",
    "PolarizationPowerGenerator",
    "RippleGenerator",
    "apply_gw_distance_to_power",
    "chirp_mass",
    "inclination_factor",
    "inspiral_polarization_power",
    "polarization_power",
    "termination_frequency",
]
