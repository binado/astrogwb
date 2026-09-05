"""Waveform power reduction, propagation, and a closed-form inspiral model."""

from astrogwb.waveform.analytical import (
    chirp_mass,
    inclination_factor,
    inspiral_polarization_power,
    termination_frequency,
)
from astrogwb.waveform.distance import apply_gw_distance_to_power
from astrogwb.waveform.metadata import FrequencyDomainWaveformMetadata
from astrogwb.waveform.polarization_power import polarization_power

__all__ = [
    "FrequencyDomainWaveformMetadata",
    "apply_gw_distance_to_power",
    "chirp_mass",
    "inclination_factor",
    "inspiral_polarization_power",
    "polarization_power",
    "termination_frequency",
]
