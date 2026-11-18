"""Estimators of the background from cross-correlation data, and their variances."""

from astrogwb.statistics.amplitude import (
    background_amplitude_estimate,
    background_amplitude_variance,
    background_amplitude_weights,
)
from astrogwb.statistics.cross_correlation import combine_pair_spectra

__all__ = [
    "background_amplitude_estimate",
    "background_amplitude_variance",
    "background_amplitude_weights",
    "combine_pair_spectra",
]
