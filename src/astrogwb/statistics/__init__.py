"""Estimators of the background from cross-correlation data, and their variances."""

from astrogwb.statistics.amplitude import (
    amplitude,
    amplitude_variance,
    amplitude_variance_budget,
    amplitude_weights,
)
from astrogwb.statistics.cross_correlation import combine_pair_spectra

__all__ = [
    "amplitude",
    "amplitude_variance",
    "amplitude_variance_budget",
    "amplitude_weights",
    "combine_pair_spectra",
]
