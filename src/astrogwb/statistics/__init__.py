"""Estimators of the background from cross-correlation data, and their variances."""

from astrogwb.statistics.amplitude import (
    background_amplitude_estimate,
    background_amplitude_variance,
    background_amplitude_weights,
)
from astrogwb.statistics.cross_correlation import cross_correlation_estimator

__all__ = [
    "background_amplitude_estimate",
    "background_amplitude_variance",
    "background_amplitude_weights",
    "cross_correlation_estimator",
]
