"""Forward-model spectral-density draws, drawn from a :class:`SpectraMetadata`."""

from astrogwb.simulators.spectra.catalog import SpectralDensityCatalog
from astrogwb.simulators.spectra.draws import (
    SpectralDensityDraws,
    draw_spectral_density,
    padded_event_capacity,
)
from astrogwb.simulators.spectra.fixed_counts import fixed_counts_forward_model
from astrogwb.simulators.spectra.forward import validate_source_model
from astrogwb.simulators.spectra.metadata import Hyperparameter, SpectraMetadata
from astrogwb.simulators.spectra.poisson_counts import poisson_counts_forward_model
from astrogwb.simulators.spectra.simulator import SpectrumGenerator

__all__ = [
    "Hyperparameter",
    "SpectraMetadata",
    "SpectralDensityCatalog",
    "SpectralDensityDraws",
    "SpectrumGenerator",
    "draw_spectral_density",
    "fixed_counts_forward_model",
    "padded_event_capacity",
    "poisson_counts_forward_model",
    "validate_source_model",
]
