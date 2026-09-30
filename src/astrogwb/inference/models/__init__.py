"""The NumPyro models this package exposes, one module per model family.

- :mod:`~astrogwb.inference.models.poisson_counts_forward_model` -- the exact
  Poisson-catalog forward model of the spectrum
- :mod:`~astrogwb.inference.models.fixed_counts_forward_model` -- the fixed-count
  forward model with a population-rate normalization
- :mod:`~astrogwb.inference.models.gaussian_gwb_model` -- the per-frequency
  Gaussian likelihood over any spectrum callable
- :mod:`~astrogwb.inference.models.gaussian_gwb_marginalized_amplitude` -- the
  same likelihood with one multiplicative parameter integrated out, plus the
  generative reconstruction that samples it back

Every function here returns ``None``; each is handed to NUTS or
:class:`~numpyro.infer.Predictive` through a ``functools.partial``. The Gaussian
likelihoods take a ``SpectralDensityFn`` and know nothing about which
populations exist; the forward models draw sources.
"""

from ._forward import validate_source_model
from .fixed_counts_forward_model import fixed_counts_forward_model
from .gaussian_gwb_marginalized_amplitude import (
    amplitude_reconstruction_model,
    gwb_amplitude_marginalized_model,
)
from .gaussian_gwb_model import gwb_spectral_density_model
from .poisson_counts_forward_model import poisson_counts_forward_model

__all__ = [
    "amplitude_reconstruction_model",
    "fixed_counts_forward_model",
    "gwb_amplitude_marginalized_model",
    "gwb_spectral_density_model",
    "poisson_counts_forward_model",
    "validate_source_model",
]
