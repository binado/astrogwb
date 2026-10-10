"""The NumPyro models this package exposes, one module per model family.

- :mod:`~astrogwb.inference.models.gaussian_gwb_model` -- the per-frequency
  Gaussian likelihood over any spectrum callable

Every function here returns ``None``; each is handed to NUTS or
:class:`~numpyro.infer.Predictive` through a ``functools.partial``. The Gaussian
likelihoods take a ``SpectralDensityFn`` and know nothing about which
populations exist. The forward models that draw sources live in
:mod:`astrogwb.simulators.spectra`.
"""

from .gaussian_gwb_model import gwb_likelihood_model

__all__ = [
    "gwb_likelihood_model",
]
