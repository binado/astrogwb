"""The NumPyro models this package exposes, one module per model family.

- :mod:`~astrogwb.inference.models.gaussian_gwb_model` -- the priors around any
  pure :data:`~astrogwb.inference.protocol.LogLikelihood`

Every function here returns ``None``; each is handed to NUTS through a
``functools.partial``. The shell knows nothing about which populations exist. The forward models that draw sources live in
:mod:`astrogwb.simulators.spectra`.
"""

from .gaussian_gwb_model import gwb_likelihood_model

__all__ = [
    "gwb_likelihood_model",
]
