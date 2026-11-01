"""NumPyro models wrapping a pure :data:`~astrogwb.inference.protocol.LogLikelihood`.

Each returns ``None`` and is handed to NUTS through ``functools.partial``.
"""

from .gaussian_gwb_model import gwb_likelihood_model

__all__ = [
    "gwb_likelihood_model",
]
