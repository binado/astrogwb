"""Callable boundary between spectral predictions and sampling models."""

from collections.abc import Mapping
from typing import Protocol

import jax
from jax.typing import ArrayLike


class SpectralDensityFn(Protocol):
    """Predict a spectrum of shape ``(F,)`` and optional diagnostics.

    An analytic calculator may return ``{}`` for diagnostics. Diagnostic keys
    and shapes must stay stable during JAX tracing; keys must not collide with
    prior sites or likelihood-owned sites. Values are recorded unchanged as
    NumPyro deterministics by the sampling model.
    """

    def __call__(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, Mapping[str, ArrayLike]]: ...
