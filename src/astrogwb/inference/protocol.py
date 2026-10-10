"""Callable boundary between spectral predictions and inference models."""

from collections.abc import Callable, Mapping

import jax
from jax.typing import ArrayLike

#: A pure likelihood: ``params -> (log L, extras)``, with ``log L`` of shape ``()``
#: and ``extras`` the diagnostics, whose keys and shapes stay stable under tracing.
#: NUTS, blackjax, optimizers and grids all consume this one signature.
type LogLikelihood = Callable[
    [Mapping[str, ArrayLike]], tuple[jax.Array, Mapping[str, jax.Array]]
]

#: A spectrum on the observed grid: ``params -> (F,)``.
type SpectrumFn = Callable[[Mapping[str, ArrayLike]], jax.Array]

__all__ = ["LogLikelihood", "SpectrumFn"]
