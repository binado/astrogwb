"""Callable boundary between spectral predictions and inference models."""

from collections.abc import Mapping
from typing import Protocol

import jax
from jax.typing import ArrayLike


class SpectralDensityFn(Protocol):
    """Predict a spectrum of shape ``(F,)`` and optional diagnostics.

    An analytic calculator may return ``{}`` for diagnostics. Diagnostic keys
    and shapes must stay stable during JAX tracing; keys must not collide with
    prior sites or likelihood-owned sites. Values are recorded unchanged as
    NumPyro deterministics by the inference model.

    A function bound to data should be a pytree whose array state is its
    leaves (:class:`jax.tree_util.Partial`), as the importance builders return:
    passed as an argument to a jitted evaluator, the data is then a traced input
    rather than a compiled-in constant. A plain closure is still valid; what it
    captures is compiled in.
    """

    def __call__(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, Mapping[str, ArrayLike]]: ...


class SpectralVarianceFn(Protocol):
    """Predict the per-bin shot-noise variance of a spectrum, shape ``(F,)``.

    The variance of a catalog realization about the spectrum a matching
    :class:`SpectralDensityFn` predicts at the same ``params`` and on the same
    grid. Bound to data, it is a pytree for the same reason.
    """

    def __call__(self, params: Mapping[str, ArrayLike]) -> jax.Array: ...


__all__ = ["SpectralDensityFn", "SpectralVarianceFn"]
