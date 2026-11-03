"""Grid evaluation of a likelihood and of its prior, compiled once per structure."""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.typing import ArrayLike

from .gaussian import DenseGaussianLikelihood

__all__ = ["grid_log_likelihood", "grid_log_prior"]


def _points(
    names: tuple[str, ...], grids: tuple[jax.Array, ...]
) -> dict[str, jax.Array]:
    """The cartesian product of ``grids``, flattened, one entry per name."""
    mesh = jnp.meshgrid(*grids, indexing="ij")
    return {name: values.ravel() for name, values in zip(names, mesh, strict=True)}


def _grid_log_likelihood(
    names: tuple[str, ...],
    chunk_size: int,
    likelihood: DenseGaussianLikelihood,
    grids: tuple[jax.Array, ...],
    fixed: dict[str, ArrayLike],
) -> jax.Array:
    flat = jax.lax.map(
        lambda point: likelihood.log_likelihood(point | fixed)[0],
        _points(names, grids),
        batch_size=chunk_size,
    )
    return flat.reshape(*(grid.size for grid in grids), *flat.shape[1:])


# Built once, so every call with the same likelihood static fields and array
# shapes shares one compilation. `names` is static and `grids` a tuple: a dict
# argument would be flattened in sorted-key order, losing the axes' order.
_grid_log_likelihood_jit = jax.jit(
    _grid_log_likelihood, static_argnames=("names", "chunk_size")
)


def grid_log_likelihood(
    likelihood: DenseGaussianLikelihood,
    grids: Mapping[str, ArrayLike],
    *,
    fixed: Mapping[str, ArrayLike] | None = None,
    chunk_size: int = 8,
) -> jax.Array:
    """The log-likelihood over the cartesian product of ``grids``.

    Parameters
    ----------
    likelihood
        The likelihood, a pytree argument: its arrays are traced, so another
        catalog or data set of the same shapes reuses the compilation.
    grids
        Values of each swept parameter.
    fixed
        Values of the remaining parameters, pinned at every point; traced.
    chunk_size
        Grid points evaluated together (:func:`jax.lax.map`'s ``batch_size``),
        bounding peak memory at ``chunk_size`` covariance matrices.

    Returns
    -------
    jax.Array
        Shape ``(*grid sizes, *batch)``, grid axes in ``grids`` order and
        ``batch`` the leading axes of ``likelihood.observed``.
    """
    return _grid_log_likelihood_jit(
        names=tuple(grids),
        chunk_size=chunk_size,
        likelihood=likelihood,
        grids=tuple(jnp.asarray(grid) for grid in grids.values()),
        fixed=dict(fixed) if fixed is not None else {},
    )


def grid_log_prior(
    priors: Mapping[str, dist.Distribution],
    grids: Mapping[str, ArrayLike],
    *,
    fixed: Mapping[str, ArrayLike] | None = None,
) -> jax.Array:
    """The log prior over the cartesian product of ``grids``.

    Every prior contributes its log density at the grid or ``fixed`` value, so
    a pinned parameter adds a constant.

    Parameters
    ----------
    priors
        Prior of every parameter.
    grids
        Values of each swept parameter.
    fixed
        Values of the remaining parameters.

    Returns
    -------
    jax.Array
        Shape ``(*grid sizes)``, axes in ``grids`` order.

    Raises
    ------
    KeyError
        If a prior's name is in neither ``grids`` nor ``fixed``.
    """
    values: dict[str, ArrayLike] = {
        **_points(tuple(grids), tuple(jnp.asarray(grid) for grid in grids.values())),
        **(fixed if fixed is not None else {}),
    }
    total = sum(
        (prior.log_prob(values[name]) for name, prior in priors.items()),
        start=jnp.zeros(()),
    )
    sizes = tuple(int(np.size(grid)) for grid in grids.values())
    return jnp.broadcast_to(total, (math.prod(sizes),)).reshape(sizes)
