"""Reusable, jit-cached grid evaluators: a NumPyro model's log density, and a likelihood's posterior."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.infer.util import log_density

from .likelihood import Network


class LogDensityFn:
    """Grid-evaluate a NumPyro model's constrained log density.

    The model given to :meth:`__init__` can take positional or keyword
    arguments. Callers supply positional inputs via ``model_args`` and
    keyword inputs via ``model_kwargs`` (the convention every model in
    :mod:`astrogwb.inference.models` follows).

    Parameters
    ----------
    model:
        A NumPyro model called as ``model(*model_args, **model_kwargs)``. Every
        model in :mod:`astrogwb.inference.models` returns ``None``; a
        ``functools.partial`` or a ``numpyro.handlers.Messenger`` wrapper
        (e.g. from ``handlers.block``/``handlers.condition``) around one
        satisfies this too.
    chunk_size:
        Forwarded to :func:`jax.lax.map` as ``batch_size``, bounding peak
        memory to ``chunk_size`` grid points' worth of intermediates rather
        than materializing the whole grid via a single ``vmap``. ``None``
        (the default) evaluates one point at a time.
    """

    def __init__(
        self, model: Callable[..., None], *, chunk_size: int | None = None
    ) -> None:
        def evaluate(
            grids: dict[str, jax.Array],
            fixed_params: dict[str, ArrayLike],
            model_args: tuple[Any, ...],
            model_kwargs: dict[str, Any],
        ) -> jax.Array:
            mesh = jnp.meshgrid(*grids.values(), indexing="ij")
            points = {
                name: values.ravel() for name, values in zip(grids, mesh, strict=True)
            }

            def score(point: dict[str, jax.Array]) -> jax.Array:
                log_joint, _ = log_density(
                    model, model_args, model_kwargs, point | fixed_params
                )
                return log_joint

            flat = jax.lax.map(score, points, batch_size=chunk_size)
            return flat.reshape(tuple(grid.size for grid in grids.values()))

        # Built once: this is what makes a sweep over model_args/model_kwargs
        # pay for a single compilation instead of one per iteration.
        self._evaluator = jax.jit(evaluate)

    def __call__(
        self,
        grids: Mapping[str, jax.Array],
        *,
        fixed: Mapping[str, ArrayLike] | None = None,
        model_args: Any = (),
        **model_kwargs: Any,
    ) -> jax.Array:
        """Constrained log density over the cartesian product of 1-2 grids.

        Returns an array of shape ``tuple(len(g) for g in grids.values())``,
        axes in ``grids`` insertion order. ``fixed`` pins additional sample
        sites at every grid point and is threaded through as a *traced*
        argument, so sweeping its value never triggers a recompile -- only a
        change to its set of keys does, since that changes the argument
        pytree's structure.

        ``model_args`` and ``model_kwargs`` are forwarded to ``model``.
        """
        fixed_params = dict(fixed) if fixed is not None else {}
        return self._evaluator(grids, fixed_params, tuple(model_args), model_kwargs)


#: A callable's static half: its pytree structure, and every non-array leaf in
#: order with ``None`` holding each array's place.
type _Structure = tuple[Any, tuple[Any, ...]]


def _is_array(leaf: Any) -> bool:
    return isinstance(leaf, (jax.Array, np.ndarray))


def _partition(fn: Any) -> tuple[list[Any], _Structure]:
    """Split ``fn`` into its array leaves and a hashable static rest.

    A pytree callable (:class:`jax.tree_util.Partial`) yields its arrays; a
    plain function is one non-array leaf, so it is static whole, as before.
    Flattening drops ``None``, so ``None`` cannot be a real static leaf.
    """
    leaves, treedef = jax.tree_util.tree_flatten(fn)
    arrays = [leaf for leaf in leaves if _is_array(leaf)]
    statics = tuple(None if _is_array(leaf) else leaf for leaf in leaves)
    return arrays, (treedef, statics)


def _combine(arrays: list[Any], structure: _Structure) -> Any:
    """Inverse of :func:`_partition`."""
    treedef, statics = structure
    remaining = iter(arrays)
    return jax.tree_util.tree_unflatten(
        treedef, [next(remaining) if leaf is None else leaf for leaf in statics]
    )


def _grid_log_posterior(
    names: tuple[str, ...],
    structure: _Structure,
    priors: tuple[tuple[str, dist.Distribution], ...],
    chunk_size: int | None,
    arrays: list[Any],
    grids: tuple[jax.Array, ...],
    fixed: dict[str, ArrayLike],
    networks: Network,
) -> jax.Array:
    likelihood = _combine(arrays, structure)
    mesh = jnp.meshgrid(*grids, indexing="ij")
    points = {name: values.ravel() for name, values in zip(names, mesh, strict=True)}

    def point_fn(point: dict[str, jax.Array]) -> jax.Array:
        values = point | fixed
        log_prior = sum(
            (prior.log_prob(values[name]) for name, prior in priors),
            start=jnp.zeros(()),
        )
        # The prediction does not depend on the network, so it is not batched:
        # `out_axes=None` for the extras asserts exactly that.
        log_likelihood, _ = jax.vmap(
            likelihood.log_likelihood, in_axes=(None, 0), out_axes=(0, None)
        )(values, networks)
        return log_likelihood + log_prior

    flat = jax.lax.map(point_fn, points, batch_size=chunk_size)
    sizes = tuple(grid.size for grid in grids)
    return flat.T.reshape(flat.shape[1], *sizes)


# Built once, so every call with the same static half and array shapes shares
# one compilation. `names` is static and `grids` a tuple: a dict argument would
# be flattened in sorted-key order, losing the insertion order of the axes.
_grid_log_posterior_jit = jax.jit(
    _grid_log_posterior, static_argnames=("names", "structure", "priors", "chunk_size")
)


def grid_log_posterior(
    likelihood: Any,
    priors: Mapping[str, dist.Distribution],
    grids: Mapping[str, jax.Array],
    *,
    fixed: Mapping[str, ArrayLike] | None = None,
    networks: Network,
    chunk_size: int | None = None,
) -> jax.Array:
    """Log posterior over the cartesian product of 1-2 grids, for K networks.

    The prediction is the expensive stage and does not depend on the network, so
    it is computed once per grid point and every network's likelihood is applied
    to it (``jax.vmap`` of ``likelihood.log_likelihood`` over ``networks``).

    Parameters
    ----------
    likelihood:
        An object with ``log_likelihood(params, network) -> (log L, extras)``, such
        as :class:`~astrogwb.inference.likelihood.ImportanceGaussianLikelihood`.
        Its array leaves are traced inputs, held once and not baked into the
        compiled function; its static half is part of the compilation key.
    priors:
        Prior of every parameter, static. Each contributes its log density at
        the grid or ``fixed`` value, so a pinned site adds a constant, as in
        :class:`LogDensityFn`.
    grids:
        Values of each swept parameter.
    fixed:
        Values of the remaining prior sites, pinned at every point.
    networks:
        K stacked :class:`~astrogwb.inference.likelihood.Network`: each leaf has
        a leading axis K, and all share one structure.
    chunk_size:
        Forwarded to :func:`jax.lax.map` as ``batch_size``. Peak memory is
        ``chunk_size * (K, F)``. ``None`` evaluates one point at a time.

    Returns
    -------
    jax.Array
        Shape ``(K, *grid sizes)``, grid axes in ``grids`` order.

    Raises
    ------
    KeyError
        At trace time, if a prior name is in neither ``grids`` nor ``fixed``.
    """
    arrays, structure = _partition(likelihood)
    return _grid_log_posterior_jit(
        names=tuple(grids),
        structure=structure,
        priors=tuple(priors.items()),
        chunk_size=chunk_size,
        arrays=arrays,
        grids=tuple(jnp.asarray(grid) for grid in grids.values()),
        fixed=dict(fixed) if fixed is not None else {},
        networks=networks,
    )


__all__ = ["LogDensityFn", "grid_log_posterior"]
