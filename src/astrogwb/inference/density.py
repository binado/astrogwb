"""A reusable, jit-cached grid evaluator for a NumPyro model's log density."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.infer.util import log_density

from .protocol import SpectralDensityFn


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


class GaussianGWBBatchedLikelihood:
    """Grid-evaluate the Gaussian GWB log density for K networks at once.

    The spectral density is the expensive stage and does not depend on the
    network; only ``scale`` and ``frequency_mask`` do, at O(F) per point. This
    evaluator predicts the spectrum once per grid point and applies the
    likelihood of every network to it.

    The spectral density function is a call argument, not part of the
    evaluator. Its array leaves -- a catalog bound by
    :func:`~astrogwb.gwb.importance.build_rescaled_spectrum`, say -- are traced
    inputs, so they are held once and not baked into the compiled function,
    and one compilation serves every function with the same static half and
    array shapes. A plain closure has no array leaves and is static whole: it
    still works, with whatever it captures compiled in as constants.

    It duplicates the Gaussian likelihood of
    :func:`astrogwb.inference.gwb_spectral_density_model` for speed; the
    equivalence test against :class:`LogDensityFn` keeps the two in sync. The
    amplitude-marginalized likelihood is not covered.

    Parameters
    ----------
    priors:
        Prior of every parameter, static. Each contributes its log density at
        the grid or ``fixed`` value, so a pinned site adds a constant, as in
        :class:`LogDensityFn`.
    chunk_size:
        Forwarded to :func:`jax.lax.map` as ``batch_size`` for the prediction
        stage. ``None`` evaluates one point at a time.

    Notes
    -----
    Memory is ``(G, F)`` for the predictions plus ``(K, G, F)`` likelihood
    intermediates. If a larger ``G`` needs it, ``lax.map`` over K instead of
    ``vmap``.
    """

    def __init__(
        self,
        priors: Mapping[str, dist.Distribution],
        *,
        chunk_size: int | None = None,
    ) -> None:
        def evaluate(
            names: tuple[str, ...],
            structure: _Structure,
            arrays: list[Any],
            grids: tuple[jax.Array, ...],
            fixed: dict[str, ArrayLike],
            observed: jax.Array,
            scale: jax.Array,
            mask: jax.Array,
        ) -> jax.Array:
            spectral_density_fn = _combine(arrays, structure)
            mesh = jnp.meshgrid(*grids, indexing="ij")
            points = {
                name: values.ravel() for name, values in zip(names, mesh, strict=True)
            }
            pred = jax.lax.map(
                lambda point: spectral_density_fn(point | fixed)[0],
                points,
                batch_size=chunk_size,
            )

            values = points | fixed
            log_prior = jnp.asarray(
                sum(
                    (prior.log_prob(values[name]) for name, prior in priors.items()),
                    start=jnp.zeros(()),
                )
            )

            def likelihood(scale_k: jax.Array, mask_k: jax.Array) -> jax.Array:
                logp = dist.Normal(pred, scale_k).log_prob(observed)
                return jnp.where(mask_k, logp, 0.0).sum(-1)

            total = log_prior[None] + jax.vmap(likelihood)(scale, mask)
            sizes = tuple(grid.size for grid in grids)
            return jnp.asarray(total).reshape(scale.shape[0], *sizes)

        # `names` is static and `grids` a tuple: a dict argument would be
        # flattened in sorted-key order, losing the insertion order of the axes.
        # `structure` is the spectral density function's static half.
        self._evaluator = jax.jit(evaluate, static_argnums=(0, 1))

    def __call__(
        self,
        grids: Mapping[str, jax.Array],
        *,
        spectral_density_fn: SpectralDensityFn,
        fixed: Mapping[str, ArrayLike] | None = None,
        observed_spectral_density: jax.Array,
        scale: jax.Array,
        frequency_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Log density over the cartesian product of 1-2 grids, per network.

        Parameters
        ----------
        grids:
            Values of each swept parameter.
        spectral_density_fn:
            ``params -> (prediction, extras)`` with ``prediction`` of shape
            ``(F,)``; ``extras`` are discarded. Its array leaves are traced, its
            static rest is part of the compilation key.
        fixed:
            Values of the remaining prior sites, pinned at every point.
        observed_spectral_density:
            Observed spectrum, ``(F,)``.
        scale:
            Per-bin standard deviation of each network, ``(K, F)``.
        frequency_mask:
            Boolean ``(K, F)`` of the bins each network counts; ``None``
            counts all. Masked bins contribute exactly zero, whatever their
            ``scale``.

        Returns
        -------
        jax.Array
            Shape ``(K, *grid sizes)``, grid axes in ``grids`` order.

        Raises
        ------
        KeyError
            At trace time, if a prior name is in neither ``grids`` nor ``fixed``.
        """
        scale = jnp.asarray(scale)
        mask = (
            jnp.ones(scale.shape, dtype=bool)
            if frequency_mask is None
            else jnp.asarray(frequency_mask)
        )
        arrays, structure = _partition(spectral_density_fn)
        return self._evaluator(
            tuple(grids),
            structure,
            arrays,
            tuple(jnp.asarray(grid) for grid in grids.values()),
            dict(fixed) if fixed is not None else {},
            jnp.asarray(observed_spectral_density),
            scale,
            mask,
        )


__all__ = ["GaussianGWBBatchedLikelihood", "LogDensityFn"]
