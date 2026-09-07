"""A reusable, jit-cached grid evaluator for a NumPyro model's log density."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro.infer.util import log_density


class LogDensityFn:
    """Grid-evaluate a NumPyro model's constrained log density.

    The model given to :meth:`__init__` can take positional or keyword
    arguments. Callers supply positional inputs via ``model_args`` and
    keyword inputs via ``model_kwargs`` (the convention every model in
    :mod:`astrogwb.sampling.models` follows).

    Parameters
    ----------
    model:
        A NumPyro model called as ``model(*model_args, **model_kwargs)``. Every
        model in :mod:`astrogwb.sampling.models` returns ``None``; a
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


__all__ = ["LogDensityFn"]
