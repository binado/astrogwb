"""A reusable, jit-cached log-density callable for a NumPyro model.

Building a model with :func:`functools.partial` bakes every input array into
the closure, so each new dataset produces a new closure object and therefore
a new :func:`jax.jit` cache key -- looping a sampler or a grid evaluation over
several datasets of the same shape (for example, one per detector network)
recompiles once per iteration for no physical reason. :class:`LogPosterior`
fixes this by keeping the model itself fixed and routing the data through as
call *arguments* instead: ``potential_fn``/``log_density`` already accept a
``params`` dict (a valid JAX pytree, no flattening needed), and JIT caches on
argument shape/dtype, not value, so a sweep over same-shaped data pays for one
compilation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.util import (
    constrain_fn,
    initialize_model,
    log_density,
    unconstrain_fn,
)


class LogPosterior:
    """A NumPyro model's log density, with its data as arguments, not captures.

    The model given to :meth:`__init__` must take only keyword arguments (the
    convention every model in :mod:`astrogwb.sampling.models` follows), since
    every method here calls it with ``model_args=()`` and the caller's
    ``model_kwargs`` supplied as keywords.

    Deliberately a plain class, not a ``NamedTuple`` (implicitly a pytree --
    passing this into a jitted function would silently flatten it, which is
    the exact retrace hazard this class exists to avoid) or a frozen
    dataclass (would force the jitted closures to become fields with a
    meaningless generated ``__eq__``, and block the :meth:`evaluate_grid`
    cache). It is also deliberately *not* registered as a pytree: the
    ``jax.jit`` wrappers below are built once, here, and close over the
    model, so ``self`` never crosses a trace boundary. ``jax.jit(lambda lp,
    p: lp(p))`` therefore fails loudly rather than silently retracing --
    the same discipline documented on
    :class:`~astrogwb.importance.estimator.SpectralDensityImportanceEstimator`.
    ``jax.vmap(lp, in_axes=(None, 0))`` still works, since there too ``lp``
    is captured rather than passed.

    Parameters
    ----------
    model:
        A NumPyro model called as ``model(**model_kwargs)``.
    template_kwargs:
        The prototype ``model_kwargs`` :func:`~numpyro.infer.util.initialize_model`
        traces once to derive the unconstrained-space bijectors and an initial
        point. Later calls may pass different data, provided it does not
        change which sites exist or their supports.
    rng_key, init_strategy:
        Forwarded to :func:`~numpyro.infer.util.initialize_model` to build
        :attr:`init_params`.
    """

    init_params: dict[str, jax.Array]
    """Unconstrained initial values, validated by ``initialize_model``."""

    def __init__(
        self,
        model: Any,
        *,
        template_kwargs: Mapping[str, Any],
        rng_key: jax.Array | None = None,
        init_strategy: Any = init_to_uniform,
    ) -> None:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        param_info, potential_fn, _postprocess_fn, _model_trace = initialize_model(
            rng_key,
            model,
            init_strategy=init_strategy,
            dynamic_args=True,
            model_args=(),
            model_kwargs=dict(template_kwargs),
        )
        self.init_params = param_info.z
        self._model = model

        # Built once: this is what makes a sweep over model_kwargs pay for a
        # single compilation instead of one per iteration.
        self._potential_jit = jax.jit(
            lambda params, **model_kwargs: potential_fn(**model_kwargs)(params)
        )
        self._log_density_jit = jax.jit(
            lambda params, **model_kwargs: log_density(model, (), model_kwargs, params)[
                0
            ]
        )
        self._constrain_jit = jax.jit(
            lambda unconstrained, **model_kwargs: constrain_fn(
                model, (), model_kwargs, unconstrained
            )
        )
        self._unconstrain_jit = jax.jit(
            lambda constrained, **model_kwargs: unconstrain_fn(
                model, (), model_kwargs, constrained
            )
        )
        self._grid_evaluators: dict[tuple[Any, int | None], Any] = {}

    def __call__(
        self, params: Mapping[str, ArrayLike], **model_kwargs: Any
    ) -> jax.Array:
        """Constrained (physical) log posterior, up to a normalizing constant."""
        return self._log_density_jit(params, **model_kwargs)

    def potential(
        self, params: Mapping[str, ArrayLike], **model_kwargs: Any
    ) -> jax.Array:
        """Unconstrained potential energy, bijector Jacobian included.

        Differs from ``-self(constrained)`` by exactly the log-Jacobian of the
        unconstraining transform at each site; use this one for HMC/NUTS or
        an unconstrained-space optimizer, :meth:`__call__` for a physical grid.
        """
        return self._potential_jit(params, **model_kwargs)

    def constrain(
        self, unconstrained: Mapping[str, ArrayLike], **model_kwargs: Any
    ) -> dict[str, jax.Array]:
        """Map unconstrained values to each sample site's constrained support."""
        return self._constrain_jit(unconstrained, **model_kwargs)

    def unconstrain(
        self, constrained: Mapping[str, ArrayLike], **model_kwargs: Any
    ) -> dict[str, jax.Array]:
        """Map constrained values to unconstrained space; the inverse of :meth:`constrain`."""
        return self._unconstrain_jit(constrained, **model_kwargs)

    def evaluate_grid(
        self,
        grids: Mapping[str, jax.Array],
        *,
        fixed: Mapping[str, ArrayLike] | None = None,
        chunk_size: int | None = None,
        **model_kwargs: Any,
    ) -> jax.Array:
        """Constrained log density over the cartesian product of 1-2 grids.

        Returns an array of shape ``tuple(len(g) for g in grids.values())``,
        axes in ``grids`` insertion order. ``fixed`` pins additional sample
        sites at every grid point and is threaded through as a *traced*
        argument, so sweeping its value never triggers a recompile.

        The evaluator is built once per distinct ``(tuple(grids), chunk_size)``
        -- the swept parameter *names* and the chunk size -- and cached on the
        instance; a later call with the same key reuses it, and JAX's own
        shape-based dispatch inside that one jit object then decides on its
        own whether ``model_kwargs`` of a new shape needs a fresh trace.
        Peak memory is bounded by chunking through :func:`jax.lax.map` rather
        than a single ``vmap`` over the whole grid, which would materialize
        every intermediate array at every point simultaneously.
        """
        fixed_params = dict(fixed) if fixed is not None else {}
        cache_key = (tuple(grids), chunk_size)
        evaluator = self._grid_evaluators.get(cache_key)
        if evaluator is None:
            evaluator = jax.jit(self._build_grid_evaluator(chunk_size))
            self._grid_evaluators[cache_key] = evaluator

        mesh = jnp.meshgrid(*grids.values(), indexing="ij")
        points = {
            name: values.ravel() for name, values in zip(grids, mesh, strict=True)
        }
        flat = evaluator(points, fixed_params, model_kwargs)
        return flat.reshape(tuple(grid.size for grid in grids.values()))

    def _build_grid_evaluator(self, chunk_size: int | None):
        model = self._model

        def evaluate(
            points: dict[str, jax.Array],
            fixed_params: dict[str, ArrayLike],
            model_kwargs: dict[str, Any],
        ) -> jax.Array:
            def score(point: dict[str, jax.Array]) -> jax.Array:
                log_joint, _ = log_density(
                    model, (), model_kwargs, point | fixed_params
                )
                return log_joint

            return jax.lax.map(score, points, batch_size=chunk_size)

        return evaluate


__all__ = ["LogPosterior"]
