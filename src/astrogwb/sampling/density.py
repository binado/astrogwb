"""A reusable, jit-cached grid evaluator for a NumPyro model's log density.

Building a model with :func:`functools.partial` bakes every input array into
the closure, so each new dataset produces a new closure object and therefore
a new :func:`jax.jit` cache key -- looping a grid evaluation over several
datasets of the same shape (for example, one per detector network) recompiles
once per iteration for no physical reason. :class:`LogDensityFn` fixes this by
keeping the model itself fixed and routing the data through as call
*arguments* instead: :func:`~numpyro.infer.util.log_density` already accepts a
``params`` dict (a valid JAX pytree, no flattening needed), and JIT caches on
argument shape/dtype, not value, so a sweep over same-shaped data pays for one
compilation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro.infer.util import log_density


class LogDensityFn:
    """Grid-evaluate a NumPyro model's constrained log density.

    The model given to :meth:`__init__` must take only keyword arguments (the
    convention every model in :mod:`astrogwb.sampling.models` follows), since
    it is always called with ``model_args=()`` and the caller's
    ``model_kwargs`` supplied as keywords. ``spectral_density_fn`` and
    ``priors`` are baked into ``model`` with :func:`functools.partial` at the
    call site (as :func:`astrogwb.paper.inference.build_model` already does),
    not passed as ``model_kwargs`` -- only array data varies between calls.

    Deliberately a plain class, not a ``NamedTuple`` (implicitly a pytree --
    passing this into a jitted function would silently flatten it, which is
    the exact retrace hazard this class exists to avoid) or a frozen
    dataclass (would force the jitted evaluator to become a field with a
    meaningless generated ``__eq__``). It is also deliberately *not*
    registered as a pytree: the ``jax.jit`` wrapper below is built once, here,
    and closes over the model, so ``self`` never crosses a trace boundary.
    ``jax.jit(lambda lp, grids: lp(grids))`` therefore fails loudly rather
    than silently retracing -- the same discipline documented on
    :class:`~astrogwb.importance.estimator.SpectralDensityImportanceEstimator`.

    There is deliberately no manual cache keyed by grid names or shapes: a
    single ``jax.jit`` object already holds a compiled program per distinct
    argument pytree structure/shape it has seen (verified directly -- calling
    it with a grid, then a different grid, then the first grid again reuses
    the first compile rather than producing a third one), so revisiting a
    previously-seen combination of swept parameter names and data shapes is
    already a cache hit with no bookkeeping of our own. ``chunk_size`` is the
    one exception: it configures :func:`jax.lax.map`'s ``batch_size``, which
    must be a concrete Python int baked into the closure rather than a traced
    argument, so it cannot ride along inside ``jax.jit``'s own dispatch key.
    It is therefore fixed per instance; a caller who wants a different chunk
    size constructs a second :class:`LogDensityFn`.

    Parameters
    ----------
    model:
        A NumPyro model called as ``model(**model_kwargs)``. Every model in
        :mod:`astrogwb.sampling.models` returns ``None``; a
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
            model_kwargs: dict[str, Any],
        ) -> jax.Array:
            mesh = jnp.meshgrid(*grids.values(), indexing="ij")
            points = {
                name: values.ravel() for name, values in zip(grids, mesh, strict=True)
            }

            def score(point: dict[str, jax.Array]) -> jax.Array:
                log_joint, _ = log_density(
                    model, (), model_kwargs, point | fixed_params
                )
                return log_joint

            flat = jax.lax.map(score, points, batch_size=chunk_size)
            return flat.reshape(tuple(grid.size for grid in grids.values()))

        # Built once: this is what makes a sweep over model_kwargs pay for a
        # single compilation instead of one per iteration.
        self._evaluator = jax.jit(evaluate)

    def __call__(
        self,
        grids: Mapping[str, jax.Array],
        *,
        fixed: Mapping[str, ArrayLike] | None = None,
        **model_kwargs: Any,
    ) -> jax.Array:
        """Constrained log density over the cartesian product of 1-2 grids.

        Returns an array of shape ``tuple(len(g) for g in grids.values())``,
        axes in ``grids`` insertion order. ``fixed`` pins additional sample
        sites at every grid point and is threaded through as a *traced*
        argument, so sweeping its value never triggers a recompile -- only a
        change to its set of keys does, since that changes the argument
        pytree's structure.
        """
        fixed_params = dict(fixed) if fixed is not None else {}
        return self._evaluator(grids, fixed_params, model_kwargs)


__all__ = ["LogDensityFn"]
