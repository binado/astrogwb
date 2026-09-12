"""NumPyro effect handlers for single-pass model evaluation.

Replaces :func:`numpyro.infer.util.compute_log_probs` for the one call pattern
the source populations use: condition every sample site, execute once, and read
back both the model's return mapping and the sum of the selected sites' log
densities. Distinct from the sibling :mod:`astrogwb.sampling` package, which
composes populations into inference models; nothing here depends on it.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import Any, Self

import jax
import jax.numpy as jnp
from numpyro import handlers
from numpyro.distributions.util import is_identically_one
from numpyro.primitives import Messenger


class DensityAccumulator(Messenger):
    """Sum ``fn.log_prob(value)`` over selected sample sites in one execution.

    Only sample sites named in ``sites`` are recognised; every other site is
    ignored, matching the ``handlers.block(hide_fn=...)`` filter this replaces.
    Applied around a conditioned model, the sum reproduces
    :func:`numpyro.infer.util.compute_log_probs` operation-for-operation, so
    values are bit-identical -- including the ``intermediates`` and
    plate-subsample ``scale`` handling.
    """

    def __init__(
        self, fn: Callable[..., Any] | None = None, *, sites: Sequence[str]
    ) -> None:
        self.sites = frozenset(sites)
        self._log_probs: list[jax.Array] = []
        super().__init__(fn)

    def __enter__(self) -> Self:
        super().__enter__()
        self._log_probs = []
        return self

    def postprocess_message(self, msg: dict[str, Any]) -> None:
        if msg["type"] != "sample" or msg["name"] not in self.sites:
            return
        intermediates = msg["intermediates"]
        if intermediates:
            log_prob = msg["fn"].log_prob(msg["value"], intermediates)
        else:
            log_prob = msg["fn"].log_prob(msg["value"])
        scale = msg["scale"]
        if scale is not None and not is_identically_one(scale):
            log_prob = scale * log_prob
        self._log_probs.append(log_prob)

    @property
    def log_prob(self) -> jax.Array:
        """Selected density, shape ``(N,)`` or ``()`` when no sites matched."""
        return jax.tree.reduce(operator.add, self._log_probs, initializer=jnp.zeros(()))


def compute_model_and_log_probs(
    model: Callable[..., Any],
    density_sites: Sequence[str],
    *model_args: Any,
    **model_kwargs: Any,
) -> tuple[Any, jax.Array]:
    """Run ``model`` once in isolation; return its result and selected density.

    ``model`` must already be wrapped for whatever the caller needs -- such as
    ``handlers.condition`` supplying fixed inputs -- because the helper cannot
    know a model's arguments. Execution is isolated from enclosing handlers
    with ``handlers.block``, matching
    :meth:`astrogwb.populations.SourceModel.evaluate`.

    Returns ``(model return value, log_prob)``. ``log_prob`` has one entry per
    source, or is the scalar zero if no ``density_sites`` matched.
    """
    with handlers.block():
        accumulator = DensityAccumulator(model, sites=density_sites)
        result = accumulator(*model_args, **model_kwargs)
    return result, accumulator.log_prob


__all__ = ["DensityAccumulator", "compute_model_and_log_probs"]
