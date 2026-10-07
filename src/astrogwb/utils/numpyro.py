"""NumPyro handler utilities shared across the package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from numpyro import handlers

__all__ = ["sample_model"]


def sample_model(
    model: Callable[..., Any], key: jax.Array, *args: Any, **kwargs: Any
) -> dict[str, jax.Array]:
    """Values of every sample site of one seeded run of ``model``.

    Parameters
    ----------
    model
        A NumPyro model.
    key
        PRNG key the model is seeded with.
    *args, **kwargs
        Passed to ``model``.

    Returns
    -------
    dict[str, jax.Array]
        Sampled value per sample site; deterministic sites are omitted.
        The run is blocked from any enclosing handler.
    """
    with handlers.block():
        trace = handlers.trace(handlers.seed(model, key)).get_trace(*args, **kwargs)
    return {
        name: site["value"] for name, site in trace.items() if site["type"] == "sample"
    }
