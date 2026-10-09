"""NumPyro handler utilities shared across the package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from numpyro import handlers
from numpyro.distributions import Distribution
from numpyro.primitives import Messenger

__all__ = ["SiteDistribution", "sample_model"]


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


class SiteDistribution(Messenger):
    """Record the distribution a sample site was declared with.

    A custom effect handler: wrap a model, run it, and read
    :attr:`distribution`. The recorded ``fn`` is the one the model passed to
    ``numpyro.sample``, before any enclosing ``plate`` expands it, so it is the
    model's own object -- a distribution class with its methods, not a batched
    stand-in. Apply it to the model itself, inside any plate or condition.

    Parameters
    ----------
    fn
        The model to wrap.
    site
        Name of the sample site to record.

    Attributes
    ----------
    distribution
        The recorded distribution, or ``None`` if the site has not been run.
        Under tracing it holds that trace's values, so read it in the same
        trace that ran the model.
    """

    def __init__(self, fn: Callable[..., Any] | None, site: str) -> None:
        self.site = site
        self.distribution: Distribution | None = None
        super().__init__(fn)

    def process_message(self, msg: dict[str, Any]) -> None:
        if msg["type"] == "sample" and msg["name"] == self.site:
            self.distribution = msg["fn"]
