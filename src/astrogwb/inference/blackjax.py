"""Glue from a NumPyro model to blackjax: a log density in unconstrained space.

The likelihood is already pure (:mod:`astrogwb.inference.likelihood`); NumPyro
only supplies the priors and the constraining transforms. This module extracts
exactly that, so any blackjax kernel can sample the model. It needs the
optional ``blackjax`` extra, and importing the module without it fails.

Example
-------
NUTS with window adaptation on a model built around a likelihood::

    model = partial(gwb_likelihood_model, likelihood=likelihood, priors=priors)
    logdensity_fn, initial_position, postprocess_fn = potential_from_model(
        model, {}, jax.random.key(0)
    )
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
    (state, parameters), _ = warmup.run(jax.random.key(1), initial_position, 500)
    kernel = blackjax.nuts(logdensity_fn, **parameters)

    def step(state, key):
        state, _ = kernel.step(key, state)
        return state, state.position

    _, positions = jax.lax.scan(step, state, jax.random.split(jax.random.key(2), 1000))
    samples = jax.vmap(postprocess_fn)(positions)  # constrained draws by site
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
from numpyro.infer.util import initialize_model

__all__ = ["potential_from_model"]


def potential_from_model(
    model: Callable[..., None],
    model_kwargs: Mapping[str, Any],
    rng_key: jax.Array,
) -> tuple[
    Callable[[Mapping[str, jax.Array]], jax.Array],
    dict[str, jax.Array],
    Callable[[Mapping[str, jax.Array]], dict[str, jax.Array]],
]:
    """The unconstrained log density of a NumPyro model, for blackjax.

    Parameters
    ----------
    model:
        A NumPyro model called as ``model(**model_kwargs)``, such as
        :func:`~astrogwb.inference.models.gaussian_gwb_model.gwb_likelihood_model`
        wrapped in a ``functools.partial``.
    model_kwargs:
        Keyword arguments of ``model``, bound into the returned functions.
    rng_key:
        Key for the initial position, drawn from the prior.

    Returns
    -------
    logdensity_fn
        ``position -> log density``, with ``position`` a dict of unconstrained
        values by sample site; the Jacobian of the constraining transforms is
        included, as blackjax expects.
    initial_position
        An initial unconstrained position.
    postprocess_fn
        ``position -> samples``: the constrained values of every sample and
        deterministic site. Apply it to one position, or ``jax.vmap`` it over a
        chain.
    """
    info = initialize_model(rng_key, model, model_kwargs=dict(model_kwargs))

    def logdensity_fn(position: Mapping[str, jax.Array]) -> jax.Array:
        return -info.potential_fn(position)

    return logdensity_fn, info.param_info.z, info.postprocess_fn
