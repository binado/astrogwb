"""blackjax NUTS on a pure likelihood, against the analytic Gaussian posterior."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax.typing import ArrayLike

from astrogwb.inference import GaussianLikelihood, Network, gwb_likelihood_model
from astrogwb.inference.blackjax import potential_from_model

# Optional extra: `just test-core` does not install it, `just test-integration` does.
blackjax = pytest.importorskip("blackjax")

pytestmark = pytest.mark.integration

SHAPE = jnp.array([1.0, 1.5, 2.0, 0.5])
PRIOR_SD = 10.0


def _spectrum(params: Mapping[str, ArrayLike]) -> jax.Array:
    return jnp.asarray(params["amplitude"]) * SHAPE


@pytest.fixture
def scale() -> jax.Array:
    return jnp.array([0.6, 0.9, 1.2, 0.4])


@pytest.fixture
def observed(scale: jax.Array) -> jax.Array:
    return 1.7 * SHAPE + 0.3 * scale * jnp.array([1.0, -1.0, 0.5, -0.5])


def test_blackjax_nuts_recovers_the_analytic_gaussian_posterior(
    observed: jax.Array, scale: jax.Array
) -> None:
    """The amplitude posterior is Gaussian: linear model, Gaussian prior."""
    model = partial(
        gwb_likelihood_model,
        likelihood=GaussianLikelihood(_spectrum, observed, Network(scale)),
        priors={"amplitude": dist.Normal(0.0, PRIOR_SD)},
    )
    logdensity_fn, initial_position, postprocess_fn = potential_from_model(
        model, {}, jax.random.key(0)
    )

    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
    (state, parameters), _ = warmup.run(
        jax.random.key(1),
        initial_position,
        300,
    )
    kernel = blackjax.nuts(logdensity_fn, **parameters)

    def step(state, key):
        state, _ = kernel.step(key, state)
        return state, state.position

    _, positions = jax.lax.scan(step, state, jax.random.split(jax.random.key(2), 1000))
    draws = np.asarray(jax.vmap(postprocess_fn)(positions)["amplitude"])

    precision = float(jnp.sum((SHAPE / scale) ** 2)) + PRIOR_SD**-2
    mean = float(jnp.sum(SHAPE * observed / scale**2)) / precision
    sd = precision**-0.5
    # About 500 effective draws put the Monte Carlo error near 0.05 sd.
    np.testing.assert_allclose(draws.mean(), mean, atol=0.25 * sd, rtol=0.0)
    np.testing.assert_allclose(draws.std(), sd, rtol=0.15)
