"""Tests for the NumPyro handler helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

from astrogwb.utils.numpyro import SiteDistribution, sample_model


def _model(scale: float = 1.0) -> None:
    x = numpyro.sample("x", dist.Normal(0.0, scale))
    numpyro.sample("y", dist.Uniform(0.0, 1.0))
    numpyro.deterministic("twice_x", 2.0 * x)


def test_returns_sample_sites_only() -> None:
    values = sample_model(_model, jax.random.key(0))
    assert set(values) == {"x", "y"}


def test_deterministic_per_key_and_distinct_across_keys() -> None:
    first = sample_model(_model, jax.random.key(0))
    again = sample_model(_model, jax.random.key(0))
    other = sample_model(_model, jax.random.key(1))
    np.testing.assert_array_equal(first["x"], again["x"])
    assert not np.array_equal(first["x"], other["x"])


def test_arguments_reach_the_model() -> None:
    wide = sample_model(_model, jax.random.key(0), scale=100.0)
    narrow = sample_model(_model, jax.random.key(0), 1.0)
    np.testing.assert_allclose(wide["x"], 100.0 * narrow["x"])


def test_ignores_enclosing_handlers_and_vmaps() -> None:
    with handlers.condition(data={"x": jnp.asarray(5.0)}):
        values = sample_model(_model, jax.random.key(0))
    assert float(values["x"]) != 5.0
    keys = jax.random.split(jax.random.key(0), 3)
    batched = jax.vmap(lambda k: sample_model(_model, k))(keys)
    assert batched["x"].shape == (3,)


def test_site_distribution_ignores_other_sites() -> None:
    def model() -> None:
        numpyro.sample("y", dist.Normal())

    recorded = SiteDistribution(model, "x")
    with handlers.seed(rng_seed=0):
        recorded()
    assert recorded.distribution is None


def test_site_distribution_sees_the_unexpanded_site_inside_a_plate() -> None:
    def model() -> None:
        numpyro.sample("x", dist.Normal(1.0, 2.0))

    inner = SiteDistribution(model, "x")

    def plated() -> None:
        with numpyro.plate("sources", 5):
            inner()

    outer = SiteDistribution(plated, "x")
    with handlers.seed(rng_seed=0):
        outer()
    assert inner.distribution is not None
    assert outer.distribution is not None
    assert inner.distribution.batch_shape == ()
    assert outer.distribution.batch_shape == (5,)
