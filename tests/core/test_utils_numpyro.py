"""Tests for the NumPyro handler helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

from astrogwb.populations.mass.uniform_mass_pair import uniform_mass_pair_model
from astrogwb.utils.numpyro import InverseCDF, sample_model


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


def _mass_pair() -> None:
    uniform_mass_pair_model({"minimum_mass": 1.0, "mass_width": 1.5})


def _inverse_cdf_values(model, points) -> dict:
    trace = handlers.trace(InverseCDF(model, points=points)).get_trace()
    return {name: site["value"] for name, site in trace.items()}


def test_inverse_cdf_maps_points_through_the_conditional_inverse_cdfs() -> None:
    values = _inverse_cdf_values(_mass_pair, jnp.full((1, 2), 0.5))

    mass_1 = 1.0 + 1.5 * np.sqrt(0.5)
    np.testing.assert_allclose(values["source_frame_mass_1"], [mass_1], rtol=1e-15)
    np.testing.assert_allclose(
        values["source_frame_mass_2"], [1.0 + (mass_1 - 1.0) / 2.0], rtol=1e-15
    )


def test_inverse_cdf_skips_a_site_conditioned_inside_it() -> None:
    conditioned = handlers.condition(_model, data={"x": 3.0})
    values = _inverse_cdf_values(conditioned, jnp.array([[0.25]]))

    assert float(values["x"]) == 3.0
    np.testing.assert_allclose(values["y"], [0.25], rtol=1e-15)


def test_inverse_cdf_restarts_its_coordinates_on_every_run() -> None:
    handler = InverseCDF(_model, points=jnp.array([[0.5, 0.25]]))
    first = handlers.trace(handler).get_trace()
    second = handlers.trace(handler).get_trace()

    np.testing.assert_array_equal(first["y"]["value"], second["y"]["value"])
    np.testing.assert_allclose(second["y"]["value"], [0.25], rtol=1e-15)
