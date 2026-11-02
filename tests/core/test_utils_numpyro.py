"""Tests for the NumPyro handler helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

from astrogwb.populations.mass.uniform_mass_pair import uniform_mass_pair_model
from astrogwb.utils.numpyro import InverseCDF


def _model(scale: float = 1.0) -> None:
    x = numpyro.sample("x", dist.Normal(0.0, scale))
    numpyro.sample("y", dist.Uniform(0.0, 1.0))
    numpyro.deterministic("twice_x", 2.0 * x)


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
