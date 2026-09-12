"""Contract tests for :mod:`astrogwb.utils.sampling`.

``DensityAccumulator`` must reproduce numpyro's ``compute_log_probs``
bit-for-bit -- including the ``intermediates`` and plate-subsample ``scale``
paths -- and ``compute_model_and_log_probs`` must hand back the model's return
value from that same single execution.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.infer.util import compute_log_probs

from astrogwb.utils.sampling import (
    DensityAccumulator,
    compute_model_and_log_probs,
)

FIXED: dict[str, ArrayLike] = {
    "z": jnp.array([0.1, 0.2, 0.3]),
    "m": jnp.array([1.4, 1.5, 1.6]),
}


def _two_site_model(params: Mapping[str, Any]) -> dict[str, ArrayLike]:
    z = numpyro.sample("z", dist.Uniform(0.0, 2.0))
    m = numpyro.sample("m", dist.Normal(0.5, 1.5))
    numpyro.deterministic("detector_mass", m * (1.0 + z))
    return {"z": z, "m": m, "detector_mass": m * (1.0 + z)}


def _plated_model(params: Mapping[str, Any]) -> dict[str, ArrayLike]:
    with numpyro.plate("events", 6, subsample_size=2):
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        b = numpyro.sample("b", dist.Uniform(0.0, 1.0))
    return {"a": a, "b": b}


def _reference(model: Any, sites: tuple[str, ...]) -> jax.Array:
    """The path the accumulator replaces: condition + block(hide_fn) + sum."""
    with handlers.block():
        bound = handlers.condition(model, data=FIXED)
        filtered = handlers.block(
            bound,
            hide_fn=lambda site: site["type"] == "sample" and site["name"] not in sites,
        )
        log_probs, _ = compute_log_probs(filtered, ({},), {}, {}, sum_log_prob=False)
    return jax.tree.reduce(operator.add, log_probs, initializer=jnp.zeros(()))


def _accumulate(model: Any, sites: tuple[str, ...]) -> jax.Array:
    with handlers.block():
        accumulator = DensityAccumulator(
            handlers.condition(model, data=FIXED), sites=sites
        )
        accumulator({})
    return accumulator.log_prob


def test_accumulator_matches_compute_log_probs_bit_for_bit() -> None:
    sites = ("z", "m")
    assert jnp.array_equal(
        _accumulate(_two_site_model, sites), _reference(_two_site_model, sites)
    )


def test_only_selected_sites_contribute() -> None:
    partial = _accumulate(_two_site_model, ("z",))
    full = _accumulate(_two_site_model, ("z", "m"))

    assert jnp.array_equal(partial, _reference(_two_site_model, ("z",)))
    assert not jnp.array_equal(partial, full)


def test_deterministic_sites_are_ignored_even_if_listed() -> None:
    sites = ("z", "detector_mass")
    assert jnp.array_equal(
        _accumulate(_two_site_model, sites), _reference(_two_site_model, sites)
    )


def test_accumulator_matches_compute_log_probs_under_a_subsampled_plate() -> None:
    """Exercises the ``scale`` branch: only a subsampled plate sets it."""
    sites = ("a", "b")
    with handlers.block():
        seeded = handlers.seed(_plated_model, jax.random.PRNGKey(0))
        log_probs, _ = compute_log_probs(seeded, ({},), {}, {}, sum_log_prob=False)
        expected = jax.tree.reduce(operator.add, log_probs, initializer=jnp.zeros(()))
    with handlers.block():
        seeded = handlers.seed(_plated_model, jax.random.PRNGKey(0))
        accumulator = DensityAccumulator(seeded, sites=sites)
        accumulator({})
    assert jnp.array_equal(accumulator.log_prob, expected)


def test_compute_model_and_log_probs_returns_the_model_value_and_density() -> None:
    bound = handlers.condition(_two_site_model, data=FIXED)
    result, log_prob = compute_model_and_log_probs(bound, ("z", "m"), {})

    assert set(result) == {"z", "m", "detector_mass"}
    assert jnp.array_equal(log_prob, _reference(_two_site_model, ("z", "m")))
    np.testing.assert_array_equal(
        result["detector_mass"], FIXED["m"] * (1.0 + FIXED["z"])
    )


def test_empty_sites_sum_to_the_scalar_zero() -> None:
    bound = handlers.condition(_two_site_model, data=FIXED)
    _, log_prob = compute_model_and_log_probs(bound, (), {})

    assert log_prob.shape == ()
    assert float(log_prob) == 0.0


def test_execution_is_isolated_from_enclosing_handlers() -> None:
    bound = handlers.condition(_two_site_model, data=FIXED)
    with handlers.trace() as outer:
        compute_model_and_log_probs(bound, ("z",), {})
    assert outer == {}


def test_helper_evaluates_under_jit() -> None:
    bound = handlers.condition(_two_site_model, data=FIXED)
    eager = compute_model_and_log_probs(bound, ("z", "m"), {})[1]
    traced = jax.jit(lambda: compute_model_and_log_probs(bound, ("z", "m"), {})[1])()

    assert jnp.array_equal(eager, traced)
