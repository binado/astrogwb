"""Contract tests for :mod:`astrogwb.utils.sampling`.

``DensityAccumulator`` must reproduce numpyro's ``compute_log_probs``
bit-for-bit -- including the ``intermediates`` and plate-subsample ``scale``
paths -- and ``compute_model_and_log_probs`` must hand back the model's return
value from that same single execution. ``evaluate_sources`` and
``sample_sources`` must evaluate a per-source model in isolation: nothing about
the sources may reach an enclosing inference trace.
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
import pytest
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import compute_log_probs, log_density

from astrogwb.utils.sampling import (
    DensityAccumulator,
    compute_model_and_log_probs,
    evaluate_sources,
    sample_sources,
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


# --------------------------------------------------------------------------- #
# evaluate_sources / sample_sources
# --------------------------------------------------------------------------- #
def _source_model(params: Mapping[str, ArrayLike]) -> dict[str, jax.Array]:
    """A per-source toy: two sample sites and one deterministic output."""
    z = numpyro.sample("z", dist.Uniform(0.0, 2.0))
    m = numpyro.sample("m", dist.Normal(params["mu"], 1.5))
    detector_mass = numpyro.deterministic("detector_mass", m * (1.0 + z))
    return {
        "z": jnp.asarray(z),
        "m": jnp.asarray(m),
        "detector_mass": jnp.asarray(detector_mass),
    }


def _hierarchical_model() -> None:
    mu = numpyro.sample("mu", dist.Normal(0.5, 1.0))
    log_prob, _ = evaluate_sources(
        _source_model, {"mu": mu}, FIXED, density_sites=("z", "m")
    )
    numpyro.factor("source_density", jnp.sum(log_prob))


def test_evaluate_sources_matches_the_hand_written_density() -> None:
    log_prob, outputs = evaluate_sources(
        _source_model, {"mu": 0.5}, FIXED, density_sites=("z", "m")
    )
    expected = dist.Uniform(0.0, 2.0).log_prob(FIXED["z"]) + dist.Normal(
        0.5, 1.5
    ).log_prob(FIXED["m"])

    np.testing.assert_allclose(log_prob, expected, rtol=1e-12)
    np.testing.assert_array_equal(
        outputs["detector_mass"],
        jnp.asarray(FIXED["m"]) * (1.0 + jnp.asarray(FIXED["z"])),
    )


def test_outer_log_density_excludes_source_sites() -> None:
    mu = 0.8
    total, trace = log_density(_hierarchical_model, (), {}, {"mu": mu})
    source_log_prob, _ = evaluate_sources(
        _source_model, {"mu": mu}, FIXED, density_sites=("z", "m")
    )

    assert set(trace) == {"mu", "source_density"}
    np.testing.assert_allclose(
        total,
        dist.Normal(0.5, 1.0).log_prob(mu) + jnp.sum(source_log_prob),
        rtol=1e-12,
    )


def test_mcmc_collects_no_per_source_sites() -> None:
    mcmc = MCMC(
        NUTS(_hierarchical_model), num_warmup=5, num_samples=5, progress_bar=False
    )
    mcmc.run(jax.random.PRNGKey(0))

    assert set(mcmc.get_samples()) == {"mu"}


def test_a_missing_column_raises_naming_the_site() -> None:
    with pytest.raises(KeyError, match="'m'"):
        evaluate_sources(
            _source_model, {"mu": 0.5}, {"z": FIXED["z"]}, density_sites=()
        )
    # An enclosing seed must not turn the missing column into fresh draws.
    with handlers.seed(rng_seed=0), pytest.raises(KeyError, match="'m'"):
        evaluate_sources(
            _source_model, {"mu": 0.5}, {"z": FIXED["z"]}, density_sites=()
        )


def test_a_length_one_column_is_rejected() -> None:
    columns = {"z": FIXED["z"], "m": jnp.array([1.4])}
    with pytest.raises(ValueError, match=r"\(N,\)"):
        evaluate_sources(_source_model, {"mu": 0.5}, columns, density_sites=())


def test_a_non_vector_column_is_rejected() -> None:
    columns = {"z": jnp.reshape(jnp.asarray(FIXED["z"]), (3, 1)), "m": FIXED["m"]}
    with pytest.raises(ValueError, match=r"\(N,\)"):
        evaluate_sources(_source_model, {"mu": 0.5}, columns, density_sites=())


def test_empty_density_sites_give_per_source_zeros() -> None:
    log_prob, _ = evaluate_sources(_source_model, {"mu": 0.5}, FIXED, density_sites=())

    assert log_prob.shape == (3,)
    np.testing.assert_array_equal(log_prob, jnp.zeros(3))


def test_stored_deterministic_columns_are_recomputed() -> None:
    stale = {**FIXED, "detector_mass": jnp.full(3, -1.0)}
    _, outputs = evaluate_sources(_source_model, {"mu": 0.5}, stale, density_sites=())

    np.testing.assert_array_equal(
        outputs["detector_mass"],
        jnp.asarray(FIXED["m"]) * (1.0 + jnp.asarray(FIXED["z"])),
    )


def test_evaluate_sources_is_differentiable_and_jittable() -> None:
    def total(mu: jax.Array) -> jax.Array:
        log_prob, _ = evaluate_sources(
            _source_model, {"mu": mu}, FIXED, density_sites=("m",)
        )
        return jnp.sum(log_prob)

    mu = jnp.asarray(0.5)
    gradient = jax.grad(total)(mu)
    expected = jnp.sum((jnp.asarray(FIXED["m"]) - 0.5) / 1.5**2)
    np.testing.assert_allclose(gradient, expected, rtol=1e-12)
    assert jnp.array_equal(jax.jit(total)(mu), total(mu))


def test_sample_sources_replays_through_evaluation_bit_for_bit() -> None:
    key = jax.random.PRNGKey(3)
    first = sample_sources(_source_model, key, {"mu": 0.5}, num_samples=32)
    second = sample_sources(_source_model, key, {"mu": 0.5}, num_samples=32)
    _, replay = evaluate_sources(
        _source_model,
        {"mu": 0.5},
        {"z": first["z"], "m": first["m"]},
        density_sites=(),
    )

    assert set(first) == {"z", "m", "detector_mass"}
    assert all(values.shape == (32,) for values in first.values())
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        np.testing.assert_array_equal(first[name], replay[name])


def test_sample_sources_is_isolated_from_enclosing_handlers() -> None:
    with handlers.trace() as outer:
        sample_sources(_source_model, jax.random.PRNGKey(0), {"mu": 0.5}, num_samples=4)
    assert outer == {}
