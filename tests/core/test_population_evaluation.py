"""Contract tests for isolated, trace-based source sampling and density evaluation."""

from __future__ import annotations

import operator
from collections.abc import Callable
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

from astrogwb.populations.evaluation import (
    sample_and_logprob,
    sample_sources_qmc,
)

FIXED: dict[str, ArrayLike] = {
    "z": jnp.array([0.1, 0.2, 0.3]),
    "m": jnp.array([1.4, 1.5, 1.6]),
}


def _two_site_model() -> dict[str, ArrayLike]:
    z = numpyro.sample("z", dist.Uniform(0.0, 2.0))
    m = numpyro.sample("m", dist.Normal(0.5, 1.5))
    numpyro.deterministic("detector_mass", m * (1.0 + z))
    return {"z": z, "m": m, "detector_mass": m * (1.0 + z)}


def _reference(model: Any, sites: tuple[str, ...]) -> jax.Array:
    """The path the accumulator replaces: condition + block(hide_fn) + sum."""
    with handlers.block():
        bound = handlers.condition(model, data=FIXED)
        filtered = handlers.block(
            bound,
            hide_fn=lambda site: site["type"] == "sample" and site["name"] not in sites,
        )
        log_probs, _ = compute_log_probs(filtered, (), {}, {}, sum_log_prob=False)
    return jax.tree.reduce(operator.add, log_probs, initializer=jnp.zeros(()))


@pytest.mark.parametrize("sites", [("z",), ("z", "m"), ("z", "detector_mass"), ()])
@pytest.mark.parametrize("scale", [1.0, 3.0])
def test_selected_density_matches_numpyro(sites: tuple[str, ...], scale: float) -> None:
    model = handlers.scale(_two_site_model, scale=scale)
    outputs, log_prob = sample_and_logprob(model, values=FIXED, density_sites=sites)

    expected = jnp.broadcast_to(_reference(model, sites), (3,))
    np.testing.assert_array_equal(log_prob, expected)
    assert set(outputs) == {"z", "m", "detector_mass"}


def test_sampled_transformed_density_matches_numpyro_intermediates() -> None:
    def model() -> None:
        numpyro.sample(
            "x",
            dist.TransformedDistribution(
                dist.Normal(0.0, 1.0), dist.transforms.ExpTransform()
            ),
        )

    def plated() -> None:
        with numpyro.plate("sources", 8):
            model()

    key = jax.random.key(0)
    outputs, log_prob = sample_and_logprob(
        model, key, num_samples=8, density_sites=("x",)
    )
    with handlers.block():
        expected, trace = compute_log_probs(
            handlers.seed(plated, key), (), {}, {}, sum_log_prob=False
        )
    assert trace["x"]["intermediates"]
    np.testing.assert_array_equal(outputs["x"], trace["x"]["value"])
    np.testing.assert_array_equal(log_prob, expected["x"])


# --------------------------------------------------------------------------- #
# Population sampling and density evaluation
# --------------------------------------------------------------------------- #
def _source_model(mu: ArrayLike) -> Callable[[], None]:
    """A per-source toy, with ``mu`` closed over: two sample sites, one deterministic."""

    def model() -> None:
        z = numpyro.sample("z", dist.Uniform(0.0, 2.0))
        m = numpyro.sample("m", dist.Normal(mu, 1.5))
        numpyro.deterministic("detector_mass", m * (1.0 + z))

    return model


def _hierarchical_model() -> None:
    mu = numpyro.sample("mu", dist.Normal(0.5, 1.0))
    _, log_prob = sample_and_logprob(
        _source_model(mu), density_sites=("z", "m"), values=FIXED
    )
    numpyro.factor("source_density", jnp.sum(log_prob))


def test_sample_and_logprob_matches_the_hand_written_density() -> None:
    outputs, log_prob = sample_and_logprob(
        _source_model(0.5), density_sites=("z", "m"), values=FIXED
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
    _, source_log_prob = sample_and_logprob(
        _source_model(mu), density_sites=("z", "m"), values=FIXED
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
        sample_and_logprob(
            _source_model(0.5), density_sites=(), values={"z": FIXED["z"]}
        )
    # An enclosing seed must not turn the missing column into fresh draws.
    with handlers.seed(rng_seed=0), pytest.raises(KeyError, match="'m'"):
        sample_and_logprob(
            _source_model(0.5), density_sites=(), values={"z": FIXED["z"]}
        )


def test_a_length_one_column_is_rejected() -> None:
    columns = {"z": FIXED["z"], "m": jnp.array([1.4])}
    with pytest.raises(ValueError, match=r"\(N,\)"):
        sample_and_logprob(_source_model(0.5), density_sites=(), values=columns)


def test_a_non_vector_column_is_rejected() -> None:
    columns = {"z": jnp.reshape(jnp.asarray(FIXED["z"]), (3, 1)), "m": FIXED["m"]}
    with pytest.raises(ValueError, match=r"\(N,\)"):
        sample_and_logprob(_source_model(0.5), density_sites=(), values=columns)


def test_empty_density_sites_give_per_source_zeros() -> None:
    _, log_prob = sample_and_logprob(_source_model(0.5), density_sites=(), values=FIXED)

    assert log_prob.shape == (3,)
    np.testing.assert_array_equal(log_prob, jnp.zeros(3))


def test_stored_deterministic_columns_are_recomputed() -> None:
    stale = {**FIXED, "detector_mass": jnp.full(3, -1.0)}
    outputs, _ = sample_and_logprob(_source_model(0.5), density_sites=(), values=stale)

    np.testing.assert_array_equal(
        outputs["detector_mass"],
        jnp.asarray(FIXED["m"]) * (1.0 + jnp.asarray(FIXED["z"])),
    )


def test_sample_and_logprob_is_differentiable_and_jittable() -> None:
    def total(mu: jax.Array) -> jax.Array:
        _, log_prob = sample_and_logprob(
            _source_model(mu), density_sites=("m",), values=FIXED
        )
        return jnp.sum(log_prob)

    mu = jnp.asarray(0.5)
    gradient = jax.grad(total)(mu)
    expected = jnp.sum((jnp.asarray(FIXED["m"]) - 0.5) / 1.5**2)
    np.testing.assert_allclose(gradient, expected, rtol=1e-12)
    assert jnp.array_equal(jax.jit(total)(mu), total(mu))


def test_sample_sources_matches_evaluation_bit_for_bit() -> None:
    key = jax.random.PRNGKey(3)
    first = sample_and_logprob(
        _source_model(0.5), key, num_samples=32, density_sites=()
    )[0]
    second = sample_and_logprob(
        _source_model(0.5), key, num_samples=32, density_sites=()
    )[0]
    replay, _ = sample_and_logprob(
        _source_model(0.5), density_sites=(), values={"z": first["z"], "m": first["m"]}
    )

    assert set(first) == {"z", "m", "detector_mass"}
    assert all(values.shape == (32,) for values in first.values())
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        np.testing.assert_array_equal(first[name], replay[name])


def test_sample_sources_executes_model_once() -> None:
    calls = []
    source = _source_model(0.5)

    def model() -> None:
        calls.append(None)
        source()

    sample_and_logprob(model, jax.random.PRNGKey(0), num_samples=4, density_sites=())[0]
    assert len(calls) == 1


def test_sample_sources_broadcasts_conditioned_sites() -> None:
    model = handlers.condition(_source_model(0.5), data={"z": 0.7})
    sources = sample_and_logprob(
        model, jax.random.PRNGKey(0), num_samples=8, density_sites=()
    )[0]
    np.testing.assert_array_equal(sources["z"], np.full(8, 0.7))
    assert all(value.shape == (8,) for value in sources.values())


def test_sample_sources_is_isolated_from_enclosing_handlers() -> None:
    with handlers.trace() as outer:
        sample_and_logprob(
            _source_model(0.5), jax.random.PRNGKey(0), num_samples=4, density_sites=()
        )[0]
    assert outer == {}


@pytest.mark.parametrize("exponent", [3, 8])
def test_sample_sources_qmc_puts_one_point_in_every_dyadic_interval(
    exponent: int,
) -> None:
    """The first coordinate of a base-2 Sobol net is exactly stratified."""
    count = 2**exponent
    sources = sample_sources_qmc(
        _source_model(0.5), jax.random.PRNGKey(3), num_samples=count
    )
    # z ~ Uniform(0, 2) takes the first coordinate, and z / 2 recovers it exactly.
    cells = np.floor(np.asarray(sources["z"]) / 2.0 * count)
    np.testing.assert_array_equal(np.sort(cells), np.arange(count))


def test_sample_sources_qmc_replays_through_evaluation_bit_for_bit() -> None:
    sources = sample_sources_qmc(
        _source_model(0.5), jax.random.PRNGKey(3), num_samples=32
    )
    replay, _ = sample_and_logprob(
        _source_model(0.5),
        density_sites=(),
        values={"z": sources["z"], "m": sources["m"]},
    )

    for name in sources:
        np.testing.assert_array_equal(sources[name], replay[name])


def test_sample_sources_qmc_gives_a_conditioned_site_no_coordinate() -> None:
    model = handlers.condition(_source_model(0.5), data={"z": 0.7})
    sources = sample_sources_qmc(model, jax.random.PRNGKey(3), num_samples=16)

    np.testing.assert_array_equal(sources["z"], np.full(16, 0.7))
    # m now takes the first coordinate: one point per cell of its CDF.
    cells = np.floor(np.asarray(dist.Normal(0.5, 1.5).cdf(sources["m"])) * 16)
    np.testing.assert_array_equal(np.sort(cells), np.arange(16))


def test_sample_sources_qmc_scramble_is_fixed_by_the_key() -> None:
    draw = lambda seed: sample_sources_qmc(
        _source_model(0.5), jax.random.PRNGKey(seed), num_samples=8
    )
    np.testing.assert_array_equal(draw(1)["m"], draw(1)["m"])
    assert not np.array_equal(draw(1)["m"], draw(2)["m"])


@pytest.mark.parametrize("count", [0, 3, 24])
def test_sample_sources_qmc_rejects_a_non_power_of_two(count: int) -> None:
    with pytest.raises(ValueError, match="power of two"):
        sample_sources_qmc(_source_model(0.5), jax.random.PRNGKey(0), num_samples=count)


def test_partial_conditioning_samples_and_scores_in_one_pass() -> None:
    outputs, sampled_density = sample_and_logprob(
        _source_model(0.5),
        jax.random.key(2),
        values={"z": FIXED["z"]},
        density_sites=("z", "m"),
    )
    replay, replay_density = sample_and_logprob(
        _source_model(0.5),
        values=outputs,
        density_sites=("z", "m"),
    )
    np.testing.assert_array_equal(outputs["z"], FIXED["z"])
    np.testing.assert_array_equal(sampled_density, replay_density)
    for name in outputs:
        np.testing.assert_array_equal(outputs[name], replay[name])


def test_auxiliary_factors_are_scored_but_not_output_columns() -> None:
    def model() -> None:
        x = jnp.asarray(numpyro.sample("x", dist.Normal(0.0, 1.0)))
        numpyro.factor("extra", -x * x)
        numpyro.deterministic("twice_x", 2.0 * x)

    outputs, log_prob = sample_and_logprob(
        model,
        jax.random.key(0),
        num_samples=8,
        density_sites=("extra",),
    )
    assert set(outputs) == {"x", "twice_x"}
    np.testing.assert_array_equal(log_prob, -(outputs["x"] ** 2))


@pytest.mark.parametrize("num_samples", [None, 0, -1])
def test_sampling_requires_a_positive_source_count(num_samples: int | None) -> None:
    with pytest.raises(ValueError, match="num_samples"):
        sample_and_logprob(
            _source_model(0.5),
            jax.random.key(0),
            num_samples=num_samples,
            density_sites=(),
        )


def test_explicit_count_must_agree_with_columns() -> None:
    with pytest.raises(ValueError, match="match"):
        sample_and_logprob(
            _source_model(0.5), num_samples=4, values=FIXED, density_sites=()
        )
