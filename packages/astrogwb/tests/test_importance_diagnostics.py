from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb.importance.diagnostics import log_prior_reweighting, relative_ess


def test_relative_ess_is_one_for_equal_weights() -> None:
    np.testing.assert_allclose(float(relative_ess(jnp.zeros(16))), 1.0, rtol=1e-6)


def test_relative_ess_approaches_one_over_n_for_a_dominant_weight() -> None:
    log_weights = jnp.array([0.0, -60.0, -60.0, -60.0])

    np.testing.assert_allclose(float(relative_ess(log_weights)), 0.25, rtol=1e-6)


def test_relative_ess_matches_the_direct_kish_expression() -> None:
    log_weights = jnp.log(jnp.array([6.0, 8.0]))

    np.testing.assert_allclose(float(relative_ess(log_weights)), 49.0 / 50.0, rtol=1e-6)


@pytest.mark.parametrize("offset", [-1000.0, 1000.0])
def test_relative_ess_is_invariant_under_a_constant_log_offset(offset: float) -> None:
    """Exponentiating first would overflow (or flush to zero) at this offset."""
    log_weights = jnp.array([0.5, -1.0, 2.0, 0.0])

    np.testing.assert_allclose(
        float(relative_ess(log_weights + offset)),
        float(relative_ess(log_weights)),
        rtol=1e-6,
    )


def test_relative_ess_reduces_over_the_trailing_axis_only() -> None:
    log_weights = jnp.stack([jnp.zeros(8), jnp.array([0.0] + [-60.0] * 7)])

    result = relative_ess(log_weights)

    assert result.shape == (2,)
    np.testing.assert_allclose(np.asarray(result), [1.0, 0.125], rtol=1e-6)


def test_log_prior_reweighting_is_zero_for_identical_priors() -> None:
    prior = dist.Uniform(0.5, 2.0)
    amplitude = jnp.array([0.6, 1.0, 1.9])

    weights = log_prior_reweighting(amplitude, used=prior, target=prior)

    np.testing.assert_array_equal(np.asarray(weights), np.zeros(3))


def test_log_prior_reweighting_matches_the_log_density_ratio() -> None:
    used = dist.Uniform(0.5, 2.0)
    target = dist.Normal(1.0, 0.2)
    amplitude = jnp.array([0.8, 1.1])

    weights = log_prior_reweighting(amplitude, used=used, target=target)

    np.testing.assert_allclose(
        np.asarray(weights),
        np.asarray(target.log_prob(amplitude) - used.log_prob(amplitude)),
        rtol=1e-6,
    )
