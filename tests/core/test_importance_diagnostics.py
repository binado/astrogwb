from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from astrogwb.importance.diagnostics import power_weighted_relative_ess, relative_ess


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


def test_power_weighted_relative_ess_matches_relative_ess_for_uniform_power() -> None:
    """A constant polarization_power adds a uniform log offset and changes nothing."""
    log_weights = jnp.array([0.5, -1.0, 2.0, 0.0])
    uniform_power = 3.0 * jnp.ones(4)

    np.testing.assert_allclose(
        float(power_weighted_relative_ess(log_weights, uniform_power)),
        float(relative_ess(log_weights)),
        rtol=1e-6,
    )


def test_power_weighted_relative_ess_is_low_when_power_is_heavy_tailed() -> None:
    """Uniform weights but one sample with wildly larger power dominates the sum.

    One dominant term among N drives the Kish ESS down to 1/N, matching
    `test_relative_ess_approaches_one_over_n_for_a_dominant_weight` above.
    """
    log_weights = jnp.zeros(4)
    power = jnp.array([1.0, 1.0, 1.0, 1.0e6])

    result = float(power_weighted_relative_ess(log_weights, power))

    np.testing.assert_allclose(result, 0.25, rtol=1e-3)
    # Equivalent to relative_ess of the weight*power products directly.
    np.testing.assert_allclose(
        result, float(relative_ess(log_weights + jnp.log(power))), rtol=1e-6
    )


def test_power_weighted_relative_ess_broadcasts_per_frequency() -> None:
    log_weights = jnp.zeros(4)
    polarization_power = jnp.stack([jnp.ones(4), jnp.array([1.0, 1.0, 1.0, 1.0e6])])

    result = power_weighted_relative_ess(log_weights, polarization_power)

    assert result.shape == (2,)
    np.testing.assert_allclose(float(result[0]), 1.0, rtol=1e-6)
    np.testing.assert_allclose(float(result[1]), 0.25, rtol=1e-3)
