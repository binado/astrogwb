from __future__ import annotations

import jax.numpy as jnp
import pytest
from tolerances import assert_close

from astrogwb.background.importance.diagnostics import relative_ess


def test_relative_ess_matches_the_direct_kish_expression() -> None:
    log_weights = jnp.log(jnp.array([6.0, 8.0]))

    assert_close(relative_ess(log_weights), 49.0 / 50.0)


@pytest.mark.parametrize("offset", [-1000.0, 1000.0])
def test_relative_ess_is_invariant_under_a_constant_log_offset(offset: float) -> None:
    """Exponentiating first would overflow (or flush to zero) at this offset."""
    log_weights = jnp.array([0.5, -1.0, 2.0, 0.0])

    assert_close(relative_ess(log_weights + offset), relative_ess(log_weights))


def test_relative_ess_reduces_over_the_trailing_axis_only() -> None:
    log_weights = jnp.stack([jnp.zeros(8), jnp.array([0.0] + [-60.0] * 7)])

    result = relative_ess(log_weights)

    assert result.shape == (2,)
    assert_close(result, [1.0, 0.125])
