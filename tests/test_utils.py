"""Tests for shared utility helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tolerances import assert_close

from astrogwb.utils import (
    cumulative_trapezoid,
    gauss_legendre_nodes_weights,
    require_x64,
    stack_pairs,
    unstack_pairs,
)


def test_require_x64_raises_when_disabled() -> None:
    @require_x64
    def identity(value: float) -> float:
        return value

    enabled = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="x64"):
            identity(1.0)
    finally:
        jax.config.update("jax_enable_x64", enabled)


def test_cumulative_trapezoid_is_exact_for_linear_samples() -> None:
    x = jnp.array([1.0, 2.0, 4.0, 7.0])
    y = 2.0 * x + 3.0

    actual = cumulative_trapezoid(y, x)
    antiderivative = x**2 + 3.0 * x
    expected = antiderivative - antiderivative[0]

    assert_close(actual, expected)


@pytest.mark.parametrize("order", [2, 4, 8])
def test_nodes_weights_is_exact_for_the_highest_representable_degree(
    order: int,
) -> None:
    """An ``order``-point rule integrates degree ``2 * order - 1`` exactly."""
    lower, upper = 0.5, 3.0
    degree = 2 * order - 1

    points, weights = gauss_legendre_nodes_weights(lower, upper, order)
    actual = jnp.sum(weights * points**degree, axis=-1)

    expected = (upper ** (degree + 1) - lower ** (degree + 1)) / (degree + 1)
    assert float(actual) == pytest.approx(expected, rel=2e-14)


def test_nodes_weights_covers_every_interval_of_a_grid() -> None:
    """Per-interval integrals must sum to the integral over the whole range."""
    order = 4
    grid = jnp.linspace(0.0, 2.0, 9)

    points, weights = gauss_legendre_nodes_weights(grid[:-1], grid[1:], order)
    assert points.shape == weights.shape == (grid.size - 1, order)

    interval_integrals = jnp.sum(weights * jnp.exp(points), axis=-1)
    total = float(jnp.sum(interval_integrals))

    assert total == pytest.approx(float(jnp.exp(2.0) - 1.0), rel=2e-14)


def test_nodes_weights_promotes_integer_bounds_instead_of_truncating() -> None:
    """Integer bounds cast to a float dtype must not collapse the rule to zeros.

    Cast to an integer dtype, the nodes and weights would truncate to zero.
    """
    integer_grid = jnp.arange(0, 5)
    dtype = jnp.result_type(integer_grid, float)

    points, weights = gauss_legendre_nodes_weights(
        integer_grid[:-1], integer_grid[1:], 4
    )

    assert jnp.issubdtype(points.dtype, jnp.floating)
    assert_close(jnp.sum(weights, axis=-1), np.ones(4))
    assert_close(
        points,
        gauss_legendre_nodes_weights(
            integer_grid[:-1].astype(dtype), integer_grid[1:].astype(dtype), 4
        )[0],
    )


PAIRS = (("A", "B"), ("A", "C"))


def test_stack_pairs_round_trips_through_unstack_pairs() -> None:
    values = {pair: jnp.arange(6.0).reshape(2, 3) + i for i, pair in enumerate(PAIRS)}

    stacked = stack_pairs(values, PAIRS)
    restored = unstack_pairs(stacked, PAIRS)

    assert stacked.shape == (2, 2, 3)
    assert list(restored) == list(PAIRS)
    for pair in PAIRS:
        assert_close(restored[pair], values[pair])


def test_stack_pairs_follows_the_order_of_pairs() -> None:
    values = {PAIRS[0]: jnp.zeros(3), PAIRS[1]: jnp.ones(3)}

    assert_close(stack_pairs(values, PAIRS[::-1])[0], jnp.ones(3))


@pytest.mark.parametrize(
    "values",
    [
        {PAIRS[0]: jnp.zeros(3)},
        {PAIRS[0]: jnp.zeros(3), ("B", "A"): jnp.zeros(3)},
        {PAIRS[0]: jnp.zeros(3), PAIRS[1]: jnp.zeros(4)},
    ],
    ids=["missing", "wrong-key", "ragged"],
)
def test_stack_pairs_with_invalid_values_raises(
    values: dict[tuple[str, str], jax.Array],
) -> None:
    with pytest.raises(ValueError):
        stack_pairs(values, PAIRS)


def test_unstack_pairs_with_the_wrong_pair_count_raises() -> None:
    with pytest.raises(ValueError, match="pair axis"):
        unstack_pairs(jnp.zeros((3, 4)), PAIRS)
