"""Tests for shared utility helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astrogwb.utils import (
    array_dict_shape,
    cumulative_trapezoid,
    gauss_legendre_rule,
    mapped_gauss_legendre_rule,
    require_x64,
)


def test_array_dict_shape_returns_the_common_shape() -> None:
    assert array_dict_shape({"mass": np.ones(3), "redshift": jnp.ones(3)}) == (3,)


def test_array_dict_shape_accepts_matching_arbitrary_shapes() -> None:
    assert array_dict_shape({"first": np.ones((2, 3)), "second": jnp.ones((2, 3))}) == (
        2,
        3,
    )


def test_array_dict_shape_accepts_matching_scalar_values() -> None:
    assert array_dict_shape({"first": 1.0, "second": np.asarray(2.0)}) == ()


def test_array_dict_shape_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match=r"mass.*\(2,\).*redshift.*\(3,\)"):
        array_dict_shape({"mass": np.ones(2), "redshift": np.ones(3)})


def test_array_dict_shape_rejects_empty_mappings() -> None:
    with pytest.raises(ValueError, match="at least one array"):
        array_dict_shape({})


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

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-15)


def test_cumulative_trapezoid_is_jittable_and_differentiable() -> None:
    x = jnp.linspace(1.0, 4.0, 7)

    def evaluate(scale: jax.Array) -> jax.Array:
        return jnp.sum(cumulative_trapezoid(scale * x, x))

    actual = jax.jit(evaluate)(jnp.asarray(2.0))
    derivative = jax.grad(evaluate)(jnp.asarray(2.0))
    expected_derivative = evaluate(jnp.asarray(1.0))

    assert actual == pytest.approx(2.0 * float(expected_derivative), rel=2e-15)
    assert derivative == pytest.approx(float(expected_derivative), rel=2e-15)


def test_cumulative_trapezoid_broadcasts_over_leading_batch_dimensions() -> None:
    x = jnp.linspace(0.0, 2.0, 5)
    y = jnp.stack([x**2, jnp.sin(x)])

    actual = cumulative_trapezoid(y, x)

    assert actual.shape == y.shape
    np.testing.assert_allclose(np.asarray(actual[:, 0]), np.zeros(2), atol=0.0)
    for row in range(y.shape[0]):
        np.testing.assert_allclose(
            np.asarray(actual[row]),
            np.asarray(cumulative_trapezoid(y[row], x)),
            atol=2e-15,
        )


def test_gauss_legendre_rule_is_cached() -> None:
    assert gauss_legendre_rule(4) is gauss_legendre_rule(4)


@pytest.mark.parametrize("order", [2, 4, 8])
def test_mapped_rule_is_exact_for_the_highest_representable_degree(
    order: int,
) -> None:
    """An ``order``-point rule integrates degree ``2 * order - 1`` exactly."""
    lower, upper = 0.5, 3.0
    degree = 2 * order - 1

    points, weights = mapped_gauss_legendre_rule(order, lower, upper)
    actual = jnp.sum(weights * points**degree, axis=-1)

    expected = (upper ** (degree + 1) - lower ** (degree + 1)) / (degree + 1)
    assert float(actual) == pytest.approx(expected, rel=2e-14)


def test_mapped_rule_weights_carry_the_interval_jacobian() -> None:
    lower = jnp.array([0.0, 1.0, 3.0])
    upper = jnp.array([1.0, 3.0, 7.0])

    _, weights = mapped_gauss_legendre_rule(4, lower, upper)

    np.testing.assert_allclose(
        np.asarray(jnp.sum(weights, axis=-1)),
        np.asarray(upper - lower),
        rtol=2e-15,
    )


def test_mapped_rule_covers_every_interval_of_a_grid() -> None:
    """Per-interval integrals must sum to the integral over the whole range."""
    order = 4
    grid = jnp.linspace(0.0, 2.0, 9)

    points, weights = mapped_gauss_legendre_rule(order, grid[:-1], grid[1:])
    assert points.shape == weights.shape == (grid.size - 1, order)

    interval_integrals = jnp.sum(weights * jnp.exp(points), axis=-1)
    total = float(jnp.sum(interval_integrals))

    assert total == pytest.approx(float(jnp.exp(2.0) - 1.0), rel=2e-14)


def test_mapped_rule_honours_the_requested_dtype() -> None:
    points, weights = mapped_gauss_legendre_rule(4, 0.0, 1.0, dtype=jnp.float32)

    assert points.dtype == jnp.float32
    assert weights.dtype == jnp.float32


def test_mapped_rule_promotes_integer_bounds_instead_of_truncating() -> None:
    """Integer bounds cast to a float dtype must not collapse the rule to zeros.

    This is the failure mode guarded in
    :func:`astrogwb.cosmology.distance_and_volume_grid`: cast to an integer
    dtype the nodes and weights all truncate to zero and the integral silently
    evaluates to zeros, with no warning and no NaN.
    """
    integer_grid = jnp.arange(0, 5)
    dtype = jnp.result_type(integer_grid, float)

    points, weights = mapped_gauss_legendre_rule(
        4, integer_grid[:-1], integer_grid[1:], dtype=dtype
    )

    assert jnp.issubdtype(points.dtype, jnp.floating)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(weights, axis=-1)), np.ones(4), rtol=2e-15
    )
    np.testing.assert_allclose(
        np.asarray(points),
        np.asarray(
            mapped_gauss_legendre_rule(
                4, integer_grid[:-1].astype(dtype), integer_grid[1:].astype(dtype)
            )[0]
        ),
        rtol=2e-15,
    )


def test_mapped_rule_is_jittable_and_differentiable_in_its_bounds() -> None:
    def integrate(upper: jax.Array) -> jax.Array:
        points, weights = mapped_gauss_legendre_rule(4, 0.0, upper)
        return jnp.sum(weights * points**2, axis=-1)

    upper = jnp.asarray(3.0)

    actual = jax.jit(integrate)(upper)
    derivative = jax.grad(integrate)(upper)

    # d/db of the integral of x^2 from 0 to b is b^2.
    assert float(actual) == pytest.approx(3.0**3 / 3.0, rel=2e-15)
    assert float(derivative) == pytest.approx(3.0**2, rel=2e-15)
