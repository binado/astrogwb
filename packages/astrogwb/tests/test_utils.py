"""Tests for shared utility helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.utils import cumulative_trapezoid, require_x64

jax.config.update("jax_enable_x64", True)


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
