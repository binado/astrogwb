"""Tests for quadrupolar Stokes factors."""

import jax.numpy as jnp
import numpy as np
import pytest

from astrogwb.waveform import quadrupolar_stokes_factors


@pytest.mark.parametrize(
    ("inclination", "expected"),
    [
        (0.0, (2.0, 0.0, 0.0, -2.0)),
        (np.pi, (2.0, 0.0, 0.0, 2.0)),
        (np.pi / 2, (0.25, 0.25, 0.0, 0.0)),
    ],
    ids=["face-on", "face-off", "edge-on"],
)
def test_quadrupolar_stokes_factors_at_special_inclinations(
    inclination: float, expected: tuple[float, ...]
) -> None:
    """Face-on and face-off are circular with opposite V, edge-on is linear."""
    factors = quadrupolar_stokes_factors(jnp.cos(jnp.array([inclination])))

    np.testing.assert_allclose(
        np.asarray(factors)[:, 0],
        np.array(expected),
        atol=1e-15,
    )
