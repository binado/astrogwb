from __future__ import annotations

import jax
import numpy as np

from astrogwb.waveform import polarization_power


def _power() -> jax.Array:
    plus = np.array([[1.0 + 1.0j, 2.0, 3.0], [4.0, 5.0 + 1.0j, 6.0]])
    cross = np.array([[0.0, 1.0, 0.0], [1.0j, 0.0, 2.0]])
    return polarization_power(plus, cross)


def test_polarization_power_reduces_plus_cross() -> None:
    actual = _power()

    expected = np.array(
        [
            [2.0, 17.0],
            [5.0, 26.0],
            [9.0, 40.0],
        ]
    )
    assert actual.shape == (3, 2)
    assert actual.dtype == np.float64
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, expected)
