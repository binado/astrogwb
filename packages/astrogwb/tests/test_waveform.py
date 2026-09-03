from __future__ import annotations

import numpy as np
import pytest
from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.waveform import apply_gw_distance_to_power, polarization_power


def _power_and_redshift() -> tuple[np.ndarray, np.ndarray]:
    plus = np.array([[1.0 + 1.0j, 2.0, 3.0], [4.0, 5.0 + 1.0j, 6.0]])
    cross = np.array([[0.0, 1.0, 0.0], [1.0j, 0.0, 2.0]])
    return polarization_power(plus, cross), np.array([0.1, 0.8])


def test_polarization_power_reduces_plus_cross() -> None:
    actual, _ = _power_and_redshift()

    expected = np.array(
        [
            [2.0, 17.0],
            [5.0, 26.0],
            [9.0, 40.0],
        ]
    )
    assert actual.shape == (3, 2)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected)


def test_apply_gw_distance_gr_limit_is_identity_and_returns_fresh_array() -> None:
    power, redshift = _power_and_redshift()

    corrected = apply_gw_distance_to_power(power, redshift, xi_0=1.0, xi_n=1.91)

    assert corrected is not power
    np.testing.assert_allclose(corrected, power)


def test_apply_gw_distance_scales_power_by_inverse_xi_squared() -> None:
    power, redshift = _power_and_redshift()
    original = power.copy()
    xi_0, xi_n = 1.5, 1.91
    xi = np.exp(log_gw_em_ratio(redshift, xi_0=xi_0, xi_n=xi_n))

    corrected = apply_gw_distance_to_power(power, redshift, xi_0=xi_0, xi_n=xi_n)

    np.testing.assert_allclose(corrected, power / xi[None, :] ** 2)
    np.testing.assert_array_equal(power, original)


@pytest.mark.parametrize(
    ("power", "redshift", "message"),
    [
        (np.ones(2), np.ones(2), "two-dimensional"),
        (np.ones((2, 2)), np.ones((2, 1)), "one-dimensional"),
        (np.ones((2, 2)), np.ones(3), "sample axis"),
    ],
)
def test_apply_gw_distance_rejects_malformed_shapes(
    power: np.ndarray, redshift: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        apply_gw_distance_to_power(power, redshift, xi_0=1.5, xi_n=1.91)
