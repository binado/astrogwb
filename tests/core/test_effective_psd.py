from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.detector import gaussian_bin_scale, log_frequency_noise_scale


def test_gaussian_bin_scale_uses_explicit_df() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    observation_time_yr = 5.0 / SECONDS_PER_YEAR

    actual = gaussian_bin_scale(eff, observation_time_yr, 10.0)

    np.testing.assert_allclose(np.asarray(actual), np.array([0.2, 0.4, 0.6]))


def test_gaussian_bin_scale_scales_with_observation_time_in_years() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])

    one_year = gaussian_bin_scale(eff, 1.0, 10.0)
    two_years = gaussian_bin_scale(eff, 2.0, 10.0)

    np.testing.assert_allclose(
        np.asarray(two_years), np.asarray(one_year / jnp.sqrt(2.0))
    )


def test_log_frequency_noise_scale_hand_computed() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    freqs = jnp.array([10.0, 40.0, 90.0])
    observation_time_yr = 5.0 / SECONDS_PER_YEAR

    # sqrt(2 * T * f) = sqrt(100), sqrt(400), sqrt(900)
    actual = log_frequency_noise_scale(eff, freqs, observation_time_yr)

    np.testing.assert_allclose(np.asarray(actual), np.array([0.2, 0.2, 0.2]))


def test_log_frequency_noise_scale_is_bin_scale_with_df_equal_to_f() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    freqs = jnp.array([3.0, 30.0, 300.0])

    actual = log_frequency_noise_scale(eff, freqs, 2.0)
    expected = jnp.array(
        [gaussian_bin_scale(eff[i], 2.0, freqs[i]) for i in range(freqs.shape[0])]
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
