from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from astrogwb.gwb import (
    gaussian_bin_scale,
    spectral_snr_squared,
)
from astrogwb.utils import SECONDS_PER_YEAR, years_to_seconds


def test_spectral_snr_squared_matches_gaussian_bin_scale() -> None:
    freqs = jnp.array([10.0, 20.0, 30.0])
    eff = jnp.array([2.0, 4.0, 6.0])
    sd = jnp.array([0.1, 0.2, 0.3])
    observation_time_yr = 5.0 / SECONDS_PER_YEAR
    df = 10.0

    scale = gaussian_bin_scale(eff, freqs, observation_time_yr, df=df)
    observation_time_sec = years_to_seconds(observation_time_yr)

    expected = jnp.sum((sd / scale) ** 2)
    actual = spectral_snr_squared(sd, eff, observation_time_sec, df)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_spectral_snr_squared_hand_computed() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    sd = jnp.array([0.2, 0.4, 0.6])
    observation_time_sec = 5.0
    df = 10.0

    # prefactor = 2 * T * df = 100; per-bin: 100 * sd_i^2 / eff_i^2
    expected = 100.0 * ((0.2 / 2.0) ** 2 + (0.4 / 4.0) ** 2 + (0.6 / 6.0) ** 2)
    actual = spectral_snr_squared(sd, eff, observation_time_sec, df)

    np.testing.assert_allclose(np.asarray(actual), expected)
