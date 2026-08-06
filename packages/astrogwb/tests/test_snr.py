from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from astrogwb.detector import gaussian_bin_scale
from astrogwb.gwb import (
    noise_weighted_inner_product,
    spectral_snr,
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


def test_noise_weighted_inner_product_matches_explicit_sum() -> None:
    a = jnp.array([1.3, 1.7, 4.6, 2.8])
    b = jnp.array([1.0, 2.0, 4.0, 3.0])
    psd = jnp.array([0.5, 0.4, 0.8, 0.6])
    df = 2.5

    actual = noise_weighted_inner_product(a, b, psd, df)
    expected = df * np.sum(np.asarray(a) * np.asarray(b) / np.asarray(psd) ** 2)

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)


def test_noise_weighted_inner_product_scales_linearly_with_df() -> None:
    a = jnp.array([1.3, 1.7, 4.6, 2.8])
    b = jnp.array([1.0, 2.0, 4.0, 3.0])
    psd = jnp.array([0.5, 0.4, 0.8, 0.6])

    baseline = noise_weighted_inner_product(a, b, psd, 1.0)
    scaled = noise_weighted_inner_product(a, b, psd, 3.0)

    np.testing.assert_allclose(
        np.asarray(scaled), 3.0 * np.asarray(baseline), rtol=1e-6
    )


def test_noise_weighted_inner_product_broadcasts_over_leading_axes() -> None:
    a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    psd = jnp.array([0.5, 1.0, 2.0])
    df = 2.0

    actual = noise_weighted_inner_product(a, a, psd, df)

    expected = df * np.sum(np.asarray(a) ** 2 / np.asarray(psd) ** 2, axis=-1)
    assert actual.shape == (2,)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)


def test_noise_weighted_inner_product_honors_the_axis_keyword() -> None:
    a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    psd = jnp.array([0.5, 1.0])
    df = 2.0

    actual = noise_weighted_inner_product(a, a, psd[:, None], df, axis=0)

    expected = df * np.sum(np.asarray(a) ** 2 / np.asarray(psd)[:, None] ** 2, axis=0)
    assert actual.shape == (3,)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)


def test_spectral_snr_is_sqrt_of_spectral_snr_squared() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    sd = jnp.array([0.2, 0.4, 0.6])
    observation_time_sec = 5.0
    df = 10.0

    snr = spectral_snr(sd, eff, observation_time_sec, df)
    snr_squared = spectral_snr_squared(sd, eff, observation_time_sec, df)

    np.testing.assert_allclose(np.asarray(snr), np.sqrt(np.asarray(snr_squared)))
