from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.detector import gaussian_bin_scale
from astrogwb.gwb import (
    spectral_snr,
    spectral_snr_squared,
    spectral_snr_squared_per_bin,
)
from astrogwb.utils import years_to_seconds


def test_spectral_snr_squared_matches_gaussian_bin_scale() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    sd = jnp.array([0.1, 0.2, 0.3])
    observation_time_yr = 5.0 / SECONDS_PER_YEAR
    df = 10.0

    scale = gaussian_bin_scale(eff, observation_time_yr, df)
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


def test_spectral_snr_squared_per_bin_matches_hand_computed_terms() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    sd = jnp.array([0.2, 0.4, 0.6])
    observation_time_sec = 5.0
    df = 10.0

    expected = 100.0 * (sd / eff) ** 2
    actual = spectral_snr_squared_per_bin(sd, eff, observation_time_sec, df)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_spectral_snr_squared_sums_per_bin_contributions() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    sd = jnp.array([0.1, 0.2, 0.3])
    observation_time_sec = 5.0
    df = 10.0

    per_bin = spectral_snr_squared_per_bin(sd, eff, observation_time_sec, df)
    actual = spectral_snr_squared(sd, eff, observation_time_sec, df)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(jnp.sum(per_bin)))


def test_spectral_snr_squared_broadcasts_batch_factors_along_frequency() -> None:
    """Batch ``T`` and ``df`` multiply after the frequency reduction.

    A trailing ``(batch,)`` scale would otherwise align to the frequency
    axis of a ``(batch, frequency)`` array. The batch and frequency sizes
    differ here so a wrong-axis broadcast would raise.
    """
    sd = jnp.array(
        [
            [0.2, 0.4, 0.6, 0.8],
            [0.1, 0.1, 0.1, 0.1],
            [0.3, 0.0, 0.3, 0.0],
        ]
    )
    eff = jnp.full(sd.shape, 2.0)
    observation_time_sec = jnp.array([5.0, 10.0, 2.5])
    df = jnp.array([10.0, 1.0, 4.0])

    expected = np.array(
        [
            float(spectral_snr_squared(sd[i], eff[i], observation_time_sec[i], df[i]))
            for i in range(sd.shape[0])
        ]
    )
    actual = spectral_snr_squared(sd, eff, observation_time_sec, df)
    per_bin = spectral_snr_squared_per_bin(sd, eff, observation_time_sec, df)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    assert per_bin.shape == sd.shape
    np.testing.assert_allclose(
        np.asarray(jnp.sum(per_bin, axis=-1)), np.asarray(actual)
    )


def test_spectral_snr_squared_batch_factors_do_not_align_to_frequency() -> None:
    """When batch size equals the number of bins, a trailing broadcast is silent.

    Multiplying ``T`` and ``df`` into the unreduced array would weight the
    frequency axis; the factors must apply per batch item instead.
    """
    sd = jnp.arange(9.0).reshape(3, 3) * 0.1 + 0.1
    eff = jnp.full((3, 3), 2.0)
    observation_time_sec = jnp.array([1.0, 2.0, 4.0])
    df = jnp.array([1.0, 10.0, 100.0])
    ratio_squared = sd**2 / eff**2
    expected = 2.0 * observation_time_sec * df * jnp.sum(ratio_squared, axis=-1)
    wrong_axis = jnp.sum(2.0 * observation_time_sec * df * ratio_squared, axis=-1)

    actual = spectral_snr_squared(sd, eff, observation_time_sec, df)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    assert not np.allclose(np.asarray(actual), np.asarray(wrong_axis))


def test_spectral_snr_is_sqrt_of_spectral_snr_squared() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    sd = jnp.array([0.2, 0.4, 0.6])
    observation_time_sec = 5.0
    df = 10.0

    snr = spectral_snr(sd, eff, observation_time_sec, df)
    snr_squared = spectral_snr_squared(sd, eff, observation_time_sec, df)

    np.testing.assert_allclose(np.asarray(snr), np.sqrt(np.asarray(snr_squared)))
