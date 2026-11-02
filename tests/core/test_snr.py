from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from astrogwb.background import (
    spectral_snr_squared,
    spectral_snr_squared_per_bin,
    spectral_snr_squared_per_log_frequency,
)
from astrogwb.detector import log_frequency_noise_scale
from astrogwb.frequency import bin_widths
from astrogwb.utils import years_to_seconds


def test_spectral_snr_squared_sums_per_bin_contributions() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    sd = jnp.array([0.1, 0.2, 0.3])
    observation_time_sec = 5.0
    frequencies = 10.0 * jnp.arange(3)

    per_bin = spectral_snr_squared_per_bin(sd, eff, observation_time_sec, frequencies)
    actual = spectral_snr_squared(sd, eff, observation_time_sec, frequencies)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(jnp.sum(per_bin)))


def test_spectral_snr_squared_batch_factors_do_not_align_to_frequency() -> None:
    """When batch size equals the number of bins, a trailing broadcast is silent.

    Multiplying ``T`` into the unreduced array would weight the frequency
    axis; the factor must apply per batch item instead.
    """
    sd = jnp.arange(9.0).reshape(3, 3) * 0.1 + 0.1
    eff = jnp.full((3, 3), 2.0)
    observation_time_sec = jnp.array([1.0, 2.0, 4.0])
    frequencies = jnp.array([0.0, 10.0, 20.0])
    ratio_squared = sd**2 / eff**2
    expected = 2.0 * observation_time_sec * 10.0 * jnp.sum(ratio_squared, axis=-1)
    wrong_axis = jnp.sum(2.0 * observation_time_sec * 10.0 * ratio_squared, axis=-1)

    actual = spectral_snr_squared(sd, eff, observation_time_sec, frequencies)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    assert not np.allclose(np.asarray(actual), np.asarray(wrong_axis))


def test_snr_per_log_frequency_rescales_per_bin_terms_by_f_over_bin_width() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    sd = jnp.array([0.1, 0.2, 0.3])
    freqs = jnp.array([10.0, 20.0, 40.0])
    observation_time_sec = 5.0
    widths = bin_widths(freqs)

    density = spectral_snr_squared_per_log_frequency(
        sd, eff, freqs, observation_time_sec
    )
    per_bin = spectral_snr_squared_per_bin(sd, eff, observation_time_sec, freqs)

    np.testing.assert_allclose(
        np.asarray(density * widths / freqs), np.asarray(per_bin)
    )
    np.testing.assert_allclose(
        np.asarray(jnp.sum(density * widths / freqs)),
        np.asarray(spectral_snr_squared(sd, eff, observation_time_sec, freqs)),
    )


def test_snr_per_log_frequency_is_squared_ratio_to_log_frequency_scale() -> None:
    """The sensitivity plot and the SNR-density plot show the same quantity."""
    eff = jnp.array([2.0, 4.0, 6.0])
    sd = jnp.array([0.1, 0.2, 0.3])
    freqs = jnp.array([5.0, 50.0, 500.0])
    observation_time_yr = 0.5

    scale = log_frequency_noise_scale(eff, freqs, observation_time_yr)
    density = spectral_snr_squared_per_log_frequency(
        sd, eff, freqs, years_to_seconds(observation_time_yr)
    )

    np.testing.assert_allclose(np.asarray(density), np.asarray((sd / scale) ** 2))


def test_spectral_snr_squared_on_a_non_uniform_grid_weights_bins_by_width() -> None:
    eff = jnp.full(4, 2.0)
    sd = jnp.array([2.0, 2.0, 2.0, 2.0])
    frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

    per_bin = spectral_snr_squared_per_bin(sd, eff, 5.0, frequencies)

    # 2 * T * width * (sd / eff)^2 with widths [1, 1.5, 3, 4].
    np.testing.assert_allclose(np.asarray(per_bin), [10.0, 15.0, 30.0, 40.0])


def test_spectral_snr_squared_mask_keeps_the_widths_of_the_full_grid() -> None:
    """A gappy mask must not re-derive widths from the surviving bins."""
    eff = jnp.full(4, 2.0)
    sd = jnp.full(4, 2.0)
    frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])
    mask = jnp.array([True, False, True, True])

    masked = spectral_snr_squared(sd, eff, 5.0, frequencies, frequency_mask=mask)
    sliced = spectral_snr_squared(sd[mask], eff[mask], 5.0, frequencies[mask])

    np.testing.assert_allclose(np.asarray(masked), 10.0 + 30.0 + 40.0)
    assert not np.allclose(np.asarray(masked), np.asarray(sliced))


def test_spectral_snr_squared_mask_excludes_bins_with_infinite_psd() -> None:
    eff = jnp.array([2.0, jnp.inf, 2.0])
    sd = jnp.array([2.0, 2.0, 2.0])
    frequencies = jnp.array([0.0, 10.0, 20.0])
    mask = jnp.array([True, False, True])

    snr_squared = spectral_snr_squared(sd, eff, 5.0, frequencies, frequency_mask=mask)

    assert np.isfinite(np.asarray(snr_squared))
