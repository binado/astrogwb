from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.detector import gaussian_bin_scale, log_frequency_noise_scale


def test_gaussian_bin_scale_uses_the_grids_bin_widths() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    frequencies = jnp.array([0.0, 10.0, 20.0])
    observation_time_yr = 5.0 / SECONDS_PER_YEAR

    actual = gaussian_bin_scale(eff, observation_time_yr, frequencies)

    np.testing.assert_allclose(np.asarray(actual), np.array([0.2, 0.4, 0.6]))


def test_gaussian_bin_scale_scales_with_observation_time_in_years() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    frequencies = jnp.array([0.0, 10.0, 20.0])

    one_year = gaussian_bin_scale(eff, 1.0, frequencies)
    two_years = gaussian_bin_scale(eff, 2.0, frequencies)

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


def test_gaussian_bin_scale_narrows_on_a_wider_bin() -> None:
    """A bin ``k`` times wider averages ``k`` times the modes: its scale is ``1/sqrt(k)``."""
    eff = jnp.full(3, 2.0)
    frequencies = jnp.array([1.0, 2.0, 4.0])
    widths = jnp.array([1.0, 1.5, 2.0])

    scale = gaussian_bin_scale(eff, 1.0, frequencies)

    np.testing.assert_allclose(
        np.asarray(scale[0] / scale), np.asarray(jnp.sqrt(widths / widths[0]))
    )
