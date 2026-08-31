from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.detector import gaussian_bin_scale


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
