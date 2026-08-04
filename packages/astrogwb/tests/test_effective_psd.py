from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from astrogwb.detector import gaussian_bin_scale
from astrogwb.utils import SECONDS_PER_YEAR


def test_gaussian_bin_scale_infers_and_accepts_df() -> None:
    freqs = jnp.array([10.0, 20.0, 30.0])
    eff = jnp.array([2.0, 4.0, 6.0])
    observation_time_yr = 5.0 / SECONDS_PER_YEAR

    inferred = gaussian_bin_scale(eff, freqs, observation_time_yr)
    explicit = gaussian_bin_scale(eff, freqs, observation_time_yr, df=10.0)

    np.testing.assert_allclose(np.asarray(inferred), np.array([0.2, 0.4, 0.6]))
    np.testing.assert_allclose(np.asarray(explicit), np.asarray(inferred))


def test_gaussian_bin_scale_scales_with_observation_time_in_years() -> None:
    freqs = jnp.array([10.0, 20.0, 30.0])
    eff = jnp.array([2.0, 4.0, 6.0])

    one_year = gaussian_bin_scale(eff, freqs, 1.0)
    two_years = gaussian_bin_scale(eff, freqs, 2.0)

    np.testing.assert_allclose(
        np.asarray(two_years), np.asarray(one_year / jnp.sqrt(2.0))
    )
