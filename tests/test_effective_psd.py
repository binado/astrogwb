from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.detector import gaussian_bin_scale


def test_gaussian_bin_scale_uses_the_grids_bin_widths() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    frequencies = jnp.array([0.0, 10.0, 20.0])
    observation_time_yr = 5.0 / SECONDS_PER_YEAR

    actual = gaussian_bin_scale(eff, observation_time_yr, frequencies)

    np.testing.assert_allclose(np.asarray(actual), np.array([0.2, 0.4, 0.6]))
