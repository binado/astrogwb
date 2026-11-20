from __future__ import annotations

import jax.numpy as jnp
from tolerances import assert_close

from astrogwb.detector import network_cross_power_std


def test_gaussian_bin_scale_uses_the_grids_bin_widths() -> None:
    eff = jnp.array([2.0, 4.0, 6.0])
    frequencies = jnp.array([0.0, 10.0, 20.0])
    observation_seconds = 5.0

    actual = network_cross_power_std(eff, observation_seconds, frequencies)

    assert_close(actual, [0.2, 0.4, 0.6])
