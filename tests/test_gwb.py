from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from astrogwb.background import spectral_density


def test_spectral_density_is_the_rate_times_the_weighted_mean_intensity() -> None:
    intensity = jnp.array([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]])
    weights = jnp.array([1.0, 1.0, 2.0])

    contracted = jax.jit(spectral_density)(intensity, weights, 10.0)

    np.testing.assert_allclose(np.asarray(contracted), np.array([60.0, 140.0 / 3.0]))
