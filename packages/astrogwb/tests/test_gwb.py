from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from astrogwb.gwb import (
    omega_gw_from_spectral_density,
    spectral_density,
    spectral_density_from_omega_gw,
)


def test_spectral_density_uses_average_mode_factor() -> None:
    power = jnp.array([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]])
    weights = jnp.array([1.0, 1.0, 2.0])

    analytic = spectral_density(
        power,
        weights,
        10.0,
        average_mode="analytic_inclination",
    )
    catalog = spectral_density(
        power,
        weights,
        10.0,
        average_mode="catalog_inclination",
    )

    np.testing.assert_allclose(np.asarray(analytic), np.array([24.0, 56.0 / 3.0]))
    np.testing.assert_allclose(np.asarray(catalog), np.array([60.0, 140.0 / 3.0]))


def test_omega_gw_round_trip() -> None:
    freqs = jnp.array([10.0, 20.0, 40.0])
    strain = jnp.array([1e-8, 2e-9, 3e-10])

    omega = omega_gw_from_spectral_density(strain, freqs, hubble_constant_si=1.0)
    actual = spectral_density_from_omega_gw(omega, freqs, hubble_constant_si=1.0)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(strain))
