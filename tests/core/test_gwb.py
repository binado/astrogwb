from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from astrogwb.cosmology import hubble_constant_si
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

    omega = omega_gw_from_spectral_density(strain, freqs, hubble_constant=1.0)
    actual = spectral_density_from_omega_gw(omega, freqs, hubble_constant=1.0)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(strain))


def test_omega_gw_uses_hubble_constant_in_km_s_mpc() -> None:
    freqs = jnp.array([10.0, 20.0, 40.0])
    strain = jnp.array([1e-8, 2e-9, 3e-10])

    coefficient = 4.0 * jnp.pi**2 / (3.0 * hubble_constant_si(67.66) ** 2)
    omega = omega_gw_from_spectral_density(strain, freqs, hubble_constant=67.66)

    np.testing.assert_allclose(
        np.asarray(omega), np.asarray(coefficient * freqs**3 * strain)
    )
