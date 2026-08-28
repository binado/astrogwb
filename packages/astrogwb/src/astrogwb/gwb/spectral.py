from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from astrogwb.cosmology import H0, hubble_constant_si

AverageMode = Literal["analytic_inclination", "catalog_inclination"]


def spectral_density(
    polarization_power: jax.Array,
    weights: jax.Array,
    total_merger_rate: float | jax.Array,
    *,
    average_mode: AverageMode,
) -> jax.Array:
    factor = 0.4 if average_mode == "analytic_inclination" else 1.0
    return (
        factor
        * total_merger_rate
        * jnp.dot(polarization_power, weights)
        / weights.shape[0]
    )


def omega_gw_from_spectral_density(
    spectral_density: jax.Array,
    frequencies: jax.Array,
    *,
    hubble_constant: float = H0,
) -> jax.Array:
    coefficient = 4.0 * jnp.pi**2 / (3.0 * hubble_constant_si(hubble_constant) ** 2)
    return coefficient * frequencies**3 * spectral_density


def spectral_density_from_omega_gw(
    omega_gw: jax.Array,
    frequencies: jax.Array,
    *,
    hubble_constant: float = H0,
) -> jax.Array:
    coefficient = 3.0 * hubble_constant_si(hubble_constant) ** 2 / (4.0 * jnp.pi**2)
    return coefficient * omega_gw / frequencies**3
