from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

AverageMode = Literal["analytic_inclination", "catalog_inclination"]

H0_SI: float = 67.74 * 1000.0 / 3.0856775814913673e22


def spectral_density(
    polarization_power: jax.Array,
    weights: jax.Array,
    total_merger_rate: float | jax.Array,
    *,
    average_mode: AverageMode,
) -> jax.Array:
    factor = 0.4 if average_mode == "analytic_inclination" else 1.0
    normalized_weights = weights / jnp.sum(weights)
    return factor * total_merger_rate * jnp.dot(polarization_power, normalized_weights)


def gaussian_bin_scale(
    effective_psd: jax.Array,
    frequencies: jax.Array,
    observation_time: float,
    *,
    df: float | jax.Array | None = None,
) -> jax.Array:
    if df is None:
        df = jnp.mean(jnp.diff(frequencies))
    return effective_psd / jnp.sqrt(2.0 * observation_time * df)


def frequency_mask(
    frequencies: jax.Array,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
) -> jax.Array:
    mask = jnp.ones_like(frequencies, dtype=bool)
    if fmin is not None:
        mask = mask & (frequencies >= fmin)
    if fmax is not None:
        mask = mask & (frequencies <= fmax)
    return mask


def omega_gw_from_spectral_density(
    spectral_density: jax.Array,
    frequencies: jax.Array,
    *,
    hubble_constant_si: float = H0_SI,
) -> jax.Array:
    coefficient = 10.0 * jnp.pi**2 / (3.0 * hubble_constant_si**2)
    return coefficient * frequencies**3 * spectral_density


def spectral_density_from_omega_gw(
    omega_gw: jax.Array,
    frequencies: jax.Array,
    *,
    hubble_constant_si: float = H0_SI,
) -> jax.Array:
    coefficient = 3.0 * hubble_constant_si**2 / (10.0 * jnp.pi**2)
    return coefficient * omega_gw / frequencies**3
