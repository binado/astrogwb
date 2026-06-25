from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

AverageMode = Literal["analytic_inclination", "catalog_inclination"]

H0_SI = 67.74 * 1000.0 / 3.0856775814913673e22


def spectral_density(
    polarization_power,
    weights,
    total_merger_rate,
    *,
    average_mode: AverageMode,
):
    factor = 0.4 if average_mode == "analytic_inclination" else 1.0
    normalized_weights = weights / jnp.sum(weights)
    return factor * total_merger_rate * jnp.dot(polarization_power, normalized_weights)


def gaussian_bin_scale(effective_psd, frequencies, observation_time, *, df=None):
    if df is None:
        df = jnp.mean(jnp.diff(frequencies))
    return effective_psd / jnp.sqrt(2.0 * observation_time * df)


def frequency_mask(frequencies, *, fmin=None, fmax=None):
    mask = jnp.ones_like(frequencies, dtype=bool)
    if fmin is not None:
        mask = mask & (frequencies >= fmin)
    if fmax is not None:
        mask = mask & (frequencies <= fmax)
    return mask


def omega_gw_from_spectral_density(
    spectral_density, frequencies, *, hubble_constant_si=H0_SI
):
    coefficient = 10.0 * jnp.pi**2 / (3.0 * hubble_constant_si**2)
    return coefficient * frequencies**3 * spectral_density


def spectral_density_from_omega_gw(omega_gw, frequencies, *, hubble_constant_si=H0_SI):
    coefficient = 3.0 * hubble_constant_si**2 / (10.0 * jnp.pi**2)
    return coefficient * omega_gw / frequencies**3
