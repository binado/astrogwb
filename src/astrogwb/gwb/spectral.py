from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.constants import INCLINATION_AVERAGE_TO_FACE_ON_RATIO
from astrogwb.cosmology import hubble_constant_si


def inclination_averaging_factor(source_parameters: Mapping[str, ArrayLike]) -> float:
    """Return the static inclination factor implied by source columns.

    A source mapping with an ``inclination`` column has already folded each
    source's orientation into its waveform power.  Without that column,
    waveform generators use the face-on orientation and this factor converts
    the resulting power to the isotropic inclination average.

    Mapping keys are part of a JAX pytree's static structure, so this ordinary
    Python branch is resolved while tracing and is safe under :func:`jax.jit`.
    """
    return (
        1.0
        if "inclination" in source_parameters
        else INCLINATION_AVERAGE_TO_FACE_ON_RATIO
    )


def spectral_density(
    polarization_power: jax.Array,
    weights: jax.Array,
    total_merger_rate: float | jax.Array,
    *,
    source_parameters: Mapping[str, ArrayLike],
) -> jax.Array:
    return (
        inclination_averaging_factor(source_parameters)
        * total_merger_rate
        * jnp.dot(polarization_power, weights)
        / weights.shape[0]
    )


def omega_gw_from_spectral_density(
    spectral_density: jax.Array,
    frequencies: jax.Array,
    *,
    hubble_constant: float,
) -> jax.Array:
    coefficient = 4.0 * jnp.pi**2 / (3.0 * hubble_constant_si(hubble_constant) ** 2)
    return coefficient * frequencies**3 * spectral_density


def spectral_density_from_omega_gw(
    omega_gw: jax.Array,
    frequencies: jax.Array,
    *,
    hubble_constant: float,
) -> jax.Array:
    coefficient = 3.0 * hubble_constant_si(hubble_constant) ** 2 / (4.0 * jnp.pi**2)
    return coefficient * omega_gw / frequencies**3
