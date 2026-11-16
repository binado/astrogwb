"""Energy-density conversions against ``pygwb.util.omega_to_power``."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
from conftest import PYGWB_POWER_FRACTION
from numpy.typing import NDArray

from astrogwb.background.spectral import (
    omega_gw_from_spectral_density,
    spectral_density_from_omega_gw,
)


def test_spectral_density_of_unit_omega_matches_pygwb(
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    frequencies: NDArray[np.float64],
    hubble_constant: float,
) -> None:
    spectral_density = spectral_density_from_omega_gw(
        jnp.ones_like(frequencies),
        jnp.asarray(frequencies),
        hubble_constant=hubble_constant,
    )

    np.testing.assert_allclose(
        PYGWB_POWER_FRACTION * spectral_density,
        pygwb_reference["power_per_unit_omega"],
        rtol=1e-12,
    )


def test_omega_of_pygwb_unit_power_is_one(
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    frequencies: NDArray[np.float64],
    hubble_constant: float,
) -> None:
    spectral_density = jnp.asarray(
        pygwb_reference["power_per_unit_omega"] / PYGWB_POWER_FRACTION
    )

    omega = omega_gw_from_spectral_density(
        spectral_density, jnp.asarray(frequencies), hubble_constant=hubble_constant
    )

    np.testing.assert_allclose(omega, 1.0, rtol=1e-12)
