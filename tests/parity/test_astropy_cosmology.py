"""Flat-LCDM distances and volumes against astropy.

``FlatLambdaCDM`` evaluates the same quantities with its own quadrature, so
agreement pins down both the integrands and the Gauss-Legendre grids. Note
astropy's differential comoving volume is per steradian while the library's
is all-sky, hence the ``4 pi``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from tolerances import assert_close

from astrogwb.cosmology import (
    distance_and_volume_grid,
    normalized_hubble_parameter,
)

REDSHIFT = np.linspace(0.0, 20.0, 128)
OMEGA_M = np.linspace(0.2, 0.4, 8)


def _flat_lcdm(omega_m: float) -> Any:
    """Reference flat-LCDM cosmology at ``H0 = 70``.

    Typed ``Any`` because astropy builds this class dynamically, which the
    type checker cannot follow.
    """
    return FlatLambdaCDM(H0=70.0, Om0=omega_m)  # ty: ignore[unknown-argument]


def test_normalized_hubble_parameter_matches_astropy() -> None:
    """E(z) is closed form on both sides, so agreement is near-exact."""
    expected = np.array([_flat_lcdm(float(omega)).efunc(REDSHIFT) for omega in OMEGA_M])
    assert_close(normalized_hubble_parameter(REDSHIFT, OMEGA_M[:, None]), expected)


def test_distance_and_volume_grid_matches_astropy() -> None:
    """Both grids agree to the references' own discretization error."""
    cosmology = _flat_lcdm(0.3)

    luminosity_distance, differential_comoving_volume = distance_and_volume_grid(
        REDSHIFT, hubble_constant=70.0, omega_m=0.3
    )

    assert_close(luminosity_distance, cosmology.luminosity_distance(REDSHIFT).value)
    assert_close(
        differential_comoving_volume,
        4.0 * np.pi * cosmology.differential_comoving_volume(REDSHIFT).value,
    )
