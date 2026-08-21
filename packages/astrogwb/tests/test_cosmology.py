from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.cosmology import (
    H0_SI,
    distance_and_volume_grid,
    hubble_constant_si,
    hubble_distance,
)

_FIDUCIALS = {"H0": 67.66, "Omega_m": 0.3096}


def test_hubble_constant_si_matches_h0_si_default() -> None:
    np.testing.assert_allclose(hubble_constant_si(67.74), H0_SI)
    np.testing.assert_allclose(
        hubble_constant_si(67.66),
        H0_SI * (67.66 / 67.74),
    )


def test_distance_and_volume_grid_vanishes_at_redshift_zero() -> None:
    # Slicing cumsum(trapezoids)[redshift.size - 1:] keeps only the last
    # comoving distance and broadcasts it onto every grid point, so d_L(0)
    # would equal d_c(z_max) instead of 0.
    redshift = jnp.array([0.0, 0.3, 1.0, 2.7, 8.0, 20.0])
    luminosity_distance, differential_comoving_volume = distance_and_volume_grid(
        _FIDUCIALS, redshift
    )
    luminosity_distance = np.asarray(luminosity_distance)
    differential_comoving_volume = np.asarray(differential_comoving_volume)

    assert float(luminosity_distance[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(differential_comoving_volume[0]) == pytest.approx(0.0, abs=1e-9)
    comoving_distance = luminosity_distance / (1.0 + np.asarray(redshift))
    assert np.all(np.diff(comoving_distance) > 0)
    assert not np.allclose(comoving_distance, comoving_distance[-1])


def test_distance_and_volume_grid_matches_low_redshift_limit() -> None:
    redshift = jnp.array([0.0, 1.0e-3])
    luminosity_distance, _ = distance_and_volume_grid(_FIDUCIALS, redshift)
    hubble = float(hubble_distance(_FIDUCIALS["H0"]))
    np.testing.assert_allclose(
        luminosity_distance[1],
        hubble * 1.0e-3 * 1.001,
        rtol=1e-3,
    )


def test_distance_and_volume_grid_offset_matches_from_zero_prefix() -> None:
    # The virtual z = 0 knot keeps grids that start above zero equal to the
    # matching tail of a from-zero grid, rather than integrating only from z_min.
    from_zero = jnp.array([0.0, 0.5, 1.2, 3.0])
    offset = jnp.array([0.5, 1.2, 3.0])
    d_l_from_zero, dvc_from_zero = distance_and_volume_grid(_FIDUCIALS, from_zero)
    d_l_offset, dvc_offset = distance_and_volume_grid(_FIDUCIALS, offset)
    np.testing.assert_allclose(d_l_offset, d_l_from_zero[1:])
    np.testing.assert_allclose(dvc_offset, dvc_from_zero[1:])
