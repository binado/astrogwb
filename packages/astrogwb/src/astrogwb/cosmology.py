"""Cosmology helpers for the importance-weighting reference models.

These functions are pure and JAX-traceable so they may run both at
catalog-build time (concrete arrays) and inside the jitted NUTS model
(traced arrays). They wrap ``gwmock_pop.cosmology.flat_lambda_cdm``.

The grid-based helpers are traceable because they evaluate on the exact
``z_grid`` array they are given: no static Python scalars (``max_redshift`` /
``n_grid``) need to be extracted from traced values inside the jitted NUTS
model. Callers build ``z_grid`` once at factory time and capture it in the
closure; see
:func:`astrogwb.importance.models.bns_madau_dickinson_modified_propagation.make_merger_rate_and_log_weights_fn`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, overload

import jax
import jax.numpy as jnp
import numpy as np
from gwmock_pop.cosmology.flat_lambda_cdm import (
    SPEED_OF_LIGHT,
    compute_normalized_hubble_parameter,
)
from numpy.typing import NDArray

MPC_IN_METERS: float = 3.0856775814913673e22


def hubble_constant_si(h0_km_s_mpc: float) -> float:
    """Convert $H_0$ from $\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}$ to SI ($\\mathrm{s^{-1}}$)."""
    return h0_km_s_mpc * 1000.0 / MPC_IN_METERS


H0_SI: float = hubble_constant_si(67.74)


@overload
def log_gw_em_ratio(
    z: NDArray[np.float64],
    xi_0: float | jax.Array,
    xi_n: float | jax.Array,
) -> NDArray[np.float64]: ...


@overload
def log_gw_em_ratio(
    z: jax.Array,
    xi_0: float | jax.Array,
    xi_n: float | jax.Array,
) -> jax.Array: ...


def log_gw_em_ratio(
    z: jax.Array | NDArray[np.float64],
    xi_0: float | jax.Array,
    xi_n: float | jax.Array,
) -> jax.Array | NDArray[np.float64]:
    """Log of the modified-propagation GW-to-EM luminosity-distance ratio.

    Models a departure from the standard ``d_GW = d_EM`` propagation as
    ``xi_0 + (1 - xi_0) * (1 + z)^(-xi_n)`` and returns its natural log.
    With ``xi_0 = 1`` the ratio is identically 1 (GR propagation); the
    function returns 0 everywhere in that case.

    Parameters
    ----------
    z:
        Redshift array. Accepts either a JAX array (JAX-traceable, for use
        inside jitted models) or a NumPy array (returned as NumPy).
    xi_0, xi_n:
        Modified-propagation parameters.

    Returns
    -------
    jax.Array or numpy.ndarray
        ``log(xi_0 + (1 - xi_0) * exp(-xi_n * log1p(z)))``, same shape as ``z``
        and matching the input array type.
    """
    value = jnp.log(xi_0 + (1.0 - xi_0) * jnp.exp(-xi_n * jnp.log1p(z)))
    if isinstance(z, np.ndarray):
        return np.asarray(value)
    return value


def distance_and_volume_grid(
    params: Mapping[str, Any],
    redshift: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Luminosity distance and differential comoving volume on a redshift grid.

    Evaluates both quantities on the exact ``redshift`` grid passed by the caller, so
    arrays that are combined element-wise with the outputs (e.g.
    ``rate_shape_grid`` and ``dvc_dz_grid``) are guaranteed to share the same
    grid. Currently only supports flat LCDM cosmology.

    The grid must be sorted ascending and start at ``0.0``; the comoving
    distance is accumulated by trapezoidal integration along ``redshift``
    assuming ``d_c(0) = 0``.

    Parameters
    ----------
    params:
        Mapping with keys ``"H0"`` (dimensionless Hubble constant) and
        ``"Omega_m"`` (matter density). May contain tracers during NUTS.
    redshift:
        Redshift grid on which both arrays are evaluated, shape ``(n_grid,)``.
        JAX-traceable; no static Python scalars are required.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(luminosity_distance, differential_comoving_volume)`` on the grid,
        each of shape ``(n_grid,)``. The differential comoving volume is the
        full-sky value in ``Mpc^3`` (includes the ``4 pi`` factor and the
        ``SPEED_OF_LIGHT / 1000`` factor).
    """
    h0 = params["H0"]
    omega_m = params["Omega_m"]

    inv_e = 1.0 / compute_normalized_hubble_parameter(
        redshift=redshift, omega_m=omega_m
    )
    delta_z = jnp.diff(redshift)
    trapezoids = 0.5 * (inv_e[1:] + inv_e[:-1]) * delta_z
    integral = jnp.concatenate(
        [jnp.zeros(1, dtype=trapezoids.dtype), jnp.cumsum(trapezoids)]
    )
    comoving_distance = SPEED_OF_LIGHT / 1000 / h0 * integral
    luminosity_distance = (1.0 + redshift) * comoving_distance
    differential_comoving_volume = (
        4.0 * jnp.pi * comoving_distance**2 * inv_e / h0 * SPEED_OF_LIGHT / 1000
    )
    return luminosity_distance, differential_comoving_volume
