"""Cosmology helpers for the importance-weighting reference models.

These functions are pure, backend-agnostic, and JAX-traceable: array inputs
are dispatched through :func:`array_api_compat.array_namespace`, so the same
code runs at catalog-build time (NumPy arrays, returned as NumPy) and inside
the jitted NUTS model (JAX arrays, concrete or traced, returned as JAX).

The grid-based helpers are traceable because they evaluate on the exact
``z_grid`` array they are given: no static Python scalars (``max_redshift`` /
``n_grid``) need to be extracted from traced values inside the jitted NUTS
model. Callers build ``z_grid`` once at factory time and capture it in the
closure; see
:func:`astrogwb.importance.models.bns_madau_dickinson_modified_propagation.make_merger_rate_and_log_weights_fn`.
"""

from __future__ import annotations

from typing import overload

import jax
import numpy as np
from array_api_compat import array_namespace
from numpy.typing import NDArray

SPEED_OF_LIGHT: float = 299792458.0
MPC_IN_METERS: float = 3.0856775814913673e22


def hubble_constant_si(h0_km_s_mpc: float) -> float:
    """Convert $H_0$ from $\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}$ to SI ($\\mathrm{s^{-1}}$)."""
    return h0_km_s_mpc * 1000.0 / MPC_IN_METERS


H0: float = 67.74  # km/s/Mpc


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
    xp = array_namespace(z)
    return xp.log(xi_0 + (1.0 - xi_0) * xp.exp(-xi_n * xp.log1p(z)))


@overload
def normalized_hubble_parameter(
    redshift: NDArray[np.float64],
    omega_m: float | NDArray[np.float64],
) -> NDArray[np.float64]: ...


@overload
def normalized_hubble_parameter(
    redshift: jax.Array,
    omega_m: float | jax.Array,
) -> jax.Array: ...


def normalized_hubble_parameter(
    redshift: jax.Array | NDArray[np.float64],
    omega_m: float | jax.Array | NDArray[np.float64],
) -> jax.Array | NDArray[np.float64]:
    r"""Normalized Hubble parameter $E(z) = H(z)/H_0$ for flat $\Lambda$CDM.

    .. math::

        E(z) = \sqrt{\Omega_m (1 + z)^3 + (1 - \Omega_m)}

    Parameters
    ----------
    redshift:
        Redshift array. Accepts either a JAX array (JAX-traceable, for use
        inside jitted models) or a NumPy array (returned as NumPy). It must be
        broadcastable with ``omega_m`` according to the backend's usual
        broadcasting rules.
    omega_m:
        Matter density parameter, broadcastable with ``redshift``.

    Returns
    -------
    jax.Array or numpy.ndarray
        ``E(z)``, with the broadcasted shape of ``redshift`` and ``omega_m``
        and matching the input array type.
    """
    xp = array_namespace(redshift)
    return xp.sqrt(omega_m * (1.0 + redshift) ** 3 + (1.0 - omega_m))


@overload
def hubble_distance(h0: float) -> float: ...


@overload
def hubble_distance(h0: NDArray[np.float64]) -> NDArray[np.float64]: ...


@overload
def hubble_distance(h0: jax.Array) -> jax.Array: ...


def hubble_distance(
    h0: float | jax.Array | NDArray[np.float64],
) -> float | jax.Array | NDArray[np.float64]:
    r"""Hubble distance $c / H_0$ in Mpc.

    With $H_0$ given in $\mathrm{km\,s^{-1}\,Mpc^{-1}}$ and the speed of light
    in $\mathrm{km\,s^{-1}}$, the result is the Hubble distance in Mpc.

    Pure scalar arithmetic, so it is backend-agnostic without needing the
    array namespace: array inputs of any backend work through operator
    overloading and preserve their array type.
    """
    return SPEED_OF_LIGHT / 1000 / h0


@overload
def distance_and_volume_grid(
    redshift: NDArray[np.float64],
    hubble_constant: float | NDArray[np.float64],
    omega_m: float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


@overload
def distance_and_volume_grid(
    redshift: jax.Array,
    hubble_constant: float | jax.Array,
    omega_m: float | jax.Array,
) -> tuple[jax.Array, jax.Array]: ...


def distance_and_volume_grid(
    redshift: jax.Array | NDArray[np.float64],
    hubble_constant: float | jax.Array | NDArray[np.float64],
    omega_m: float | jax.Array | NDArray[np.float64],
) -> tuple[jax.Array | NDArray[np.float64], jax.Array | NDArray[np.float64]]:
    r"""Luminosity distance and differential comoving volume on a redshift grid.

    Evaluates both quantities on the exact ``redshift`` grid.
    Currently only supports a flat LCDM cosmology.

    The grid must be sorted ascending and nonnegative.

    Parameters
    ----------
    redshift:
        Redshift grid on which both arrays are evaluated. The final axis is
        the grid axis, with shape ``(..., n_grid)``. Accepts either a JAX
        array (JAX-traceable; no static Python scalars are required) or a
        NumPy array (outputs returned as NumPy).
    hubble_constant:
        Hubble constant $H_0$ in $\mathrm{km\,s^{-1}\,Mpc^{-1}}$. May be an
        array and must be broadcastable with the redshift-dependent terms.
    omega_m:
        Matter density parameter. May be an array and must be broadcastable
        with the redshift-dependent terms.

    Returns
    -------
    tuple[jax.Array, jax.Array] or tuple[numpy.ndarray, numpy.ndarray]
        ``(luminosity_distance, differential_comoving_volume)`` on the grid,
        with the final axis corresponding to ``n_grid`` and any leading axes
        determined by backend broadcasting. The differential comoving volume
        is integrated over the full sky to give a value in ``Mpc^3``.
    """
    xp = array_namespace(redshift)

    extended = xp.concat(
        [xp.zeros_like(redshift[..., :1]), redshift],
        axis=-1,
    )
    inv_e_extended = 1.0 / normalized_hubble_parameter(
        redshift=extended, omega_m=omega_m
    )
    inv_e = inv_e_extended[..., 1:]
    delta_z = xp.diff(extended, axis=-1)
    trapezoids = 0.5 * (inv_e_extended[..., 1:] + inv_e_extended[..., :-1]) * delta_z
    integral = xp.cumsum(trapezoids, axis=-1)
    comoving_distance = hubble_distance(hubble_constant) * integral
    luminosity_distance = (1.0 + redshift) * comoving_distance
    differential_comoving_volume = (
        4.0 * xp.pi * comoving_distance**2 * inv_e * hubble_distance(hubble_constant)
    )
    return luminosity_distance, differential_comoving_volume
