"""Cosmology helpers for the importance-weighting reference models.

These functions are pure and JAX-traceable: they accept anything JAX accepts
(:data:`jax.typing.ArrayLike` -- Python scalars, NumPy arrays, JAX arrays,
concrete or traced) and always return :class:`jax.Array`, so the same code runs
at catalog-build time and inside the jitted NUTS model.

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
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpy.polynomial.legendre import leggauss

SPEED_OF_LIGHT: float = 299792458.0
MPC_IN_METERS: float = 3.0856775814913673e22

#: Fixed Gauss-Legendre quadrature rule (host-side constants, converted to the
#: working dtype inside the grid helpers). 4 nodes per interval reach near
#: machine precision for the smooth flat-LCDM integrand 1/E(z).
GAUSS_LEGENDRE_NODES, GAUSS_LEGENDRE_WEIGHTS = leggauss(4)


@overload
def hubble_constant_si(h0_km_s_mpc: float) -> float: ...


@overload
def hubble_constant_si(h0_km_s_mpc: jax.Array) -> jax.Array: ...


def hubble_constant_si(h0_km_s_mpc: float | jax.Array) -> float | jax.Array:
    """Convert $H_0$ from $\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}$ to SI ($\\mathrm{s^{-1}}$)."""
    return h0_km_s_mpc * 1000.0 / MPC_IN_METERS


H0: float = 67.74  # km/s/Mpc


def log_gw_em_ratio(z: ArrayLike, xi_0: ArrayLike, xi_n: ArrayLike) -> jax.Array:
    """Log of the modified-propagation GW-to-EM luminosity-distance ratio.

    Models a departure from the standard ``d_GW = d_EM`` propagation as
    ``xi_0 + (1 - xi_0) * (1 + z)^(-xi_n)`` and returns its natural log.
    With ``xi_0 = 1`` the ratio is identically 1 (GR propagation); the
    function returns 0 everywhere in that case.

    Parameters
    ----------
    z:
        Redshift array.
    xi_0, xi_n:
        Modified-propagation parameters.

    Returns
    -------
    jax.Array
        ``log(xi_0 + (1 - xi_0) * exp(-xi_n * log1p(z)))``, same shape as ``z``.
    """
    return jnp.log(xi_0 + (1.0 - xi_0) * jnp.exp(-(xi_n * jnp.log1p(z))))


def normalized_hubble_parameter(redshift: ArrayLike, omega_m: ArrayLike) -> jax.Array:
    r"""Normalized Hubble parameter $E(z) = H(z)/H_0$ for flat $\Lambda$CDM.

    .. math::

        E(z) = \sqrt{\Omega_m (1 + z)^3 + (1 - \Omega_m)}

    Parameters
    ----------
    redshift:
        Redshift array. It must be broadcastable with ``omega_m`` according to
        the usual broadcasting rules.
    omega_m:
        Matter density parameter, broadcastable with ``redshift``.

    Returns
    -------
    jax.Array
        ``E(z)``, with the broadcasted shape of ``redshift`` and ``omega_m``.
    """
    return jnp.sqrt(omega_m * (1.0 + redshift) ** 3 + (1.0 - omega_m))


def hubble_distance(h0: ArrayLike) -> jax.Array:
    r"""Hubble distance $c / H_0$ in Mpc.

    With $H_0$ given in $\mathrm{km\,s^{-1}\,Mpc^{-1}}$ and the speed of light
    in $\mathrm{km\,s^{-1}}$, the result is the Hubble distance in Mpc.
    """
    return SPEED_OF_LIGHT / 1000.0 / jnp.asarray(h0)


def distance_and_volume_grid(
    redshift: ArrayLike,
    hubble_constant: ArrayLike,
    omega_m: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    r"""Luminosity distance and differential comoving volume on a redshift grid.

    Evaluates both quantities on the exact ``redshift`` grid.
    Currently only supports a flat LCDM cosmology.

    The grid must be sorted ascending and nonnegative.

    Parameters
    ----------
    redshift:
        Redshift grid on which both arrays are evaluated. The final axis is
        the grid axis, with shape ``(..., n_grid)``.
    hubble_constant:
        Hubble constant $H_0$ in $\mathrm{km\,s^{-1}\,Mpc^{-1}}$. May be an
        array and must be broadcastable with the redshift-dependent terms.
    omega_m:
        Matter density parameter. May be an array and must be broadcastable
        with the redshift-dependent terms.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(luminosity_distance, differential_comoving_volume)`` on the grid,
        with the final axis corresponding to ``n_grid`` and any leading axes
        determined by broadcasting. The differential comoving volume is
        integrated over the full sky to give a value in ``Mpc^3``.

    The redshift integral is evaluated with a fixed 4-point Gauss-Legendre
    rule within each grid interval (nodes/weights are the module-level
    :data:`GAUSS_LEGENDRE_NODES` / :data:`GAUSS_LEGENDRE_WEIGHTS` constants,
    so no static scalars are extracted from traced values). For the smooth
    flat-LCDM integrand :math:`1/E(z)` this reaches near machine precision
    while keeping the computation a single cumulative pass over the grid.
    """
    redshift = jnp.asarray(redshift)
    omega_m = jnp.asarray(omega_m)

    # Promote through float so an integer redshift grid cannot truncate the
    # quadrature nodes to zeros.
    dtype = jnp.result_type(redshift, float)
    nodes = jnp.asarray(GAUSS_LEGENDRE_NODES, dtype=dtype)
    weights = jnp.asarray(GAUSS_LEGENDRE_WEIGHTS, dtype=dtype)

    extended = jnp.concatenate(
        [jnp.zeros_like(redshift[..., :1]), redshift],
        axis=-1,
    )
    lower, upper = extended[..., :-1], extended[..., 1:]
    midpoint, half_width = 0.5 * (lower + upper), 0.5 * (upper - lower)
    quadrature_points = midpoint[..., None] + half_width[..., None] * nodes
    e_quadrature = normalized_hubble_parameter(
        redshift=quadrature_points, omega_m=omega_m[..., None]
    )
    interval_integrals = (half_width[..., None] / e_quadrature) @ weights
    integral = jnp.cumsum(interval_integrals, axis=-1)
    inv_e = 1.0 / normalized_hubble_parameter(redshift=redshift, omega_m=omega_m)
    comoving_distance = hubble_distance(hubble_constant) * integral
    luminosity_distance = (1.0 + redshift) * comoving_distance
    differential_comoving_volume = (
        4.0 * jnp.pi * comoving_distance**2 * inv_e * hubble_distance(hubble_constant)
    )
    return luminosity_distance, differential_comoving_volume
