"""Cosmology helpers for the source-population models.

Pure and JAX-traceable. The grid helpers evaluate on the exact grid they are
given, so no static scalar is extracted from a traced value.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import overload

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.constants import MPC_IN_METERS, SPEED_OF_LIGHT
from astrogwb.utils import gauss_legendre_nodes_weights

#: Fixed Gauss-Legendre quadrature order used by the grid helpers. 4 nodes per
#: interval reach near machine precision for the smooth flat-LCDM integrand
#: 1/E(z).
GAUSS_LEGENDRE_ORDER: int = 4


@overload
def hubble_constant_si(h0_km_s_mpc: float) -> float: ...


@overload
def hubble_constant_si(h0_km_s_mpc: jax.Array) -> jax.Array: ...


def hubble_constant_si(h0_km_s_mpc: float | jax.Array) -> float | jax.Array:
    """Convert $H_0$ from $\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}$ to SI ($\\mathrm{s^{-1}}$).

    Parameters
    ----------
    h0_km_s_mpc
        Hubble constant in km s^-1 Mpc^-1.
    """
    return h0_km_s_mpc * 1000.0 / MPC_IN_METERS


def log_gw_em_ratio(z: ArrayLike, xi_0: ArrayLike, xi_n: ArrayLike) -> jax.Array:
    """Log of the modified-propagation GW-to-EM luminosity-distance ratio.

    Models a departure from the standard ``d_GW = d_EM`` propagation as
    ``xi_0 + (1 - xi_0) * (1 + z)^(-xi_n)`` and returns its natural log.
    With ``xi_0 = 1`` the ratio is identically 1 (GR propagation); the
    function returns 0 everywhere in that case.

    Parameters
    ----------
    z
        Redshift array.
    xi_0, xi_n
        Modified-propagation parameters.

    Returns
    -------
    jax.Array
        ``log(xi_0 + (1 - xi_0) * exp(-xi_n * log1p(z)))``, same shape as ``z``.
    """
    return jnp.log(xi_0 + (1.0 - xi_0) * jnp.exp(-(xi_n * jnp.log1p(z))))


def gw_distance_ratio(
    redshift: ArrayLike, params: Mapping[str, ArrayLike]
) -> jax.Array:
    r"""The GW-to-EM luminosity-distance ratio :math:`\Xi(z)`.

    :math:`\Xi(z)` of :func:`log_gw_em_ratio` when ``xi_0`` is in ``params``
    (``xi_n`` must then be too), else one: plain propagation.

    Parameters
    ----------
    redshift
        Redshift.
    params
        Hyperparameter values.

    Returns
    -------
    jax.Array
        The ratio, same shape as ``redshift``.
    """
    redshift = jnp.asarray(redshift)
    if "xi_0" not in params:
        return jnp.ones_like(redshift)
    return jnp.exp(log_gw_em_ratio(redshift, params["xi_0"], params["xi_n"]))


def normalized_hubble_parameter(redshift: ArrayLike, omega_m: ArrayLike) -> jax.Array:
    r"""Normalized Hubble parameter $E(z) = H(z)/H_0$ for flat $\Lambda$CDM.

    .. math::

        E(z) = \sqrt{\Omega_m (1 + z)^3 + (1 - \Omega_m)}

    Parameters
    ----------
    redshift
        Redshift array. It must be broadcastable with ``omega_m`` according to
        the usual broadcasting rules.
    omega_m
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

    Parameters
    ----------
    h0
        Hubble constant in km s^-1 Mpc^-1.
    """
    return SPEED_OF_LIGHT / 1000.0 / jnp.asarray(h0)


def distance_and_volume_grid(
    redshift: ArrayLike,
    hubble_constant: ArrayLike,
    omega_m: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    r"""Luminosity distance and differential comoving volume on a redshift grid.

    Flat LCDM. The grid must be sorted ascending and nonnegative.

    Parameters
    ----------
    redshift
        Redshift grid on which both arrays are evaluated. The final axis is
        the grid axis, with shape ``(..., n_grid)``.
    hubble_constant
        Hubble constant $H_0$ in $\mathrm{km\,s^{-1}\,Mpc^{-1}}$. May be an
        array and must be broadcastable with the redshift-dependent terms.
    omega_m
        Matter density parameter. May be an array and must be broadcastable
        with the redshift-dependent terms.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(luminosity_distance, differential_comoving_volume)`` on the grid,
        with the final axis corresponding to ``n_grid`` and any leading axes
        determined by broadcasting. The differential comoving volume is
        all-sky, in ``Mpc^3``.

    Each grid interval is integrated with a
    :data:`GAUSS_LEGENDRE_ORDER`-point Gauss-Legendre rule.
    """
    redshift = jnp.asarray(redshift)
    omega_m = jnp.asarray(omega_m)

    extended = jnp.concatenate(
        [jnp.zeros_like(redshift[..., :1]), redshift],
        axis=-1,
    )
    quadrature_points, quadrature_weights = gauss_legendre_nodes_weights(
        extended[..., :-1], extended[..., 1:], GAUSS_LEGENDRE_ORDER
    )
    interval_integrals = jnp.sum(
        quadrature_weights
        * _inverse_hubble_parameter(quadrature_points, omega_m[..., None]),
        axis=-1,
    )
    integral = jnp.cumsum(interval_integrals, axis=-1)
    comoving_distance_grid = hubble_distance(hubble_constant) * integral
    return (
        (1.0 + redshift) * comoving_distance_grid,
        _comoving_volume_element(
            comoving_distance_grid, redshift, hubble_constant, omega_m
        ),
    )


def _inverse_hubble_parameter(redshift: ArrayLike, omega_m: ArrayLike) -> jax.Array:
    """The integrand :math:`1 / E(z)` of the comoving distance."""
    return 1.0 / normalized_hubble_parameter(redshift, omega_m)


def _comoving_volume_element(
    comoving_distance: jax.Array,
    redshift: ArrayLike,
    hubble_constant: ArrayLike,
    omega_m: ArrayLike,
) -> jax.Array:
    r"""All-sky :math:`4 \pi D_H \chi^2 / E(z)` in :math:`\mathrm{Mpc}^3`."""
    return (
        4.0
        * jnp.pi
        * comoving_distance**2
        * _inverse_hubble_parameter(redshift, omega_m)
        * hubble_distance(hubble_constant)
    )


#: Gauss-Legendre order of the pointwise comoving distance. Each point is one
#: interval ``[0, z]`` reaching ``z`` of order 10, not a grid cell, so it needs
#: far more nodes than :data:`GAUSS_LEGENDRE_ORDER`.
POINTWISE_GAUSS_LEGENDRE_ORDER: int = 48


def comoving_distance(
    redshift: ArrayLike, hubble_constant: ArrayLike, omega_m: ArrayLike
) -> jax.Array:
    r"""Comoving distance in Mpc at each redshift, for flat :math:`\Lambda`CDM.

    Unlike :func:`distance_and_volume_grid` the points need not form a sorted
    grid: each is integrated on its own :math:`[0, z]` with a fixed
    :data:`POINTWISE_GAUSS_LEGENDRE_ORDER`-point rule. The three arguments
    broadcast.

    Parameters
    ----------
    redshift
        Redshift.
    hubble_constant
        Hubble constant in km s^-1 Mpc^-1.
    omega_m
        Matter density parameter.
    """
    redshift = jnp.asarray(redshift)
    points, weights = gauss_legendre_nodes_weights(
        jnp.zeros_like(redshift), redshift, POINTWISE_GAUSS_LEGENDRE_ORDER
    )
    integral = jnp.sum(
        weights * _inverse_hubble_parameter(points, jnp.asarray(omega_m)[..., None]),
        axis=-1,
    )
    return hubble_distance(hubble_constant) * integral


def luminosity_distance(
    redshift: ArrayLike, hubble_constant: ArrayLike, omega_m: ArrayLike
) -> jax.Array:
    r"""Luminosity distance in Mpc, :math:`(1 + z)\,\chi(z)`; see :func:`comoving_distance`.

    Parameters
    ----------
    redshift
        Redshift.
    hubble_constant
        Hubble constant in km s^-1 Mpc^-1.
    omega_m
        Matter density parameter.
    """
    return (1.0 + jnp.asarray(redshift)) * comoving_distance(
        redshift, hubble_constant, omega_m
    )


def differential_comoving_volume(
    redshift: ArrayLike, hubble_constant: ArrayLike, omega_m: ArrayLike
) -> jax.Array:
    r"""All-sky :math:`\mathrm{d}V_c/\mathrm{d}z` in :math:`\mathrm{Mpc}^3`.

    Parameters
    ----------
    redshift
        Redshift.
    hubble_constant
        Hubble constant in km s^-1 Mpc^-1.
    omega_m
        Matter density parameter.
    """
    return _comoving_volume_element(
        comoving_distance(redshift, hubble_constant, omega_m),
        redshift,
        hubble_constant,
        omega_m,
    )
