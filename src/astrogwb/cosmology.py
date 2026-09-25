"""Cosmology helpers for the source-population models.

These functions are pure and JAX-traceable: they accept anything JAX accepts
(:data:`jax.typing.ArrayLike` -- Python scalars, NumPy arrays, JAX arrays,
concrete or traced) and always return :class:`jax.Array`, so the same code runs
at catalog-build time and inside the jitted NUTS model.

The grid-based helpers are traceable because they evaluate on the exact
``z_grid`` array they are given: no static Python scalars (``max_redshift`` /
``n_grid``) need to be extracted from traced values inside the jitted NUTS
model. A population model binds its window and grid size as construction
settings and rebuilds the grid from them; see
:mod:`astrogwb.populations.bns_madau_dickinson`.
"""

from __future__ import annotations

from typing import overload

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.constants import MPC_IN_METERS, SECONDS_PER_YEAR, SPEED_OF_LIGHT
from astrogwb.utils import mapped_gauss_legendre_rule

#: Fixed Gauss-Legendre quadrature order used by the grid helpers. 4 nodes per
#: interval reach near machine precision for the smooth flat-LCDM integrand
#: 1/E(z).
GAUSS_LEGENDRE_ORDER: int = 4


@overload
def hubble_constant_si(h0_km_s_mpc: float) -> float: ...


@overload
def hubble_constant_si(h0_km_s_mpc: jax.Array) -> jax.Array: ...


def hubble_constant_si(h0_km_s_mpc: float | jax.Array) -> float | jax.Array:
    """Convert $H_0$ from $\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}$ to SI ($\\mathrm{s^{-1}}$)."""
    return h0_km_s_mpc * 1000.0 / MPC_IN_METERS


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

    The redshift integral is evaluated with a fixed
    :data:`GAUSS_LEGENDRE_ORDER`-point Gauss-Legendre rule within each grid
    interval, via :func:`astrogwb.utils.mapped_gauss_legendre_rule`. The order
    is a module-level Python constant, so no static scalars are extracted from
    traced values. For the smooth flat-LCDM integrand :math:`1/E(z)` this
    reaches near machine precision while keeping the computation a single
    cumulative pass over the grid.
    """
    redshift = jnp.asarray(redshift)
    omega_m = jnp.asarray(omega_m)

    # Promote through float so an integer redshift grid cannot truncate the
    # quadrature nodes to zeros.
    dtype = jnp.result_type(redshift, float)

    extended = jnp.concatenate(
        [jnp.zeros_like(redshift[..., :1]), redshift],
        axis=-1,
    )
    quadrature_points, quadrature_weights = mapped_gauss_legendre_rule(
        GAUSS_LEGENDRE_ORDER,
        extended[..., :-1],
        extended[..., 1:],
        dtype=dtype,
    )
    e_quadrature = normalized_hubble_parameter(
        redshift=quadrature_points, omega_m=omega_m[..., None]
    )
    interval_integrals = jnp.sum(quadrature_weights / e_quadrature, axis=-1)
    integral = jnp.cumsum(interval_integrals, axis=-1)
    inv_e = 1.0 / normalized_hubble_parameter(redshift=redshift, omega_m=omega_m)
    comoving_distance = hubble_distance(hubble_constant) * integral
    luminosity_distance = (1.0 + redshift) * comoving_distance
    differential_comoving_volume = (
        4.0 * jnp.pi * comoving_distance**2 * inv_e * hubble_distance(hubble_constant)
    )
    return luminosity_distance, differential_comoving_volume


def hubble_time_gyr(h0: ArrayLike) -> jax.Array:
    r"""Hubble time $1 / H_0$ in Gyr, with $H_0$ in $\mathrm{km\,s^{-1}\,Mpc^{-1}}$."""
    return 1.0 / (hubble_constant_si(jnp.asarray(h0)) * SECONDS_PER_YEAR * 1e9)


def _lambda_matter_ratio(omega_m: ArrayLike) -> tuple[jax.Array, jax.Array]:
    r"""$\sqrt{\Omega_\Lambda}$ and $\sqrt{\Omega_\Lambda / \Omega_m}$ for flat $\Lambda$CDM."""
    omega_m = jnp.asarray(omega_m)
    sqrt_omega_lambda = jnp.sqrt(1.0 - omega_m)
    return sqrt_omega_lambda, sqrt_omega_lambda / jnp.sqrt(omega_m)


def lookback_time(
    redshift: ArrayLike, hubble_constant: ArrayLike, omega_m: ArrayLike
) -> jax.Array:
    r"""Lookback time in Gyr for flat $\Lambda$CDM without radiation.

    The closed form of $t_L(z) = t_H \int_0^z \mathrm{d}z' / [(1+z') E(z')]$:

    .. math::

        t_L(z) = \frac{2 t_H}{3\sqrt{\Omega_\Lambda}} \left[
            \operatorname{arsinh}\sqrt{\Omega_\Lambda/\Omega_m}
            - \operatorname{arsinh}\left(\sqrt{\Omega_\Lambda/\Omega_m}
                \,(1+z)^{-3/2}\right)\right],

    with $t_H = 1/H_0$ and $\Omega_\Lambda = 1 - \Omega_m$, which must be
    positive. As $z \to \infty$ it tends to the age of the universe.
    Broadcasts ``redshift``, ``hubble_constant`` and ``omega_m``.
    """
    sqrt_omega_lambda, ratio = _lambda_matter_ratio(omega_m)
    scale = 2.0 * hubble_time_gyr(hubble_constant) / (3.0 * sqrt_omega_lambda)
    return scale * (
        jnp.arcsinh(ratio) - jnp.arcsinh(ratio * (1.0 + jnp.asarray(redshift)) ** -1.5)
    )


def redshift_at_lookback_time(
    time: ArrayLike, hubble_constant: ArrayLike, omega_m: ArrayLike
) -> jax.Array:
    r"""Invert :func:`lookback_time`: redshift at a lookback time in Gyr.

    .. math::

        1 + z = \left[\frac{\sqrt{\Omega_\Lambda/\Omega_m}}{\sinh A}\right]^{2/3},
        \qquad
        A = \operatorname{arsinh}\sqrt{\Omega_\Lambda/\Omega_m}
            - \frac{3\sqrt{\Omega_\Lambda}}{2}\,\frac{t}{t_H}.

    Returns ``inf`` at and beyond the age of the universe ($A \le 0$). The
    masking is gradient-safe: the out-of-range entries never feed a ``nan``
    into the backward pass.
    """
    sqrt_omega_lambda, ratio = _lambda_matter_ratio(omega_m)
    angle = jnp.arcsinh(ratio) - 1.5 * sqrt_omega_lambda * jnp.asarray(
        time
    ) / hubble_time_gyr(hubble_constant)
    in_range = angle > 0.0
    safe_angle = jnp.where(in_range, angle, 1.0)
    redshift = (ratio / jnp.sinh(safe_angle)) ** (2.0 / 3.0) - 1.0
    return jnp.where(in_range, redshift, jnp.inf)
