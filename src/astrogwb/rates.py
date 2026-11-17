r"""Madau-Dickinson source-frame merger-rate density and its quadrature total.

Kept free of NumPyro, so analytic callers of :math:`\psi(z)` do not import it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import differential_comoving_volume
from astrogwb.utils import gauss_legendre_nodes_weights

#: A source-frame merger-rate shape :math:`\psi(z)`, called as
#: ``fn(redshift, params)`` and returning an array shaped like ``redshift``.
type SourceFrameDistributionFn = Callable[
    [ArrayLike, Mapping[str, ArrayLike]], jax.Array
]


def madau_dickinson_rate(
    redshift: ArrayLike,
    gamma: ArrayLike,
    kappa: ArrayLike,
    z_peak: ArrayLike,
    local_merger_rate: ArrayLike = 1.0,
) -> jax.Array:
    r"""Madau-like source-frame rate density in ``Gpc^-3 yr^-1``.

    The ``local_merger_rate`` argument sets the value at zero redshift. With
    the default ``local_merger_rate=1.0``, this returns the dimensionless shape
    :math:`\psi(z)` with :math:`\psi(0) = 1`.

    .. math::

        \psi(z) = \mathcal{R}_0 \mathcal{C}\,
            \frac{(1+z)^{\gamma}}{1 + \left(\frac{1+z}{1+z_p}\right)^{\gamma+\kappa}},
        \qquad
        \mathcal{C} = 1 + (1+z_p)^{-(\gamma+\kappa)}.

    Same parametrization as ``gwmock_pop.distributions.madau_dickinson``
    (Leuven Gravity Institute, BSD-3-Clause).

    Parameters
    ----------
    redshift
        Redshift.
    gamma
        Low-redshift rate exponent.
    kappa
        High-redshift rate exponent.
    z_peak
        Redshift of the rate turnover.
    local_merger_rate
        Local merger rate density.
    """
    one_plus_z = 1.0 + jnp.asarray(redshift)
    # Promoted so the unary minus below type-checks (`numpy.bool_` has no `__neg__`).
    exponent = jnp.asarray(gamma) + jnp.asarray(kappa)
    normalization = 1.0 + (1.0 + z_peak) ** (-exponent)
    return jnp.asarray(local_merger_rate) * (
        normalization
        * one_plus_z**gamma
        / (1.0 + (one_plus_z / (1.0 + z_peak)) ** exponent)
    )


def differential_merger_rate(
    redshift: ArrayLike,
    merger_rate: ArrayLike,
    hubble_constant: ArrayLike,
    omega_m: ArrayLike,
) -> jax.Array:
    """Observer-frame merger rate in mergers per second per unit redshift.

    Parameters
    ----------
    redshift
        Redshift at which to evaluate the integrand.
    merger_rate
        Source-frame merger-rate density at ``redshift``, in Gpc^-3 yr^-1.
    hubble_constant, omega_m
        Flat LambdaCDM cosmology, as in :mod:`astrogwb.cosmology`.
    """
    redshift = jnp.asarray(redshift)
    volume = differential_comoving_volume(redshift, hubble_constant, omega_m)
    rate = jnp.asarray(merger_rate) * volume / (1.0 + redshift)
    return 1e-9 * rate / SECONDS_PER_YEAR


def total_merger_rate(
    minimum_redshift: ArrayLike,
    maximum_redshift: ArrayLike,
    merger_rate: Callable[[ArrayLike], ArrayLike],
    hubble_constant: ArrayLike,
    omega_m: ArrayLike,
    *,
    num_nodes: int = 64,
) -> jax.Array:
    """Integrate the observer-frame merger rate with Gauss-Legendre quadrature.

    Parameters
    ----------
    minimum_redshift, maximum_redshift
        Lower and upper redshift bounds. Array bounds broadcast to give
        one integral per interval.
    merger_rate
        Callable returning the source-frame rate density in Gpc^-3 yr^-1
        at the supplied redshifts.
    hubble_constant, omega_m
        Flat LambdaCDM cosmology, as in :mod:`astrogwb.cosmology`.
    num_nodes
        Number of Gauss-Legendre nodes per interval. Static under JIT.

    Returns
    -------
    jax.Array
        Observer-frame rate in mergers per second.
    """
    redshift, weights = gauss_legendre_nodes_weights(
        minimum_redshift, maximum_redshift, num_nodes
    )
    integrand = differential_merger_rate(
        redshift, merger_rate(redshift), hubble_constant, omega_m
    )
    return jnp.sum(weights * integrand, axis=-1)
