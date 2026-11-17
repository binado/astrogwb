r"""Madau-Dickinson source-frame merger-rate densities.

Kept free of NumPyro, so analytic callers of :math:`\psi(z)` do not import it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

if TYPE_CHECKING:
    import numpyro.distributions as dist

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import (
    differential_comoving_volume,
    lookback_time,
    redshift_at_lookback_time,
)
from astrogwb.utils import gauss_legendre_nodes_weights

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


def total_merger_rate(
    redshift: ArrayLike,
    weights: ArrayLike,
    merger_rate: ArrayLike,
    hubble_constant: ArrayLike,
    omega_m: ArrayLike,
) -> jax.Array:
    r"""Total merger rate in mergers per second, by quadrature over given nodes.

    .. math::

        \mathcal{R} = \int \frac{\psi(z)}{1 + z}
            \frac{\mathrm{d}V_c}{\mathrm{d}z}\,\mathrm{d}z
        \approx \sum_j w_j \frac{\psi(z_j)}{1 + z_j}
            \frac{\mathrm{d}V_c}{\mathrm{d}z}(z_j),

    the same quantity as
    :meth:`~astrogwb.distributions.redshift.base.RedshiftDistribution.total_merger_rate`,
    but with no distribution, grid or normalization table: only the rate
    :math:`\psi` at the nodes and the cosmology.

    Parameters
    ----------
    redshift, weights
        Quadrature nodes and weights of :math:`\int \mathrm{d}z`, shape ``(Z,)``.
    merger_rate
        The source-frame rate :math:`\psi` at ``redshift``, in
        :math:`\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}`, shape ``(Z,)``.
    hubble_constant, omega_m
        Flat :math:`\Lambda`CDM cosmology, as in :mod:`astrogwb.cosmology`.
    """
    redshift = jnp.asarray(redshift)
    volume = differential_comoving_volume(redshift, hubble_constant, omega_m)
    integral = jnp.sum(
        jnp.asarray(weights) * jnp.asarray(merger_rate) / (1.0 + redshift) * volume
    )
    return 1e-9 * integral / SECONDS_PER_YEAR


def time_delayed_merger_rate(
    redshift: ArrayLike,
    params: Mapping[str, ArrayLike],
    *,
    formation_rate: SourceFrameDistributionFn,
    delay: dist.Distribution,
    n_delay_nodes: int = 48,
    maximum_formation_redshift: ArrayLike = 20.0,
) -> jax.Array:
    r"""Merger rate :math:`R_m(z)` of ``formation_rate`` delayed by ``delay``.

    Pointwise and shape-agnostic: ``redshift`` need not be a grid, and the
    result has its shape. The rate is rescaled to :math:`R_m(0) = \psi(0)`
    by a separate evaluation at :math:`z = 0`. ``params`` must carry ``H0``
    and ``Omega_m`` and is passed on to ``formation_rate``;
    ``n_delay_nodes`` sets an array shape and so must be a static integer.

    Parameters
    ----------
    redshift
        Redshift.
    params
        Model parameter values.
    formation_rate
        Source-frame formation rate function.
    delay
        Delay-time distribution.
    n_delay_nodes
        Number of delay quadrature nodes.
    maximum_formation_redshift
        Upper formation-redshift bound.
    """
    hubble_constant, omega_m = params["H0"], params["Omega_m"]
    longest_available = lookback_time(
        maximum_formation_redshift, hubble_constant, omega_m
    )

    def delayed(merger_redshift: jax.Array) -> jax.Array:
        merger_time = lookback_time(merger_redshift, hubble_constant, omega_m)
        # Clip into the support so `cdf` saturates at 0 and 1 instead of warning.
        available_probability = delay.cdf(
            jnp.clip(
                longest_available - merger_time,
                getattr(delay.support, "lower_bound", -jnp.inf),
                getattr(delay.support, "upper_bound", jnp.inf),
            )
        )
        # The weights carry the Jacobian F(tau_avail) / 2, so the probability
        # mass of the available delays is built in.
        quantiles, weights = gauss_legendre_nodes_weights(
            jnp.zeros_like(available_probability),
            available_probability,
            n_delay_nodes,
        )
        # Every node forms at or before the cut-off, which precedes the age of
        # the universe, so no redshift here is infinite.
        formation_redshift = redshift_at_lookback_time(
            merger_time[..., None] + delay.icdf(quantiles), hubble_constant, omega_m
        )
        return jnp.sum(weights * formation_rate(formation_redshift, params), axis=-1)

    redshift = jnp.asarray(redshift)
    local_rate = formation_rate(jnp.zeros(()), params)
    return delayed(redshift) * (local_rate / delayed(jnp.zeros((), redshift.dtype)))
