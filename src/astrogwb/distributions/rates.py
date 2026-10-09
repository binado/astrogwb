r"""Madau-Dickinson source-frame merger-rate densities.

The canonical home for the rate shapes the distribution classes in
:mod:`astrogwb.distributions` build on. NumPyro-free by design: it must stay
importable without loading NumPyro, so a caller that only wants
:math:`\psi(z)` -- an analytic spectrum, a test oracle -- does not inherit a
NumPyro import from the distribution subpackage. That is also why
:mod:`astrogwb.distributions` re-exports nothing from its package root.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import differential_comoving_volume


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
    (Leuven Gravity Institute, BSD-3-Clause), restated here so that neither the
    distribution classes nor the importance-weighting reference callbacks
    depend on the optional ``simulation`` extra.
    """
    one_plus_z = 1.0 + jnp.asarray(redshift)
    # Promoted rather than added as-is: `ArrayLike` admits `numpy.bool_`, which
    # has no `__neg__`, so the unary minus below is not well typed otherwise.
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
    redshift, weights:
        Quadrature nodes and weights of :math:`\int \mathrm{d}z`, shape ``(Z,)``.
    merger_rate:
        The source-frame rate :math:`\psi` at ``redshift``, in
        :math:`\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}`, shape ``(Z,)``.
    hubble_constant, omega_m:
        Flat :math:`\Lambda`CDM cosmology, as in :mod:`astrogwb.cosmology`.
    """
    redshift = jnp.asarray(redshift)
    volume = differential_comoving_volume(redshift, hubble_constant, omega_m)
    integral = jnp.sum(
        jnp.asarray(weights) * jnp.asarray(merger_rate) / (1.0 + redshift) * volume
    )
    return 1e-9 * integral / SECONDS_PER_YEAR
