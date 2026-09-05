r"""Dimensionless merger-rate shapes :math:`\psi(z)`.

The canonical home for the rate shapes shared by the distribution classes in
:mod:`astrogwb.distributions` and the importance-weighting reference callbacks
in :mod:`astrogwb.importance.models`. NumPyro-free by design: it must stay
importable without loading NumPyro, so that the reference callbacks can share
these definitions without inheriting a NumPyro import from the distribution
subpackage. That is also why :mod:`astrogwb.distributions` re-exports nothing
from its package root.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def madau_dickinson_rate(
    redshift: ArrayLike,
    gamma: ArrayLike,
    kappa: ArrayLike,
    z_peak: ArrayLike,
) -> jax.Array:
    r"""Dimensionless Madau-like rate shape :math:`\psi(z)` with :math:`\psi(0) = 1`.

    .. math::

        \psi(z) = \mathcal{C}\,
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
    return (
        normalization
        * one_plus_z**gamma
        / (1.0 + (one_plus_z / (1.0 + z_peak)) ** exponent)
    )
