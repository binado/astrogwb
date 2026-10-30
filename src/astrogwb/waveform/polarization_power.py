r"""Reductions of per-source polarizations to power and Stokes parameters.

The Stokes parameters of one source, in its radiation frame at zero
polarization angle, are

.. math::

    I = |h_+|^2 + |h_\times|^2, \qquad Q = |h_+|^2 - |h_\times|^2, \qquad
    U = 2\,\mathrm{Re}(h_+^* h_\times), \qquad V = 2\,\mathrm{Im}(h_+^* h_\times).

:math:`I` is the polarization power. :math:`Q, U, V` carry what the power drops
and what a detector pair needs: :math:`Q` and :math:`U` the linear polarization
and :math:`V` the circular one. One source is fully polarized at every
frequency, :math:`I^2 = Q^2 + U^2 + V^2`, whatever its mode content; only a sum
over sources is partially polarized, :math:`I^2 > Q^2 + U^2 + V^2`. A rotation of the polarization basis by
:math:`\psi` leaves :math:`I` and :math:`V` alone and rotates :math:`(Q, U)` by
:math:`4\psi`; callers apply :math:`\psi` through the antenna patterns instead.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = [
    "polarization_power",
    "quadrupolar_polarizations",
    "quadrupolar_stokes_factors",
]


def polarization_power(plus: ArrayLike, cross: ArrayLike) -> jax.Array:
    """Reduce ``(n, F)`` complex polarization arrays to ``(F, n)`` float64 power.

    Returns ``|h+|^2 + |hx|^2`` per sample and frequency, transposed to the
    on-disk ``(frequency, sample)`` layout.
    """
    power = jnp.abs(jnp.asarray(plus)) ** 2 + jnp.abs(jnp.asarray(cross)) ** 2
    return power.T.astype(jnp.float64)


def quadrupolar_polarizations(
    amplitude: ArrayLike, inclination: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    r"""Polarizations ``(n, F)`` of the quadrupolar :math:`(2, 2)` mode at zero phase.

    :math:`h_+ = -\tfrac{1 + c^2}{2} A` and :math:`h_\times = i c A` with
    :math:`c = \cos\iota`, whose Stokes parameters are :math:`A^2` times
    :func:`quadrupolar_stokes_factors`. The orbital phase of :math:`h_0` is set
    to zero: it cancels in every product :math:`h^*_a h_b` of one source.

    Parameters
    ----------
    amplitude:
        :math:`A`, ``(n, F)``.
    inclination:
        :math:`\iota` in radians, ``(n,)``.

    Returns
    -------
    tuple of jax.Array
        :math:`(\tilde h_+, \tilde h_\times)`, each ``(n, F)`` complex128.
    """
    amplitude = jnp.asarray(amplitude).astype(jnp.complex128)
    cosine = jnp.cos(jnp.asarray(inclination))[:, None]
    return -0.5 * (1.0 + cosine**2) * amplitude, 1j * cosine * amplitude


def quadrupolar_stokes_factors(cos_inclination: ArrayLike) -> jax.Array:
    r"""The :math:`(2, 2)` Stokes parameters per unit :math:`A^2`, ``(4, ...)``.

    :math:`(g, (1 - c^2)^2/4, 0, -c(1 + c^2))` with :math:`c = \cos\iota`, the
    quadrupolar inclination dependence, for any shape of
    ``cos_inclination``. :math:`g` is formed as the quadrupolar power's
    inclination factor is.
    """
    cosine = jnp.asarray(cos_inclination)
    cos_squared = cosine**2
    return jnp.stack(
        [
            ((1.0 + cos_squared) / 2.0) ** 2 + cos_squared,
            (1.0 - cos_squared) ** 2 / 4.0,
            jnp.zeros_like(cosine),
            -cosine * (1.0 + cos_squared),
        ]
    )
