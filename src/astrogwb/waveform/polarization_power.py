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
    "polarization_stokes",
    "quadrupolar_stokes",
    "quadrupolar_stokes_factors",
]


def polarization_power(plus: ArrayLike, cross: ArrayLike) -> jax.Array:
    """Reduce ``(n, F)`` complex polarization arrays to ``(F, n)`` float64 power.

    Returns ``|h+|^2 + |hx|^2`` per sample and frequency, transposed to the
    on-disk ``(frequency, sample)`` layout.
    """
    power = jnp.abs(jnp.asarray(plus)) ** 2 + jnp.abs(jnp.asarray(cross)) ** 2
    return power.T.astype(jnp.float64)


def polarization_stokes(plus: ArrayLike, cross: ArrayLike) -> jax.Array:
    r"""Reduce ``(n, F)`` complex polarizations to ``(4, F, n)`` Stokes parameters.

    :math:`(I, Q, U, V)` as in the module docstring, each transposed to the
    ``(frequency, sample)`` layout of :func:`polarization_power`, whose output
    is the :math:`I` slice.

    Parameters
    ----------
    plus, cross:
        :math:`\tilde h_+` and :math:`\tilde h_\times`, ``(n, F)``.

    Returns
    -------
    jax.Array
        ``(4, F, n)`` float64.
    """
    plus = jnp.asarray(plus)
    cross = jnp.asarray(cross)
    plus_power = jnp.abs(plus) ** 2
    cross_power = jnp.abs(cross) ** 2
    correlation = 2.0 * jnp.conj(plus) * cross
    stokes = jnp.stack(
        [
            plus_power + cross_power,
            plus_power - cross_power,
            jnp.real(correlation),
            jnp.imag(correlation),
        ]
    )
    return jnp.swapaxes(stokes, 1, 2).astype(jnp.float64)


def quadrupolar_stokes(amplitude: ArrayLike, inclination: ArrayLike) -> jax.Array:
    r"""Stokes parameters ``(4, F, n)`` of the quadrupolar :math:`(2, 2)` mode.

    For a pre-polarization strain :math:`h_0` of amplitude :math:`A` and the
    LAL convention Ripple follows, :math:`h_+ = -\tfrac{1 + c^2}{2}\, h_0` and
    :math:`h_\times = i c\, h_0` with :math:`c = \cos\iota`, so

    .. math::

        I = A^2 g(\iota), \quad
        Q = A^2 \frac{(1 - c^2)^2}{4}, \quad U = 0, \quad
        V = -A^2 c\, (1 + c^2),

    with :math:`g = ((1 + c^2)/2)^2 + c^2`. The phase of :math:`h_0` cancels, so
    the amplitude alone determines them. :math:`I` is formed exactly as the
    quadrupolar power is, ``A**2 * g``, so the two agree bit for bit.

    Parameters
    ----------
    amplitude:
        :math:`A`, ``(n, F)``.
    inclination:
        :math:`\iota` in radians, ``(n,)``.

    Returns
    -------
    jax.Array
        ``(4, F, n)`` float64.
    """
    factors = quadrupolar_stokes_factors(jnp.cos(jnp.asarray(inclination)))
    stokes = jnp.asarray(amplitude)[None] ** 2 * factors[:, :, None]
    return jnp.swapaxes(stokes, 1, 2).astype(jnp.float64)


def quadrupolar_stokes_factors(cos_inclination: ArrayLike) -> jax.Array:
    r"""The :math:`(2, 2)` Stokes parameters per unit :math:`A^2`, ``(4, ...)``.

    :math:`(g, (1 - c^2)^2/4, 0, -c(1 + c^2))` with :math:`c = \cos\iota`, the
    inclination dependence of :func:`quadrupolar_stokes`, for any shape of
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
