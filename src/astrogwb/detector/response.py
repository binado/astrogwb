r"""The response of a cross-correlation network to one source.

A single source does not contribute its sky-averaged power to a pair's cross
correlation: it contributes its power times the per-source ORF
:math:`\gamma_{ab}(\hat n, \psi, \iota; f)`
(:func:`~astrogwb.detector.overlap.point_source_overlap_reduction_function`), whose mean
over an isotropic sky, uniform polarization and uniform :math:`\cos\iota` is
the isotropic ORF :math:`\gamma_{ab}(f)`. Its scatter adds to the shot noise.

:func:`pair_response` stacks the per-source ORF of every pair, unnormalized:
it is the response of the stacked per-pair data, whose mean is
:math:`\gamma_{ab} S_h`, and does not divide by :math:`\gamma_{ab}`, which
crosses zero. The connected covariance of the data is a sum over Stokes
pairs :math:`\mu\nu` of the population's source moments
:math:`\langle \mu \nu' \rangle` times the network's generalized ORFs
:math:`\Gamma^{\mu\nu}` (:func:`generalized_overlap_reduction_function`):
the sky average of the pair responses to each Stokes parameter. The
polarization angle and the inclination belong to the source moments, not to
the ORFs.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from astrogwb.waveform.intensity import StokesPair

from .config import Network, map_over_pairs, network_pairs
from .overlap import point_source_overlap_reduction_function, stokes_cross_power

__all__ = [
    "generalized_overlap_reduction_function",
    "pair_response",
]

#: The prefactor of the generalized ORFs.
_GENERALIZED_ORF_PREFACTOR = 4.0


def pair_response(
    frequencies: ArrayLike,
    network: Network,
    right_ascension: ArrayLike,
    declination: ArrayLike,
    polarization_angle: ArrayLike,
    cos_inclination: ArrayLike,
    *,
    gmst: ArrayLike = 0.0,
) -> jax.Array:
    r"""The per-source ORF of every pair of ``network``, ``(..., P, F)``.

    :func:`~astrogwb.detector.overlap.point_source_overlap_reduction_function` of each pair
    :math:`a < b`, in the order of
    :func:`~astrogwb.detector.network_pairs`, stacked on the
    second-to-last axis. Its isotropic mean is the isotropic ORF of each pair;
    it is not normalized by it (see the module docstring), and does not depend
    on the noise curves.

    Parameters
    ----------
    frequencies
        Frequencies in Hz, ``(F,)``.
    network
        Two or more detectors keyed by name.
    right_ascension, declination, polarization_angle, cos_inclination
        The celestial source angles in radians of
        :func:`~astrogwb.detector.overlap.point_source_overlap_reduction_function`,
        broadcasting to ``(...)``.
    gmst
        Greenwich mean sidereal time in radians.

    Returns
    -------
    jax.Array
        Shape ``(..., P, F)`` with ``P = n (n - 1) / 2`` for ``n`` detectors.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    values = map_over_pairs(
        lambda first, second: point_source_overlap_reduction_function(
            frequencies,
            first,
            second,
            right_ascension,
            declination,
            polarization_angle,
            cos_inclination,
            gmst=gmst,
        ),
        network,
    )
    return jnp.stack(tuple(values.values()), axis=-2)


def _isotropic_angles(key: jax.Array, num_draws: int) -> tuple[jax.Array, ...]:
    r"""Isotropic direction, uniform :math:`\psi` and :math:`\cos\iota`, ``(num_draws,)`` each."""
    keys = jax.random.split(key, 4)
    shape = (num_draws,)
    return (
        jax.random.uniform(keys[1], shape, maxval=2.0 * jnp.pi),
        jnp.arcsin(jax.random.uniform(keys[0], shape, minval=-1.0, maxval=1.0)),
        jax.random.uniform(keys[2], shape, maxval=jnp.pi),
        jax.random.uniform(keys[3], shape, minval=-1.0, maxval=1.0),
    )


#: Unit radiation-frame Stokes vectors (I, Q, U, V).
_UNIT_STOKES = {name: np.eye(4)[index] for index, name in enumerate("IQUV")}


def generalized_overlap_reduction_function(
    key: jax.Array,
    num_draws: int,
    frequencies: ArrayLike,
    network: Network,
    *,
    components: Sequence[StokesPair] = ("II", "QQ", "UU", "VV"),
    chunk_size: int = 4096,
) -> dict[StokesPair, jax.Array]:
    r"""The generalized ORFs :math:`\Gamma^{\mu\nu}_{ab,cd}(f_k, f_l)`, each ``(P, F, P, F)``.

    .. math::

        \Gamma^{\mu\nu}_{ab,cd}(f, f') = 4 \int \frac{d^2\hat n}{4\pi}
            F^\mu_{(ab)}(f, \hat n)\, F^\nu_{(cd)}(f', \hat n),

    with :math:`F^\mu_{(ab)}` the response of pair :math:`ab` to unit Stokes
    parameter :math:`\mu` (:func:`~astrogwb.detector.overlap.stokes_cross_power`)
    at zero polarization angle; the polarization angle and the inclination
    belong to the source moments :math:`\langle \mu \nu' \rangle` they
    multiply. The sky average runs over ``num_draws`` isotropic directions
    from ``key``. Each diagonal component :math:`\Gamma^{\mu\mu}` is a sum of
    outer products, symmetric positive semidefinite, and none depends on
    ``chunk_size``; :math:`\Gamma^{VI}` is :math:`\Gamma^{IV}` with
    :math:`(ab, f)` and :math:`(cd, f')` swapped.

    Multiply by :data:`~astrogwb.detector.overlap.NORMALIZED_ORF_SCALE` squared
    for the units of the per-pair data, whose mean carries the normalized
    isotropic ORF.

    Parameters
    ----------
    key
        JAX random key.
    num_draws
        Number of isotropic directions.
    frequencies
        Frequency grid in Hz, ``(F,)``.
    network
        Detectors keyed by name.
    components
        The Stokes pairs :math:`\mu\nu` to compute.
    chunk_size
        Directions reduced at once, bounding peak memory at
        ``S * chunk_size * P * F`` for ``S`` distinct Stokes parameters.

    Returns
    -------
    dict
        :math:`\Gamma^{\mu\nu}` keyed by ``components``, pairs in the order of
        :func:`pair_response`.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    components = tuple(dict.fromkeys(components))
    stokes = sorted({name for pair in components for name in pair}, key="IQUV".index)
    index = {name: position for position, name in enumerate(stokes)}
    chunk_size = min(chunk_size, num_draws)
    num_chunks = -(-num_draws // chunk_size)
    padding = num_chunks * chunk_size - num_draws
    right_ascension, declination, _, _ = _isotropic_angles(key, num_draws)
    angles = [
        jnp.pad(angle, (0, padding)).reshape(num_chunks, chunk_size)
        for angle in (right_ascension, declination)
    ]
    # Padded draws are zero-weighted, so they add nothing to the sum.
    counted = (jnp.arange(num_chunks * chunk_size) < num_draws).reshape(
        num_chunks, chunk_size
    )
    unit_stokes = jnp.asarray(np.stack([_UNIT_STOKES[name] for name in stokes]))
    first, second = (
        jnp.asarray([index[pair[position]] for pair in components])
        for position in (0, 1)
    )

    def responses(ra: jax.Array, dec: jax.Array) -> jax.Array:
        r""":math:`F^\mu_{ab}` of every requested Stokes parameter, ``(S, chunk, P F)``."""
        values = map_over_pairs(
            lambda first, second: stokes_cross_power(
                frequencies,
                first,
                second,
                ra,
                dec,
                0.0,
                unit_stokes.T[:, :, None, None],
            ),
            network,
        )
        per_stokes = jnp.stack(tuple(values.values()), axis=-2)
        return jnp.reshape(per_stokes, (len(stokes), chunk_size, -1))

    def accumulate(
        total: jax.Array, chunk: tuple[jax.Array, ...]
    ) -> tuple[jax.Array, None]:
        ra, dec, weight = chunk
        flat = responses(ra, dec) * weight[None, :, None]
        return total + jnp.einsum("gni,gnj->gij", flat[first], flat[second]), None

    shape = (len(network_pairs(network)), frequencies.size)
    size = shape[0] * shape[1]
    total, _ = jax.lax.scan(
        accumulate,
        jnp.zeros((len(components), size, size)),
        (*angles, counted.astype(float)),
    )
    moments = _GENERALIZED_ORF_PREFACTOR * total / num_draws
    result: dict[StokesPair, jax.Array] = {}
    for pair, moment in zip(components, moments, strict=True):
        if pair[0] == pair[1]:
            # Symmetric to round-off only; made exact for the Cholesky downstream.
            moment = 0.5 * (moment + moment.T)
        result[pair] = jnp.reshape(moment, (*shape, *shape))
    return result
