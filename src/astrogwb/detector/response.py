r"""The response of a cross-correlation network to one source.

A single source does not contribute its sky-averaged power to a pair's cross
correlation: it contributes its power times the per-source ORF
:math:`\gamma_{ab}(\hat n, \psi, \iota; f)`
(:func:`~astrogwb.detector.overlap.overlap_reduction_function`), whose mean
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

from .geometry import Detector
from .overlap import overlap_reduction_function, stokes_cross_power

__all__ = [
    "generalized_overlap_reduction_function",
    "pair_names",
    "pair_response",
    "sample_pair_response",
]

#: The prefactor of the generalized ORFs.
_GENERALIZED_ORF_PREFACTOR = 4.0


def pair_names(detectors: Sequence[Detector]) -> tuple[tuple[str, str], ...]:
    """The sorted name pair of every pair :math:`a < b`, ``(P,)``.

    In the order of :func:`pair_response` and
    :func:`~astrogwb.detector.effective_psd.pair_terms`, the keys of
    per-pair data.

    Parameters
    ----------
    detectors
        Detectors in the network, with distinct names.

    Raises
    ------
    ValueError
        If two detectors share a name.
    """
    names = [detector.name for detector in detectors]
    if len(set(names)) != len(names):
        raise ValueError(f"detector names must be distinct, got {names}")
    return tuple(
        (min(names[first], names[second]), max(names[first], names[second]))
        for first in range(len(names))
        for second in range(first + 1, len(names))
    )


def pair_response(
    frequencies: ArrayLike,
    detectors: Sequence[Detector],
    right_ascension: ArrayLike,
    declination: ArrayLike,
    polarization_angle: ArrayLike,
    cos_inclination: ArrayLike,
    *,
    gmst: ArrayLike = 0.0,
) -> jax.Array:
    r"""The per-source ORF of every pair of ``detectors``, ``(..., P, F)``.

    :func:`~astrogwb.detector.overlap.overlap_reduction_function` of each pair
    :math:`a < b`, in the order of
    :func:`~astrogwb.detector.effective_psd.pair_terms`, stacked on the
    second-to-last axis. Its isotropic mean is the isotropic ORF of each pair;
    it is not normalized by it (see the module docstring), and does not depend
    on the noise curves.

    Parameters
    ----------
    frequencies
        Frequencies in Hz, ``(F,)``.
    detectors
        Two or more detectors.
    right_ascension, declination, polarization_angle, cos_inclination
        The celestial source angles in radians of
        :func:`~astrogwb.detector.overlap.overlap_reduction_function`,
        broadcasting to ``(...)``.
    gmst
        Greenwich mean sidereal time in radians.

    Returns
    -------
    jax.Array
        Shape ``(..., P, F)`` with ``P = n (n - 1) / 2`` for ``n`` detectors.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    detectors = list(detectors)
    return jnp.stack(
        [
            overlap_reduction_function(
                frequencies,
                detectors[first],
                detectors[second],
                right_ascension,
                declination,
                polarization_angle,
                cos_inclination,
                gmst=gmst,
            )
            for first in range(len(detectors))
            for second in range(first + 1, len(detectors))
        ],
        axis=-2,
    )


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


def sample_pair_response(
    key: jax.Array,
    num_draws: int,
    frequencies: ArrayLike,
    detectors: Sequence[Detector],
) -> jax.Array:
    r""":func:`pair_response` at isotropic draws, ``(num_draws, P, F)``.

    The direction is uniform on the sphere, :math:`\psi` uniform on
    :math:`[0, \pi)` and :math:`\cos\iota` uniform on :math:`[-1, 1]`.

    Parameters
    ----------
    key
        JAX random key.
    num_draws
        Number of random draws.
    frequencies
        Frequency grid in Hz.
    detectors
        Detectors in the network.
    """
    return pair_response(frequencies, detectors, *_isotropic_angles(key, num_draws))


#: Unit radiation-frame Stokes vectors (I, Q, U, V).
_UNIT_STOKES = {name: np.eye(4)[index] for index, name in enumerate("IQUV")}


def generalized_overlap_reduction_function(
    key: jax.Array,
    num_draws: int,
    frequencies: ArrayLike,
    detectors: Sequence[Detector],
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
    detectors
        Detectors in the network.
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
    detectors = list(detectors)
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
        per_stokes = jnp.stack(
            [
                jnp.stack(
                    [
                        stokes_cross_power(
                            frequencies,
                            detectors[a],
                            detectors[b],
                            ra,
                            dec,
                            0.0,
                            unit[:, None, None],
                        )
                        for a in range(len(detectors))
                        for b in range(a + 1, len(detectors))
                    ],
                    axis=-2,
                )
                for unit in unit_stokes
            ]
        )
        return jnp.reshape(per_stokes, (len(stokes), chunk_size, -1))

    def accumulate(
        total: jax.Array, chunk: tuple[jax.Array, ...]
    ) -> tuple[jax.Array, None]:
        ra, dec, weight = chunk
        flat = responses(ra, dec) * weight[None, :, None]
        return total + jnp.einsum("gni,gnj->gij", flat[first], flat[second]), None

    shape = (len(detectors) * (len(detectors) - 1) // 2, frequencies.size)
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
