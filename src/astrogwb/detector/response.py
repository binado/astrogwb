r"""The response of a cross-correlation network to one source.

A stochastic search weights each detector pair optimally,
:math:`w_{ab} = \gamma_{ab} / (S_a S_b)` with :math:`\gamma_{ab}` the isotropic
ORF (:func:`~astrogwb.detector.effective_psd.pair_terms`). A single source does
not contribute its sky-averaged power to that estimator: it contributes its
power times the realised network response

.. math::

    r(\hat n, \psi, \iota; f) = \frac{\sum_{a<b} w_{ab}\,
        \gamma_{ab}(\hat n, \psi, \iota; f)}{\sum_{a<b} w_{ab}\, \gamma_{ab}(f)},

with :math:`\gamma_{ab}(\hat n, \psi, \iota; f)` the per-source
:func:`~astrogwb.detector.overlap.overlap_reduction_function`. Its mean over an
isotropic sky, uniform polarization and uniform :math:`\cos\iota` is one at every
frequency, and its scatter adds to the shot noise. It includes the inclination,
replacing the quadrupolar factor :math:`g(\iota)` normalized to its mean.

Sky position and orientation are independent of distance and masses, so draws
of :math:`r` enter the catalog covariance as a Hadamard product with the second
moment of the power
(:class:`~astrogwb.inference.likelihood.Network`).

:func:`pair_response` keeps the pairs apart: the per-source ORF
:math:`\gamma_{ab}(\hat n, \psi, \iota; f)` of every pair, unnormalized, with
isotropic mean :math:`\gamma_{ab}(f)`. It is the response of the stacked
per-pair data, whose mean is :math:`\gamma_{ab} S_h`, and does not divide by
:math:`\gamma_{ab}`, which crosses zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .effective_psd import pair_terms
from .geometry import Detector
from .overlap import overlap_reduction_function

__all__ = [
    "network_response",
    "pair_response",
    "pair_response_moment",
    "sample_network_response",
    "sample_pair_response",
]


def network_response(
    frequencies: ArrayLike,
    detectors: Sequence[Detector],
    right_ascension: ArrayLike,
    declination: ArrayLike,
    polarization_angle: ArrayLike,
    cos_inclination: ArrayLike,
    *,
    gmst: ArrayLike = 0.0,
) -> jax.Array:
    r"""The network response :math:`r(\hat n, \psi, \iota; f)` to one source.

    Normalized to unit isotropic mean at every frequency (see the module
    docstring). Bins where no pair has a finite weight return zero; they carry
    no signal-to-noise either.

    Parameters
    ----------
    frequencies
        Frequencies in Hz, ``(F,)``.
    detectors
        Two or more detectors with sensitivities.
    right_ascension, declination, polarization_angle, cos_inclination
        The celestial source angles in radians of
        :func:`~astrogwb.detector.overlap.overlap_reduction_function`,
        broadcasting to ``(...)``.
    gmst
        Greenwich mean sidereal time in radians.

    Returns
    -------
    jax.Array
        Shape ``(..., F)``.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    terms = pair_terms(frequencies, detectors)
    weights = np.stack([term.weight for term in terms])
    denominator = np.sum([term.weight * term.overlap for term in terms], axis=0)
    numerator = jnp.sum(
        weights
        * pair_response(
            frequencies,
            detectors,
            right_ascension,
            declination,
            polarization_angle,
            cos_inclination,
            gmst=gmst,
        ),
        axis=-2,
    )
    usable = denominator > 0.0
    return jnp.where(usable, numerator / np.where(usable, denominator, 1.0), 0.0)


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


def sample_network_response(
    key: jax.Array,
    num_draws: int,
    frequencies: ArrayLike,
    detectors: Sequence[Detector],
) -> jax.Array:
    r""":func:`network_response` at isotropic draws, ``(num_draws, F)``.

    The direction is uniform on the sphere, :math:`\psi` uniform on
    :math:`[0, \pi)` and :math:`\cos\iota` uniform on :math:`[-1, 1]`. Pass the
    likelihood's frequency grid to use the draws as a
    :class:`~astrogwb.inference.likelihood.Network` ``response``.

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
    return network_response(frequencies, detectors, *_isotropic_angles(key, num_draws))


def sample_pair_response(
    key: jax.Array,
    num_draws: int,
    frequencies: ArrayLike,
    detectors: Sequence[Detector],
) -> jax.Array:
    r""":func:`pair_response` at isotropic draws, ``(num_draws, P, F)``.

    The angles are those :func:`sample_network_response` draws from the same
    ``key``.

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


def pair_response_moment(
    key: jax.Array,
    num_draws: int,
    frequencies: ArrayLike,
    detectors: Sequence[Detector],
    *,
    chunk_size: int = 4096,
) -> jax.Array:
    r"""The isotropic second moment of :func:`pair_response`, ``(P, F, P, F)``.

    :math:`\langle \gamma_{ab}(\hat n, \psi, \iota; f_k)\,
    \gamma_{cd}(\hat n, \psi, \iota; f_l) \rangle` over the ``num_draws``
    isotropic draws of :func:`sample_pair_response` from the same ``key``.
    The draws are reduced ``chunk_size`` at a time, so the ``(L, P, F)`` array
    is never formed; the result does not depend on ``chunk_size``. A sample
    second moment, it is symmetric positive semidefinite.

    Parameters
    ----------
    key
        JAX random key.
    num_draws
        Number of isotropic draws.
    frequencies
        Frequency grid in Hz, ``(F,)``.
    detectors
        Detectors in the network.
    chunk_size
        Draws reduced at once, bounding peak memory at ``chunk_size * P * F``.

    Returns
    -------
    jax.Array
        Shape ``(P, F, P, F)``, pairs in the order of :func:`pair_response`.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    detectors = list(detectors)
    chunk_size = min(chunk_size, num_draws)
    num_chunks = -(-num_draws // chunk_size)
    padding = num_chunks * chunk_size - num_draws
    angles = [
        jnp.pad(angle, (0, padding)).reshape(num_chunks, chunk_size)
        for angle in _isotropic_angles(key, num_draws)
    ]
    # Padded draws are zero-weighted, so they add nothing to the sum.
    counted = (jnp.arange(num_chunks * chunk_size) < num_draws).reshape(
        num_chunks, chunk_size
    )

    def accumulate(
        total: jax.Array, chunk: tuple[jax.Array, ...]
    ) -> tuple[jax.Array, None]:
        *chunk_angles, weight = chunk
        draws = pair_response(frequencies, detectors, *chunk_angles)
        flat = jnp.reshape(draws, (chunk_size, -1)) * weight[:, None]
        return total + flat.T @ flat, None

    shape = (len(detectors) * (len(detectors) - 1) // 2, frequencies.size)
    size = shape[0] * shape[1]
    total, _ = jax.lax.scan(
        accumulate, jnp.zeros((size, size)), (*angles, counted.astype(float))
    )
    # Symmetric to round-off only; made exact for the Cholesky downstream.
    moment = 0.5 * (total + total.T) / num_draws
    return jnp.reshape(moment, (*shape, *shape))
