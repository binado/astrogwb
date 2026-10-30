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
isotropic sky, uniform polarization and uniform :math:`\cos\iota` is approximately one at
every frequency under the analytic spherical-Earth, level-arm approximation.
Native WGS-84 geometry, elevation and tilts change that mean slightly. Its scatter adds to
the shot noise. It carries the inclination too: it replaces the quadrupolar
factor :math:`g(\iota)` normalized to its mean, which a catalog pinned at the
effective inclination stands in for.

Sky position and orientation are independent of distance and masses, so draws
of :math:`r` enter the catalog covariance as a Hadamard product with the second
moment of the power
(:class:`~astrogwb.inference.likelihood.Network`).

:func:`pair_response` keeps the pairs apart: the per-source ORF
:math:`\gamma_{ab}(\hat n, \psi, \iota; f)` of every pair, unnormalized, with
approximate isotropic mean :math:`\gamma_{ab}(f)`. It is the response of the stacked
per-pair data, whose mean is :math:`\gamma_{ab} S_h`, and does not divide by
:math:`\gamma_{ab}`, which crosses zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ._types import DetectorSpec
from .effective_psd import pair_terms
from .overlap import overlap_reduction_function

__all__ = [
    "network_response",
    "pair_response",
    "sample_network_response",
    "sample_pair_response",
]


def network_response(
    frequencies: ArrayLike,
    detectors: Sequence[DetectorSpec],
    right_ascension: ArrayLike,
    declination: ArrayLike,
    polarization_angle: ArrayLike,
    cos_inclination: ArrayLike,
    *,
    gmst: ArrayLike = 0.0,
) -> jax.Array:
    r"""The network response :math:`r(\hat n, \psi, \iota; f)` to one source.

    Normalized to approximately unit isotropic mean at every frequency (see the module
    docstring). Bins where no pair has a finite weight return zero; they carry
    no signal-to-noise either.

    Parameters
    ----------
    frequencies:
        Frequencies in Hz, ``(F,)``.
    detectors:
        Two or more packaged names or owned detectors with sensitivities.
    right_ascension, declination, polarization_angle, cos_inclination:
        The celestial source angles in radians of
        :func:`~astrogwb.detector.overlap.overlap_reduction_function`,
        broadcasting to ``(...)``.

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
    detectors: Sequence[DetectorSpec],
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
    second-to-last axis. Its isotropic mean approximates the analytic ORF of each pair;
    it is not normalized by it (see the module docstring), and does not depend
    on the noise curves.

    Parameters
    ----------
    frequencies:
        Frequencies in Hz, ``(F,)``.
    detectors:
        Two or more detectors.
    right_ascension, declination, polarization_angle, cos_inclination:
        The celestial source angles in radians of
        :func:`~astrogwb.detector.overlap.overlap_reduction_function`,
        broadcasting to ``(...)``.

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
    detectors: Sequence[DetectorSpec],
) -> jax.Array:
    r""":func:`network_response` at isotropic draws, ``(num_draws, F)``.

    The direction is uniform on the sphere, :math:`\psi` uniform on
    :math:`[0, \pi)` and :math:`\cos\iota` uniform on :math:`[-1, 1]`. Pass the
    likelihood's frequency grid to use the draws as a
    :class:`~astrogwb.inference.likelihood.Network` ``response``.
    """
    return network_response(frequencies, detectors, *_isotropic_angles(key, num_draws))


def sample_pair_response(
    key: jax.Array,
    num_draws: int,
    frequencies: ArrayLike,
    detectors: Sequence[DetectorSpec],
) -> jax.Array:
    r""":func:`pair_response` at isotropic draws, ``(num_draws, P, F)``.

    The angles are those :func:`sample_network_response` draws from the same
    ``key``, so combining these draws with the optimal pair weights gives its
    draws.
    """
    return pair_response(frequencies, detectors, *_isotropic_angles(key, num_draws))
