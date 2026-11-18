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
crosses zero. The connected covariance of the data factorizes into the
intensity second moment of the population and the network's connected
overlap, :func:`connected_overlap`: the sky and polarization average of the
pair responses to each Stokes parameter, weighted by the quadrupole's
inclination moments. The inclination scatter belongs to the intensity second
moment, not to the overlap.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from astrogwb.waveform.intensity import quadrupolar_stokes_factors

from .geometry import Detector
from .overlap import overlap_reduction_function, stokes_cross_power

__all__ = [
    "connected_overlap",
    "pair_names",
    "pair_response",
    "sample_pair_response",
]

#: Gauss-Legendre nodes in cos(inclination); the Stokes products are
#: polynomials of degree eight, integrated exactly.
_INCLINATION_NODES = 8

#: The prefactor 4 of the generalized ORFs, and 25 for the normalized isotropic
#: ORF: the response to unit intensity has isotropic mean gamma / 5.
_PAPER_PREFACTOR = 4.0
_NORMALIZED_PREFACTOR = 25.0


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


def _inclination_moment_factor() -> np.ndarray:
    r""":math:`L` with :math:`L L^T = \langle s s^T\rangle_\iota / \langle s_I^2\rangle_\iota`, ``(4, 4)``.

    :math:`s` the quadrupolar Stokes factors over a uniform :math:`\cos\iota`.
    """
    nodes, weights = np.polynomial.legendre.leggauss(_INCLINATION_NODES)
    stokes = np.asarray(quadrupolar_stokes_factors(nodes), dtype=float)
    moment = (stokes * weights) @ stokes.T
    moment /= moment[0, 0]
    values, vectors = np.linalg.eigh(moment)
    return vectors * np.sqrt(np.clip(values, 0.0, None))


def connected_overlap(
    key: jax.Array,
    num_draws: int,
    frequencies: ArrayLike,
    detectors: Sequence[Detector],
    *,
    normalize: bool = True,
    chunk_size: int = 4096,
) -> jax.Array:
    r"""The quadrupole's connected overlap :math:`\Gamma^c_{ab,cd}(f_k, f_l)`, ``(P, F, P, F)``.

    .. math::

        \Gamma^c_{ab,cd} = \Gamma^{II}_{ab,cd}
            + \frac{\Gamma^{QQ}_{ab,cd} + \Gamma^{UU}_{ab,cd}}{71}
            + \frac{69}{71}\Gamma^{VV}_{ab,cd},

    the generalized ORFs :math:`\Gamma^{\mu\nu}_{ab,cd} = 4\langle
    F^\mu_{ab} F^\nu_{cd}\rangle` of each Stokes parameter, weighted by the
    quadrupole's inclination moments relative to :math:`\langle II'\rangle`.
    The pair response :math:`F^\mu_{ab}` to unit Stokes parameter :math:`\mu`
    is :func:`~astrogwb.detector.overlap.stokes_cross_power`. The average runs
    over ``num_draws`` isotropic directions and uniform polarization angles
    from ``key``; the inclination is averaged exactly, and the coefficients
    above emerge from the polarization average. A sum of outer products, the
    result is symmetric positive semidefinite, and does not depend on
    ``chunk_size``.

    Multiplied elementwise by :math:`4\mathcal R\langle I_k I_l\rangle / T`, it
    is the connected covariance of the per-pair data.

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
    normalize
        If true, in the units of the normalized isotropic ORF
        (:func:`~astrogwb.detector.overlap.isotropic_overlap_reduction_function`,
        one for co-located, aligned, orthogonal arms), the ORF the per-pair
        data's mean carries. If false, the paper's :math:`\Gamma^{\mu\nu}`.
    chunk_size
        Draws reduced at once, bounding peak memory at ``4 * chunk_size * P * F``.

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
    right_ascension, declination, polarization_angle, _ = _isotropic_angles(
        key, num_draws
    )
    angles = [
        jnp.pad(angle, (0, padding)).reshape(num_chunks, chunk_size)
        for angle in (right_ascension, declination, polarization_angle)
    ]
    # Padded draws are zero-weighted, so they add nothing to the sum.
    counted = (jnp.arange(num_chunks * chunk_size) < num_draws).reshape(
        num_chunks, chunk_size
    )
    factor = jnp.asarray(_inclination_moment_factor())
    unit_stokes = jnp.eye(4)[:, :, None, None]

    def responses(chunk_angles: list[jax.Array]) -> jax.Array:
        r""":math:`F^\mu_{ab}` of every unit Stokes vector, ``(4, chunk, P, F)``."""
        ra, dec, psi = chunk_angles
        return jnp.stack(
            [
                jnp.stack(
                    [
                        stokes_cross_power(
                            frequencies,
                            detectors[first],
                            detectors[second],
                            ra,
                            dec,
                            psi,
                            stokes,
                        )
                        for first in range(len(detectors))
                        for second in range(first + 1, len(detectors))
                    ],
                    axis=-2,
                )
                for stokes in unit_stokes
            ]
        )

    def accumulate(
        total: jax.Array, chunk: tuple[jax.Array, ...]
    ) -> tuple[jax.Array, None]:
        *chunk_angles, weight = chunk
        weighted = jnp.einsum("mk,mlpf->klpf", factor, responses(chunk_angles))
        flat = jnp.reshape(weighted * weight[None, :, None, None], (4 * chunk_size, -1))
        return total + flat.T @ flat, None

    shape = (len(detectors) * (len(detectors) - 1) // 2, frequencies.size)
    size = shape[0] * shape[1]
    total, _ = jax.lax.scan(
        accumulate, jnp.zeros((size, size)), (*angles, counted.astype(float))
    )
    prefactor = _NORMALIZED_PREFACTOR if normalize else _PAPER_PREFACTOR
    # Symmetric to round-off only; made exact for the Cholesky downstream.
    moment = prefactor * 0.5 * (total + total.T) / num_draws
    return jnp.reshape(moment, (*shape, *shape))
