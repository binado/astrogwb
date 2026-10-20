r"""The response of a cross-correlation network to one source.

A stochastic search weights each detector pair optimally,
:math:`w_{ab} = \gamma_{ab} / (S_a S_b)` with :math:`\gamma_{ab}` the isotropic
ORF (:func:`~astrogwb.detector.effective_psd.effective_psd`). A single source
does not contribute its sky-averaged power to that estimator: it contributes
its power times the realised network response

.. math::

    r(\hat n, \psi, \iota; f) = \frac{\sum_{a<b} w_{ab}\,
        \gamma_{ab}(\hat n, \psi, \iota; f)}{\sum_{a<b} w_{ab}\, \gamma_{ab}(f)},

with :math:`\gamma_{ab}(\hat n, \psi, \iota; f)` the per-source
:func:`~astrogwb.detector.overlap.overlap_reduction_function`. Its mean over an
isotropic sky, uniform polarization and uniform :math:`\cos\iota` is one, so the
expected spectrum is unchanged, but its scatter adds to the shot noise. It
carries the inclination too: :math:`r` replaces the quadrupolar factor
:math:`g(\iota)` normalized to its mean, which the response of a single
co-located aligned L-shaped pair reduces to.

Sky position and orientation are independent of distance and masses, so the
response enters a second moment of the power as one factor,
:func:`response_second_moment`, of draws projected on the search template
(:func:`projected_network_response`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from gwmock_signal.stochastic.overlap import detector_names
from jax.typing import ArrayLike

from astrogwb.frequency import bin_widths

from ._types import DetectorSpec
from .effective_psd import effective_psd
from .overlap import isotropic_overlap_reduction_function, overlap_reduction_function
from .sensitivity import Sensitivity

__all__ = [
    "network_response",
    "projected_network_response",
    "response_nodes",
    "response_second_moment",
    "sample_network_response",
]

#: Inspiral power-law index of the strain spectrum, the default template.
_INSPIRAL_INDEX = -7.0 / 3.0


def _pair_weights(
    frequencies: np.ndarray,
    detectors: Sequence[DetectorSpec],
    sensitivities: Mapping[str, Sensitivity],
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """``(a, b, w_ab, gamma_ab)`` per pair, with non-finite weights zeroed."""
    psds = [sensitivities[name](frequencies) for name in detector_names(detectors)]
    pairs = []
    for a in range(len(detectors)):
        for b in range(a + 1, len(detectors)):
            gamma = isotropic_overlap_reduction_function(
                frequencies, detectors[a], detectors[b]
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                weight = gamma / (psds[a] * psds[b])
            pairs.append((a, b, np.where(np.isfinite(weight), weight, 0.0), gamma))
    return pairs


def network_response(
    frequencies: ArrayLike,
    detectors: Sequence[DetectorSpec],
    sensitivities: Mapping[str, Sensitivity],
    polar_cosine: ArrayLike,
    azimuth: ArrayLike,
    polarization: ArrayLike,
    cos_inclination: ArrayLike,
) -> jax.Array:
    r"""The network response :math:`r(\hat n, \psi, \iota; f)` to one source.

    Normalized to unit isotropic mean at every frequency (see the module
    docstring). Bins where no pair has a finite weight return zero; they carry
    no signal-to-noise either.

    Parameters
    ----------
    frequencies:
        Frequencies in Hz, ``(F,)``.
    detectors:
        Two or more detectors, matched to ``sensitivities`` by public name as in
        :func:`~astrogwb.detector.effective_psd.effective_psd`.
    sensitivities:
        Name -> :class:`~astrogwb.detector.sensitivity.Sensitivity`.
    polar_cosine, azimuth, polarization, cos_inclination:
        The source angles of
        :func:`~astrogwb.detector.overlap.overlap_reduction_function`,
        broadcasting to ``(...)``.

    Returns
    -------
    jax.Array
        Shape ``(..., F)``.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    detectors = list(detectors)
    numerator = jnp.zeros(())
    denominator = np.zeros(frequencies.shape)
    for a, b, weight, gamma in _pair_weights(frequencies, detectors, sensitivities):
        numerator = numerator + jnp.asarray(weight) * overlap_reduction_function(
            frequencies,
            detectors[a],
            detectors[b],
            polar_cosine,
            azimuth,
            polarization,
            cos_inclination,
        )
        denominator = denominator + weight * gamma
    usable = denominator > 0.0
    return jnp.where(usable, numerator / np.where(usable, denominator, 1.0), 0.0)


def sample_network_response(
    key: jax.Array,
    num_draws: int,
    frequencies: ArrayLike,
    detectors: Sequence[DetectorSpec],
    sensitivities: Mapping[str, Sensitivity],
) -> jax.Array:
    r""":func:`network_response` at isotropic draws, ``(num_draws, F)``.

    The direction is uniform on the sphere, :math:`\psi` uniform on
    :math:`[0, \pi)` and :math:`\cos\iota` uniform on :math:`[-1, 1]`.
    """
    keys = jax.random.split(key, 4)
    shape = (num_draws,)
    return network_response(
        frequencies,
        detectors,
        sensitivities,
        polar_cosine=jax.random.uniform(keys[0], shape, minval=-1.0, maxval=1.0),
        azimuth=jax.random.uniform(keys[1], shape, maxval=2.0 * jnp.pi),
        polarization=jax.random.uniform(keys[2], shape, maxval=jnp.pi),
        cos_inclination=jax.random.uniform(keys[3], shape, minval=-1.0, maxval=1.0),
    )


def projected_network_response(
    key: jax.Array,
    num_draws: int,
    frequencies: ArrayLike,
    detectors: Sequence[DetectorSpec],
    sensitivities: Mapping[str, Sensitivity],
    spectrum: ArrayLike | None = None,
) -> jax.Array:
    r"""Isotropic draws of the response projected on the search template, ``(num_draws,)``.

    .. math::

        R = \frac{\sum_f u_f\, r_f}{\sum_f u_f}, \qquad
        u_f = \frac{\mu_f^2\, \Delta f}{S_{\mathrm{eff}, f}^2},

    the SNR weights of the amplitude estimator on the template :math:`\mu`
    (``spectrum``, :math:`f^{-7/3}` by default), with the bin widths of
    :func:`~astrogwb.frequency.bin_widths` and the network
    :func:`~astrogwb.detector.effective_psd.effective_psd`. Every source is
    projected on the template, not on its own spectrum, which is exact where the
    response is flat across the band each source fills.

    The draws are divided by their sample mean, whose expectation is one, so
    :func:`response_second_moment` estimates :math:`1 + \mathrm{cv}^2[R]`
    without the Monte Carlo error of the mean.

    Parameters
    ----------
    key:
        PRNG key of the draws.
    num_draws:
        Number of isotropic draws.
    frequencies, detectors, sensitivities:
        As for :func:`network_response`; ``frequencies`` is the search's full
        grid, ``(F,)``.
    spectrum:
        The template :math:`\mu`, ``(F,)``, typically the expected spectrum.

    Returns
    -------
    jax.Array
        Shape ``(num_draws,)``, sample mean one.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    template = (
        frequencies**_INSPIRAL_INDEX if spectrum is None else np.asarray(spectrum)
    )
    psd = effective_psd(frequencies, detectors, sensitivities)
    weights = np.where(
        np.isfinite(psd),
        template**2 * np.asarray(bin_widths(frequencies)) / psd**2,
        0.0,
    )
    response = sample_network_response(
        key, num_draws, frequencies, detectors, sensitivities
    )
    projected = response @ jnp.asarray(weights / weights.sum())
    return projected / jnp.mean(projected)


def response_second_moment(projected_response: ArrayLike) -> jax.Array:
    r""":math:`\langle R^2 \rangle / \langle R \rangle^2` of equally weighted draws, ``()``.

    The network counterpart of
    :data:`~astrogwb.gwb.importance.INCLINATION_SECOND_MOMENT`, which it
    replaces in a second moment of the power.
    """
    projected_response = jnp.asarray(projected_response)
    return jnp.mean(projected_response**2) / jnp.mean(projected_response) ** 2


def response_nodes(projected_response: ArrayLike, num_nodes: int) -> jax.Array:
    """Compress draws to ``num_nodes`` equally weighted nodes, ``(num_nodes,)``.

    The means of ``num_nodes`` equal-count groups of the sorted draws: the mean
    is kept exactly and the second moment loses only the within-group scatter.
    A compact point set for
    :class:`~astrogwb.inference.likelihood.ImportanceCompoundPoissonLikelihood`,
    whose cost grows with the node count. ``num_nodes`` must divide the number
    of draws.
    """
    ordered = jnp.sort(jnp.asarray(projected_response))
    return jnp.mean(ordered.reshape(num_nodes, -1), axis=1)
