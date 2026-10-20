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
isotropic sky, uniform polarization and uniform :math:`\cos\iota` is one at
every frequency, so the expected spectrum is unchanged, but its scatter adds to
the shot noise. It carries the inclination too: it replaces the quadrupolar
factor :math:`g(\iota)` normalized to its mean, which a catalog pinned at the
effective inclination stands in for.

Sky position and orientation are independent of distance and masses, so draws
of :math:`r` enter the catalog covariance as a Hadamard product with the second
moment of the power
(:class:`~astrogwb.inference.likelihood.Network`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ._types import DetectorSpec
from .effective_psd import pair_terms
from .overlap import overlap_reduction_function
from .sensitivity import Sensitivity

__all__ = ["network_response", "sample_network_response"]


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
    for pair in pair_terms(frequencies, detectors, sensitivities):
        numerator = numerator + jnp.asarray(pair.weight) * overlap_reduction_function(
            frequencies,
            detectors[pair.first],
            detectors[pair.second],
            polar_cosine,
            azimuth,
            polarization,
            cos_inclination,
        )
        denominator = denominator + pair.weight * pair.overlap
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
    :math:`[0, \pi)` and :math:`\cos\iota` uniform on :math:`[-1, 1]`. Pass the
    likelihood's frequency grid to use the draws as a
    :class:`~astrogwb.inference.likelihood.Network` ``response``.
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
