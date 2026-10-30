"""Setup shared by the posterior notebooks.

The per-network Gaussian data, read from a
:class:`~astrogwb.simulators.NetworkNoise` and cut to an analysis band, and
the shot-noise variants: everything a grid or a sampler needs besides its
catalogs before it evaluates a likelihood. Only NumPy and :mod:`astrogwb` are
imported at module scope, and ``jax.numpy`` inside the functions that use it,
so importing this module leaves the XLA backend alone.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.frequency import frequency_mask
from astrogwb.inference import ImportanceGaussianLikelihood
from astrogwb.simulators import NetworkNoise

__all__ = [
    "network_data",
    "network_pair_data",
    "pad_pairs",
    "shot_noise_variant",
]


def _band(
    frequencies: ArrayLike, minimum_frequency: float, maximum_frequency: float
) -> NDArray[np.bool_]:
    """The bins of ``frequencies`` inside ``[minimum_frequency, maximum_frequency]``."""
    import jax.numpy as jnp

    return np.asarray(
        frequency_mask(
            jnp.asarray(frequencies),
            fmin=minimum_frequency,
            fmax=maximum_frequency,
        )
    )


def network_data(
    frequencies: ArrayLike,
    noise: NetworkNoise,
    *,
    minimum_frequency: float,
    maximum_frequency: float,
) -> dict[str, object]:
    """The per-bin Gaussian scale and usable-bin mask of one network.

    Bins no detector pair measures have infinite effective PSD; they are
    dropped from the mask and given a finite scale of one, as
    ``InferenceInputs.model_kwargs`` does.

    Parameters
    ----------
    frequencies
        The grid of ``noise`` in Hz, ``(F,)``.
    noise
        The network's noise, from
        :meth:`~astrogwb.simulators.BackgroundArtifact.noise`.
    minimum_frequency, maximum_frequency
        The analysis band in Hz.

    Returns
    -------
    dict
        ``scale`` and ``frequency_mask``, shape ``(F,)`` on ``frequencies``,
        as model keyword arguments.
    """
    import jax.numpy as jnp

    psd = noise.effective_psd
    mask = (
        _band(frequencies, minimum_frequency, maximum_frequency)
        & np.isfinite(psd)
        & (psd > 0.0)
    )
    return {
        "scale": jnp.asarray(np.where(mask, noise.scale, 1.0)),
        "frequency_mask": jnp.asarray(mask),
    }


#: The most detector pairs of a paper network (ET triangle + CE); networks
#: with fewer are padded to it so they stack.
MAX_PAIRS = 6


def pad_pairs(values: ArrayLike, num_pairs: int = MAX_PAIRS) -> NDArray:
    """Pad the pair axis (second to last) of ``values`` with zeros to ``num_pairs``."""
    values = np.asarray(values)
    widths = [(0, 0)] * values.ndim
    widths[-2] = (0, num_pairs - values.shape[-2])
    return np.pad(values, widths)


def network_pair_data(
    frequencies: ArrayLike,
    noise: NetworkNoise,
    *,
    minimum_frequency: float,
    maximum_frequency: float,
    num_pairs: int = MAX_PAIRS,
) -> dict[str, object]:
    """The per-pair Gaussian scale, usable-entry mask and ORF of one network.

    The stacked-pair counterpart of :func:`network_data`: ``scale`` is
    :func:`~astrogwb.detector.pair_bin_scale`, ``pair_overlap`` the isotropic
    ORF of each pair, pairs in the order of the network's background, padded
    to ``num_pairs`` with masked rows of zero overlap and unit scale so
    networks stack. Entries outside the band or with an infinite scale are
    masked and given a scale of one.

    Parameters are those of :func:`network_data`, with ``num_pairs`` the padded
    pair count.

    Returns
    -------
    dict
        ``scale``, ``frequency_mask`` and ``pair_overlap``, ``(num_pairs, F)``
        on ``frequencies``, as :class:`~astrogwb.inference.Network` fields.
    """
    import jax.numpy as jnp

    scale = noise.pair_scale
    band = _band(frequencies, minimum_frequency, maximum_frequency)
    mask = band & np.isfinite(scale) & (scale > 0.0)
    padded_scale = np.ones((num_pairs, np.size(frequencies)))
    padded_scale[: scale.shape[0]] = np.where(mask, scale, 1.0)
    return {
        "scale": jnp.asarray(padded_scale),
        "frequency_mask": jnp.asarray(pad_pairs(mask, num_pairs)),
        "pair_overlap": jnp.asarray(pad_pairs(noise.pair_overlap, num_pairs)),
    }


def shot_noise_variant(
    variant: str,
    likelihood: ImportanceGaussianLikelihood,
    fiducial_relative_variance: NDArray[np.float64],
) -> tuple[ImportanceGaussianLikelihood, NDArray[np.float64] | None]:
    r"""The likelihood and per-network :math:`s^2` of one shot-noise variant.

    Parameters
    ----------
    variant
        ``"detector"``, ``"amplitude"``, ``"fixed"``, ``"per_frequency"`` or
        ``"modes"``.
    likelihood
        Built with ``shot_noise="amplitude"``, so the squared power is held
        for every variant.
    fiducial_relative_variance
        :math:`s^2` at the fiducials per network, in ``per_network`` order;
        used by ``"fixed"`` only.

    Returns
    -------
    tuple
        The likelihood with the variant's mode, and the ``"fixed"`` variant's
        per-network :math:`s^2` (else ``None``).
    """
    if variant == "detector":
        return dataclasses.replace(likelihood, shot_noise=None), None
    if variant == "fixed":
        return (
            dataclasses.replace(likelihood, shot_noise="fixed"),
            np.asarray(fiducial_relative_variance, dtype=np.float64),
        )
    return dataclasses.replace(likelihood, shot_noise=variant), None
