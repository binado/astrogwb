"""Setup shared by the posterior notebooks.

Per-network Gaussian data cut to an analysis band.
``jax.numpy`` is imported inside functions, so importing this module leaves the
XLA backend alone.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike as JaxArrayLike
from numpy.typing import ArrayLike, NDArray

from astrogwb.frequency import frequency_mask
from astrogwb.inference.gaussian import Network
from astrogwb.simulators import NetworkNoise

__all__ = [
    "network_data",
    "pair_network",
]


def _band(
    frequencies: ArrayLike, minimum_frequency: float, maximum_frequency: float
) -> NDArray[np.bool_]:
    """The bins of ``frequencies`` inside ``[minimum_frequency, maximum_frequency]``."""

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
    dropped from the mask and given a finite scale of one.

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


def pair_network(
    frequencies: ArrayLike,
    noise: NetworkNoise,
    *,
    minimum_frequency: float,
    maximum_frequency: float,
    response_moment: JaxArrayLike | None = None,
) -> Network:
    """One network's per-pair likelihood network, cut to the analysis band.

    The stacked-pair counterpart of :func:`network_data`: the counted entries
    are the pairs and bins inside the band with a finite, positive
    :func:`~astrogwb.detector.pair_bin_scale`, in the order of the network's
    background. Compress the network's recorded per-pair spectra with
    :meth:`~astrogwb.inference.gaussian.Network.compress`.

    Parameters
    ----------
    frequencies
        The grid of ``noise`` in Hz, ``(F,)``.
    noise
        The network's noise, from
        :meth:`~astrogwb.simulators.BackgroundArtifact.noise`.
    minimum_frequency, maximum_frequency
        The analysis band in Hz.
    response_moment
        :func:`~astrogwb.detector.pair_response_moment` on ``frequencies``,
        ``(P, F, P, F)``, or ``None`` for detector noise alone.
    """
    scale = noise.pair_scale
    mask = (
        _band(frequencies, minimum_frequency, maximum_frequency)
        & np.isfinite(scale)
        & (scale > 0.0)
    )
    return Network.from_pairs(
        scale, noise.pair_overlap, mask, response_moment=response_moment
    )
