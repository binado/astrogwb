r"""Pure Gaussian likelihoods of an observed spectrum, behind one output contract.

A likelihood here is a pytree whose ``__call__`` maps ``params`` to
``(log L, extras)``: no NumPyro, no sampling sites. NumPyro keeps the priors and
the constraining transforms
(:func:`~astrogwb.inference.models.gaussian_gwb_model.gwb_likelihood_model`);
NUTS, blackjax, optimizers and grids consume the likelihood directly.

The expensive stage, predicting the spectrum, does not depend on the detector
network; only :class:`Network` does, at :math:`O(F)` per point. Each class
therefore splits the work in two so that ``jax.vmap`` over stacked networks
shares one prediction:

- ``predict(params) -> (mean, variance | None, extras)``
- ``log_likelihood_from_prediction(prediction, network) -> log L``

``log_likelihood(params, network)`` composes them and ``__call__(params)`` uses
the network the instance carries.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .shot_noise import (
    amplitude_direction,
    amplitude_shot_noise_variance,
    per_frequency_direction,
    rank_one_gaussian_log_likelihood,
)

__all__ = [
    "GaussianLikelihood",
    "Network",
    "ShotNoiseMode",
    "gaussian_log_likelihood",
    "shot_noise_log_likelihood",
]

#: The shot-noise term of a likelihood: an amplitude mode with :math:`s^2` per
#: point, the per-frequency mode, or an amplitude mode with a fixed :math:`s^2`
#: (``Network.relative_variance``). ``None`` is the diagonal Gaussian.
type ShotNoiseMode = Literal["amplitude", "per_frequency", "fixed"]

#: ``(mean, variance | None, extras)``, what ``predict`` returns.
type Prediction = tuple[jax.Array, jax.Array | None, Mapping[str, jax.Array]]


class Network(NamedTuple):
    """The detector-network half of a likelihood.

    Stacking K networks means stacking each leaf along a new leading axis, and
    every network in a stack must share one structure: all of ``mask`` and
    ``relative_variance`` ``None``, or all arrays.

    Attributes
    ----------
    scale:
        Per-bin standard deviation, ``(F,)``.
    mask:
        Boolean ``(F,)`` of the bins counted; ``None`` counts all. Masked bins
        contribute exactly zero, whatever their ``scale``.
    relative_variance:
        A fixed :math:`s^2`, shape ``()``, for the ``"fixed"`` shot-noise mode.
    """

    scale: jax.Array
    mask: jax.Array | None = None
    relative_variance: jax.Array | None = None


def gaussian_log_likelihood(
    mean: ArrayLike,
    observed: ArrayLike,
    scale: ArrayLike,
    mask: ArrayLike | None = None,
) -> jax.Array:
    """Log density of a diagonal Gaussian, summed over the counted bins.

    Masked bins contribute exactly zero, and their ``scale`` may be infinite
    without producing a non-finite value or gradient. Shape ``(...)`` for
    ``mean`` of shape ``(..., F)``.
    """
    return rank_one_gaussian_log_likelihood(
        observed, mean, scale, jnp.zeros_like(jnp.asarray(mean)), mask
    )


def shot_noise_log_likelihood(
    mean: ArrayLike,
    variance: ArrayLike | None,
    observed: ArrayLike,
    network: Network,
    mode: ShotNoiseMode | None,
) -> jax.Array:
    r"""Gaussian log-likelihood with the shot noise as a rank-one covariance term.

    The covariance is :math:`D + \mathbf{u}\mathbf{u}^T` along the direction
    ``mode`` selects (see :mod:`astrogwb.inference.shot_noise`): ``"amplitude"``
    takes :math:`s^2` from the ``variance`` at this point and network,
    ``"fixed"`` takes ``network.relative_variance``, ``"per_frequency"`` takes
    :math:`u_f = \sqrt{V_f}`. ``None`` is the diagonal Gaussian and ignores
    ``variance``. ``mode`` is static; ``variance`` is required by the first and
    third modes and is not checked.
    """
    if mode is None:
        return gaussian_log_likelihood(mean, observed, network.scale, network.mask)
    if mode == "fixed":
        assert network.relative_variance is not None
        direction = amplitude_direction(mean, network.relative_variance)
        return rank_one_gaussian_log_likelihood(
            observed, mean, network.scale, direction, network.mask
        )
    # Every other mode predicted a variance next to the spectrum.
    assert variance is not None
    if mode == "amplitude":
        direction = amplitude_direction(
            mean,
            amplitude_shot_noise_variance(mean, variance, network.scale, network.mask),
        )
    else:
        direction = per_frequency_direction(variance)
    return rank_one_gaussian_log_likelihood(
        observed, mean, network.scale, direction, network.mask
    )


@dataclass(frozen=True)
class GaussianLikelihood:
    """A diagonal Gaussian likelihood of an observed spectrum.

    Parameters
    ----------
    spectrum_fn:
        ``params -> (F,)`` or ``params -> ((F,), extras)``. Static: it is
        compared by identity, so whatever it captures is compiled in as a
        constant. Use :class:`~astrogwb.inference.likelihood.ImportanceGaussianLikelihood`
        when a large catalog should be a traced input instead.
    observed:
        Observed spectrum, ``(F,)``.
    network:
        The detector network ``log_likelihood`` and ``__call__`` default to.
    """

    spectrum_fn: Callable[[Mapping[str, ArrayLike]], Any]
    observed: jax.Array
    network: Network

    def predict(self, params: Mapping[str, ArrayLike]) -> Prediction:
        """The spectrum at ``params``: ``(mean, None, extras)``."""
        result = self.spectrum_fn(params)
        if isinstance(result, tuple):
            mean, extras = result
            return jnp.asarray(mean), None, extras
        return jnp.asarray(result), None, {}

    def log_likelihood_from_prediction(
        self, prediction: Prediction, network: Network
    ) -> jax.Array:
        """The log-likelihood of ``prediction`` under ``network``, shape ``()``."""
        mean, _, _ = prediction
        return gaussian_log_likelihood(mean, self.observed, network.scale, network.mask)

    def log_likelihood(
        self, params: Mapping[str, ArrayLike], network: Network | None = None
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        """``(log L, extras)`` at ``params`` under ``network`` (the instance's by default)."""
        prediction = self.predict(params)
        value = self.log_likelihood_from_prediction(
            prediction, self.network if network is None else network
        )
        return value, prediction[2]

    def __call__(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        return self.log_likelihood(params)


jax.tree_util.register_dataclass(
    GaussianLikelihood,
    data_fields=["observed", "network"],
    meta_fields=["spectrum_fn"],
)
