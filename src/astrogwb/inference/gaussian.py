r"""The dense Gaussian likelihood of the stacked per-pair cross powers.

The data are the cross powers :math:`d_{ab,k}` of every distinct detector pair
and frequency bin a network measures, flattened pair-major and restricted to
the entries it counts, ``(D,)``. Their mean is :math:`\gamma_{ab}(f_k) S(f_k)`
and their covariance

.. math::

    C_{ab,k;\,cd,l} = \delta_{ab,cd}\,\delta_{kl}\,\sigma_{ab,k}^2
        + M_{kl}\, R_{ab,k;\,cd,l},

with :math:`M` the power moment of a :class:`~astrogwb.inference.model.CatalogModel`,
which depends on the parameters, and :math:`R` the network's response moment
:math:`\langle \gamma_{ab}(\hat n, \psi, \iota; f_k)\,
\gamma_{cd}(\hat n, \psi, \iota; f_l) \rangle`
(:func:`~astrogwb.detector.response.pair_response_moment`), which does not.
Without shot noise :math:`C` is the diagonal detector noise.

A :class:`DenseGaussianLikelihood` is one network. Its ``log_likelihood``
predicts, factorizes :math:`C` once and scores every data realization it holds
against the one factor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from .model import CatalogModel, Prediction

__all__ = [
    "DenseGaussianLikelihood",
    "Evaluation",
    "GaussianFactor",
    "Network",
]

_LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class Network:
    r"""One detector network, on the entries it counts.

    Attributes
    ----------
    keep
        Pair-major indices of the counted entries into the ``(P, F)`` grid,
        ``(D,)``.
    bins
        The frequency index of every counted entry, ``(D,)``.
    scale
        Detector-noise standard deviation :math:`\sigma_{ab,k}`, ``(D,)``.
    pair_overlap
        Isotropic ORF :math:`\gamma_{ab}(f_k)`, the gain from the spectrum to
        each entry's mean, ``(D,)``.
    response_moment
        The response moment :math:`R` on the counted entries, ``(D, D)``, or
        ``None`` for detector noise alone.
    """

    keep: jax.Array
    bins: jax.Array
    scale: jax.Array
    pair_overlap: jax.Array
    response_moment: jax.Array | None = None

    @classmethod
    def from_pairs(
        cls,
        scale: ArrayLike,
        pair_overlap: ArrayLike,
        mask: ArrayLike | None = None,
        response_moment: ArrayLike | None = None,
    ) -> Self:
        """A network from its ``(P, F)`` leaves, keeping the entries ``mask`` counts.

        Call outside JAX transformations: the kept set fixes the shapes.

        Parameters
        ----------
        scale
            Detector-noise standard deviation, ``(P, F)``
            (:func:`~astrogwb.detector.pair_bin_scale`). May be infinite where
            ``mask`` is false.
        pair_overlap
            Isotropic ORF of every pair, ``(P, F)``.
        mask
            Boolean, ``(P, F)``, of the counted entries; ``None`` counts all.
        response_moment
            :func:`~astrogwb.detector.response.pair_response_moment`,
            ``(P, F, P, F)``, or ``None``.
        """
        scale = np.asarray(scale, dtype=np.float64)
        num_pairs, num_bins = scale.shape
        counted = (
            np.ones(scale.shape, dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool)
        )
        keep = np.flatnonzero(counted)
        moment = None
        if response_moment is not None:
            flat = np.reshape(np.asarray(response_moment), (num_pairs * num_bins,) * 2)
            moment = jnp.asarray(flat[np.ix_(keep, keep)])
        return cls(
            keep=jnp.asarray(keep),
            bins=jnp.asarray(keep % num_bins),
            scale=jnp.asarray(scale.reshape(-1)[keep]),
            pair_overlap=jnp.asarray(np.asarray(pair_overlap).reshape(-1)[keep]),
            response_moment=moment,
        )

    def compress(self, values: ArrayLike) -> jax.Array:
        """``(..., P, F)`` per-pair values on the counted entries, ``(..., D)``.

        Parameters
        ----------
        values
            Per-pair values, such as the observed cross powers.
        """
        values = jnp.asarray(values)
        flat = jnp.reshape(values, (*values.shape[:-2], -1))
        return flat[..., self.keep]


jax.tree_util.register_dataclass(
    Network,
    data_fields=["keep", "bins", "scale", "pair_overlap", "response_moment"],
    meta_fields=[],
)


class GaussianFactor(NamedTuple):
    """A Gaussian on a network's counted entries, factorized.

    Attributes
    ----------
    mean
        The mean, ``(D,)``.
    cholesky
        The lower Cholesky factor of the covariance, ``(D, D)``, or its
        diagonal, ``(D,)``, for a diagonal covariance.
    log_det
        :math:`\\ln\\det C`, ``()``.
    """

    mean: jax.Array
    cholesky: jax.Array
    log_det: jax.Array


class Evaluation(NamedTuple):
    """A likelihood's prediction and factor at one parameter point."""

    prediction: Prediction
    factor: GaussianFactor


def _factorize(prediction: Prediction, network: Network) -> GaussianFactor:
    mean = network.pair_overlap * prediction.mean[network.bins]
    if prediction.power_moment is None:
        return GaussianFactor(
            mean, network.scale, 2.0 * jnp.sum(jnp.log(network.scale))
        )
    if network.response_moment is None:
        raise ValueError("a shot-noise likelihood needs network.response_moment")
    covariance = (
        prediction.power_moment[network.bins][:, network.bins] * network.response_moment
    ) + jnp.diag(network.scale**2)
    cholesky = jnp.linalg.cholesky(covariance)
    return GaussianFactor(mean, cholesky, 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky))))


def _log_density(factor: GaussianFactor, observed: jax.Array) -> jax.Array:
    """Log density of ``observed``, ``(..., D)``, under ``factor``: ``(...)``."""
    residual = observed - factor.mean
    batch = residual.shape[:-1]
    flat = jnp.reshape(residual, (-1, residual.shape[-1]))
    if factor.cholesky.ndim == 1:
        whitened = flat / factor.cholesky
    else:
        # One triangular solve with every realization as a right-hand side.
        whitened = solve_triangular(factor.cholesky, flat.T, lower=True).T
    chi_squared = jnp.sum(whitened * whitened, axis=-1)
    size = factor.mean.shape[0]
    value = -0.5 * (chi_squared + factor.log_det + size * _LOG_TWO_PI)
    return jnp.reshape(value, batch)


@dataclass(frozen=True)
class DenseGaussianLikelihood:
    """The dense Gaussian likelihood of one network's observed cross powers.

    A pytree: ``model``, ``network`` and ``observed`` are traced, so one
    compilation serves every catalog and data set of the same shapes;
    ``shot_noise`` is static.

    Attributes
    ----------
    model
        The spectrum and power moment.
    network
        The detector network.
    observed
        Observed cross powers on the network's counted entries, ``(..., D)``
        (:meth:`Network.compress`); leading axes are independent realizations.
    shot_noise
        Whether the covariance includes the catalog's shot noise.
    """

    model: CatalogModel
    network: Network
    observed: jax.Array
    shot_noise: bool = True

    def evaluate(self, params: Mapping[str, ArrayLike]) -> Evaluation:
        """The prediction and the factorized covariance at ``params``.

        Parameters
        ----------
        params
            Model parameter values.
        """
        prediction = self.model.predict(params, shot_noise=self.shot_noise)
        return Evaluation(prediction, _factorize(prediction, self.network))

    def log_likelihood(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        """``(log L, extras)`` at ``params``, ``log L`` shaped as ``observed[..., 0]``.

        Parameters
        ----------
        params
            Model parameter values.
        """
        prediction, factor = self.evaluate(params)
        return _log_density(factor, self.observed), prediction.extras

    def __call__(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        return self.log_likelihood(params)


jax.tree_util.register_dataclass(
    DenseGaussianLikelihood,
    data_fields=["model", "network", "observed"],
    meta_fields=["shot_noise"],
)
