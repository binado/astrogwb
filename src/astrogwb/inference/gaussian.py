r"""The dense Gaussian likelihood of the stacked per-pair cross powers.

The data are the cross powers :math:`d_{ab,k}` of every distinct detector pair
and frequency bin a network measures, flattened pair-major and restricted to
the entries it counts, ``(D,)``. Their mean is :math:`\gamma_{ab}(f_k) S_h(f_k)`
and their covariance (the paper's segment-averaged covariance) is

.. math::

    C_{ab,k;\,cd,l} = \delta_{ab,cd}\,\delta_{kl}\,\sigma_{ab,k}^2
        + \frac{1}{T}\left[4\mathcal R\langle I_k I_l\rangle\right]
        \Gamma^c_{ab,k;\,cd,l},
    \qquad
    \Gamma^c = \Gamma^{II} + \frac{\Gamma^{QQ} + \Gamma^{UU}}{71}
        + \frac{69}{71}\Gamma^{VV},

with :math:`4\mathcal R\langle I_k I_l\rangle` the intensity second moment of a
:class:`~astrogwb.inference.model.SpectralDensityModel`, which depends on the
parameters, and :math:`\Gamma^c` the quadrupole's combination
(:data:`~astrogwb.waveform.intensity.QUADRUPOLE_STOKES_WEIGHTS`) of the
network's generalized ORFs
(:func:`~astrogwb.detector.generalized_overlap_reduction_function`) in the
units of the normalized isotropic ORF, which does not. Without shot noise
:math:`C` is the diagonal detector noise.

A :class:`DenseGaussianLikelihood` is one network. Its ``log_likelihood``
predicts, factorizes :math:`C` once and scores every data realization it holds
against the one factor. Built with ``keep_generalized_orfs``, its
:meth:`~DenseGaussianLikelihood.covariance_components` splits :math:`C` into
the paper's detector-noise, count, intensity, linear- and
circular-polarization terms.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.detector import (
    NORMALIZED_ORF_SCALE,
    Detector,
    generalized_overlap_reduction_function,
    pair_bin_scale,
    pair_names,
    pair_terms,
    resolve_network,
)
from astrogwb.utils import PairName, stack_pairs
from astrogwb.waveform.intensity import QUADRUPOLE_STOKES_WEIGHTS, StokesPair

from .model import SpectralDensityModel, SpectralDensityPrediction

__all__ = [
    "DenseGaussianLikelihood",
    "Evaluation",
    "GaussianFactor",
]

_LOG_TWO_PI = math.log(2.0 * math.pi)

#: The generalized ORFs behind each Stokes term of the connected covariance.
_INTENSITY_PAIRS: tuple[StokesPair, ...] = ("II",)
_LINEAR_PAIRS: tuple[StokesPair, ...] = ("QQ", "UU")
_CIRCULAR_PAIRS: tuple[StokesPair, ...] = ("IV", "VI", "VV")


def _stokes_sum(
    orfs: Mapping[StokesPair, jax.Array], pairs: Sequence[StokesPair] | None = None
) -> jax.Array:
    r""":math:`\sum_{\mu\nu} w_{\mu\nu} \Gamma^{\mu\nu}` with the quadrupole's weights.

    Over ``pairs``, or every pair of ``orfs``; pairs the quadrupole does not
    weight contribute nothing.
    """
    selected = orfs if pairs is None else [pair for pair in pairs if pair in orfs]
    terms = [
        QUADRUPOLE_STOKES_WEIGHTS[pair] * orfs[pair]
        for pair in selected
        if pair in QUADRUPOLE_STOKES_WEIGHTS
    ]
    if not terms:
        raise ValueError(f"no weighted generalized ORF among {list(selected)}")
    return jnp.sum(jnp.stack(terms), axis=0)


def _scaled_orfs(
    orfs: Mapping[StokesPair, ArrayLike], shape: tuple[int, ...]
) -> dict[StokesPair, jax.Array]:
    """``orfs`` in the units of the per-pair data, each checked against ``shape``."""
    scaled = {}
    for pair, values in orfs.items():
        values = jnp.asarray(values)
        if values.shape != shape:
            raise ValueError(
                f"the generalized ORF {pair!r} has shape {values.shape}, not "
                f"{shape} for this network and frequency grid"
            )
        scaled[pair] = NORMALIZED_ORF_SCALE**2 * values
    return scaled


@dataclass(frozen=True)
class _Network:
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
        Isotropic ORF :math:`\gamma_{ab}(f_k)`, the gain from the spectral
        density to each entry's mean, ``(D,)``.
    observation_time
        Observation time :math:`T` in years, ``()``.
    connected_overlap
        The connected overlap :math:`\Gamma^c` on the counted entries,
        ``(D, D)``, or ``None`` for detector noise alone.
    generalized_orfs
        The generalized ORFs :math:`\Gamma^{\mu\nu}` on the counted entries,
        each ``(D, D)``, in the units of :math:`\Gamma^c`, or ``None`` when not
        kept.
    pairs
        The sorted name pair of every pair, in the data's order.
    num_bins
        The frequency grid size :math:`F`.
    """

    keep: jax.Array
    bins: jax.Array
    scale: jax.Array
    pair_overlap: jax.Array
    observation_time: jax.Array
    connected_overlap: jax.Array | None
    generalized_orfs: dict[StokesPair, jax.Array] | None
    pairs: tuple[PairName, ...]
    num_bins: int

    @classmethod
    def from_leaves(
        cls,
        *,
        scale: ArrayLike,
        pair_overlap: ArrayLike,
        observation_time: float,
        pairs: Sequence[PairName],
        mask: ArrayLike | None = None,
        connected_overlap: ArrayLike | None = None,
        generalized_orfs: Mapping[StokesPair, ArrayLike] | None = None,
    ) -> Self:
        """A network from its ``(P, F)`` leaves, keeping the finite, unmasked entries.

        Call outside JAX transformations: the kept set fixes the shapes.

        Parameters
        ----------
        scale
            Detector-noise standard deviation, ``(P, F)``, infinite out of band.
        pair_overlap
            Isotropic ORF of every pair, ``(P, F)``.
        observation_time
            Observation time in years.
        pairs
            The sorted name pair of every pair, ``(P,)``.
        mask
            Boolean, ``(P, F)``, of the entries to count; ``None`` counts all
            with finite noise.
        connected_overlap
            The connected overlap, ``(P, F, P, F)``, or ``None``.
        generalized_orfs
            The generalized ORFs, each ``(P, F, P, F)``, kept and combined into
            the connected overlap; exclusive with ``connected_overlap``.

        Raises
        ------
        ValueError
            If both ``connected_overlap`` and ``generalized_orfs`` are given.
        """
        if connected_overlap is not None and generalized_orfs is not None:
            raise ValueError(
                "pass the connected overlap or the generalized ORFs it combines, "
                "not both"
            )
        scale = np.asarray(scale, dtype=np.float64)
        num_pairs, num_bins = scale.shape
        counted = np.isfinite(scale)
        if mask is not None:
            counted &= np.asarray(mask, dtype=bool)
        keep = np.flatnonzero(counted)

        def restrict(values: ArrayLike) -> jax.Array:
            flat = np.reshape(np.asarray(values), (num_pairs * num_bins,) * 2)
            return jnp.asarray(flat[np.ix_(keep, keep)])

        orfs = None
        overlap = None
        if generalized_orfs is not None:
            orfs = {pair: restrict(values) for pair, values in generalized_orfs.items()}
            overlap = _stokes_sum(orfs)
        elif connected_overlap is not None:
            overlap = restrict(connected_overlap)
        return cls(
            keep=jnp.asarray(keep),
            bins=jnp.asarray(keep % num_bins),
            scale=jnp.asarray(scale.reshape(-1)[keep]),
            pair_overlap=jnp.asarray(np.asarray(pair_overlap).reshape(-1)[keep]),
            observation_time=jnp.asarray(observation_time, dtype=float),
            connected_overlap=overlap,
            generalized_orfs=orfs,
            pairs=tuple(pairs),
            num_bins=num_bins,
        )

    def compress(self, values: ArrayLike) -> jax.Array:
        """``(..., P, F)`` per-pair values on the counted entries, ``(..., D)``."""
        values = jnp.asarray(values)
        flat = jnp.reshape(values, (*values.shape[:-2], -1))
        return flat[..., self.keep]

    def stack(self, values: Mapping[PairName, ArrayLike]) -> jax.Array:
        """Pair-keyed ``(..., F)`` arrays in the data's pair order, ``(..., P, F)``.

        Raises
        ------
        ValueError
            If the keys are not exactly the pairs, or the arrays do not share
            a shape ending in the frequency grid size.
        """
        stacked = stack_pairs(values, self.pairs)
        if stacked.shape[-1] != self.num_bins:
            raise ValueError(
                f"per-pair arrays must share a shape (..., {self.num_bins}), "
                f"got {stacked.shape[:-2] + stacked.shape[-1:]}"
            )
        return stacked

    @property
    def mask(self) -> np.ndarray:
        """The counted entries as a boolean ``(P, F)`` grid."""
        counted = np.zeros(len(self.pairs) * self.num_bins, dtype=bool)
        counted[np.asarray(self.keep)] = True
        return counted.reshape(len(self.pairs), self.num_bins)


jax.tree_util.register_dataclass(
    _Network,
    data_fields=[
        "keep",
        "bins",
        "scale",
        "pair_overlap",
        "observation_time",
        "connected_overlap",
        "generalized_orfs",
    ],
    meta_fields=["pairs", "num_bins"],
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

    def covariance(self) -> jax.Array:
        r"""The covariance :math:`C = L L^\top`, ``(D, D)``."""
        if self.cholesky.ndim == 1:
            return jnp.diag(self.cholesky**2)
        return self.cholesky @ self.cholesky.T

    def whiten(self, values: ArrayLike) -> jax.Array:
        r""":math:`L^{-1} \mathbf x` for ``values`` ``(..., D)``, ``(..., D)``."""
        values = jnp.asarray(values)
        if self.cholesky.ndim == 1:
            return values / self.cholesky
        flat = jnp.reshape(values, (-1, values.shape[-1]))
        # One triangular solve with every row as a right-hand side.
        whitened = solve_triangular(self.cholesky, flat.T, lower=True).T
        return jnp.reshape(whitened, values.shape)

    def solve(self, values: ArrayLike) -> jax.Array:
        r""":math:`C^{-1} \mathbf x` for ``values`` ``(..., D)``, ``(..., D)``."""
        whitened = self.whiten(values)
        if self.cholesky.ndim == 1:
            return whitened / self.cholesky
        flat = jnp.reshape(whitened, (-1, whitened.shape[-1]))
        solved = solve_triangular(self.cholesky, flat.T, lower=True, trans="T").T
        return jnp.reshape(solved, whitened.shape)


class Evaluation(NamedTuple):
    """A likelihood's prediction and factor at one parameter point."""

    prediction: SpectralDensityPrediction
    factor: GaussianFactor


def _expected_data(
    prediction: SpectralDensityPrediction, network: _Network
) -> jax.Array:
    r"""The mean :math:`\gamma_{ab}(f_k) S_h(f_k)` on the counted entries, ``(D,)``."""
    if prediction.spectral_density.shape != (network.num_bins,):
        raise ValueError(
            f"the spectral density has shape {prediction.spectral_density.shape}, "
            f"not the likelihood's frequency grid ({network.num_bins},)"
        )
    return network.pair_overlap * prediction.spectral_density[network.bins]


def _catalog_covariance(
    prediction: SpectralDensityPrediction,
    network: _Network,
    overlap: jax.Array | None = None,
) -> jax.Array:
    r"""The connected covariance :math:`[4\mathcal R\langle I_k I_l\rangle]\,\Gamma^c / T`, ``(D, D)``.

    ``overlap`` replaces the network's :math:`\Gamma^c`, for one Stokes
    term of it.
    """
    moment = prediction.intensity_second_moment
    if moment is None or network.connected_overlap is None:
        raise ValueError(
            "the catalog covariance needs an intensity second moment and a "
            "connected overlap; build the likelihood with shot_noise=True"
        )
    if moment.shape != (network.num_bins,) * 2:
        raise ValueError(
            f"the intensity second moment has shape {moment.shape}, not "
            f"{(network.num_bins,) * 2}"
        )
    if overlap is None:
        overlap = network.connected_overlap
    covariance = (
        moment[network.bins][:, network.bins]
        * overlap
        / (network.observation_time * SECONDS_PER_YEAR)
    )
    # Symmetric to round-off only; made exact for the Cholesky downstream.
    return 0.5 * (covariance + covariance.T)


def _factorize(
    prediction: SpectralDensityPrediction, network: _Network, shot_noise: bool
) -> GaussianFactor:
    mean = _expected_data(prediction, network)
    if not shot_noise:
        return GaussianFactor(
            mean, network.scale, 2.0 * jnp.sum(jnp.log(network.scale))
        )
    covariance = _catalog_covariance(prediction, network) + jnp.diag(network.scale**2)
    cholesky = jnp.linalg.cholesky(covariance)
    return GaussianFactor(mean, cholesky, 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky))))


def _log_density(factor: GaussianFactor, observed: jax.Array) -> jax.Array:
    """Log density of ``observed``, ``(..., D)``, under ``factor``: ``(...)``."""
    whitened = factor.whiten(observed - factor.mean)
    chi_squared = jnp.sum(whitened * whitened, axis=-1)
    size = factor.mean.shape[0]
    return -0.5 * (chi_squared + factor.log_det + size * _LOG_TWO_PI)


@jax.tree_util.register_pytree_node_class
class DenseGaussianLikelihood[M: SpectralDensityModel]:
    """The dense Gaussian likelihood of one network's observed cross powers.

    A pytree: the model, the network's arrays and the data are traced, so one
    compilation serves every model, network and data set of the same
    structure and shapes; ``shot_noise`` is static. Building one is the
    expensive part, done once outside JAX transformations.

    Parameters
    ----------
    model
        The background model, on ``frequencies``.
    network
        A registered network name (:func:`~astrogwb.detector.load_network`)
        or its detectors.
    frequencies
        Frequency grid in Hz, ``(F,)``.
    observation_time
        Observation time :math:`T` in years.
    observed
        Observed cross powers keyed by the sorted detector-name pair
        (:func:`~astrogwb.detector.pair_names`), each ``(..., F)``; leading
        axes are independent realizations. ``None`` builds a likelihood for
        diagnostics only.
    mask
        Boolean ``(F,)`` per sorted pair, of the entries to count; entries
        with infinite detector noise are never counted.
    shot_noise
        Whether the covariance includes the catalog's shot noise.
    key
        Random key of the generalized ORFs' sky draws; a fixed key by
        default.
    num_draws
        Number of sky draws of the generalized ORFs; ``2**15`` by default.
    generalized_orfs
        Precomputed generalized ORFs, as returned by
        :func:`~astrogwb.detector.generalized_overlap_reduction_function` for
        this network's detectors and ``frequencies`` (each ``(P, F, P, F)``,
        unmasked, not rescaled), which spare the sky draws; one set serves any
        mask, model or data. Mutually exclusive with ``key`` and
        ``num_draws``. Ignored, like them, without shot noise.
    keep_generalized_orfs
        Whether to keep the generalized ORFs
        (:func:`~astrogwb.detector.generalized_overlap_reduction_function`)
        for :meth:`covariance_components`, at four times the memory of their
        combination. The covariance is the same either way.

    Attributes
    ----------
    model
        The background model.
    pairs
        The sorted name pair of every pair, in the data's order.
    mask
        The counted entries, boolean ``(P, F)``: the user mask without the
        entries of infinite detector noise. The compressed data axis is
        ``stack_pairs(values, pairs)[..., mask]``, pair-major.
    observed
        Observed cross powers on the counted entries, ``(..., D)``, or ``None``.
    shot_noise
        Whether the covariance includes the catalog's shot noise.
    """

    model: M
    observed: jax.Array | None
    shot_noise: bool
    _network: _Network

    def __init__(
        self,
        model: M,
        network: str | Sequence[Detector],
        *,
        frequencies: ArrayLike,
        observation_time: float,
        observed: Mapping[PairName, ArrayLike] | None = None,
        mask: Mapping[PairName, ArrayLike] | None = None,
        shot_noise: bool = True,
        key: jax.Array | None = None,
        num_draws: int | None = None,
        generalized_orfs: Mapping[StokesPair, ArrayLike] | None = None,
        keep_generalized_orfs: bool = False,
    ) -> None:
        detectors = resolve_network(network)
        frequencies = np.asarray(frequencies, dtype=float)
        pairs = pair_names(detectors)
        num_bins = frequencies.size
        scale = pair_bin_scale(frequencies, detectors, observation_time)
        counted = np.isfinite(scale)
        if mask is not None:
            counted &= np.asarray(stack_pairs(mask, pairs), dtype=bool)
        overlaps: dict[str, Any] = {}
        if shot_noise:
            if generalized_orfs is None:
                generalized_orfs = generalized_overlap_reduction_function(
                    jax.random.key(0) if key is None else key,
                    2**15 if num_draws is None else num_draws,
                    frequencies,
                    detectors,
                )
            elif key is not None or num_draws is not None:
                raise ValueError(
                    "key and num_draws set the sky draws of the generalized "
                    "ORFs; do not pass them with generalized_orfs"
                )
            orfs = _scaled_orfs(generalized_orfs, (len(pairs), num_bins) * 2)
            overlaps = (
                {"generalized_orfs": orfs}
                if keep_generalized_orfs
                else {"connected_overlap": _stokes_sum(orfs)}
            )
        resolved = _Network.from_leaves(
            scale=scale,
            pair_overlap=np.stack(
                [term.overlap for term in pair_terms(frequencies, detectors)]
            ),
            observation_time=observation_time,
            pairs=pairs,
            mask=counted,
            **overlaps,
        )
        self._assign(
            model,
            resolved,
            None if observed is None else resolved.compress(resolved.stack(observed)),
            shot_noise,
        )

    def _assign(
        self,
        model: M,
        network: _Network,
        observed: jax.Array | None,
        shot_noise: bool,
    ) -> None:
        self.model = model
        self._network = network
        self.observed = observed
        self.shot_noise = shot_noise

    @classmethod
    def _from_network(
        cls,
        model: M,
        network: _Network,
        observed: jax.Array | None,
        *,
        shot_noise: bool = True,
    ) -> DenseGaussianLikelihood[M]:
        """A likelihood on a prebuilt network, with data already on its entries."""
        likelihood = object.__new__(cls)
        likelihood._assign(model, network, observed, shot_noise)
        return likelihood

    def tree_flatten(self) -> tuple[tuple[Any, ...], bool]:
        return (self.model, self._network, self.observed), self.shot_noise

    @classmethod
    def tree_unflatten(
        cls, shot_noise: bool, children: tuple[Any, ...]
    ) -> DenseGaussianLikelihood[M]:
        # Rebuilt without __init__: tracing must not reload the network or
        # redraw the generalized ORFs.
        model, network, observed = children
        return cls._from_network(model, network, observed, shot_noise=shot_noise)

    @property
    def pairs(self) -> tuple[PairName, ...]:
        """The sorted name pair of every pair, in the data's order."""
        return self._network.pairs

    @property
    def mask(self) -> np.ndarray:
        """The counted entries, boolean ``(P, F)``."""
        return self._network.mask

    @property
    def observation_time(self) -> jax.Array:
        """Observation time :math:`T` in years, ``()``."""
        return self._network.observation_time

    def mean(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        r"""The expected data :math:`\gamma_{ab}(f_k) S_h(f_k)` at ``params``, ``(D,)``.

        Parameters
        ----------
        params
            Model parameter values.
        """
        return _expected_data(self.model(params, shot_noise=False), self._network)

    def covariance(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        r"""The total covariance :math:`C` at ``params``, ``(D, D)``.

        Needs neither ``keep_generalized_orfs`` nor data. Without shot noise
        it is the diagonal detector noise.

        Parameters
        ----------
        params
            Model parameter values.
        """
        return self.evaluate(params).factor.covariance()

    def covariance_components(
        self, params: Mapping[str, ArrayLike]
    ) -> dict[str, jax.Array]:
        r"""The terms of the covariance at ``params``, each ``(D, D)``.

        .. math::

            C = \Sigma^g + \Sigma^{c,{\rm count}} + \Sigma^{c,{\rm int}}
                + \Sigma^{c,{\rm lin}} + \Sigma^{c,{\rm circ}},

        keyed ``detector``, ``count``, ``intensity``, ``linear`` and
        ``circular``. The count term is the covariance of the expected data
        conditioned on the event count,

        .. math::

            \Sigma^{c,{\rm count}}_{ab,k;\,cd,l}
                = \frac{\gamma_{ab}(f_k) S_h(f_k)\,\gamma_{cd}(f_l) S_h(f_l)}
                    {\mathcal R T},

        and the intensity term the Stokes-I connected covariance less it.

        Parameters
        ----------
        params
            Model parameter values.

        Raises
        ------
        ValueError
            If the likelihood was built without shot noise or without
            ``keep_generalized_orfs``, or the model does not predict its
            merger rate.
        """
        orfs = self._network.generalized_orfs
        if not self.shot_noise or orfs is None:
            raise ValueError(
                "the covariance components need a likelihood built with "
                "shot_noise=True and keep_generalized_orfs=True"
            )
        prediction = self.model(params, shot_noise=True)
        if prediction.merger_rate is None:
            raise ValueError(
                "the count term needs the model to predict its merger rate"
            )
        mean = _expected_data(prediction, self._network)
        seconds = self._network.observation_time * SECONDS_PER_YEAR
        count = jnp.outer(mean, mean) / (prediction.merger_rate * seconds)

        def term(pairs: Sequence[StokesPair]) -> jax.Array:
            return _catalog_covariance(
                prediction, self._network, _stokes_sum(orfs, pairs)
            )

        return {
            "detector": jnp.diag(self._network.scale**2),
            "count": count,
            "intensity": term(_INTENSITY_PAIRS) - count,
            "linear": term(_LINEAR_PAIRS),
            "circular": term(_CIRCULAR_PAIRS),
        }

    def evaluate(self, params: Mapping[str, ArrayLike]) -> Evaluation:
        """The prediction and the factorized covariance at ``params``.

        Parameters
        ----------
        params
            Model parameter values.
        """
        prediction = self.model(params, shot_noise=self.shot_noise)
        return Evaluation(
            prediction, _factorize(prediction, self._network, self.shot_noise)
        )

    def log_likelihood(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        """``(log L, extras)`` at ``params``, ``log L`` shaped as ``observed[..., 0]``.

        Parameters
        ----------
        params
            Model parameter values.

        Raises
        ------
        ValueError
            If the likelihood holds no data.
        """
        if self.observed is None:
            raise ValueError("the likelihood was built without observed data")
        prediction, factor = self.evaluate(params)
        return _log_density(factor, self.observed), prediction.extras

    def __call__(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        return self.log_likelihood(params)
