r"""The Gaussian likelihoods of the stacked per-pair cross powers.

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
units of the normalized isotropic ORF, which does not.

A :class:`DenseGaussianLikelihood` is one network with the full covariance;
its :meth:`~DenseGaussianLikelihood.covariance_components` split :math:`C`
into the paper's detector-noise, count, intensity, linear- and
circular-polarization terms. A :class:`DiagonalGaussianLikelihood` is its
limit without shot noise, the diagonal detector noise alone. Either one's
``log_likelihood`` predicts, factorizes :math:`C` once and scores every data
realization it holds against the one factor; :class:`GaussianLikelihood` is
what their consumers need of them.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple, Protocol, Self

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
    "DiagonalGaussianLikelihood",
    "GaussianLikelihood",
]

_LOG_TWO_PI = math.log(2.0 * math.pi)

#: The generalized ORFs the quadrupole weights, the ones a dense likelihood keeps.
_WEIGHTED_PAIRS: tuple[StokesPair, ...] = tuple(QUADRUPOLE_STOKES_WEIGHTS)

#: The generalized ORFs behind each Stokes term of the connected covariance.
_INTENSITY_PAIRS: tuple[StokesPair, ...] = ("II",)
_LINEAR_PAIRS: tuple[StokesPair, ...] = ("QQ", "UU")
_CIRCULAR_PAIRS: tuple[StokesPair, ...] = ("VV",)


def _stokes_sum(
    orfs: Mapping[StokesPair, jax.Array],
    pairs: Sequence[StokesPair] = _WEIGHTED_PAIRS,
) -> jax.Array:
    r""":math:`\sum_{\mu\nu} w_{\mu\nu} \Gamma^{\mu\nu}` over ``pairs``, with the quadrupole's weights."""
    return jnp.sum(
        jnp.stack([QUADRUPOLE_STOKES_WEIGHTS[pair] * orfs[pair] for pair in pairs]),
        axis=0,
    )


def _weighted_orfs[T](orfs: Mapping[StokesPair, T]) -> dict[StokesPair, T]:
    """The generalized ORFs of ``orfs`` the quadrupole weights.

    Raises
    ------
    ValueError
        If one of them is missing.
    """
    missing = [pair for pair in _WEIGHTED_PAIRS if pair not in orfs]
    if missing:
        raise ValueError(
            f"the dense likelihood needs the generalized ORFs {list(_WEIGHTED_PAIRS)}; "
            f"missing {missing}"
        )
    return {pair: orfs[pair] for pair in _WEIGHTED_PAIRS}


def _scaled_orfs(
    orfs: Mapping[StokesPair, ArrayLike], shape: tuple[int, ...]
) -> dict[StokesPair, np.ndarray]:
    """``orfs`` in the units of the per-pair data, each checked against ``shape``."""
    scaled = {}
    for pair, values in orfs.items():
        values = np.asarray(values)
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
        """
        scale = np.asarray(scale, dtype=np.float64)
        num_bins = scale.shape[1]
        counted = np.isfinite(scale)
        if mask is not None:
            counted &= np.asarray(mask, dtype=bool)
        keep = np.flatnonzero(counted)
        return cls(
            keep=jnp.asarray(keep),
            bins=jnp.asarray(keep % num_bins),
            scale=jnp.asarray(scale.reshape(-1)[keep]),
            pair_overlap=jnp.asarray(np.asarray(pair_overlap).reshape(-1)[keep]),
            observation_time=jnp.asarray(observation_time, dtype=float),
            pairs=tuple(pairs),
            num_bins=num_bins,
        )

    def compress(self, values: ArrayLike) -> jax.Array:
        """``(..., P, F)`` per-pair values on the counted entries, ``(..., D)``."""
        values = jnp.asarray(values)
        flat = jnp.reshape(values, (*values.shape[:-2], -1))
        return flat[..., self.keep]

    def restrict(self, values: ArrayLike) -> jax.Array:
        """``(P, F, P, F)`` pair-pair values on the counted entries, ``(D, D)``.

        Call outside JAX transformations.
        """
        size = len(self.pairs) * self.num_bins
        flat = np.reshape(np.asarray(values), (size, size))
        keep = np.asarray(self.keep)
        return jnp.asarray(flat[np.ix_(keep, keep)])

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
    ],
    meta_fields=["pairs", "num_bins"],
)


class _Factor(Protocol):
    """A Gaussian on a network's counted entries, factorized."""

    @property
    def mean(self) -> jax.Array:
        """The mean, ``(D,)``."""
        ...

    @property
    def log_det(self) -> jax.Array:
        r""":math:`\ln\det C`, ``()``."""
        ...

    def covariance(self) -> jax.Array:
        """The covariance :math:`C`, ``(D, D)``."""
        ...

    def whiten(self, values: ArrayLike) -> jax.Array:
        r""":math:`L^{-1} \mathbf x` for ``values`` ``(..., D)``, ``(..., D)``."""
        ...

    def solve(self, values: ArrayLike) -> jax.Array:
        r""":math:`C^{-1} \mathbf x` for ``values`` ``(..., D)``, ``(..., D)``."""
        ...


class _DiagonalFactor(NamedTuple):
    r"""A Gaussian of diagonal covariance :math:`C = \mathrm{diag}(\sigma^2)`.

    Attributes
    ----------
    mean
        The mean, ``(D,)``.
    scale
        The standard deviation :math:`\sigma`, ``(D,)``.
    """

    mean: jax.Array
    scale: jax.Array

    @property
    def log_det(self) -> jax.Array:
        r""":math:`\ln\det C = 2 \sum \ln\sigma`, ``()``."""
        return 2.0 * jnp.sum(jnp.log(self.scale))

    def covariance(self) -> jax.Array:
        r"""The covariance :math:`\mathrm{diag}(\sigma^2)`, ``(D, D)``."""
        return jnp.diag(self.scale**2)

    def whiten(self, values: ArrayLike) -> jax.Array:
        r""":math:`\mathbf x / \sigma` for ``values`` ``(..., D)``, ``(..., D)``."""
        return jnp.asarray(values) / self.scale

    def solve(self, values: ArrayLike) -> jax.Array:
        r""":math:`\mathbf x / \sigma^2` for ``values`` ``(..., D)``, ``(..., D)``."""
        return jnp.asarray(values) / self.scale**2


class _CholeskyFactor(NamedTuple):
    r"""A Gaussian of dense covariance :math:`C = L L^\top`.

    Attributes
    ----------
    mean
        The mean, ``(D,)``.
    cholesky
        The lower Cholesky factor :math:`L` of the covariance, ``(D, D)``.
    log_det
        :math:`\ln\det C`, ``()``.
    """

    mean: jax.Array
    cholesky: jax.Array
    log_det: jax.Array

    @classmethod
    def from_covariance(cls, mean: jax.Array, covariance: jax.Array) -> Self:
        """The factor of a symmetric positive definite ``covariance``, ``(D, D)``."""
        cholesky = jnp.linalg.cholesky(covariance)
        return cls(mean, cholesky, 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky))))

    def covariance(self) -> jax.Array:
        r"""The covariance :math:`C = L L^\top`, ``(D, D)``."""
        return self.cholesky @ self.cholesky.T

    def whiten(self, values: ArrayLike) -> jax.Array:
        r""":math:`L^{-1} \mathbf x` for ``values`` ``(..., D)``, ``(..., D)``."""
        values = jnp.asarray(values)
        flat = jnp.reshape(values, (-1, values.shape[-1]))
        # One triangular solve with every row as a right-hand side.
        whitened = solve_triangular(self.cholesky, flat.T, lower=True).T
        return jnp.reshape(whitened, values.shape)

    def solve(self, values: ArrayLike) -> jax.Array:
        r""":math:`C^{-1} \mathbf x` for ``values`` ``(..., D)``, ``(..., D)``."""
        whitened = self.whiten(values)
        flat = jnp.reshape(whitened, (-1, whitened.shape[-1]))
        solved = solve_triangular(self.cholesky, flat.T, lower=True, trans="T").T
        return jnp.reshape(solved, whitened.shape)


class _Evaluation(NamedTuple):
    """A likelihood's prediction and factor at one parameter point."""

    prediction: SpectralDensityPrediction
    factor: _Factor


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
    prediction: SpectralDensityPrediction, network: _Network, overlap: jax.Array
) -> jax.Array:
    r"""The connected covariance :math:`[4\mathcal R\langle I_k I_l\rangle]\,\Gamma / T`, ``(D, D)``.

    ``overlap`` is :math:`\Gamma` on the counted entries: the connected
    overlap, or one Stokes term of it.
    """
    moment = prediction.intensity_second_moment
    if moment is None:
        raise ValueError("the model predicted no intensity second moment")
    if moment.shape != (network.num_bins,) * 2:
        raise ValueError(
            f"the intensity second moment has shape {moment.shape}, not "
            f"{(network.num_bins,) * 2}"
        )
    covariance = (
        moment[network.bins][:, network.bins]
        * overlap
        / (network.observation_time * SECONDS_PER_YEAR)
    )
    # Symmetric to round-off only; made exact for the Cholesky downstream.
    return 0.5 * (covariance + covariance.T)


def _log_density(factor: _Factor, observed: jax.Array) -> jax.Array:
    """Log density of ``observed``, ``(..., D)``, under ``factor``: ``(...)``."""
    whitened = factor.whiten(observed - factor.mean)
    chi_squared = jnp.sum(whitened * whitened, axis=-1)
    size = factor.mean.shape[0]
    return -0.5 * (chi_squared + factor.log_det + size * _LOG_TWO_PI)


def _resolve(
    network: str | Sequence[Detector],
    frequencies: np.ndarray,
    observation_time: float,
    mask: Mapping[PairName, ArrayLike] | None,
) -> tuple[list[Detector], _Network]:
    """The detectors of ``network`` and the network on the entries it counts."""
    detectors = list(resolve_network(network))
    pairs = pair_names(detectors)
    scale = pair_bin_scale(frequencies, detectors, observation_time)
    counted = np.isfinite(scale)
    if mask is not None:
        counted &= np.asarray(stack_pairs(mask, pairs), dtype=bool)
    resolved = _Network.from_leaves(
        scale=scale,
        pair_overlap=np.stack(
            [term.overlap for term in pair_terms(frequencies, detectors)]
        ),
        observation_time=observation_time,
        pairs=pairs,
        mask=counted,
    )
    return detectors, resolved


def _compressed(
    network: _Network, observed: Mapping[PairName, ArrayLike] | None
) -> jax.Array | None:
    """Pair-keyed observations on the network's counted entries, ``(..., D)``."""
    return None if observed is None else network.compress(network.stack(observed))


class GaussianLikelihood(Protocol):
    """What the grid and the amplitude statistic need of a Gaussian likelihood."""

    def mean(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        """The expected data at ``params``, ``(D,)``."""
        ...

    def covariance(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        """The total covariance at ``params``, ``(D, D)``."""
        ...

    def solve(self, x: ArrayLike, params: Mapping[str, ArrayLike]) -> jax.Array:
        r"""The solution of :math:`C\mathbf a = \mathbf x` at ``params``, ``(..., D)``."""
        ...

    def log_likelihood(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        """``(log L, extras)`` at ``params``."""
        ...


class _GaussianLikelihood[M: SpectralDensityModel](ABC):
    """The parts the Gaussian likelihoods share; subclasses factorize.

    Attributes
    ----------
    model
        The background model.
    observed
        Observed cross powers on the counted entries, ``(..., D)``, or ``None``.
    """

    #: Whether the model is asked for the moments of the shot noise.
    _shot_noise: ClassVar[bool]

    model: M
    observed: jax.Array | None
    _network: _Network

    def _assign(self, model: M, network: _Network, observed: jax.Array | None) -> None:
        self.model = model
        self._network = network
        self.observed = observed

    @abstractmethod
    def _factorize(self, prediction: SpectralDensityPrediction) -> _Factor:
        """The factorized Gaussian of ``prediction`` on the counted entries."""

    @property
    def pairs(self) -> tuple[PairName, ...]:
        """The sorted name pair of every pair, in the data's order."""
        return self._network.pairs

    @property
    def mask(self) -> np.ndarray:
        """The counted entries, boolean ``(P, F)``.

        The user mask without the entries of infinite detector noise. The
        compressed data axis is ``stack_pairs(values, pairs)[..., mask]``,
        pair-major.
        """
        return self._network.mask

    def mean(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        r"""The expected data :math:`\gamma_{ab}(f_k) S_h(f_k)` at ``params``, ``(D,)``.

        Parameters
        ----------
        params
            Model parameter values.
        """
        return _expected_data(self.model(params, shot_noise=False), self._network)

    def covariance(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        r"""The total covariance :math:`C` at ``params``, ``(D, D)``; needs no data.

        Parameters
        ----------
        params
            Model parameter values.
        """
        return self._evaluate(params).factor.covariance()

    def solve(self, x: ArrayLike, params: Mapping[str, ArrayLike]) -> jax.Array:
        r"""The solution :math:`\mathbf a = C^{-1}\mathbf x` of :math:`C\mathbf a = \mathbf x`, ``(..., D)``.

        Uses the factor of the covariance at ``params``; the inverse is never
        formed. Leading axes of ``x`` are solved together.

        Parameters
        ----------
        x
            Vectors on the counted entries, ``(..., D)``, in the order of
            :meth:`mean`.
        params
            Model parameter values.
        """
        return self._evaluate(params).factor.solve(x)

    def _evaluate(self, params: Mapping[str, ArrayLike]) -> _Evaluation:
        """The prediction and the factorized covariance at ``params``."""
        prediction = self.model(params, shot_noise=self._shot_noise)
        return _Evaluation(prediction, self._factorize(prediction))

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
        prediction, factor = self._evaluate(params)
        return _log_density(factor, self.observed), prediction.extras

    def __call__(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        return self.log_likelihood(params)


@jax.tree_util.register_pytree_node_class
class DiagonalGaussianLikelihood[M: SpectralDensityModel](_GaussianLikelihood[M]):
    r"""The Gaussian likelihood of one network's cross powers, detector noise alone.

    The limit of :class:`DenseGaussianLikelihood` without shot noise: the
    covariance is the diagonal :math:`\sigma_{ab,k}^2`.

    A pytree: the model, the network's arrays and the data are traced, so one
    compilation serves every model, network and data set of the same
    structure and shapes. Building one is done once outside JAX
    transformations.

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

    Attributes
    ----------
    model
        The background model.
    observed
        Observed cross powers on the counted entries, ``(..., D)``, or ``None``.
    """

    _shot_noise: ClassVar[bool] = False

    def __init__(
        self,
        model: M,
        network: str | Sequence[Detector],
        *,
        frequencies: ArrayLike,
        observation_time: float,
        observed: Mapping[PairName, ArrayLike] | None = None,
        mask: Mapping[PairName, ArrayLike] | None = None,
    ) -> None:
        frequencies = np.asarray(frequencies, dtype=float)
        _, resolved = _resolve(network, frequencies, observation_time, mask)
        self._assign(model, resolved, _compressed(resolved, observed))

    @classmethod
    def _from_network(
        cls, model: M, network: _Network, observed: jax.Array | None
    ) -> DiagonalGaussianLikelihood[M]:
        """A likelihood on a prebuilt network, with data already on its entries."""
        likelihood = object.__new__(cls)
        likelihood._assign(model, network, observed)
        return likelihood

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return (self.model, self._network, self.observed), None

    @classmethod
    def tree_unflatten(
        cls, _aux: None, children: tuple[Any, ...]
    ) -> DiagonalGaussianLikelihood[M]:
        # Rebuilt without __init__: tracing must not reload the network.
        return cls._from_network(*children)

    def _factorize(self, prediction: SpectralDensityPrediction) -> _DiagonalFactor:
        return _DiagonalFactor(
            _expected_data(prediction, self._network), self._network.scale
        )


@jax.tree_util.register_pytree_node_class
class DenseGaussianLikelihood[M: SpectralDensityModel](_GaussianLikelihood[M]):
    """The dense Gaussian likelihood of one network's cross powers, with shot noise.

    A pytree: the model, the network's arrays, the generalized ORFs and the
    data are traced, so one compilation serves every model, network and data
    set of the same structure and shapes. Building one is the expensive
    part, done once outside JAX transformations.

    Parameters
    ----------
    model
        The background model, on ``frequencies``; it must predict the
        intensity second moment.
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
        mask, model or data. Those the quadrupole does not weight are
        dropped. Mutually exclusive with ``key`` and ``num_draws``.

    Attributes
    ----------
    model
        The background model.
    observed
        Observed cross powers on the counted entries, ``(..., D)``, or ``None``.

    Raises
    ------
    ValueError
        If ``generalized_orfs`` is passed with ``key`` or ``num_draws``, lacks
        one of the quadrupole's weighted ORFs, or does not match the network
        and frequency grid.
    """

    _shot_noise: ClassVar[bool] = True

    _orfs: dict[StokesPair, jax.Array]

    def __init__(
        self,
        model: M,
        network: str | Sequence[Detector],
        *,
        frequencies: ArrayLike,
        observation_time: float,
        observed: Mapping[PairName, ArrayLike] | None = None,
        mask: Mapping[PairName, ArrayLike] | None = None,
        key: jax.Array | None = None,
        num_draws: int | None = None,
        generalized_orfs: Mapping[StokesPair, ArrayLike] | None = None,
    ) -> None:
        frequencies = np.asarray(frequencies, dtype=float)
        detectors, resolved = _resolve(network, frequencies, observation_time, mask)
        if generalized_orfs is None:
            generalized_orfs = generalized_overlap_reduction_function(
                jax.random.key(0) if key is None else key,
                2**15 if num_draws is None else num_draws,
                frequencies,
                detectors,
                components=_WEIGHTED_PAIRS,
            )
        elif key is not None or num_draws is not None:
            raise ValueError(
                "key and num_draws set the sky draws of the generalized "
                "ORFs; do not pass them with generalized_orfs"
            )
        scaled = _scaled_orfs(
            _weighted_orfs(generalized_orfs),
            (len(resolved.pairs), resolved.num_bins) * 2,
        )
        self._assign_orfs(
            model,
            resolved,
            _compressed(resolved, observed),
            {pair: resolved.restrict(values) for pair, values in scaled.items()},
        )

    def _assign_orfs(
        self,
        model: M,
        network: _Network,
        observed: jax.Array | None,
        orfs: dict[StokesPair, jax.Array],
    ) -> None:
        self._assign(model, network, observed)
        self._orfs = orfs

    @classmethod
    def _from_network(
        cls,
        model: M,
        network: _Network,
        observed: jax.Array | None,
        orfs: Mapping[StokesPair, jax.Array],
    ) -> DenseGaussianLikelihood[M]:
        """A likelihood on a prebuilt network, with data and ORFs on its entries.

        ``orfs`` are in the units of the per-pair data, each ``(D, D)``; those
        the quadrupole does not weight are dropped.
        """
        likelihood = object.__new__(cls)
        likelihood._assign_orfs(model, network, observed, _weighted_orfs(orfs))
        return likelihood

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return (self.model, self._network, self.observed, self._orfs), None

    @classmethod
    def tree_unflatten(
        cls, _aux: None, children: tuple[Any, ...]
    ) -> DenseGaussianLikelihood[M]:
        # Rebuilt without __init__: tracing must not reload the network or
        # redraw the generalized ORFs.
        likelihood = object.__new__(cls)
        likelihood._assign_orfs(*children)
        return likelihood

    @property
    def generalized_orfs(self) -> Mapping[StokesPair, jax.Array]:
        r"""The generalized ORFs :math:`\Gamma^{\mu\nu}` the quadrupole weights.

        Keyed ``II``, ``QQ``, ``UU`` and ``VV``, each ``(D, D)`` on the
        counted entries, in the units of the per-pair data.
        """
        return self._orfs

    @property
    def connected_overlap(self) -> jax.Array:
        r"""The connected overlap :math:`\Gamma^c` on the counted entries, ``(D, D)``."""
        return _stokes_sum(self._orfs)

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
            If the model does not predict its merger rate.
        """
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
                prediction, self._network, _stokes_sum(self._orfs, pairs)
            )

        return {
            "detector": jnp.diag(self._network.scale**2),
            "count": count,
            "intensity": term(_INTENSITY_PAIRS) - count,
            "linear": term(_LINEAR_PAIRS),
            "circular": term(_CIRCULAR_PAIRS),
        }

    def _factorize(self, prediction: SpectralDensityPrediction) -> _CholeskyFactor:
        mean = _expected_data(prediction, self._network)
        covariance = _catalog_covariance(
            prediction, self._network, self.connected_overlap
        ) + jnp.diag(self._network.scale**2)
        return _CholeskyFactor.from_covariance(mean, covariance)
