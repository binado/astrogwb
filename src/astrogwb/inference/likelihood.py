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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import luminosity_distance
from astrogwb.gwb.importance import (
    INCLINATION_SECOND_MOMENT,
    _node_power,
    _rescaling_arrays,
    _shared_terms,
    _SharedTerms,
    _weighted_mean_power,
)
from astrogwb.importance.diagnostics import relative_ess
from astrogwb.populations._types import Population
from astrogwb.simulators.polarization_power.metadata import CatalogMetadata
from astrogwb.simulators.polarization_power.simulator import PolarizationPowerData

from .shot_noise import (
    _keep,
    amplitude_direction,
    per_frequency_direction,
    rank_one_gaussian_log_likelihood,
)

__all__ = [
    "GaussianLikelihood",
    "ImportanceGaussianLikelihood",
    "Network",
    "ShotNoiseMode",
    "gaussian_log_likelihood",
    "shot_noise_log_likelihood",
]

#: The shot-noise term of a likelihood: an amplitude mode with the projected
#: :math:`s^2` per point and network, the per-frequency mode, or an amplitude
#: mode with a fixed :math:`s^2` (``Network.relative_variance``). ``None`` is
#: the diagonal Gaussian.
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
    and ``"fixed"`` take :math:`s^2` from ``network.relative_variance`` (the
    projected variance at this point for the first, a constant for the second),
    ``"per_frequency"`` takes :math:`u_f = \sqrt{V_f}` from ``variance``.
    ``None`` is the diagonal Gaussian and ignores ``variance``. ``mode`` is
    static; the inputs each mode reads are required and not checked.
    """
    if mode is None:
        return gaussian_log_likelihood(mean, observed, network.scale, network.mask)
    if mode in ("amplitude", "fixed"):
        assert network.relative_variance is not None
        direction = amplitude_direction(mean, network.relative_variance)
    else:
        assert variance is not None
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


@dataclass(frozen=True)
class ImportanceGaussianLikelihood:
    r"""A Gaussian likelihood of the rescaled importance spectrum.

    Binds a :func:`~astrogwb.gwb.importance.reference_catalog` to a target
    population on the observed grid. The spectrum is the redshift quadrature of
    :mod:`astrogwb.gwb.importance`; with ``shot_noise`` set, the covariance of
    each network gains a rank-one term from the catalog's Poisson variance
    (:mod:`astrogwb.inference.shot_noise`).

    Every array is a data field, so the instance is a pytree whose catalog is a
    traced input: one compilation serves every catalog of the same shape.
    ``population``, ``density_sites`` and ``shot_noise`` are static and compare
    by value. Build with :meth:`from_catalog`; attach data later, or switch
    the shot-noise mode, with :func:`dataclasses.replace`. Call outside JAX
    transformations only to build; every method is traceable.

    Attributes
    ----------
    power:
        Reference-catalog polarization power, ``(F_ref, N)``.
    squared_power:
        ``power`` squared, held once so it is not formed on every evaluation;
        ``None`` unless ``shot_noise`` is ``"amplitude"`` or ``"per_frequency"``.
    mean_power, mean_squared_power:
        The mean over draws of ``power`` and of ``squared_power``, ``(F_ref,)``.
        Held only when ``density_sites`` is empty: every importance weight is
        then one, the mean does not depend on the parameters, and it is formed
        once here and not on every evaluation. ``mean_squared_power`` is also
        ``None`` unless a variance is predicted. ``None`` otherwise, and the
        weighted mean is taken at every evaluation. They are a cache of
        ``power``: :func:`dataclasses.replace` of ``power`` must replace them
        too.
    intrinsic, proposal_log_prob, log_reference_frequencies, query_log_frequencies,
    amplitude, redshift, redshift_weights, reference_distance:
        The arrays of :func:`~astrogwb.gwb.importance._rescaling_arrays`.
    observation_seconds:
        Observation time in seconds, ``()``; ``None`` unless a variance is
        predicted.
    observed:
        Observed spectrum, ``(F,)``, or ``None`` when only the spectrum is used.
    network:
        The detector network ``log_likelihood`` defaults to, or ``None``.
    population:
        The target's bound :class:`~astrogwb.populations.Population`.
    density_sites:
        The intrinsic factors the importance weights include.
    shot_noise:
        The shot-noise mode, or ``None`` for the diagonal Gaussian.
    """

    power: jax.Array
    squared_power: jax.Array | None
    mean_power: jax.Array | None
    mean_squared_power: jax.Array | None
    intrinsic: Mapping[str, jax.Array]
    proposal_log_prob: jax.Array
    log_reference_frequencies: jax.Array
    query_log_frequencies: jax.Array
    amplitude: jax.Array
    redshift: jax.Array
    redshift_weights: jax.Array
    reference_distance: jax.Array
    observation_seconds: jax.Array | None
    observed: jax.Array | None
    network: Network | None
    population: Population
    density_sites: tuple[str, ...]
    shot_noise: ShotNoiseMode | None

    @classmethod
    def from_catalog(
        cls,
        data: PolarizationPowerData,
        metadata: CatalogMetadata,
        *,
        population: Population,
        frequencies: ArrayLike,
        num_redshift_nodes: int,
        density_sites: Sequence[str],
        observed: ArrayLike | None = None,
        network: Network | None = None,
        shot_noise: ShotNoiseMode | None = None,
        observation_time: float | None = None,
        minimum_redshift: float | None = None,
    ) -> Self:
        """Bind a reference catalog to a target on the observed ``frequencies``.

        The spectrum is predicted on ``frequencies``, which need not be the
        catalog's own; redshift is integrated on ``num_redshift_nodes`` nodes
        of :func:`~astrogwb.gwb.importance.redshift_quadrature` over the
        catalog population's window, whose lower edge is ``minimum_redshift``
        (the population's own by default). It must not be below the catalog's
        :func:`~astrogwb.gwb.importance.reference_redshift`, else ``ValueError``,
        so one catalog generated low serves every window above it.
        ``density_sites`` names the intrinsic
        factors the weights include; leave it empty when no intrinsic
        hyperparameter varies, and every weight is one. Call outside JAX
        transformations.

        ``observed`` and ``network`` default to ``None`` for spectrum-only use;
        :meth:`log_likelihood` and :meth:`__call__` require them, and nothing
        checks this. ``shot_noise="amplitude"`` or ``"per_frequency"`` stores
        the squared power and needs ``observation_time`` in years;
        ``"fixed"`` takes :math:`s^2` from ``network.relative_variance``.

        The caller is trusted to pass a reference catalog of ``metadata`` and a
        waveform the contracts of :mod:`astrogwb.gwb.importance` hold for.
        """
        power = np.asarray(data["polarization_power"], dtype=np.float64)
        needs_variance = shot_noise in ("amplitude", "per_frequency")
        # No density site: every weight is one, so the means are constants.
        constant = not density_sites
        return cls(
            power=jnp.asarray(power),
            squared_power=jnp.asarray(power * power) if needs_variance else None,
            mean_power=jnp.asarray(power.mean(axis=1)) if constant else None,
            mean_squared_power=(
                jnp.asarray((power * power).mean(axis=1))
                if constant and needs_variance
                else None
            ),
            observation_seconds=(
                None
                if observation_time is None
                else jnp.asarray(observation_time * SECONDS_PER_YEAR)
            ),
            observed=None if observed is None else jnp.asarray(observed),
            network=network,
            population=population,
            density_sites=tuple(density_sites),
            shot_noise=shot_noise,
            **_rescaling_arrays(
                data,
                metadata,
                frequencies,
                num_redshift_nodes,
                density_sites,
                minimum_redshift,
            ),
        )

    def _shared(self, params: Mapping[str, ArrayLike]):
        return _shared_terms(
            params,
            population=self.population,
            intrinsic=self.intrinsic,
            proposal_log_prob=self.proposal_log_prob,
            density_sites=self.density_sites,
            redshift=self.redshift,
            redshift_weights=self.redshift_weights,
            reference_distance=self.reference_distance,
        )

    def log_weights(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        """Per-draw intrinsic log importance weights at ``params``, ``(N,)``."""
        return self._shared(params).log_weights

    def _variance_kernel(self, shared: _SharedTerms) -> jax.Array:
        r"""The spectrum's kernel times :math:`(d^{\mathrm{ref}}/d_{GW})^2`, ``(Z,)``.

        A second moment of the power carries the distance rescaling twice. The
        extra factor uses the pointwise
        :func:`~astrogwb.cosmology.luminosity_distance`, as the kernel does.
        """
        gw_distance = shared.distance_ratio * luminosity_distance(
            self.redshift, shared.hubble_constant, shared.omega_m
        )
        return shared.kernel * (self.reference_distance / gw_distance) ** 2

    def predict(self, params: Mapping[str, ArrayLike]) -> Prediction:
        r"""``(mean, variance | None, extras)`` at ``params``.

        ``mean`` is the spectrum, ``(F,)``. ``variance`` is the shot-noise
        variance of a Poisson catalog realization about it, ``(F,)``, for the
        ``"amplitude"`` and ``"per_frequency"`` modes: the weighted mean of
        ``squared_power`` is rescaled with ``amplitude**2`` (:math:`s_j^8`),
        integrated with :meth:`_variance_kernel` and multiplied by
        :data:`~astrogwb.gwb.importance.INCLINATION_SECOND_MOMENT` over
        ``observation_seconds``. ``extras`` holds ``total_merger_rate``
        (observer frame, mergers per second) and ``importance_relative_ess``,
        both ``()``.
        """
        return self._predict(self._shared(params))

    def _predict(self, shared: _SharedTerms) -> Prediction:
        mean_power = self.mean_power
        if mean_power is None:
            mean_power = _weighted_mean_power(self.power, shared.log_weights)
        node_power = _node_power(
            mean_power,
            self.log_reference_frequencies,
            self.query_log_frequencies,
            self.amplitude,
        )
        extras = {
            "total_merger_rate": shared.total_rate,
            "importance_relative_ess": relative_ess(shared.log_weights),
        }
        variance = None
        if self.shot_noise in ("amplitude", "per_frequency"):
            assert self.squared_power is not None
            assert self.observation_seconds is not None
            mean_squared_power = self.mean_squared_power
            if mean_squared_power is None:
                mean_squared_power = _weighted_mean_power(
                    self.squared_power, shared.log_weights
                )
            squared_node_power = _node_power(
                mean_squared_power,
                self.log_reference_frequencies,
                self.query_log_frequencies,
                self.amplitude**2,
            )
            variance = (
                INCLINATION_SECOND_MOMENT
                * (squared_node_power @ self._variance_kernel(shared))
                / self.observation_seconds
            )
        return node_power @ shared.kernel, variance, extras

    def _projected_relative_variance(
        self, shared: _SharedTerms, mean: jax.Array, network: Network
    ) -> jax.Array:
        r"""The variance of the catalog's fractional amplitude along ``mean``.

        Each source is projected on the noise-weighted mean,
        :math:`a = \sum_f c_f P_f` with :math:`c_f = S_f / \sigma_f^2` over the
        counted bins, before squaring, so

        .. math::

            s^2 = \frac{\langle a^2 \rangle}{\mathcal{R} T \langle a \rangle^2}
                = \frac{\mathbf{c}^T \mathrm{Cov}[\mathbf{S}^{\mathrm{cat}}]\,
                  \mathbf{c}}{(\mathbf{c}^T \mathbf{S})^2},

        the second cumulant of the compound-Poisson amplitude. At node
        :math:`j` the projection is the transposed interpolation of
        :math:`\mathbf{c}` contracted with the catalog, ``(Z, F_ref) @ (F_ref, N)``,
        once per network.
        """
        assert self.observation_seconds is not None
        keep = _keep(network.mask, mean) & (mean > 0.0)
        safe_scale = jnp.where(keep, network.scale, 1.0)
        projector = jnp.where(keep, mean / safe_scale**2, 0.0)

        def node_projector(query: jax.Array) -> jax.Array:
            """:math:`\\mathbf{c}` pulled back through one node's interpolation."""
            _, pullback = jax.vjp(
                lambda reference: jnp.interp(
                    query, self.log_reference_frequencies, reference, right=0.0
                ),
                jnp.zeros_like(self.log_reference_frequencies),
            )
            return pullback(projector)[0]

        projectors = jax.vmap(node_projector, in_axes=1)(self.query_log_frequencies)
        projected = self.amplitude[:, None] * (projectors @ self.power)
        second_moment = (projected**2 @ jnp.exp(shared.log_weights)) / (
            shared.log_weights.shape[0]
        )
        variance = (
            INCLINATION_SECOND_MOMENT
            * (second_moment @ self._variance_kernel(shared))
            / self.observation_seconds
        )
        return variance / (projector @ mean) ** 2

    def relative_variance(
        self, params: Mapping[str, ArrayLike], network: Network | None = None
    ) -> jax.Array:
        r"""The ``"amplitude"`` mode's :math:`s^2` at ``params`` under ``network``.

        The exact variance of the fractional amplitude estimator, ``()``. It is
        bounded above by the SNR-weighted
        :func:`~astrogwb.inference.shot_noise.amplitude_shot_noise_variance` of
        the predicted per-bin variance, and equals it when the shot noise is
        perfectly correlated across bins. Requires ``observation_seconds``.
        """
        shared = self._shared(params)
        mean = self._predict(shared)[0]
        network = self.network if network is None else network
        assert network is not None
        return self._projected_relative_variance(shared, mean, network)

    def spectrum(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        """The spectrum at ``params``, ``(F,)``: a :data:`~astrogwb.inference.protocol.SpectrumFn`."""
        return self.predict(params)[0]

    def log_likelihood_from_prediction(
        self, prediction: Prediction, network: Network
    ) -> jax.Array:
        """The log-likelihood of ``prediction`` under ``network``, shape ``()``.

        In the ``"amplitude"`` mode :math:`s^2` is ``network.relative_variance``,
        which :meth:`log_likelihood` sets from :meth:`relative_variance`.
        """
        mean, variance, _ = prediction
        assert self.observed is not None
        return shot_noise_log_likelihood(
            mean, variance, self.observed, network, self.shot_noise
        )

    def log_likelihood(
        self, params: Mapping[str, ArrayLike], network: Network | None = None
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        """``(log L, extras)`` at ``params`` under ``network`` (the instance's by default)."""
        shared = self._shared(params)
        prediction = self._predict(shared)
        network = self.network if network is None else network
        assert network is not None
        if self.shot_noise == "amplitude":
            network = network._replace(
                relative_variance=self._projected_relative_variance(
                    shared, prediction[0], network
                )
            )
        return self.log_likelihood_from_prediction(prediction, network), prediction[2]

    def __call__(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        return self.log_likelihood(params)


jax.tree_util.register_dataclass(
    ImportanceGaussianLikelihood,
    data_fields=[
        "power",
        "squared_power",
        "mean_power",
        "mean_squared_power",
        "intrinsic",
        "proposal_log_prob",
        "log_reference_frequencies",
        "query_log_frequencies",
        "amplitude",
        "redshift",
        "redshift_weights",
        "reference_distance",
        "observation_seconds",
        "observed",
        "network",
    ],
    meta_fields=["population", "density_sites", "shot_noise"],
)
