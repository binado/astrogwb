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

from .compound_poisson import (
    AmplitudeDensityMethod,
    PointSet,
    amplitude_log_density,
    inclination_quadrature,
)
from .shot_noise import (
    _keep,
    amplitude_direction,
    low_rank_gaussian_log_likelihood,
    pad_modes,
    per_frequency_direction,
    rank_one_gaussian_log_likelihood,
    shot_noise_modes,
)

__all__ = [
    "GaussianLikelihood",
    "ImportanceCompoundPoissonLikelihood",
    "ImportanceGaussianLikelihood",
    "Network",
    "ShotNoiseMode",
    "gaussian_log_likelihood",
    "shot_noise_log_likelihood",
]

#: The shot-noise term of a likelihood: an amplitude mode with the projected
#: :math:`s^2` per point and network, the per-frequency mode, an amplitude
#: mode with a fixed :math:`s^2` (``Network.relative_variance``), or ``k``
#: eigenmodes of the catalog covariance (``Network.modes``). ``None`` is the
#: diagonal Gaussian.
type ShotNoiseMode = Literal["amplitude", "per_frequency", "fixed", "modes"]

#: ``(mean, variance | None, extras)``, what ``predict`` returns.
type Prediction = tuple[jax.Array, jax.Array | None, Mapping[str, jax.Array]]

_HALF_LOG_TWO_PI = 0.5 * float(np.log(2.0 * np.pi))


class Network(NamedTuple):
    r"""The detector-network half of a likelihood.

    Stacking K networks means stacking each leaf along a new leading axis, and
    every network in a stack must share one structure: each optional leaf
    ``None`` in all of them, or arrays of one shape (pad ``modes`` with
    :func:`~astrogwb.inference.shot_noise.pad_modes`).

    Attributes
    ----------
    scale:
        Per-bin standard deviation, ``(F,)``.
    mask:
        Boolean ``(F,)`` of the bins counted; ``None`` counts all. Masked bins
        contribute exactly zero, whatever their ``scale``.
    relative_variance:
        A fixed :math:`s^2`, shape ``()``, for the ``"fixed"`` shot-noise mode.
    response:
        Equally weighted isotropic draws of the network response
        :math:`r_l(f)` on the likelihood's frequency grid, ``(L, F)``
        (:func:`~astrogwb.detector.response.sample_network_response`). Sky
        position and orientation are independent of the catalog, so the
        catalog covariance becomes the Hadamard product

        .. math::

            \mathrm{Cov}[S^{\mathrm{cat}}]_{f f'} = \frac{\mathcal{R}}{T}
                \langle P_f P_{f'} \rangle\,
                \frac{1}{L} \sum_l r_l(f)\, r_l(f'),

        with :math:`P` the catalog power at the effective inclination. The
        response carries the inclination, so it replaces
        :data:`~astrogwb.gwb.importance.INCLINATION_SECOND_MOMENT` everywhere
        rather than multiplying it: in the projected variance, the per-bin
        variance of the ``"per_frequency"`` mode and the inclination nodes of
        the compound-Poisson point set. ``None`` keeps the sky-averaged power.
    modes:
        The shot-noise basis :math:`U` of the ``"modes"`` mode, ``(F, k)``:
        orthonormal over the counted bins in whitened coordinates
        (:func:`~astrogwb.inference.shot_noise.shot_noise_modes`). Zero
        columns are inert.
    mode_blocks:
        The per-node blocks :math:`M_z` of the shot noise in the basis,
        ``(Z, k, k)``
        (:meth:`ImportanceGaussianLikelihood.mode_blocks`), held to skip
        recomputing them at every point. They do not depend on the parameters
        while the importance weights do not, that is with no ``density_sites``;
        the caller is trusted to hold them only then. ``None`` computes them.
    mode_covariance:
        The whitened shot-noise covariance in the basis, :math:`B`,
        ``(k, k)``, which
        :meth:`ImportanceGaussianLikelihood.log_likelihood` sets per point
        in the ``"modes"`` mode, as it sets ``relative_variance`` in the
        ``"amplitude"`` mode.
    """

    scale: jax.Array
    mask: jax.Array | None = None
    relative_variance: jax.Array | None = None
    response: jax.Array | None = None
    modes: jax.Array | None = None
    mode_blocks: jax.Array | None = None
    mode_covariance: jax.Array | None = None


def _orientation_covariance(network: Network | None) -> jax.Array | float:
    r"""The orientation factor of a second moment of the power, ``()`` or ``(F, F)``.

    :data:`~astrogwb.gwb.importance.INCLINATION_SECOND_MOMENT` for the
    sky-averaged power, else :math:`\langle r_f r_{f'} \rangle` of the
    network's response draws, which multiplies elementwise.
    """
    if network is None or network.response is None:
        return INCLINATION_SECOND_MOMENT
    return network.response.T @ network.response / network.response.shape[0]


def _orientation_variance(network: Network) -> jax.Array | float:
    r"""The diagonal of :func:`_orientation_covariance`, ``()`` or ``(F,)``."""
    if network.response is None:
        return INCLINATION_SECOND_MOMENT
    return jnp.mean(network.response**2, axis=0)


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
    ``"modes"`` is the low-rank covariance of ``network.modes`` and
    ``network.mode_covariance``
    (:func:`~astrogwb.inference.shot_noise.low_rank_gaussian_log_likelihood`).
    ``None`` is the diagonal Gaussian and ignores ``variance``. ``mode`` is
    static; the inputs each mode reads are required and not checked.
    """
    if mode is None:
        return gaussian_log_likelihood(mean, observed, network.scale, network.mask)
    if mode == "modes":
        assert network.modes is not None
        assert network.mode_covariance is not None
        return low_rank_gaussian_log_likelihood(
            observed,
            mean,
            network.scale,
            network.modes,
            network.mode_covariance,
            network.mask,
        )
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
class _ImportanceCatalog:
    r"""What every likelihood of a rescaled reference catalog shares.

    The fields of :func:`~astrogwb.gwb.importance._rescaling_arrays` plus the
    data, and the stages that do not depend on how the shot noise is modelled:
    the importance terms of ``population(params)``, the predicted spectrum, and
    the projection of every catalog source on a network's noise-weighted
    template. Subclasses register themselves as pytrees with their own fields.
    """

    power: jax.Array
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

    def _shared(self, params: Mapping[str, ArrayLike]) -> _SharedTerms:
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

    def _squared_distance_ratio(self, shared: _SharedTerms) -> jax.Array:
        r""":math:`(d^{\mathrm{ref}}/d_{GW})^2` at each redshift node, ``(Z,)``.

        A second moment of the power carries the distance rescaling twice. The
        extra factor uses the pointwise
        :func:`~astrogwb.cosmology.luminosity_distance`, as the kernel does.
        """
        gw_distance = shared.distance_ratio * luminosity_distance(
            self.redshift, shared.hubble_constant, shared.omega_m
        )
        return (self.reference_distance / gw_distance) ** 2

    def _variance_kernel(self, shared: _SharedTerms) -> jax.Array:
        r"""The spectrum's kernel times :meth:`_squared_distance_ratio`, ``(Z,)``."""
        return shared.kernel * self._squared_distance_ratio(shared)

    def _mean_power(self, shared: _SharedTerms) -> jax.Array:
        """The importance-weighted mean of ``power`` over draws, ``(F_ref,)``."""
        return _weighted_mean_power(self.power, shared.log_weights)

    def _mean(self, shared: _SharedTerms) -> tuple[jax.Array, dict[str, jax.Array]]:
        """The spectrum, ``(F,)``, and the extras every prediction carries."""
        node_power = _node_power(
            self._mean_power(shared),
            self.log_reference_frequencies,
            self.query_log_frequencies,
            self.amplitude,
        )
        extras = {
            "total_merger_rate": shared.total_rate,
            "importance_relative_ess": relative_ess(shared.log_weights),
        }
        return node_power @ shared.kernel, extras

    def spectrum(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        """The spectrum at ``params``, ``(F,)``: a :data:`~astrogwb.inference.protocol.SpectrumFn`."""
        return self._mean(self._shared(params))[0]

    def _node_projectors(self, projector: jax.Array) -> jax.Array:
        r""":math:`\mathbf{c}` pulled back through every node's interpolation, ``(Z, F_ref)``.

        The transposed interpolation of ``projector``, :math:`\mathbf{c}` with
        shape ``(F,)``, from node :math:`m`'s observed frequencies to the
        reference grid. It is linear in :math:`\mathbf{c}` and does not touch
        the catalog.
        """

        def node_projector(query: jax.Array) -> jax.Array:
            """:math:`\\mathbf{c}` pulled back through one node's interpolation."""
            _, pullback = jax.vjp(
                lambda reference: jnp.interp(
                    query, self.log_reference_frequencies, reference, right=0.0
                ),
                jnp.zeros_like(self.log_reference_frequencies),
            )
            return pullback(projector)[0]

        return jax.vmap(node_projector, in_axes=1)(self.query_log_frequencies)

    def _project(self, projector: jax.Array) -> jax.Array:
        r"""Every catalog source projected on ``projector``, per node, ``(Z, N)``.

        Entry :math:`(m, j)` is :math:`\sum_f c_f P_{f j}(z_m)`, source ``j``'s
        power interpolated to node ``m``'s observed frequencies, rescaled by
        ``amplitude`` and contracted with :math:`\mathbf{c}` = ``projector``,
        ``(F,)``. The contraction is :meth:`_node_projectors` against the
        catalog, ``(Z, F_ref) @ (F_ref, N)``.
        """
        return self.amplitude[:, None] * (self._node_projectors(projector) @ self.power)


def _template_projector(mean: jax.Array, network: Network) -> jax.Array:
    r""":math:`c_f = S_f / \sigma_f^2` over the counted bins with signal, else 0, ``(F,)``.

    :math:`\mathbf{c}^T \mathbf{S} = \rho^2`, the template's squared SNR, and
    :math:`\mathbf{c}^T \mathbf{d} / \rho^2` the amplitude estimator.
    """
    keep = _keep(network.mask, mean) & (mean > 0.0)
    safe_scale = jnp.where(keep, network.scale, 1.0)
    return jnp.where(keep, mean / safe_scale**2, 0.0)


@dataclass(frozen=True)
class ImportanceGaussianLikelihood(_ImportanceCatalog):
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
    gram:
        The Gram matrix of the catalog, :math:`\sum_i P_i P_i^T / N`,
        ``(F_ref, F_ref)``. Held only when ``density_sites`` is empty and
        ``shot_noise`` is ``"amplitude"`` or ``"modes"``: the second moment of the projected
        power is then a quadratic form in it, whose cost does not grow with the
        catalog, and it is formed once here. ``None`` otherwise, and every
        source is projected at every evaluation. A cache of ``power`` as well.
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

    squared_power: jax.Array | None
    mean_power: jax.Array | None
    mean_squared_power: jax.Array | None
    gram: jax.Array | None
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
        ``"fixed"`` takes :math:`s^2` from ``network.relative_variance``;
        ``"modes"`` needs ``observation_time`` and ``network.modes``.

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
            gram=(
                jnp.asarray(power @ power.T / power.shape[1])
                if constant and shot_noise in ("amplitude", "modes")
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

    def predict(self, params: Mapping[str, ArrayLike]) -> Prediction:
        r"""``(mean, variance | None, extras)`` at ``params``.

        ``mean`` is the spectrum, ``(F,)``. ``variance`` is the shot-noise
        variance of a Poisson catalog realization about it, ``(F,)``, for the
        ``"amplitude"`` and ``"per_frequency"`` modes: the weighted mean of
        ``squared_power`` is rescaled with ``amplitude**2`` (:math:`s_j^8`),
        integrated with :meth:`_variance_kernel` and multiplied by
        :data:`~astrogwb.gwb.importance.INCLINATION_SECOND_MOMENT` over
        ``observation_seconds``; it does not see the network, and
        :meth:`log_likelihood_from_prediction` swaps that factor for the
        network's response when one is set. ``extras`` holds ``total_merger_rate``
        (observer frame, mergers per second) and ``importance_relative_ess``,
        both ``()``.
        """
        return self._predict(self._shared(params))

    def _mean_power(self, shared: _SharedTerms) -> jax.Array:
        if self.mean_power is None:
            return super()._mean_power(shared)
        return self.mean_power

    def _predict(self, shared: _SharedTerms) -> Prediction:
        mean, extras = self._mean(shared)
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
        return mean, variance, extras

    def _projected_second_moment(
        self, shared: _SharedTerms, projector: jax.Array
    ) -> jax.Array:
        r"""The weighted mean over draws of the squared projection, per node, ``(Z,)``.

        :math:`\langle a_m^2 \rangle` with :math:`a_{m j}` the entry of
        :meth:`_project`. With the :attr:`gram` held it is
        :math:`s_m^{2} \mathbf{p}_m^T G \mathbf{p}_m` for the node projector
        :math:`\mathbf{p}_m` and the rescaling :math:`s_m`; otherwise the
        projection of every source is formed.
        """
        if self.gram is None:
            projected = self._project(projector)
            return (projected**2 @ jnp.exp(shared.log_weights)) / (
                shared.log_weights.shape[0]
            )
        nodes = self._node_projectors(projector)
        return self.amplitude**2 * jnp.sum((nodes @ self.gram) * nodes, axis=1)

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
        once per network. Under a ``network.response`` each draw :math:`l`
        projects :math:`P_f\, r_l(f)` instead, on the projector
        :math:`c_f\, r_l(f)`, and the squares are averaged over draws; this is
        :math:`\mathbf{c}^T \mathrm{Cov}\, \mathbf{c}` of the Hadamard
        covariance. Otherwise the inclination factor
        :data:`~astrogwb.gwb.importance.INCLINATION_SECOND_MOMENT` multiplies.
        """
        assert self.observation_seconds is not None
        projector = _template_projector(mean, network)
        if network.response is None:
            second_moment = INCLINATION_SECOND_MOMENT * self._projected_second_moment(
                shared, projector
            )
        else:
            second_moment = jnp.mean(
                jax.lax.map(
                    lambda draw: self._projected_second_moment(
                        shared, projector * draw
                    ),
                    network.response,
                ),
                axis=0,
            )
        variance = second_moment @ self._variance_kernel(shared)
        return variance / self.observation_seconds / (projector @ mean) ** 2

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

    def catalog_covariance(
        self, params: Mapping[str, ArrayLike], network: Network | None = None
    ) -> jax.Array:
        r"""The shot-noise covariance of a Poisson catalog at ``params``, ``(F, F)``.

        Campbell's theorem for the compound-Poisson sum of per-source spectra,

        .. math::

            \mathrm{Cov}[S^{\mathrm{cat}}_k, S^{\mathrm{cat}}_{k'}]
                = \frac{\mathcal{R}}{T} \langle P_k P_{k'} \rangle,

        with the average over the redshift nodes and the weighted catalog, as
        in :meth:`predict`. A source sits at one redshift, so the nodes add
        without cross terms. At node :math:`j` the source spectrum is the
        linear interpolation :math:`W_j` of the reference power, so the
        contribution is :math:`W_j G W_j^T` with the weighted Gram matrix
        :math:`G = \sum_i w_i P_i P_i^T / N` of the catalog, ``(F_ref, F_ref)``,
        formed once.

        The quadratic form along :math:`\mathbf{c}` reproduces
        :meth:`relative_variance`, and the diagonal is the per-bin variance of
        :meth:`predict` up to the interpolation order of the square, which it
        never exceeds. Under ``network`` (the instance's by default) with a
        ``response``, the inclination factor is replaced by the Hadamard product
        with :math:`\langle r_f r_{f'} \rangle` (see :class:`Network`).
        Requires ``observation_seconds``; ``squared_power`` is not used.
        """
        assert self.observation_seconds is not None
        shared = self._shared(params)
        num_bins = self.query_log_frequencies.shape[0]
        covariance, _ = self._fold_node_moments(
            shared,
            lambda carry, moment, kernel: (carry + kernel * moment, None),
            jnp.zeros((num_bins, num_bins)),
        )
        # Symmetric to round-off only; made exact for eigh and Cholesky.
        covariance = 0.5 * (covariance + covariance.T)
        network = self.network if network is None else network
        return _orientation_covariance(network) * covariance / self.observation_seconds

    def _fold_node_moments(
        self, shared: _SharedTerms, step: Callable, init: Any
    ) -> tuple[Any, Any]:
        r"""Scan ``step(carry, moment, kernel)`` over the redshift nodes.

        ``moment`` is node :math:`z`'s second moment of the power on the observed
        grid, :math:`s_z^4 W_z G W_z^T` (the ``amplitude`` rescaling squared,
        :math:`W_z` the node's interpolation and :math:`G` the weighted Gram
        matrix of the catalog), ``(F, F)``, and ``kernel`` is
        :meth:`_variance_kernel` there. Returns the final carry and the stacked
        outputs, as :func:`jax.lax.scan` does.
        """
        if self.gram is None:
            weights = jnp.exp(shared.log_weights) / shared.log_weights.shape[0]
            gram = (self.power * weights) @ self.power.T
        else:
            gram = self.gram
        identity = jnp.eye(self.log_reference_frequencies.shape[0])

        def node(carry: Any, inputs: tuple) -> tuple[Any, Any]:
            query, amplitude, kernel = inputs
            interpolation = jax.vmap(
                lambda reference: jnp.interp(
                    query, self.log_reference_frequencies, reference, right=0.0
                ),
                out_axes=1,
            )(identity)
            moment = amplitude**2 * (interpolation @ gram @ interpolation.T)
            return step(carry, moment, kernel)

        return jax.lax.scan(
            node,
            init,
            (
                self.query_log_frequencies.T,
                self.amplitude,
                self._variance_kernel(shared),
            ),
        )

    def _projected_cross_moment(
        self, shared: _SharedTerms, projectors: jax.Array
    ) -> jax.Array:
        r""":meth:`_projected_second_moment` for ``k`` projectors at once, ``(Z, k, k)``.

        Entry :math:`(z, i, j)` is the weighted mean over draws of
        :math:`a_{z i} a_{z j}`, the source projected on ``projectors[i]`` and
        ``projectors[j]`` at node :math:`z`; ``projectors`` is ``(k, F)``.
        """
        if self.gram is None:
            projected = jax.vmap(self._project)(projectors)
            weights = jnp.exp(shared.log_weights) / shared.log_weights.shape[0]
            return jnp.einsum("izn,jzn,n->zij", projected, projected, weights)
        nodes = jax.vmap(self._node_projectors)(projectors)
        return self.amplitude[:, None, None] ** 2 * jnp.einsum(
            "izf,fg,jzg->zij", nodes, self.gram, nodes
        )

    def _mode_blocks(self, shared: _SharedTerms, network: Network) -> jax.Array:
        assert network.modes is not None
        keep = _keep(network.mask, network.scale)
        projectors = jnp.where(
            keep[:, None],
            network.modes / jnp.where(keep, network.scale, 1.0)[:, None],
            0.0,
        ).T
        if network.response is None:
            return INCLINATION_SECOND_MOMENT * self._projected_cross_moment(
                shared, projectors
            )
        orientation = _orientation_covariance(network)
        _, blocks = self._fold_node_moments(
            shared,
            lambda carry, moment, kernel: (
                carry,
                projectors @ (moment * orientation) @ projectors.T,
            ),
            0.0,
        )
        return blocks

    def mode_blocks(
        self, params: Mapping[str, ArrayLike], network: Network | None = None
    ) -> jax.Array:
        r"""The per-node shot noise in ``network.modes``, ``(Z, k, k)``.

        With :math:`\tilde U = U / \sigma` over the counted bins,

        .. math::

            M_z = \tilde U^T \left( s_z^4 W_z G W_z^T \circ R \right) \tilde U,

        with :math:`R = \langle r_f r_{f'} \rangle` under a
        ``network.response`` and the constant
        :data:`~astrogwb.gwb.importance.INCLINATION_SECOND_MOMENT` otherwise
        (then contracted as projections, without the dense moment). The
        whitened shot-noise covariance in the basis is
        :math:`B = \sum_z \kappa_z M_z / T`, with :math:`\kappa_z` the
        :meth:`_variance_kernel`, so :math:`U B U^T` is the whitened
        :meth:`catalog_covariance` restricted to the basis. Hold the result
        on ``network.mode_blocks`` to skip this at every point; that is exact
        while the importance weights do not vary. Requires
        ``observation_seconds`` for :meth:`log_likelihood`.
        """
        network = self.network if network is None else network
        assert network is not None
        return self._mode_blocks(self._shared(params), network)

    def with_shot_noise_modes(
        self,
        params: Mapping[str, ArrayLike],
        network: Network,
        *,
        tolerance: float = 1e-6,
        max_modes: int = 32,
        num_modes: int | None = None,
        hold_blocks: bool | None = None,
    ) -> Network:
        """``network`` with the shot-noise basis of the ``"modes"`` mode.

        The basis is :func:`~astrogwb.inference.shot_noise.shot_noise_modes` of
        :meth:`catalog_covariance` at ``params`` under ``network`` (with its
        ``response``), padded with zero columns to ``num_modes`` when given, so
        that networks stack. ``hold_blocks`` also holds :meth:`mode_blocks` at
        ``params``; by default it does when no importance weight varies (no
        ``density_sites``), where they are exact at every point. Call outside
        JAX transformations.
        """
        modes = shot_noise_modes(
            self.catalog_covariance(params, network),
            network.scale,
            network.mask,
            tolerance=tolerance,
            max_modes=max_modes,
        )
        if num_modes is not None:
            modes = pad_modes(modes, num_modes)
        network = network._replace(modes=modes)
        if hold_blocks is None:
            hold_blocks = not self.density_sites
        if hold_blocks:
            network = network._replace(mode_blocks=self.mode_blocks(params, network))
        return network

    def _mode_covariance(self, shared: _SharedTerms, network: Network) -> jax.Array:
        r""":math:`B = \sum_z \kappa_z M_z / T`, ``(k, k)``."""
        assert self.observation_seconds is not None
        blocks = (
            self._mode_blocks(shared, network)
            if network.mode_blocks is None
            else network.mode_blocks
        )
        covariance = (
            jnp.einsum("z,zij->ij", self._variance_kernel(shared), blocks)
            / self.observation_seconds
        )
        return 0.5 * (covariance + covariance.T)

    def log_likelihood_from_prediction(
        self, prediction: Prediction, network: Network
    ) -> jax.Array:
        r"""The log-likelihood of ``prediction`` under ``network``, shape ``()``.

        In the ``"amplitude"`` mode :math:`s^2` is ``network.relative_variance``,
        which :meth:`log_likelihood` sets from :meth:`relative_variance`. A
        ``network.response`` rescales the per-bin variance of ``prediction``
        from the inclination factor to :math:`\langle r_f^2 \rangle`.
        """
        mean, variance, _ = prediction
        assert self.observed is not None
        if variance is not None and network.response is not None:
            variance = (
                variance * _orientation_variance(network) / INCLINATION_SECOND_MOMENT
            )
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
        elif self.shot_noise == "modes":
            network = network._replace(
                mode_covariance=self._mode_covariance(shared, network)
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
        "gram",
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


@dataclass(frozen=True)
class ImportanceCompoundPoissonLikelihood(_ImportanceCatalog):
    r"""The shot-noise likelihood with the exact compound-Poisson amplitude.

    The ``"amplitude"`` mode of :class:`ImportanceGaussianLikelihood` factors
    into a shape part and a Gaussian in the amplitude estimator
    :math:`\hat B = \mathbf{c}^T \mathbf{d} / \rho^2` (with
    :math:`c_f = S_f / \sigma_f^2` and :math:`\rho^2 = \mathbf{c}^T \mathbf{S}`).
    This class keeps the shape part and replaces that Gaussian with the exact
    density of :math:`\hat B`, a compound-Poisson sum of the catalog's sources
    in Gaussian detector noise (:mod:`astrogwb.inference.compound_poisson`):

    .. math::

        \log L = -\tfrac12 \chi_\perp^2 - \sum_f \log\sigma_f
                 - \tfrac{F - 1}{2} \log 2\pi - \log\rho + \log p(\hat B),

    where :math:`\chi_\perp^2` is the whitened residual orthogonal to the
    template and :math:`F` counts the kept bins. With ``method="gaussian"``,
    :math:`p = \mathcal{N}(1, \rho^{-2} + s^2)` and this is the Gaussian
    class's ``"amplitude"`` mode.

    The point set (:meth:`point_set`) holds one point per redshift node
    :math:`m`, catalog draw :math:`j` and inclination node :math:`l`:

    .. math::

        \beta_{mjl} = \frac{q_m\, P_{mj}\, \tilde F(c_l)}{T \rho^2}, \qquad
        a_{mjl} = \frac{T K_m}{q_m}\, \frac{w_j}{N}\, u_l,

    with :math:`P_{mj} = \mathbf{c}^T \mathbf{P}_j(z_m)` source :math:`j` at
    node :math:`m` projected on the template, :math:`K_m` the spectrum's kernel,
    :math:`q_m = (d^{\mathrm{ref}}/d_{GW})^2`, :math:`w_j` the importance
    weights and :math:`(c_l, u_l)` the
    :func:`~astrogwb.inference.compound_poisson.inclination_quadrature`. Under
    a ``network.response`` the nodes :math:`l` are its draws with equal weights,
    and :math:`P_{mj}\,\tilde F(c_l)` becomes the projection of source
    :math:`j`'s power times :math:`r_l(f)`. Its first
    two moments are one and the Gaussian class's
    :meth:`~ImportanceGaussianLikelihood.relative_variance`.

    The instance is a pytree as :class:`ImportanceGaussianLikelihood` is, and
    vmaps over stacked networks the same way. ``method`` and the node counts
    are static. Its cost is a few passes over the
    :math:`Z \times N \times L` point set per network and evaluation.

    Attributes
    ----------
    power, intrinsic, proposal_log_prob, log_reference_frequencies,
    query_log_frequencies, amplitude, redshift, redshift_weights,
    reference_distance, observed, network, population, density_sites:
        As for :class:`ImportanceGaussianLikelihood`.
    observation_seconds:
        Observation time :math:`T` in seconds, ``()``; required.
    method:
        How :math:`p(\hat B)` is evaluated: ``"saddlepoint"`` (normalized),
        ``"hermite"`` (Gauss-Hermite on the contour through the saddle) or
        ``"gaussian"``.
    num_inclination_nodes:
        Gauss-Legendre nodes :math:`L` in :math:`\cos\iota`. Eight are plenty
        when :math:`\mathcal{R} T` is large; the skewness at small
        :math:`\mathcal{R} T` needs more (24). Unused under a network that
        carries a ``response``.
    num_hermite_nodes:
        Contour nodes of ``"hermite"``.
    num_normalization_nodes:
        Nodes of the ``"saddlepoint"`` normalization.
    """

    method: AmplitudeDensityMethod = "saddlepoint"
    num_inclination_nodes: int = 8
    num_hermite_nodes: int = 8
    num_normalization_nodes: int = 4

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
        observation_time: float,
        observed: ArrayLike | None = None,
        network: Network | None = None,
        minimum_redshift: float | None = None,
        method: AmplitudeDensityMethod = "saddlepoint",
        num_inclination_nodes: int = 8,
        num_hermite_nodes: int = 8,
        num_normalization_nodes: int = 4,
    ) -> Self:
        """Bind a reference catalog to a target on the observed ``frequencies``.

        As :meth:`ImportanceGaussianLikelihood.from_catalog`, with
        ``observation_time`` in years required and the density options of the
        class. Call outside JAX transformations.
        """
        return cls(
            power=jnp.asarray(np.asarray(data["polarization_power"], dtype=np.float64)),
            observation_seconds=jnp.asarray(observation_time * SECONDS_PER_YEAR),
            observed=None if observed is None else jnp.asarray(observed),
            network=network,
            population=population,
            density_sites=tuple(density_sites),
            method=method,
            num_inclination_nodes=num_inclination_nodes,
            num_hermite_nodes=num_hermite_nodes,
            num_normalization_nodes=num_normalization_nodes,
            **_rescaling_arrays(
                data,
                metadata,
                frequencies,
                num_redshift_nodes,
                density_sites,
                minimum_redshift,
            ),
        )

    def predict(self, params: Mapping[str, ArrayLike]) -> Prediction:
        """``(mean, None, extras)`` at ``params``, as the Gaussian class's."""
        mean, extras = self._mean(self._shared(params))
        return mean, None, extras

    def _point_set(
        self,
        shared: _SharedTerms,
        mean: jax.Array,
        projector: jax.Array,
        network: Network,
    ) -> PointSet:
        assert self.observation_seconds is not None
        seconds = self.observation_seconds
        squared_snr = projector @ mean
        ratio = self._squared_distance_ratio(shared)
        draw_weights = jnp.exp(shared.log_weights) / shared.log_weights.shape[0]
        if network.response is None:
            pattern, node_weights = inclination_quadrature(self.num_inclination_nodes)
            # (Z, N, L): the inclination scales every source as a whole.
            projected = self._project(projector)[..., None] * jnp.asarray(pattern)
        else:
            # (Z, N, L): each response draw reweights the source across the band.
            projected = jnp.moveaxis(
                jax.vmap(lambda draw: self._project(projector * draw))(
                    network.response
                ),
                0,
                -1,
            )
            node_weights = jnp.full(
                network.response.shape[0], 1.0 / network.response.shape[0]
            )
        amplitudes = (ratio[:, None, None] * projected) / (seconds * squared_snr)
        weights = ((seconds * shared.kernel / ratio)[:, None] * draw_weights)[
            ..., None
        ] * jnp.asarray(node_weights)
        return PointSet(weights, amplitudes, squared_snr)

    def point_set(
        self, params: Mapping[str, ArrayLike], network: Network | None = None
    ) -> PointSet:
        """The compound-Poisson point set of :math:`\\hat B` at ``params``, ``(Z, N, L)``.

        Under ``network``, the instance's by default.
        """
        shared = self._shared(params)
        mean = self._mean(shared)[0]
        network = self.network if network is None else network
        assert network is not None
        return self._point_set(
            shared, mean, _template_projector(mean, network), network
        )

    def log_likelihood(
        self, params: Mapping[str, ArrayLike], network: Network | None = None
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        """``(log L, extras)`` at ``params`` under ``network`` (the instance's by default)."""
        assert self.observed is not None
        shared = self._shared(params)
        mean, extras = self._mean(shared)
        network = self.network if network is None else network
        assert network is not None
        projector = _template_projector(mean, network)
        points = self._point_set(shared, mean, projector, network)
        squared_snr = points.noise_precision

        keep = _keep(network.mask, mean)
        safe_scale = jnp.where(keep, network.scale, 1.0)
        residual = self.observed - mean
        whitened = jnp.where(keep, residual / safe_scale, 0.0)
        along = projector @ residual
        chi_squared_perp = jnp.sum(whitened * whitened) - along**2 / squared_snr
        log_normalization = (
            jnp.sum(jnp.where(keep, jnp.log(safe_scale) + _HALF_LOG_TWO_PI, 0.0))
            - _HALF_LOG_TWO_PI
            + 0.5 * jnp.log(squared_snr)
        )
        log_amplitude = amplitude_log_density(
            1.0 + along / squared_snr,
            points,
            self.method,
            num_hermite_nodes=self.num_hermite_nodes,
            num_normalization_nodes=self.num_normalization_nodes,
        )
        value = -0.5 * chi_squared_perp - log_normalization + log_amplitude
        return value, extras

    def __call__(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        return self.log_likelihood(params)


jax.tree_util.register_dataclass(
    ImportanceCompoundPoissonLikelihood,
    data_fields=[
        "power",
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
    meta_fields=[
        "population",
        "density_sites",
        "method",
        "num_inclination_nodes",
        "num_hermite_nodes",
        "num_normalization_nodes",
    ],
)
