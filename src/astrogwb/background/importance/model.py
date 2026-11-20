r"""The spectral density and intensity second moment of a reference catalog.

A :class:`CatalogModel` binds a
:class:`~astrogwb.background.importance.catalog.ReferenceCatalog` to a target population
on the observed grid. It implements
:class:`~astrogwb.inference.model.SpectralDensityModel`: at every parameter
point it predicts the spectral density :math:`S_h(f_k) = 2\mathcal R\langle
I(f_k)\rangle` of :mod:`astrogwb.background.importance.spectrum`, and the raw intensity
second moment of the connected covariance,

.. math::

    4\mathcal R\langle I(f_k)\, I(f_l)\rangle = c_\iota \sum_z \kappa_z\, s_z^8
        \left[W_z G W_z^T\right]_{kl}, \qquad
    G = \frac{1}{N} \sum_i w_i\, \mathbf P_i \mathbf P_i^T,

with :math:`\kappa_z` the spectrum's kernel times
:math:`(d^{\mathrm{ref}}/d_{GW})^2` at node :math:`z`, :math:`s_z^4` the
node's intensity rescaling, :math:`W_z` its linear interpolation in
:math:`\ln f` from the reference grid, and :math:`G` the importance-weighted
Gram matrix of the face-on catalog intensity. The factor :math:`c_\iota`
(:data:`~astrogwb.background.importance.spectrum.INCLINATION_RAW_SECOND_MOMENT`) is the
isotropic inclination average of the face-on intensity product; the sky and
polarization scatter is the network's connected overlap
(the quadrupole's combination of
:func:`~astrogwb.detector.generalized_overlap_reduction_function`).

The redshift nodes and the reference redshift are fixed, so the query
frequencies, and with them :math:`W_z`, do not depend on the parameters. Each
row of :math:`W_z` has at most two nonzero weights, held as indices and
weights, ``(Z, F)`` and ``(Z, F, 2)``. When the target matches the proposal,
the weights :math:`w_i` are one and the node moments :math:`s_z^8 W_z G W_z^T`,
``(Z, F, F)``, are formed once; otherwise :math:`G` is weighted per point and
each node gathers four ``(F, F)`` slices of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import luminosity_distance
from astrogwb.inference.model import SpectralDensityPrediction

from .catalog import ReferenceCatalog
from .diagnostics import relative_ess
from .spectrum import (
    INCLINATION_MEAN,
    INCLINATION_RAW_SECOND_MOMENT,
    IntrinsicLogProb,
    MergerRate,
    _node_intensity,
    _rescaling_arrays,
    _shared_terms,
    _SharedTerms,
    _weighted_mean_intensity,
)

__all__ = ["CatalogModel"]


def _interpolation_weights(
    query: ArrayLike, reference: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    r"""Linear interpolation from ``reference`` to ``query`` as two weights per row.

    Returns the lower reference index :math:`i_k`, ``(F,)``, and the weights
    :math:`(1 - t_k, t_k)` of entries :math:`i_k` and :math:`i_k + 1`,
    ``(F, 2)``, such that :math:`\sum_a w_{k,a} v_{i_k + a}` equals
    ``jnp.interp(query, reference, v, right=0.0)`` for any ``v``: queries
    below ``reference`` take its first value, queries above it zero.

    Parameters
    ----------
    query
        Points to interpolate at, ``(F,)``.
    reference
        Increasing reference points, ``(F_ref,)``, at least two.
    """
    query = jnp.asarray(query)
    reference = jnp.asarray(reference)
    index = jnp.clip(
        jnp.searchsorted(reference, query, side="right") - 1,
        0,
        reference.shape[0] - 2,
    )
    lower, upper = reference[index], reference[index + 1]
    fraction = jnp.clip((query - lower) / (upper - lower), 0.0, 1.0)
    weights = jnp.stack([1.0 - fraction, fraction], axis=-1)
    return index, jnp.where((query > reference[-1])[:, None], 0.0, weights)


def _node_moment(index: jax.Array, weights: jax.Array, gram: jax.Array) -> jax.Array:
    """:math:`W G W^T` for one node's interpolation, from four gathered slices, ``(F, F)``."""
    rows = jnp.stack([index, index + 1], axis=1)
    block = gram[rows[:, :, None, None], rows[None, None, :, :]]
    return jnp.einsum("ka,kalb,lb->kl", weights, block, weights)


def _node_moments(
    index: jax.Array, weights: jax.Array, gram: jax.Array, amplitude: jax.Array
) -> jax.Array:
    r""":math:`s_z^8 W_z G W_z^T` at every node, ``(Z, F, F)``."""
    moments = jax.lax.map(
        lambda inputs: _node_moment(inputs[0], inputs[1], gram), (index, weights)
    )
    return amplitude[:, None, None] ** 2 * moments


@dataclass(frozen=True)
class CatalogModel:
    r"""The spectral density and intensity second moment of a rescaled reference catalog.

    A :class:`~astrogwb.inference.model.SpectralDensityModel`. Every array is a
    data field, so one compilation serves every catalog of the same shape and
    callables. The callables are static. Build with :meth:`from_catalog`;
    every method is traceable.
    """

    _intensity: jax.Array
    _mean_intensity: jax.Array | None
    _node_moments: jax.Array | None
    _interpolation_index: jax.Array
    _interpolation_weight: jax.Array
    _intrinsic: Mapping[str, jax.Array]
    _proposal_log_prob: jax.Array
    _log_reference_frequencies: jax.Array
    _query_log_frequencies: jax.Array
    _amplitude: jax.Array
    _redshift: jax.Array
    _redshift_weights: jax.Array
    _reference_distance: jax.Array
    merger_rate: MergerRate
    log_prob: IntrinsicLogProb | None

    @classmethod
    def from_catalog(
        cls,
        catalog: ReferenceCatalog,
        *,
        frequencies: ArrayLike,
        num_redshift_nodes: int,
        merger_rate: MergerRate,
        minimum_redshift: float,
        maximum_redshift: float,
        log_prob: IntrinsicLogProb | None = None,
        target_matches_proposal: bool = False,
    ) -> Self:
        """Bind a reference catalog to a target on the observed ``frequencies``.

        Redshift is integrated over the specified window. Call outside JAX
        transformations. Pass hashable, JAX-traceable callables and supply
        catalog-sized arrays through the reference catalog.

        Parameters
        ----------
        catalog
            Reference intensities, intrinsic draws, proposal density and geometry.
        frequencies
            Observed frequency grid in Hz, ``(F,)``.
        num_redshift_nodes
            Number of redshift quadrature nodes.
        merger_rate
            Source-frame merger rate density in Gpc^-3 yr^-1, evaluated as
            ``merger_rate(redshift, params)``.
        log_prob
            Target joint intrinsic log density, evaluated as
            ``log_prob(intrinsic, params)`` with shape ``(N,)``. Use negative
            infinity outside target support. The absolute weights are
            ``exp(log_prob - proposal_log_prob)``; the proposal must cover the
            target's support.
        minimum_redshift, maximum_redshift
            Increasing integration bounds with the lower bound at or above
            the reference redshift.
        target_matches_proposal
            Set to true in place of ``log_prob`` when both intrinsic densities
            are identical throughout the calculation. Caches the unit-weight moments.
        """
        if target_matches_proposal != (log_prob is None):
            raise TypeError("pass either log_prob or target_matches_proposal=True")
        arrays = _rescaling_arrays(
            catalog,
            frequencies,
            num_redshift_nodes,
            minimum_redshift,
            maximum_redshift,
        )
        intensity = jnp.asarray(catalog.intensity, dtype=jnp.float64)
        index, weight = jax.vmap(_interpolation_weights, in_axes=(1, None))(
            arrays["_query_log_frequencies"], arrays["_log_reference_frequencies"]
        )
        constant = log_prob is None
        return cls(
            _intensity=intensity,
            _mean_intensity=jnp.mean(intensity, axis=1) if constant else None,
            _node_moments=(
                _node_moments(
                    index,
                    weight,
                    intensity @ intensity.T / intensity.shape[1],
                    arrays["_amplitude"],
                )
                if constant
                else None
            ),
            _interpolation_index=index,
            _interpolation_weight=weight,
            merger_rate=merger_rate,
            log_prob=log_prob,
            **arrays,
        )

    def _shared(self, params: Mapping[str, ArrayLike]) -> _SharedTerms:
        return _shared_terms(
            params,
            merger_rate=self.merger_rate,
            log_prob=self.log_prob,
            intrinsic=self._intrinsic,
            proposal_log_prob=self._proposal_log_prob,
            redshift=self._redshift,
            redshift_weights=self._redshift_weights,
            reference_distance=self._reference_distance,
        )

    def _variance_kernel(self, shared: _SharedTerms) -> jax.Array:
        r""":math:`\kappa_z`: the kernel times :math:`(d^{\mathrm{ref}}/d_{GW})^2`, ``(Z,)``.

        A second moment of the intensity carries the distance rescaling twice.
        """
        return shared.kernel * self._squared_distance_ratio(shared)

    def _squared_distance_ratio(self, shared: _SharedTerms) -> jax.Array:
        r""":math:`(d^{\mathrm{ref}}/d_{GW})^2` at each redshift node, ``(Z,)``."""
        gw_distance = shared.distance_ratio * luminosity_distance(
            self._redshift, shared.hubble_constant, shared.omega_m
        )
        return (self._reference_distance / gw_distance) ** 2

    def _pullback(self, projector: jax.Array) -> jax.Array:
        r""":math:`W_z^T \mathbf v` at every node, ``(Z, F_ref)``, for ``projector`` ``(F,)``."""

        def node(index: jax.Array, weight: jax.Array) -> jax.Array:
            reference = jnp.zeros(self._log_reference_frequencies.shape[0])
            reference = reference.at[index].add(weight[:, 0] * projector)
            return reference.at[index + 1].add(weight[:, 1] * projector)

        return jax.vmap(node)(self._interpolation_index, self._interpolation_weight)

    def projected_cumulants(
        self,
        params: Mapping[str, ArrayLike],
        projectors: ArrayLike,
        *,
        observation_time: ArrayLike,
        orders: tuple[int, ...] = (2, 3),
    ) -> jax.Array:
        r"""Cumulants of a catalog realization projected on a per-source random template.

        A realization over :math:`T` is the compound-Poisson sum
        :math:`\sum_i \beta_i` over its sources, with
        :math:`\beta_i = (2/5)\, \mathbf v_{l_i}^T \mathbf P_i / T`, source
        :math:`i`'s face-on intensity on the observed grid projected on
        ``projectors[l_i]``, its row drawn uniformly per source (a response draw,
        inclination included, folded into the template). Its :math:`n`-th
        cumulant is the expected count times the raw moment,

        .. math::

            \kappa_n = \sum_{z, j} a_{zj}\, \frac{1}{L} \sum_l \beta_{zjl}^n,
            \qquad a_{zj} = \frac{T \kappa^{\rm mean}_z}{q_z} \frac{w_j}{N},

        with :math:`q_z = (d^{\mathrm{ref}}/d_{GW})^2` moved from the counts
        into :math:`\beta`. Along constant rows :math:`\mathbf v` the second
        cumulant is :math:`(2/5)^2 / c_\iota` times :math:`\mathbf v^T X
        \mathbf v / T`, with :math:`X` the intensity second moment and
        :math:`c_\iota` = :data:`~astrogwb.background.importance.spectrum.INCLINATION_RAW_SECOND_MOMENT`:
        a constant template carries no inclination scatter. Costs
        :math:`L Z F_{\rm ref} N`.

        Parameters
        ----------
        params
            Model parameter values.
        projectors
            Per-draw templates :math:`\mathbf v_l` on the observed grid, ``(L, F)``.
        observation_time
            Observation time :math:`T` in years.
        orders
            Cumulant orders, each at least one; static.

        Returns
        -------
        jax.Array
            :math:`\kappa_n` for each of ``orders``, ``(len(orders),)``.
        """
        seconds = jnp.asarray(observation_time) * SECONDS_PER_YEAR
        shared = self._shared(params)
        ratio = self._squared_distance_ratio(shared)
        weights = jnp.exp(shared.log_weights) / shared.log_weights.shape[0]
        counts = (seconds * shared.kernel / ratio)[:, None] * weights
        scale = (INCLINATION_MEAN * ratio * self._amplitude / seconds)[:, None]

        def per_draw(projector: jax.Array) -> jax.Array:
            contributions = scale * (self._pullback(projector) @ self._intensity)
            return jnp.stack(
                [jnp.sum(counts * contributions**order) for order in orders]
            )

        return jnp.mean(jax.lax.map(per_draw, jnp.asarray(projectors)), axis=0)

    def _spectral_density(self, shared: _SharedTerms) -> jax.Array:
        _mean_intensity = self._mean_intensity
        if _mean_intensity is None:
            _mean_intensity = _weighted_mean_intensity(
                self._intensity, shared.log_weights
            )
        node_intensity = _node_intensity(
            _mean_intensity,
            self._log_reference_frequencies,
            self._query_log_frequencies,
            self._amplitude,
        )
        return INCLINATION_MEAN * (node_intensity @ shared.kernel)

    def _intensity_second_moment(self, shared: _SharedTerms) -> jax.Array:
        moments = self._node_moments
        if moments is None:
            weights = jnp.exp(shared.log_weights) / shared.log_weights.shape[0]
            gram = (self._intensity * weights) @ self._intensity.T
            moments = _node_moments(
                self._interpolation_index,
                self._interpolation_weight,
                gram,
                self._amplitude,
            )
        return INCLINATION_RAW_SECOND_MOMENT * jnp.einsum(
            "z,zkl->kl", self._variance_kernel(shared), moments
        )

    def spectral_density(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        r"""The spectral density :math:`S_h` at ``params``, ``(F,)``.

        Parameters
        ----------
        params
            Model parameter values.
        """
        return self._spectral_density(self._shared(params))

    def intensity_second_moment(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        r""":math:`4\mathcal R\langle I(f_k)\, I(f_l)\rangle` at ``params``, ``(F, F)``.

        Per unit time: a realization over :math:`T` scatters by this over
        :math:`T`. Symmetric to round-off.

        Parameters
        ----------
        params
            Model parameter values.
        """
        return self._intensity_second_moment(self._shared(params))

    def log_weights(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        """Per-draw intrinsic log importance weights at ``params``, ``(N,)``.

        Parameters
        ----------
        params
            Model parameter values.
        """
        return self._shared(params).log_weights

    def __call__(
        self, params: Mapping[str, ArrayLike], *, shot_noise: bool = True
    ) -> SpectralDensityPrediction:
        """The spectral density, the intensity second moment and the extras at ``params``.

        The extras are ``total_merger_rate`` (observer frame, mergers per
        second) and ``importance_relative_ess``, both ``()``.

        Parameters
        ----------
        params
            Model parameter values.
        shot_noise
            Whether to predict the intensity second moment; static.
        """
        shared = self._shared(params)
        extras = {
            "total_merger_rate": shared.total_rate,
            "importance_relative_ess": relative_ess(shared.log_weights),
        }
        return SpectralDensityPrediction(
            self._spectral_density(shared),
            self._intensity_second_moment(shared) if shot_noise else None,
            extras,
            shared.total_rate,
        )


jax.tree_util.register_dataclass(
    CatalogModel,
    data_fields=[
        "_intensity",
        "_mean_intensity",
        "_node_moments",
        "_interpolation_index",
        "_interpolation_weight",
        "_intrinsic",
        "_proposal_log_prob",
        "_log_reference_frequencies",
        "_query_log_frequencies",
        "_amplitude",
        "_redshift",
        "_redshift_weights",
        "_reference_distance",
    ],
    meta_fields=["merger_rate", "log_prob"],
)
