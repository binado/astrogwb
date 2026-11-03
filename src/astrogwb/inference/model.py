r"""The mean and the power moment of a background predicted from a reference catalog.

A :class:`CatalogModel` binds a
:func:`~astrogwb.importance.spectrum.reference_catalog` to a target population
on the observed grid. At every parameter point it predicts the two
network-independent ingredients of the dense Gaussian likelihood
(:mod:`astrogwb.inference.gaussian`): the spectrum :math:`S(f_k)` of
:mod:`astrogwb.importance.spectrum`, and the power moment of a Poisson catalog
realization over an observation time :math:`T`,

.. math::

    M_{kl} = \frac{(2/5)^2}{T} \sum_z \kappa_z\, s_z^8
        \left[W_z G W_z^T\right]_{kl}, \qquad
    G = \frac{1}{N} \sum_i w_i\, \mathbf P_i \mathbf P_i^T,

with :math:`\kappa_z` the spectrum's kernel times
:math:`(d^{\mathrm{ref}}/d_{GW})^2` at node :math:`z`, :math:`s_z^4` the
node's intensity rescaling, :math:`W_z` its linear interpolation in
:math:`\ln f` from the reference grid, and :math:`G` the importance-weighted
Gram matrix of the face-on catalog intensity. The factor :math:`(2/5)^2`
converts the face-on intensity product to the isotropic one; the inclination,
sky and polarization scatter is the network's response moment, which
multiplies :math:`M` elementwise.

The redshift nodes and the reference redshift are fixed, so the query
frequencies, and with them :math:`W_z`, do not depend on the parameters. Each
row of :math:`W_z` has at most two nonzero weights, held as indices and
weights, ``(Z, F)`` and ``(Z, F, 2)``. Without ``density_sites`` the weights
:math:`w_i` are one and the node moments :math:`s_z^8 W_z G W_z^T`,
``(Z, F, F)``, are formed once; otherwise :math:`G` is weighted per point and
each node gathers four ``(F, F)`` slices of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import luminosity_distance
from astrogwb.importance.catalog import CatalogMetadata, IntensityData
from astrogwb.importance.diagnostics import relative_ess
from astrogwb.importance.spectrum import (
    INCLINATION_MEAN,
    _node_intensity,
    _rescaling_arrays,
    _shared_terms,
    _SharedTerms,
    _weighted_mean_intensity,
)
from astrogwb.populations._types import Population

__all__ = ["CatalogModel", "Prediction", "interpolation_weights"]


class Prediction(NamedTuple):
    """What a :class:`CatalogModel` predicts at one parameter point.

    Attributes
    ----------
    mean
        The spectrum, ``(F,)``.
    power_moment
        The power moment :math:`M`, ``(F, F)``, or ``None`` when shot noise is
        not predicted.
    extras
        ``total_merger_rate`` (observer frame, mergers per second) and
        ``importance_relative_ess``, both ``()``.
    """

    mean: jax.Array
    power_moment: jax.Array | None
    extras: Mapping[str, jax.Array]


def interpolation_weights(
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
    r"""The spectrum and power moment of a rescaled reference catalog.

    Every array is a data field, so one compilation serves every catalog of
    the same shape; ``population`` and ``density_sites`` are static. Build
    with :meth:`from_catalog`; every method is traceable.

    Attributes
    ----------
    intensity
        Literal face-on reference-catalog intensity, ``(F_ref, N)``.
    mean_intensity
        Its mean over draws, ``(F_ref,)``, held only without ``density_sites``.
    node_moments
        :math:`s_z^8 W_z G W_z^T`, ``(Z, F, F)``, held only without
        ``density_sites``.
    interpolation_index, interpolation_weight
        The two-weight form of every node's :math:`W_z`
        (:func:`interpolation_weights`), ``(Z, F)`` and ``(Z, F, 2)``.
    intrinsic, proposal_log_prob, log_reference_frequencies, query_log_frequencies,
    amplitude, redshift, redshift_weights, reference_distance
        The arrays of :func:`~astrogwb.importance.spectrum._rescaling_arrays`.
    observation_seconds
        Observation time :math:`T` in seconds, ``()``.
    population
        The target's bound :class:`~astrogwb.populations.Population`.
    density_sites
        The intrinsic factors the importance weights include.
    """

    intensity: jax.Array
    mean_intensity: jax.Array | None
    node_moments: jax.Array | None
    interpolation_index: jax.Array
    interpolation_weight: jax.Array
    intrinsic: Mapping[str, jax.Array]
    proposal_log_prob: jax.Array
    log_reference_frequencies: jax.Array
    query_log_frequencies: jax.Array
    amplitude: jax.Array
    redshift: jax.Array
    redshift_weights: jax.Array
    reference_distance: jax.Array
    observation_seconds: jax.Array
    population: Population
    density_sites: tuple[str, ...]

    @classmethod
    def from_catalog(
        cls,
        data: IntensityData,
        metadata: CatalogMetadata,
        *,
        population: Population,
        frequencies: ArrayLike,
        num_redshift_nodes: int,
        observation_time: float,
        density_sites: Sequence[str] = (),
        minimum_redshift: float | None = None,
    ) -> Self:
        """Bind a reference catalog to a target on the observed ``frequencies``.

        Redshift is integrated on ``num_redshift_nodes`` nodes of
        :func:`~astrogwb.importance.spectrum.redshift_quadrature` over the
        population's window, with lower edge ``minimum_redshift`` (default the
        population's). Leave ``density_sites`` empty when no intrinsic
        hyperparameter varies: the node moments are then formed here, once.
        Call outside JAX transformations.

        Parameters
        ----------
        data
            Reference catalog intensity data.
        metadata
            Metadata describing the catalog.
        population
            Target population model.
        frequencies
            Observed frequency grid in Hz, ``(F,)``.
        num_redshift_nodes
            Number of redshift quadrature nodes.
        observation_time
            Observation time in years.
        density_sites
            Intrinsic sites the importance weights include.
        minimum_redshift
            Lower redshift bound.
        """
        arrays = _rescaling_arrays(
            data,
            metadata,
            frequencies,
            num_redshift_nodes,
            density_sites,
            minimum_redshift,
        )
        intensity = jnp.asarray(np.asarray(data["intensity"], dtype=np.float64))
        index, weight = jax.vmap(interpolation_weights, in_axes=(1, None))(
            arrays["query_log_frequencies"], arrays["log_reference_frequencies"]
        )
        constant = not density_sites
        return cls(
            intensity=intensity,
            mean_intensity=jnp.mean(intensity, axis=1) if constant else None,
            node_moments=(
                _node_moments(
                    index,
                    weight,
                    intensity @ intensity.T / intensity.shape[1],
                    arrays["amplitude"],
                )
                if constant
                else None
            ),
            interpolation_index=index,
            interpolation_weight=weight,
            observation_seconds=jnp.asarray(observation_time * SECONDS_PER_YEAR),
            population=population,
            density_sites=tuple(density_sites),
            **arrays,
        )

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

    def _variance_kernel(self, shared: _SharedTerms) -> jax.Array:
        r""":math:`\kappa_z`: the kernel times :math:`(d^{\mathrm{ref}}/d_{GW})^2`, ``(Z,)``.

        A second moment of the intensity carries the distance rescaling twice.
        """
        return shared.kernel * self._squared_distance_ratio(shared)

    def _squared_distance_ratio(self, shared: _SharedTerms) -> jax.Array:
        r""":math:`(d^{\mathrm{ref}}/d_{GW})^2` at each redshift node, ``(Z,)``."""
        gw_distance = shared.distance_ratio * luminosity_distance(
            self.redshift, shared.hubble_constant, shared.omega_m
        )
        return (self.reference_distance / gw_distance) ** 2

    def _pullback(self, projector: jax.Array) -> jax.Array:
        r""":math:`W_z^T \mathbf v` at every node, ``(Z, F_ref)``, for ``projector`` ``(F,)``."""

        def node(index: jax.Array, weight: jax.Array) -> jax.Array:
            reference = jnp.zeros(self.log_reference_frequencies.shape[0])
            reference = reference.at[index].add(weight[:, 0] * projector)
            return reference.at[index + 1].add(weight[:, 1] * projector)

        return jax.vmap(node)(self.interpolation_index, self.interpolation_weight)

    def projected_cumulants(
        self,
        params: Mapping[str, ArrayLike],
        projectors: ArrayLike,
        orders: tuple[int, ...] = (2, 3),
    ) -> jax.Array:
        r"""Cumulants of a catalog realization projected on a per-source random template.

        A realization over :math:`T` is the compound-Poisson sum
        :math:`\sum_i \beta_i` over its sources, with
        :math:`\beta_i = (2/5)\, \mathbf v_{l_i}^T \mathbf P_i / T`, source
        :math:`i`'s face-on intensity on the observed grid projected on
        ``projectors[l_i]``, its row drawn uniformly per source (a response draw
        folded into the template). Its :math:`n`-th cumulant is the expected
        count times the raw moment,

        .. math::

            \kappa_n = \sum_{z, j} a_{zj}\, \frac{1}{L} \sum_l \beta_{zjl}^n,
            \qquad a_{zj} = \frac{T \kappa^{\rm mean}_z}{q_z} \frac{w_j}{N},

        with :math:`q_z = (d^{\mathrm{ref}}/d_{GW})^2` moved from the counts
        into :math:`\beta`. The second cumulant along constant rows
        :math:`\mathbf v` is :math:`\mathbf v^T M \mathbf v`. Costs
        :math:`L Z F_{\rm ref} N`.

        Parameters
        ----------
        params
            Model parameter values.
        projectors
            Per-draw templates :math:`\mathbf v_l` on the observed grid, ``(L, F)``.
        orders
            Cumulant orders, each at least one; static.

        Returns
        -------
        jax.Array
            :math:`\kappa_n` for each of ``orders``, ``(len(orders),)``.
        """
        shared = self._shared(params)
        ratio = self._squared_distance_ratio(shared)
        weights = jnp.exp(shared.log_weights) / shared.log_weights.shape[0]
        counts = (self.observation_seconds * shared.kernel / ratio)[:, None] * weights
        scale = (INCLINATION_MEAN * ratio * self.amplitude / self.observation_seconds)[
            :, None
        ]

        def per_draw(projector: jax.Array) -> jax.Array:
            contributions = scale * (self._pullback(projector) @ self.intensity)
            return jnp.stack(
                [jnp.sum(counts * contributions**order) for order in orders]
            )

        return jnp.mean(jax.lax.map(per_draw, jnp.asarray(projectors)), axis=0)

    def _mean(self, shared: _SharedTerms) -> jax.Array:
        mean_intensity = self.mean_intensity
        if mean_intensity is None:
            mean_intensity = _weighted_mean_intensity(
                self.intensity, shared.log_weights
            )
        node_intensity = _node_intensity(
            mean_intensity,
            self.log_reference_frequencies,
            self.query_log_frequencies,
            self.amplitude,
        )
        return INCLINATION_MEAN * (node_intensity @ shared.kernel)

    def _power_moment(self, shared: _SharedTerms) -> jax.Array:
        moments = self.node_moments
        if moments is None:
            weights = jnp.exp(shared.log_weights) / shared.log_weights.shape[0]
            gram = (self.intensity * weights) @ self.intensity.T
            moments = _node_moments(
                self.interpolation_index,
                self.interpolation_weight,
                gram,
                self.amplitude,
            )
        moment = (
            INCLINATION_MEAN**2
            * jnp.einsum("z,zkl->kl", self._variance_kernel(shared), moments)
            / self.observation_seconds
        )
        # Symmetric to round-off only; made exact for the Cholesky downstream.
        return 0.5 * (moment + moment.T)

    def spectrum(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        """The spectrum at ``params``, ``(F,)``.

        Parameters
        ----------
        params
            Model parameter values.
        """
        return self._mean(self._shared(params))

    def power_moment(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        """The power moment :math:`M` at ``params``, ``(F, F)``.

        Parameters
        ----------
        params
            Model parameter values.
        """
        return self._power_moment(self._shared(params))

    def log_weights(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        """Per-draw intrinsic log importance weights at ``params``, ``(N,)``.

        Parameters
        ----------
        params
            Model parameter values.
        """
        return self._shared(params).log_weights

    def predict(
        self, params: Mapping[str, ArrayLike], *, shot_noise: bool = True
    ) -> Prediction:
        """The spectrum, the power moment and the extras at ``params``.

        Parameters
        ----------
        params
            Model parameter values.
        shot_noise
            Whether to predict the power moment; static.
        """
        shared = self._shared(params)
        extras = {
            "total_merger_rate": shared.total_rate,
            "importance_relative_ess": relative_ess(shared.log_weights),
        }
        return Prediction(
            self._mean(shared),
            self._power_moment(shared) if shot_noise else None,
            extras,
        )


jax.tree_util.register_dataclass(
    CatalogModel,
    data_fields=[
        "intensity",
        "mean_intensity",
        "node_moments",
        "interpolation_index",
        "interpolation_weight",
        "intrinsic",
        "proposal_log_prob",
        "log_reference_frequencies",
        "query_log_frequencies",
        "amplitude",
        "redshift",
        "redshift_weights",
        "reference_distance",
        "observation_seconds",
    ],
    meta_fields=["population", "density_sites"],
)
