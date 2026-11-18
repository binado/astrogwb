r"""The background spectrum as a redshift quadrature over an intrinsic sample.

The spectrum is an expectation over sources. Only the intrinsic parameters
are left to Monte Carlo here: redshift is integrated on fixed Gauss-Legendre
nodes in the scale factor :math:`1/(1 + z)` (:func:`redshift_quadrature`),
and inclination is averaged exactly. Every one of :math:`N` intrinsic
draws is placed at each of :math:`Z` nodes, so

.. math::

    S(f; \Lambda) = \sum_j w_j\, K(z_j \mid \Lambda)
        \frac{1}{N} \sum_k \omega_k(\Lambda)\, (2/5) P(f; \theta_k, z_j),

with :math:`P` the intensity at reference distance :math:`d^{\mathrm{ref}}`,
:math:`\omega_k` the ratio of the target to the drawn intrinsic density
(one when the intrinsic hyperparameters are not varied), and :math:`K` the
closed-form :func:`phinney_kernel`: the merger rate density times
:math:`[d^{\mathrm{ref}}/d_{GW}]^2`, in which the comoving distance and the
redshift density's normalization cancel, so redshift carries no Monte Carlo
variance. Inclination is fixed at zero (face-on), and the stored intensity is literal
:math:`|h_+|^2 + |h_\times|^2`. The mean receives the explicit factor
:data:`INCLINATION_MEAN`; this is exact only for the :math:`(2, 2)` mode alone.

The intensity is not generated at every node. For a :math:`(2, 2)`-mode,
aligned-spin, quasi-circular waveform the strain depends on mass only through
:math:`M_d f` and an overall :math:`M_d^2`, with :math:`M_d = M (1 + z)`, so the
intensity of a draw generated at the reference redshift :math:`z_{\mathrm{ref}}`
(by default the window's lower edge) is, at any redshift,

.. math::

    P(f; \theta, z) = s^4 \left[\frac{d^{\mathrm{ref}}}{d_L(z)}\right]^2
        P_{\mathrm{ref}}(f s; \theta), \qquad s = \frac{1 + z}{1 + z_{\mathrm{ref}}},

with :math:`d^{\mathrm{ref}} = d_L(z_{\mathrm{ref}})` at the fiducials. One
catalog serves every window whose lower edge is at or above
:math:`z_{\mathrm{ref}}`. :func:`reference_catalog` generates one waveform per
draw, and :class:`~astrogwb.inference.model.CatalogModel` rescales their
weighted mean to the nodes, interpolating in :math:`\ln f`; the node count is
the model's, not the catalog's.

A catalog realization over an observation time :math:`T` is a Poisson sum
:math:`\sum_i P_i / T` over the real mergers, so it scatters about the spectrum
above. By Campbell's theorem its variance in each bin is the rate times the
second moment of the intensity, over :math:`T`. Waveform-intensity rescalings enter
squared; quadrature and importance weights each enter once:

.. math::

    \mathrm{Var}\, S(f; \Lambda) = \frac{c_\iota}{T} \sum_j w_j\, K(z_j \mid \Lambda)
        \left[\frac{d^{\mathrm{ref}}}{d_{GW}(z_j)}\right]^2 s_j^8\,
        \frac{1}{N} \sum_k \omega_k(\Lambda)\, P_{\mathrm{ref}}^2(f s_j; \theta_k),

with :math:`c_\iota` = :data:`INCLINATION_RAW_SECOND_MOMENT` restoring the
inclination scatter the pinned catalog lacks. Its redshift integrand is steeper
than the mean's near :math:`z_{\min}`, so its node count needs its own
convergence check.

Unchecked contracts:

- The population's source model is independent of inclination, which is
  isotropic.
- The waveform is :math:`(2, 2)`-mode, aligned-spin and quasi-circular: its
  intensity is a function of :math:`M_d f` times :math:`M_d^2`.
- The target and proposal log densities cover the same intrinsic variables and
  measure. Redshift is integrated independently.
- The catalog's reference redshift is at or below the window's lower edge.
- The reference catalog's frequency grid starts at or below the observed
  grid's and extends past the highest detector-frame merger frequency at
  :math:`z_{\min}`: intensity above it is zero.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray
from numpyro import handlers

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import (
    distance_and_volume_grid,
    gw_distance_ratio,
    hubble_distance,
    normalized_hubble_parameter,
)
from astrogwb.importance.catalog import (
    CatalogMetadata,
    IntensityData,
    ReferenceCatalog,
    intensity_data,
)
from astrogwb.populations._types import Parameters
from astrogwb.populations.sampling import simulate_sources
from astrogwb.rates import differential_merger_rate
from astrogwb.utils import gauss_legendre_nodes_weights, require_current_version
from astrogwb.waveform.factory import waveform_generator

__all__ = [
    "INCLINATION_MEAN",
    "INCLINATION_RAW_SECOND_MOMENT",
    "REFERENCE_INCLINATION",
    "phinney_kernel",
    "redshift_quadrature",
    "reference_catalog",
    "reference_redshift",
]

logger = logging.getLogger(__name__)

#: The inclination site, fixed rather than drawn.
INCLINATION_SITE = "inclination"

#: Reference waveforms are face-on; intensity remains |h_plus|^2 + |h_cross|^2.
REFERENCE_INCLINATION = 0.0

#: Isotropic mean relative to face-on intensity, for the (2, 2) mode alone.
INCLINATION_MEAN = 2.0 / 5.0

#: Isotropic raw intensity product relative to the face-on product.
INCLINATION_RAW_SECOND_MOMENT = 71.0 / 315.0

# Analytic averaging and redshift rescaling require aligned-spin,
# quasi-circular dominant-quadrupole waveforms, excluding higher modes.
_REFERENCE_APPROXIMANTS = frozenset(
    {
        "AnalyticInspiral",
        "TaylorF2",
        "IMRPhenomD",
        "IMRPhenomD_NRTidalv2",
        "IMRPhenomXAS",
        "IMRPhenomXAS_NRTidalv3",
    }
)


def _require_quadrupole_waveform(metadata: CatalogMetadata) -> None:
    """Restrict analytic inclination conversion to supported quadrupole models."""
    if metadata.waveform.approximant not in _REFERENCE_APPROXIMANTS:
        raise ValueError(
            "reference integration requires a supported dominant-quadrupole, "
            "aligned-spin approximant; higher modes and precession are unsupported"
        )


#: The luminosity distance declared by a source model.
_LUMINOSITY_DISTANCE = "luminosity_distance"


def redshift_quadrature(
    minimum_redshift: float, maximum_redshift: float, num_nodes: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Gauss-Legendre nodes and weights for :math:`\int dz` over a window.

    The rule is Gauss-Legendre in the scale factor :math:`a = 1/(1 + z)`, with
    the Jacobian :math:`|dz/da| = a^{-2}` folded into the weights, so
    :math:`\int f\, dz \approx \sum_j w_j f(z_j)`. The spectrum's kernel is
    dominated by :math:`1/d_L^2`, and :math:`d_L = \chi / a` makes
    :math:`dz / d_L^2 = da / \chi^2`: the Jacobian cancels the
    :math:`(1 + z)^2` inside :math:`d_L^2`, leaving a smooth integrand in
    :math:`a`. Nodes are returned in increasing redshift.

    Parameters
    ----------
    minimum_redshift
        Lower redshift bound.
    maximum_redshift
        Upper redshift bound.
    num_nodes
        Number of quadrature nodes.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if not 0.0 <= minimum_redshift < maximum_redshift:
        raise ValueError("redshift bounds must satisfy 0 <= minimum < maximum")
    lower, upper = 1.0 / (1.0 + maximum_redshift), 1.0 / (1.0 + minimum_redshift)
    nodes, weights = gauss_legendre_nodes_weights(lower, upper, num_nodes)
    scale_factor = np.asarray(nodes, dtype=np.float64)[::-1]
    weights = np.asarray(weights, dtype=np.float64)[::-1]
    return 1.0 / scale_factor - 1.0, weights / scale_factor**2


def phinney_kernel(
    redshift: ArrayLike,
    merger_rate: ArrayLike,
    distance_ratio: ArrayLike,
    hubble_constant: ArrayLike,
    omega_m: ArrayLike,
    reference_distance: ArrayLike,
) -> jax.Array:
    r"""The redshift kernel of the spectrum, in closed form, per unit redshift.

    .. math::

        K(z) = \mathcal{R}\, p(z) \left[\frac{d^{\mathrm{ref}}}{d_{GW}(z)}\right]^2
        = \frac{10^{-9}}{\mathrm{yr}}\, 4 \pi D_H \left(d^{\mathrm{ref}}\right)^2
        \frac{\psi(z)}{(1 + z)^3 E(z)\, \Xi(z)^2},

    where :math:`d_{GW} = (1 + z)\, \chi\, \Xi` and
    :math:`\mathrm{d}V_c/\mathrm{d}z = 4 \pi D_H \chi^2 / E`. The comoving
    distance :math:`\chi` and the normalization of :math:`p(z)` cancel
    against the rate, so neither a distance table nor a density is needed.

    Parameters
    ----------
    redshift, merger_rate, distance_ratio
        The nodes, the source-frame rate :math:`\psi` in
        :math:`\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}` there, and the ratio
        :math:`\Xi = d_{GW}/d_{EM}` of the model's distance to the
        electromagnetic one; all shape ``(Z,)``.
    hubble_constant, omega_m
        Flat :math:`\Lambda`CDM cosmology.
    reference_distance
        The distance :math:`d^{\mathrm{ref}}` in Mpc the intensity was generated at.

    Returns
    -------
    jax.Array
        :math:`K(z)` in mergers per second per unit redshift, shape ``(Z,)``.
    """
    redshift = jnp.asarray(redshift)
    prefactor = (
        1e-9
        / SECONDS_PER_YEAR
        * 4.0
        * jnp.pi
        * hubble_distance(hubble_constant)
        * jnp.asarray(reference_distance) ** 2
    )
    return (
        prefactor
        * jnp.asarray(merger_rate)
        / (
            (1.0 + redshift) ** 3
            * normalized_hubble_parameter(redshift, omega_m)
            * jnp.asarray(distance_ratio) ** 2
        )
    )


def reference_redshift(metadata: CatalogMetadata) -> float:
    """The redshift ``metadata``'s reference catalog is generated at.

    ``metadata.reference_redshift``, or the population's ``minimum_redshift``
    when that is ``None``.

    Parameters
    ----------
    metadata
        Metadata describing the object.
    """
    if metadata.reference_redshift is not None:
        return metadata.reference_redshift
    return metadata.population.minimum_redshift


def reference_catalog(
    metadata: CatalogMetadata,
    key: jax.Array,
    *,
    chunk_size: int | None = None,
) -> IntensityData:
    """Every draw of ``metadata`` placed at its reference redshift: intensity ``(F, N)``.

    The population's source model is drawn with its inclination conditioned on
    :data:`REFERENCE_INCLINATION`; it declares intrinsic sites only. Redshift
    is then set to :func:`reference_redshift` for every source, and the
    luminosity distance is interpolated on the grid physical redshift draws
    use, times the modified-propagation ratio when ``xi_0`` is among the
    fiducials. No extrinsic parameters are drawn, so none consume Sobol
    coordinates. The intrinsic parameters are drawn i.i.d. or, with
    ``metadata.sampling == "sobol"``, on a scrambled Sobol net. ``chunk_size``
    is :meth:`~astrogwb.waveform.WaveformGenerator.generate_batch`'s.

    Parameters
    ----------
    metadata
        Metadata describing the object.
    key
        JAX random key.
    chunk_size
        Number of sources per chunk.
    """
    # Before the draw: ripplegw enables x64 on import, but only once the
    # generator runs, after the sources are sampled.
    jax.config.update("jax_enable_x64", True)
    require_current_version(metadata.version)
    logger.info(
        "Reference catalog: num_samples=%d population=%r",
        metadata.num_samples,
        metadata.population,
    )
    _require_quadrupole_waveform(metadata)
    reference = reference_redshift(metadata)
    population = metadata.population
    parameters = metadata.fiducials
    source_model = population.build().source_model(parameters)
    samples = simulate_sources(
        handlers.condition(
            source_model, data={INCLINATION_SITE: REFERENCE_INCLINATION}
        ),
        key,
        metadata.num_samples,
        sampling=metadata.sampling,
    )
    # Match the distance grid used by physical redshift draws.
    grid = jnp.linspace(
        min(reference, population.minimum_redshift),
        population.maximum_redshift,
        population.n_grid,
    )
    distance_grid, _ = distance_and_volume_grid(
        grid, hubble_constant=parameters["H0"], omega_m=parameters["Omega_m"]
    )
    distance = jnp.interp(reference, grid, distance_grid) * gw_distance_ratio(
        reference, parameters
    )
    samples["redshift"] = np.full(metadata.num_samples, reference)
    samples[_LUMINOSITY_DISTANCE] = np.full(metadata.num_samples, float(distance))
    return intensity_data(
        waveform_generator(metadata.waveform), samples, chunk_size=chunk_size
    )


def _weighted_mean_intensity(intensity: jax.Array, log_weights: jax.Array) -> jax.Array:
    """The importance-weighted mean of ``intensity`` ``(F_ref, N)`` over draws: ``(F_ref,)``."""
    return intensity @ jnp.exp(log_weights) / log_weights.shape[0]


def _node_intensity(
    mean_intensity: jax.Array,
    log_reference_frequencies: jax.Array,
    query_log_frequencies: jax.Array,
    amplitude: jax.Array,
) -> jax.Array:
    """The mean intensity ``(F_ref,)`` placed at every node: ``(F, Z)``.

    Interpolated in :math:`\\ln f` at every ``f s_j`` and scaled by
    ``amplitude``, ``(Z,)``.
    """
    return amplitude * jnp.interp(
        query_log_frequencies, log_reference_frequencies, mean_intensity, right=0.0
    )


class _SharedTerms(NamedTuple):
    """What the spectrum and its variance both take from the population."""

    log_weights: jax.Array
    kernel: jax.Array
    distance_ratio: jax.Array
    hubble_constant: ArrayLike
    omega_m: ArrayLike
    total_rate: jax.Array


type MergerRate = Callable[[ArrayLike, Parameters], jax.Array]
type IntrinsicLogProb = Callable[[Mapping[str, jax.Array], Parameters], jax.Array]


def _shared_terms(
    params: Mapping[str, ArrayLike],
    *,
    merger_rate: MergerRate,
    log_prob: IntrinsicLogProb | None,
    intrinsic: Mapping[str, jax.Array],
    proposal_log_prob: jax.Array,
    redshift: jax.Array,
    redshift_weights: jax.Array,
    reference_distance: jax.Array,
) -> _SharedTerms:
    """The front half of the spectrum and of its variance, from one population call.

    ``kernel`` is the quadrature weights times :func:`phinney_kernel`, ``(Z,)``;
    ``total_rate`` is the observer-frame merger rate in mergers per second.
    """
    log_weights = (
        jnp.zeros_like(proposal_log_prob)
        if log_prob is None
        else log_prob(intrinsic, params) - proposal_log_prob
    )
    hubble_constant = params["H0"]
    omega_m = params["Omega_m"]
    rate = merger_rate(redshift, params)
    # The GW distance over the electromagnetic one: modified propagation alone.
    distance_ratio = gw_distance_ratio(redshift, params)
    kernel = redshift_weights * phinney_kernel(
        redshift,
        rate,
        distance_ratio,
        hubble_constant,
        omega_m,
        reference_distance,
    )
    total_rate = jnp.sum(
        redshift_weights
        * differential_merger_rate(redshift, rate, hubble_constant, omega_m)
    )
    return _SharedTerms(
        log_weights, kernel, distance_ratio, hubble_constant, omega_m, total_rate
    )


def _rescaling_arrays(
    catalog: ReferenceCatalog,
    frequencies: ArrayLike,
    num_redshift_nodes: int,
    minimum_redshift: float,
    maximum_redshift: float,
) -> dict[str, Any]:
    """Bind the intrinsic columns and compute quadrature and query frequencies."""
    reference = catalog.reference_redshift
    if minimum_redshift < reference:
        raise ValueError(
            f"the window's lower edge, {minimum_redshift}, is below the reference "
            f"redshift {reference} the catalog was generated at"
        )
    redshift, redshift_weights = redshift_quadrature(
        minimum_redshift, maximum_redshift, num_redshift_nodes
    )
    scale = (1.0 + redshift) / (1.0 + reference)
    query_log_frequencies = (
        np.log(np.asarray(frequencies, dtype=np.float64))[:, None]
        + np.log(scale)[None, :]
    )
    return {
        "intrinsic": {
            name: jnp.asarray(values) for name, values in catalog.intrinsic.items()
        },
        "proposal_log_prob": jnp.asarray(catalog.proposal_log_prob, dtype=jnp.float64),
        "log_reference_frequencies": jnp.log(jnp.asarray(catalog.frequencies)),
        "query_log_frequencies": jnp.asarray(query_log_frequencies),
        "amplitude": jnp.asarray(scale**4),
        "redshift": jnp.asarray(redshift),
        "redshift_weights": jnp.asarray(redshift_weights),
        "reference_distance": jnp.asarray(
            catalog.reference_distance, dtype=jnp.float64
        ),
    }
