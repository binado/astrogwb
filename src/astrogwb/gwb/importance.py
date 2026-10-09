r"""The background spectrum as a redshift quadrature over an intrinsic sample.

The spectrum is an expectation over sources. Only the intrinsic parameters
are left to Monte Carlo here: redshift is integrated on fixed Gauss-Legendre
nodes in the scale factor :math:`1/(1 + z)` (:func:`redshift_quadrature`),
and inclination is averaged exactly. Every one of :math:`N` intrinsic
draws is placed at each of :math:`Z` nodes, so

.. math::

    S(f; \Lambda) = \sum_j w_j\, K(z_j \mid \Lambda)
        \frac{1}{N} \sum_k \omega_k(\Lambda)\, P(f; \theta_k, z_j),

with :math:`P` the power at reference distance :math:`d^{\mathrm{ref}}`,
:math:`\omega_k` the ratio of the target to the drawn intrinsic density
(one when the intrinsic hyperparameters are not varied), and :math:`K` the
closed-form :func:`phinney_kernel`: the merger rate density times
:math:`[d^{\mathrm{ref}}/d_{GW}]^2`, in which the comoving distance and the
redshift density's normalization cancel. Redshift has no
Monte Carlo variance left: the :math:`1/d_L^2` tail that dominated a
redshift-sampled proposal is integrated exactly. Inclination is fixed at
:data:`EFFECTIVE_INCLINATION`, where the quadrupolar factor equals its
isotropic mean; that is exact only for waveforms carrying the :math:`(2, 2)`
mode alone.

The power is not generated at every node. For a :math:`(2, 2)`-mode,
aligned-spin, quasi-circular waveform the strain depends on mass only through
:math:`M_d f` and an overall :math:`M_d^2`, with :math:`M_d = M (1 + z)`, so the
power of a draw generated at the window's lower edge :math:`z_{\min}` is, at any
redshift,

.. math::

    P(f; \theta, z) = s^4 \left[\frac{d^{\mathrm{ref}}}{d_L(z)}\right]^2
        P_{\mathrm{ref}}(f s; \theta), \qquad s = \frac{1 + z}{1 + z_{\min}},

with :math:`d^{\mathrm{ref}} = d_L(z_{\min})` at the fiducials.
:func:`reference_catalog` generates one waveform per draw there, and
:func:`build_rescaled_spectrum` rescales the weighted mean of those to the
nodes in each call, interpolating it in :math:`\ln f`. Its node count is the
builder's, not the catalog's, so changing it needs no new waveforms.

Contracts the caller is trusted to honour (nothing here checks them):

- The population's intrinsic sites are independent of redshift and
  inclination, and inclination is isotropic.
- The waveform is :math:`(2, 2)`-mode, aligned-spin and quasi-circular: its
  power is a function of :math:`M_d f` times :math:`M_d^2`.
- ``density_sites`` names intrinsic sites only: redshift is integrated, never
  weighted.
- The population's ``redshift`` site is a
  :class:`~astrogwb.distributions.redshift.base.RedshiftDistribution`: the
  spectrum reads its rate and cosmology off that distribution, and a
  :class:`TypeError` names anything else.
- The catalog is never narrowed with
  :func:`~astrogwb.simulators.polarization_power.restrict_redshift`: its
  reference redshift is the population window's lower edge.
- The reference catalog's frequency grid starts at or below the observed
  grid's and extends past the highest detector-frame merger frequency at
  :math:`z_{\min}`: power above it is zero.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial
from jax.typing import ArrayLike
from numpy.typing import NDArray
from numpyro import handlers

from astrogwb import __version__
from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import hubble_distance, normalized_hubble_parameter
from astrogwb.distributions.rates import total_merger_rate
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.importance.diagnostics import relative_ess
from astrogwb.inference.protocol import SpectralDensityFn
from astrogwb.populations._types import Population, PopulationModel
from astrogwb.populations.evaluation import evaluate_sources, sample_sources
from astrogwb.populations.metadata import PopulationMetadata
from astrogwb.simulators.polarization_power.metadata import CatalogMetadata
from astrogwb.simulators.polarization_power.restrict import (
    REDSHIFT_SITE,
    REDSHIFT_WINDOW_KWARGS,
)
from astrogwb.simulators.polarization_power.simulator import (
    PolarizationPowerData,
    polarization_power_data,
)
from astrogwb.utils import gauss_legendre_nodes_weights
from astrogwb.utils.numpyro import SiteDistribution

__all__ = [
    "EFFECTIVE_INCLINATION",
    "LogWeightsFn",
    "build_rescaled_spectrum",
    "phinney_kernel",
    "redshift_quadrature",
    "reference_catalog",
    "reference_catalog_stem",
]

logger = logging.getLogger(__name__)

#: The inclination site, fixed rather than drawn.
INCLINATION_SITE = "inclination"

#: The inclination where the quadrupolar factor
#: :math:`g(\iota) = ((1 + \cos^2\iota)/2)^2 + \cos^2\iota` equals its
#: isotropic mean, 4/5: :math:`\cos^2\iota^* = \sqrt{11.2} - 3`.
EFFECTIVE_INCLINATION = float(np.arccos(np.sqrt(np.sqrt(11.2) - 3.0)))

#: The luminosity distance a source model returns.
_LUMINOSITY_DISTANCE = "luminosity_distance"

#: Per-draw log importance weights from hyperparameters alone, shape ``(N,)``.
type LogWeightsFn = Callable[[Mapping[str, ArrayLike]], jax.Array]


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
    :math:`a`, and the nodes crowd towards low redshift, where that weight
    is. The nodes are returned in increasing redshift.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    lower, upper = 1.0 / (1.0 + maximum_redshift), 1.0 / (1.0 + minimum_redshift)
    nodes, weights = gauss_legendre_nodes_weights(lower, upper, num_nodes)
    scale_factor = np.asarray(nodes, dtype=np.float64)[::-1]
    weights = np.asarray(weights, dtype=np.float64)[::-1]
    return 1.0 / scale_factor - 1.0, weights / scale_factor**2


def _intrinsic_log_prob(
    model: PopulationModel,
    intrinsic: Mapping[str, jax.Array],
    redshift: jax.Array,
    density_sites: Sequence[str],
) -> jax.Array:
    log_prob, _ = evaluate_sources(
        _pin_redshift_and_inclination(model, redshift),
        intrinsic,
        density_sites=density_sites,
    )
    return log_prob


@dataclass(frozen=True)
class _Bound:
    """The static half of a bound estimator: its function and non-array settings.

    Compared by value, so two builds with the same population object and sites
    are equal: :func:`_bind` wraps it in a :class:`jax.tree_util.Partial` whose
    leaves are the arrays, and a ``jit`` taking that callable as an argument
    traces once for every catalog of the same shapes.
    """

    function: Callable[..., Any]
    population: Population
    density_sites: tuple[str, ...]

    def __call__(self, params: Mapping[str, ArrayLike], /, **arrays: Any) -> Any:
        return self.function(
            params,
            population=self.population,
            density_sites=self.density_sites,
            **arrays,
        )


def _bind(
    function: Callable[..., Any],
    population: Population,
    density_sites: Sequence[str],
    **arrays: Any,
) -> Any:
    """``function`` as a pytree callable: ``arrays`` are its leaves, the rest static."""
    return Partial(_Bound(function, population, tuple(density_sites)), **arrays)


def evaluate_log_weights(
    params: Mapping[str, ArrayLike],
    *,
    population: Population,
    intrinsic: Mapping[str, jax.Array],
    redshift: jax.Array,
    proposal_log_prob: jax.Array,
    density_sites: Sequence[str],
) -> jax.Array:
    """Per-draw intrinsic log importance weights at ``params``, shape ``(N,)``."""
    _, model = population(params)
    target = _intrinsic_log_prob(model, intrinsic, redshift[0], density_sites)
    return target - proposal_log_prob


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
    redshift, merger_rate, distance_ratio:
        The nodes, the source-frame rate :math:`\psi` in
        :math:`\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}` there, and the ratio
        :math:`\Xi = d_{GW}/d_{EM}` of the model's distance to the
        electromagnetic one; all shape ``(Z,)``.
    hubble_constant, omega_m:
        Flat :math:`\Lambda`CDM cosmology.
    reference_distance:
        The distance :math:`d^{\mathrm{ref}}` in Mpc the power was generated at.

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


def _redshift_window(population: PopulationMetadata) -> tuple[float, float]:
    """The population's ``(minimum_redshift, maximum_redshift)``."""
    kwargs = population.model_kwargs
    minimum, maximum = (float(kwargs[name]) for name in REDSHIFT_WINDOW_KWARGS)
    return minimum, maximum


def _pin_redshift_and_inclination(
    model: PopulationModel, redshift: ArrayLike
) -> PopulationModel:
    """``model`` with redshift and inclination fixed rather than drawn."""
    return handlers.condition(
        model,
        data={
            REDSHIFT_SITE: jnp.asarray(redshift),
            INCLINATION_SITE: EFFECTIVE_INCLINATION,
        },
    )


def reference_catalog_stem(metadata: CatalogMetadata, seed: int | np.integer) -> str:
    """The file stem by convention: ``reference_catalog-<key>-<seed>``.

    The prefix keeps it apart from the plain catalog of the same record, which
    :func:`~astrogwb.simulators.polarization_power.catalog_stem` names.
    """
    return f"reference_catalog-{metadata.key()}-{int(seed)}"


def reference_catalog(
    metadata: CatalogMetadata,
    key: jax.Array,
    *,
    chunk_size: int | None = None,
) -> PolarizationPowerData:
    """Every draw of ``metadata`` placed at the window's lower edge: power ``(F, N)``.

    The model is conditioned on the population's ``minimum_redshift`` and
    :data:`EFFECTIVE_INCLINATION` before the draw. ``chunk_size`` is
    :meth:`~astrogwb.waveform.PolarizationPowerGenerator.generate_batch`'s.
    """
    # x64 before the draw, as for a plain catalog: see draw_catalog.
    jax.config.update("jax_enable_x64", True)
    if metadata.version != __version__:
        raise ValueError(
            f"metadata is for astrogwb {metadata.version}, but {__version__} "
            "is installed; a catalog is generated by the code its key names"
        )
    logger.info(
        "Reference catalog %s: population=%s num_samples=%d",
        metadata.key(),
        metadata.population.model_name,
        metadata.num_samples,
    )
    _, model = metadata.population.build()(metadata.fiducials)
    minimum_redshift, _ = _redshift_window(metadata.population)
    samples = sample_sources(
        _pin_redshift_and_inclination(model, minimum_redshift),
        key,
        num_samples=metadata.num_samples,
    )
    return polarization_power_data(
        metadata.waveform.build(), samples, chunk_size=chunk_size
    )


def rescaled_spectral_density(
    params: Mapping[str, ArrayLike],
    *,
    population: Population,
    polarization_power: jax.Array,
    log_reference_frequencies: jax.Array,
    query_log_frequencies: jax.Array,
    amplitude: jax.Array,
    intrinsic: Mapping[str, jax.Array],
    redshift: jax.Array,
    redshift_weights: jax.Array,
    reference_distance: jax.Array,
    proposal_log_prob: jax.Array,
    density_sites: Sequence[str],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The spectrum at ``params``, shape ``(F,)``, with its diagnostics.

    The weighted mean reference power ``(F_ref,)`` is interpolated in
    :math:`\\ln f` at every ``f s_j`` (``query_log_frequencies``, ``(F, Z)``),
    scaled by ``s_j**4`` (``amplitude``), and integrated over the nodes.
    ``extras`` holds ``total_merger_rate`` (observer frame, mergers per second)
    and ``importance_relative_ess`` over the intrinsic draws, both shape ``()``.
    """
    _, model = population(params)
    log_weights = (
        _intrinsic_log_prob(model, intrinsic, redshift[0], density_sites)
        - proposal_log_prob
    )
    mean_power = polarization_power @ jnp.exp(log_weights) / log_weights.shape[0]
    node_power = amplitude * jnp.interp(
        query_log_frequencies, log_reference_frequencies, mean_power, right=0.0
    )
    first = {name: values[0] for name, values in intrinsic.items()}
    # The model is run for the node distances (the redshift column is
    # conditioned in) and to hand over its redshift site's own distribution,
    # which knows the rate and the cosmology: no density is evaluated.
    recorded = SiteDistribution(
        handlers.condition(
            model, data={**first, INCLINATION_SITE: EFFECTIVE_INCLINATION}
        ),
        REDSHIFT_SITE,
    )
    _, outputs = evaluate_sources(recorded, {REDSHIFT_SITE: redshift}, density_sites=())
    distribution = recorded.distribution
    if not isinstance(distribution, RedshiftDistribution):
        raise TypeError(
            f"the {REDSHIFT_SITE!r} site must be a RedshiftDistribution, got "
            f"{type(distribution).__name__}"
        )
    hubble_constant = distribution.params["H0"]
    omega_m = distribution.params["Omega_m"]
    merger_rate = distribution.merger_rate(redshift)
    # The model's distance over its own table's: modified propagation alone.
    distance_ratio = outputs[_LUMINOSITY_DISTANCE] / distribution.luminosity_distance(
        redshift
    )
    kernel = redshift_weights * phinney_kernel(
        redshift,
        merger_rate,
        distance_ratio,
        hubble_constant,
        omega_m,
        reference_distance,
    )
    total_rate = total_merger_rate(
        redshift, redshift_weights, merger_rate, hubble_constant, omega_m
    )
    return node_power @ kernel, {
        "total_merger_rate": total_rate,
        "importance_relative_ess": relative_ess(log_weights),
    }


def build_rescaled_spectrum(
    data: PolarizationPowerData,
    metadata: CatalogMetadata,
    *,
    population: Population,
    frequencies: ArrayLike,
    num_redshift_nodes: int,
    density_sites: Sequence[str],
) -> tuple[SpectralDensityFn, LogWeightsFn]:
    """Bind a :func:`reference_catalog` to a target, as both callables at once.

    The spectrum is predicted on ``frequencies``, the observed grid, which need
    not be the catalog's own; redshift is integrated on ``num_redshift_nodes``
    nodes of :func:`redshift_quadrature` over the catalog population's window.
    Every step -- weights, mean power, interpolation, rescaling, quadrature --
    runs per call. Call outside JAX transformations.

    The drawn intrinsic density is the catalog's own population at its
    fiducials, evaluated once here. ``density_sites`` names the intrinsic
    factors the weights include; leave it empty when no intrinsic
    hyperparameter varies, and every weight is one. ``population`` is the
    target's bound :class:`~astrogwb.populations.Population`; build it once per
    run.

    Both callables are pytrees (:class:`jax.tree_util.Partial`): the catalog's
    arrays are their leaves, so passing one *as an argument* to a jitted
    function traces it as input buffers instead of baking it in as constants,
    and the compiled function serves every catalog of the same shapes built
    with the same ``population`` and ``density_sites``.

    The caller is trusted to pass a reference catalog of ``metadata`` and a
    waveform the module's contracts hold for; nothing here checks either.
    """
    minimum_redshift, maximum_redshift = _redshift_window(metadata.population)
    redshift, redshift_weights = redshift_quadrature(
        minimum_redshift, maximum_redshift, num_redshift_nodes
    )
    scale = (1.0 + redshift) / (1.0 + minimum_redshift)
    query_log_frequencies = (
        np.log(np.asarray(frequencies, dtype=np.float64))[:, None]
        + np.log(scale)[None, :]
    )
    columns = data["source_parameters"]
    intrinsic = {
        name: jnp.asarray(np.asarray(values))
        for name, values in columns.items()
        if name not in (REDSHIFT_SITE, INCLINATION_SITE)
    }
    redshift_nodes = jnp.asarray(redshift)
    _, fiducial_model = metadata.population.build()(metadata.fiducials)
    proposal_log_prob = _intrinsic_log_prob(
        fiducial_model, intrinsic, redshift_nodes[0], density_sites
    )
    shared: dict[str, Any] = {
        "intrinsic": intrinsic,
        "redshift": redshift_nodes,
        "proposal_log_prob": proposal_log_prob,
    }
    spectral_density_fn = _bind(
        rescaled_spectral_density,
        population,
        density_sites,
        polarization_power=jnp.asarray(data["polarization_power"]),
        log_reference_frequencies=jnp.log(jnp.asarray(data["frequencies"])),
        query_log_frequencies=jnp.asarray(query_log_frequencies),
        amplitude=jnp.asarray(scale**4),
        redshift_weights=jnp.asarray(redshift_weights),
        reference_distance=jnp.asarray(
            np.asarray(columns[_LUMINOSITY_DISTANCE], dtype=np.float64)[0]
        ),
        **shared,
    )
    return spectral_density_fn, _bind(
        evaluate_log_weights, population, density_sites, **shared
    )
