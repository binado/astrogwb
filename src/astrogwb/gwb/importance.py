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

A catalog realization over an observation time :math:`T` is a Poisson sum
:math:`\sum_i P_i / T` over the real mergers, so it scatters about the spectrum
above. By Campbell's theorem its variance in each bin is the rate times the
second moment of the power, over :math:`T`, and every factor of the quadrature
enters squared:

.. math::

    \mathrm{Var}\, S(f; \Lambda) = \frac{c_\iota}{T} \sum_j w_j\, K(z_j \mid \Lambda)
        \left[\frac{d^{\mathrm{ref}}}{d_{GW}(z_j)}\right]^2 s_j^8\,
        \frac{1}{N} \sum_k \omega_k(\Lambda)\, P_{\mathrm{ref}}^2(f s_j; \theta_k),

with :math:`c_\iota` = :data:`INCLINATION_SECOND_MOMENT` restoring the
inclination scatter the pinned catalog lacks. :func:`build_rescaled_shot_noise`
binds it as :func:`build_rescaled_spectrum` binds the spectrum. Its redshift
integrand is steeper than the mean's near :math:`z_{\min}`, so its node count
needs its own convergence check.

Contracts the caller is trusted to honour (nothing here checks them):

- The population's source model is independent of inclination, which is
  isotropic. It is independent of redshift by construction: it never sees it.
- The waveform is :math:`(2, 2)`-mode, aligned-spin and quasi-circular: its
  power is a function of :math:`M_d f` times :math:`M_d^2`.
- ``density_sites`` names intrinsic sites only: redshift is integrated, never
  weighted.
- The spectrum reads the rate, the cosmology and both distances off the
  population's
  :class:`~astrogwb.distributions.redshift.base.RedshiftDistribution` directly:
  no model is run at the nodes.
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
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial
from jax.typing import ArrayLike
from numpy.typing import NDArray
from numpyro import handlers

from astrogwb import __version__
from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import (
    hubble_distance,
    luminosity_distance,
    normalized_hubble_parameter,
)
from astrogwb.distributions.rates import total_merger_rate
from astrogwb.importance.diagnostics import relative_ess
from astrogwb.inference.protocol import SpectralDensityFn, SpectralVarianceFn
from astrogwb.populations._types import Population, PopulationModel, SourceModel
from astrogwb.populations.evaluation import (
    evaluate_sources,
    sample_sources,
    sample_sources_qmc,
)
from astrogwb.populations.joint import joint_model
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

__all__ = [
    "EFFECTIVE_INCLINATION",
    "INCLINATION_SECOND_MOMENT",
    "LogWeightsFn",
    "build_rescaled_shot_noise",
    "build_rescaled_spectrum",
    "phinney_kernel",
    "redshift_quadrature",
    "reference_catalog",
    "reference_catalog_stem",
    "rescaled_spectral_variance",
]

logger = logging.getLogger(__name__)

#: The inclination site, fixed rather than drawn.
INCLINATION_SITE = "inclination"

#: The inclination where the quadrupolar factor
#: :math:`g(\iota) = ((1 + \cos^2\iota)/2)^2 + \cos^2\iota` equals its
#: isotropic mean, 4/5: :math:`\cos^2\iota^* = \sqrt{11.2} - 3`.
EFFECTIVE_INCLINATION = float(np.arccos(np.sqrt(np.sqrt(11.2) - 3.0)))

#: The isotropic second moment of the quadrupolar factor over its squared
#: mean, :math:`\langle g^2 \rangle / \langle g \rangle^2 = 355/252`. A catalog
#: pinned at :data:`EFFECTIVE_INCLINATION` carries the mean of :math:`g` but
#: none of its scatter; this factor restores the scatter in a second moment of
#: the power. Exact for the :math:`(2, 2)` mode alone, as the pin is.
INCLINATION_SECOND_MOMENT = 355.0 / 252.0

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
    source_model: SourceModel,
    intrinsic: Mapping[str, jax.Array],
    density_sites: Sequence[str],
) -> jax.Array:
    log_prob, _ = evaluate_sources(
        handlers.condition(
            source_model, data={INCLINATION_SITE: EFFECTIVE_INCLINATION}
        ),
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
    proposal_log_prob: jax.Array,
    density_sites: Sequence[str],
) -> jax.Array:
    """Per-draw intrinsic log importance weights at ``params``, shape ``(N,)``."""
    _, source_model = population(params)
    target = _intrinsic_log_prob(source_model, intrinsic, density_sites)
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
    :data:`EFFECTIVE_INCLINATION` before the draw. The intrinsic parameters are
    drawn i.i.d., or on a scrambled Sobol net seeded by ``key`` when
    ``metadata.sampling`` is ``"sobol"``
    (:func:`~astrogwb.populations.evaluation.sample_sources_qmc`): the only
    Monte Carlo left in the rescaled spectrum is this intrinsic average, and its
    integrand is smooth and dominated by the chirp mass. ``chunk_size`` is
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
    model = joint_model(*metadata.population.build()(metadata.fiducials))
    minimum_redshift, _ = _redshift_window(metadata.population)
    draw = sample_sources_qmc if metadata.sampling == "sobol" else sample_sources
    samples = draw(
        _pin_redshift_and_inclination(model, minimum_redshift),
        key,
        num_samples=metadata.num_samples,
    )
    return polarization_power_data(
        metadata.waveform.build(), samples, chunk_size=chunk_size
    )


def _node_power(
    power: jax.Array,
    log_weights: jax.Array,
    log_reference_frequencies: jax.Array,
    query_log_frequencies: jax.Array,
    amplitude: jax.Array,
) -> jax.Array:
    """The weighted mean of ``power`` ``(F_ref, N)`` placed at every node: ``(F, Z)``.

    Interpolated in :math:`\\ln f` at every ``f s_j`` and scaled by
    ``amplitude``, ``(Z,)``.
    """
    mean_power = power @ jnp.exp(log_weights) / log_weights.shape[0]
    return amplitude * jnp.interp(
        query_log_frequencies, log_reference_frequencies, mean_power, right=0.0
    )


class _SharedTerms(NamedTuple):
    """What the spectrum and its variance both take from ``population(params)``."""

    log_weights: jax.Array
    kernel: jax.Array
    distance_ratio: jax.Array
    hubble_constant: ArrayLike
    omega_m: ArrayLike
    total_rate: jax.Array


def _shared_terms(
    params: Mapping[str, ArrayLike],
    *,
    population: Population,
    intrinsic: Mapping[str, jax.Array],
    proposal_log_prob: jax.Array,
    density_sites: Sequence[str],
    redshift: jax.Array,
    redshift_weights: jax.Array,
    reference_distance: jax.Array,
) -> _SharedTerms:
    """The front half of the spectrum and of its variance, from one population call.

    ``kernel`` is the quadrature weights times :func:`phinney_kernel`, ``(Z,)``;
    ``total_rate`` is the observer-frame merger rate in mergers per second.
    """
    distribution, source_model = population(params)
    log_weights = (
        _intrinsic_log_prob(source_model, intrinsic, density_sites) - proposal_log_prob
    )
    hubble_constant = distribution.params["H0"]
    omega_m = distribution.params["Omega_m"]
    merger_rate = distribution.merger_rate(redshift)
    # The GW distance over the electromagnetic one: modified propagation alone.
    distance_ratio = distribution.distance_ratio(redshift)
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
    return _SharedTerms(
        log_weights, kernel, distance_ratio, hubble_constant, omega_m, total_rate
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
    shared = _shared_terms(
        params,
        population=population,
        intrinsic=intrinsic,
        proposal_log_prob=proposal_log_prob,
        density_sites=density_sites,
        redshift=redshift,
        redshift_weights=redshift_weights,
        reference_distance=reference_distance,
    )
    node_power = _node_power(
        polarization_power,
        shared.log_weights,
        log_reference_frequencies,
        query_log_frequencies,
        amplitude,
    )
    return node_power @ shared.kernel, {
        "total_merger_rate": shared.total_rate,
        "importance_relative_ess": relative_ess(shared.log_weights),
    }


def rescaled_spectral_variance(
    params: Mapping[str, ArrayLike],
    *,
    population: Population,
    squared_polarization_power: jax.Array,
    log_reference_frequencies: jax.Array,
    query_log_frequencies: jax.Array,
    amplitude: jax.Array,
    intrinsic: Mapping[str, jax.Array],
    redshift: jax.Array,
    redshift_weights: jax.Array,
    reference_distance: jax.Array,
    proposal_log_prob: jax.Array,
    density_sites: Sequence[str],
    observation_seconds: jax.Array,
) -> jax.Array:
    r"""The shot-noise variance of the spectrum at ``params``, shape ``(F,)``.

    The variance of a Poisson catalog realization about
    :func:`rescaled_spectral_density`, from the module's Campbell sum: the
    weighted mean of ``squared_polarization_power`` ``(F_ref, N)`` is
    rescaled with ``amplitude**2`` (:math:`s_j^8`), integrated with the
    spectrum's kernel times :math:`(d^{\mathrm{ref}}/d_{GW})^2`, and multiplied
    by :data:`INCLINATION_SECOND_MOMENT` over ``observation_seconds``.

    The extra distance factor uses the pointwise
    :func:`~astrogwb.cosmology.luminosity_distance` rather than the
    distribution's table, as the kernel does.
    """
    shared = _shared_terms(
        params,
        population=population,
        intrinsic=intrinsic,
        proposal_log_prob=proposal_log_prob,
        density_sites=density_sites,
        redshift=redshift,
        redshift_weights=redshift_weights,
        reference_distance=reference_distance,
    )
    node_power = _node_power(
        squared_polarization_power,
        shared.log_weights,
        log_reference_frequencies,
        query_log_frequencies,
        amplitude**2,
    )
    gw_distance = shared.distance_ratio * luminosity_distance(
        redshift, shared.hubble_constant, shared.omega_m
    )
    kernel = shared.kernel * (reference_distance / gw_distance) ** 2
    return INCLINATION_SECOND_MOMENT * (node_power @ kernel) / observation_seconds


def _rescaling_arrays(
    data: PolarizationPowerData,
    metadata: CatalogMetadata,
    frequencies: ArrayLike,
    num_redshift_nodes: int,
    density_sites: Sequence[str],
) -> dict[str, Any]:
    """Every array a rescaled estimator binds except the catalog's power.

    The redshift nodes and weights of the window, the query frequencies
    ``(F, Z)``, the node amplitude :math:`s_j^4`, the intrinsic columns, the
    drawn intrinsic density at the catalog's fiducials, and the reference
    distance.
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
        if name not in (REDSHIFT_SITE, INCLINATION_SITE, _LUMINOSITY_DISTANCE)
    }
    _, fiducial_source_model = metadata.population.build()(metadata.fiducials)
    return {
        "intrinsic": intrinsic,
        "proposal_log_prob": _intrinsic_log_prob(
            fiducial_source_model, intrinsic, density_sites
        ),
        "log_reference_frequencies": jnp.log(jnp.asarray(data["frequencies"])),
        "query_log_frequencies": jnp.asarray(query_log_frequencies),
        "amplitude": jnp.asarray(scale**4),
        "redshift": jnp.asarray(redshift),
        "redshift_weights": jnp.asarray(redshift_weights),
        "reference_distance": jnp.asarray(
            np.asarray(columns[_LUMINOSITY_DISTANCE], dtype=np.float64)[0]
        ),
    }


def build_rescaled_shot_noise(
    data: PolarizationPowerData,
    metadata: CatalogMetadata,
    *,
    population: Population,
    frequencies: ArrayLike,
    num_redshift_nodes: int,
    density_sites: Sequence[str],
    observation_time: float,
) -> SpectralVarianceFn:
    """Bind a :func:`reference_catalog` to a target's shot-noise variance.

    The counterpart of :func:`build_rescaled_spectrum`, with the same arguments
    and contracts plus ``observation_time`` in years: ``params -> (F,)``, the
    per-bin variance of a Poisson catalog realization over that time about the
    spectrum on ``frequencies``. Call outside JAX transformations.

    The squared power is formed once here and held as a leaf, so the returned
    pytree carries a second ``(F_ref, N)`` array next to the spectrum's: binding
    both doubles the catalog's device memory, where squaring per call would
    allocate it on every evaluation.
    """
    power = np.asarray(data["polarization_power"], dtype=np.float64)
    return _bind(
        rescaled_spectral_variance,
        population,
        density_sites,
        squared_polarization_power=jnp.asarray(power * power),
        observation_seconds=jnp.asarray(observation_time * SECONDS_PER_YEAR),
        **_rescaling_arrays(
            data, metadata, frequencies, num_redshift_nodes, density_sites
        ),
    )


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
    arrays = _rescaling_arrays(
        data, metadata, frequencies, num_redshift_nodes, density_sites
    )
    spectral_density_fn = _bind(
        rescaled_spectral_density,
        population,
        density_sites,
        polarization_power=jnp.asarray(data["polarization_power"]),
        **arrays,
    )
    return spectral_density_fn, _bind(
        evaluate_log_weights,
        population,
        density_sites,
        intrinsic=arrays["intrinsic"],
        proposal_log_prob=arrays["proposal_log_prob"],
    )
