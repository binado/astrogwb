"""Reference importance-weights model: BNS + Madau-Dickinson rate + modified propagation.

This module packages one concrete realization of the
:mod:`astrogwb.importance.population` API and, on top of it, the
:class:`~astrogwb.importance.protocol.MergerRateAndLogWeightsFn` callback the
NumPyro models consume: a binary-neutron-star population whose merger-rate
density follows the Madau-Dickinson (2017) shape, evolved on a flat-LambdaCDM
cosmology, with a phenomenological GW-to-EM luminosity-distance ratio
(``xi_0``, ``xi_n``) capturing a modified-gravity propagation effect.

It is a *reference implementation* -- the NumPyro model itself accepts any
callback satisfying the protocol, so callers may substitute their own.

Two routes express the same redshift density here, and
``tests/core/test_distributions.py`` pins them together to rounding:

- :func:`bns_population` builds a
  :class:`~astrogwb.distributions.redshift.madau_dickinson.MadauDickinsonRedshiftDistribution`
  inside a :class:`~astrogwb.importance.population.Population`, and
  :func:`bns_population_terms` reduces it to the arrays the weights need. This
  is what :func:`make_merger_rate_and_log_weights_fn` runs.
- :func:`compute_merger_rate_distance_and_logprob` is the grid-level formula
  written out by hand. It is kept as the reference the distribution class and
  the closure are both tested against, and the paper application still calls
  it directly for the fiducial injection spectrum and the proposal density.

Factory contract: :func:`make_merger_rate_and_log_weights_fn` takes a
``redshift_grid`` array that is captured by the returned closure, so the closure
never needs to extract static Python scalars from traced values and is safe
to trace inside ``jax.jit`` during NUTS.

The callback accepts a precomputed ``proposal_logprob`` array, so the proposal
need not equal the target at its fiducial parameters. The GW distance on the
proposal side is read from the catalog's stored fiducial luminosity distances,
never recomputed; :mod:`astrogwb.importance.population` explains why.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import distance_and_volume_grid, log_gw_em_ratio

# Imported, not redefined: `astrogwb.distributions.rates` is the canonical home
# for the rate shapes, and it is NumPyro-free precisely so this module can share
# them. The name stays importable from here, which is how every existing caller
# and `tests/core/test_importance.py` reach it.
from astrogwb.distributions.rates import madau_dickinson_rate
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.distributions.redshift.madau_dickinson import (
    MadauDickinsonRedshiftDistribution,
)
from astrogwb.importance.population import (
    Population,
    PopulationTerms,
    importance_log_weights,
)
from astrogwb.importance.protocol import MergerRateAndLogWeightsFn

AMPLITUDE_PARAMETERS: tuple[str, ...] = ("H0", "local_merger_rate")
"""Parameters this callback supports marginalizing analytically."""


# Absolute scalings as module-level ``def``s (not closures over the fiducial)
# so they are singletons: ``AmplitudeConditional`` carries the amplitude
# function as pytree *aux* data, which JAX hashes into the jit cache key. A
# lambda (or a ``functools.partial`` over a float) is identity-hashed, so a
# fresh one per call would retrace the model on every construction. The
# consumer forms the ratio ``f(varphi)/f(varphi_fid)`` itself.
#
# The predicted spectrum factorizes as ``f = g_R * g_F``. ``local_merger_rate``
# enters only through ``total_merger_rate`` (linear; absent from ``log_weights``),
# so ``g_R = varphi``, ``g_F = 1``, ``f = varphi``. ``H0`` enters the rate via
# ``dV_c/dz ∝ h0^{-3}`` and the mean energy flux via
# ``exp(-2 log d_L) ∝ h0^2``, so ``g_R = varphi^{-3}``, ``g_F = varphi^2``,
# and ``f = varphi^{-1}``.
def merger_rate_H0_fn(marginalized_parameter: jax.Array) -> jax.Array:
    """Merger-rate scaling :math:`g_R(H_0) = H_0^{-3}`."""
    return marginalized_parameter**-3


def amplitude_H0_fn(marginalized_parameter: jax.Array) -> jax.Array:
    """Total amplitude scaling :math:`f(H_0) = H_0^{-1}` (:math:`g_R g_F`)."""
    return 1.0 / marginalized_parameter


def merger_rate_local_merger_rate_fn(
    marginalized_parameter: jax.Array,
) -> jax.Array:
    """Merger-rate scaling :math:`g_R(\\mathcal{R}_0) = \\mathcal{R}_0`."""
    return marginalized_parameter


def amplitude_local_merger_rate_fn(
    marginalized_parameter: jax.Array,
) -> jax.Array:
    """Total amplitude scaling :math:`f(\\mathcal{R}_0) = \\mathcal{R}_0`."""
    return marginalized_parameter


def compute_merger_rate_distance_and_logprob(
    params: Mapping[str, Any],
    samples: Mapping[str, jax.Array],
    *,
    redshift_grid: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Merger rate, luminosity distance, and redshift log-pdf at catalog samples.

    Builds the cosmology and density tables on ``redshift_grid``, then evaluates
    the redshift PDF

    :math:`\mathrm{logpdf} = \log p(z|\theta)`

    at ``samples["redshift"]``. One grid pass serves all three outputs. The
    density :math:`p(z) \propto \psi(z) / (1 + z) \times dV_c/dz` is normalized
    by its trapezoidal integral; interpolating the *complete unnormalized*
    density and dividing by that integral makes the interpolant's own integral
    exactly equal to the normalization. Samples outside ``redshift_grid``
    interpolate to zero density, hence ``-inf`` log-density. Also returns the
    interpolated luminosity distance ``d_L(z|\theta)``. This function is the
    single source of truth for the density formula: the same function is used
    for the proposal (at fiducials) and the target (at sampled ``params``), so
    the two densities can never drift apart; the importance weight is a *ratio*
    of them, and a second copy of the formula would bias every weight the
    moment either copy changed. :func:`log_weights` combines these with the
    catalog fiducial distances and the GW/EM ratio correction. Construction of
    a proposal density for a precomputed catalog must call this same function
    (on the grid the catalog was actually *sampled* from).

    Parameters
    ----------
    params:
        Hyperparameters. Must include ``H0``, ``Omega_m``, ``gamma``,
        ``kappa``, ``z_peak``, ``xi_0``, ``xi_n``, and ``local_merger_rate``
        (in ``Gpc^-3 yr^-1``).
    samples:
        Catalog arrays; must include ``redshift`` with leading dimension
        ``N``.
    redshift_grid:
        Redshift grid for the cosmology integrals and MD normalization.
        ``rate_shape_grid`` and ``dvc_dz_grid`` are evaluated on this exact
        grid, so the two can never fall out of alignment.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        ``(total_merger_rate, luminosity_distance, logpdf)``. Rate is in
        mergers per second; ``luminosity_distance`` and ``logpdf`` have shape
        ``(N,)``.
    """
    redshift = samples["redshift"]

    luminosity_distance_grid, dvc_dz_grid = distance_and_volume_grid(
        redshift_grid,
        hubble_constant=params["H0"],
        omega_m=params["Omega_m"],
    )
    rate_shape_grid = madau_dickinson_rate(
        redshift_grid, params["gamma"], params["kappa"], params["z_peak"]
    )
    unnormalized_pdf_grid = rate_shape_grid / (1.0 + redshift_grid) * dvc_dz_grid
    integral_mpc3 = jnp.trapezoid(unnormalized_pdf_grid, redshift_grid)

    unnormalized_pdf = jnp.interp(
        redshift,
        redshift_grid,
        unnormalized_pdf_grid,
        left=0.0,
        right=0.0,
    )
    # One log of the ratio, not `log(u) - log(Z)`: the operation order
    # `InterpolatedDistribution.log_prob` shares, so the two routes are
    # bit-identical and a catalog that is its own proposal has exactly zero
    # log-weight. Its docstring explains why this order is also the better
    # conditioned one.
    logpdf = jnp.log(unnormalized_pdf / integral_mpc3)
    luminosity_distance = jnp.interp(
        redshift,
        redshift_grid,
        luminosity_distance_grid,
        left=luminosity_distance_grid[0],
        right=luminosity_distance_grid[-1],
    )
    total_merger_rate = (
        1e-9 * params["local_merger_rate"] * integral_mpc3 / SECONDS_PER_YEAR
    )
    return total_merger_rate, luminosity_distance, logpdf


def bns_population(
    params: Mapping[str, ArrayLike], *, redshift_grid: jax.Array
) -> Population:
    """The BNS population at ``params``: a Madau-Dickinson redshift law.

    ``redshift_grid`` must be a concrete array -- the one the factory captured
    -- never a traced value. Its endpoints are read through NumPy rather than
    ``float(redshift_grid[0])``: under ``jax.jit`` even indexing a concrete
    closure constant is staged out and yields a tracer, whereas
    ``np.asarray`` of a concrete array is a host-side read that works inside
    or outside a trace. The grid the distribution rebuilds from those
    endpoints is bit-identical to ``redshift_grid``
    (``tests/core/test_distributions.py`` pins this).
    """
    grid = np.asarray(redshift_grid)
    redshift = MadauDickinsonRedshiftDistribution(
        params=params,
        minimum_redshift=float(grid[0]),
        maximum_redshift=float(grid[-1]),
        n_grid=grid.shape[0],
    )
    return Population(distributions={"redshift": redshift}, params=params)


def bns_population_terms(
    population: Population,
    source_parameters: Mapping[str, ArrayLike],
    *,
    luminosity_distance: ArrayLike | None = None,
) -> PopulationTerms:
    r"""Reduce a :func:`bns_population` to the arrays the weights need.

    ``luminosity_distance`` selects the side being evaluated. Left ``None``
    (the target), :math:`d_L` is interpolated from the population's own
    cosmology table. Given (the proposal), it is the catalog's stored fiducial
    distance -- the one the waveforms were generated at -- and only the GW/EM
    ratio at ``population.params`` is applied on top.
    """
    redshift_distribution = population.distributions["redshift"]
    assert isinstance(redshift_distribution, RedshiftDistribution)
    params = population.params
    redshift = jnp.asarray(source_parameters["redshift"])

    if luminosity_distance is None:
        luminosity_distance = redshift_distribution.luminosity_distance(redshift)
    log_gw_distance = jnp.log(jnp.asarray(luminosity_distance)) + log_gw_em_ratio(
        redshift, params["xi_0"], params["xi_n"]
    )
    return PopulationTerms(
        log_prob=population.log_prob(source_parameters),
        log_gw_distance=log_gw_distance,
        total_merger_rate=redshift_distribution.total_merger_rate(
            params["local_merger_rate"]
        ),
    )


def _proposal_terms(
    proposal_logprob: jax.Array,
    samples: Mapping[str, jax.Array],
    fiducials: Mapping[str, Any],
) -> PopulationTerms:
    """Proposal-side terms from a precomputed density and the stored distances.

    No :class:`Population` is built: the density arrives as an array, and the
    distance is the catalog's own. The rate is unused by the weights and is
    filled with ``nan`` so a stray read is loud rather than plausible.
    """
    redshift = samples["redshift"]
    log_gw_distance = jnp.log(samples["luminosity_distance"]) + log_gw_em_ratio(
        redshift, fiducials["xi_0"], fiducials["xi_n"]
    )
    return PopulationTerms(
        log_prob=proposal_logprob,
        log_gw_distance=log_gw_distance,
        total_merger_rate=jnp.asarray(jnp.nan),
    )


def log_weights(
    logprob: jax.Array,
    proposal_logprob: jax.Array,
    luminosity_distance: jax.Array,
    parameters: Mapping[str, Any],
    samples: Mapping[str, jax.Array],
    fiducials: Mapping[str, Any],
) -> jax.Array:
    """Log importance weights from grid-level pieces.

    A thin wrapper over
    :func:`~astrogwb.importance.population.importance_log_weights`: the
    target side is ``logprob`` and ``luminosity_distance`` at ``parameters``,
    the proposal side is ``proposal_logprob`` and the catalog's stored
    ``samples["luminosity_distance"]`` at ``fiducials``. Kept so callers
    holding :func:`compute_merger_rate_distance_and_logprob` outputs can form
    weights without building a population.
    """
    redshift = samples["redshift"]
    target = PopulationTerms(
        log_prob=logprob,
        log_gw_distance=jnp.log(luminosity_distance)
        + log_gw_em_ratio(redshift, parameters["xi_0"], parameters["xi_n"]),
        total_merger_rate=jnp.asarray(jnp.nan),
    )
    return importance_log_weights(
        target, _proposal_terms(proposal_logprob, samples, fiducials)
    )


def make_merger_rate_and_log_weights_fn(
    *,
    fiducials: Mapping[str, Any],
    redshift_grid: jax.Array,
    proposal_logprob: jax.Array,
) -> MergerRateAndLogWeightsFn:
    """Build the merger-rate + importance-log-weights callback.

    The returned closure reweights a fixed proposal catalog to arbitrary
    sampled hyperparameters. It is JAX-traceable and intended to be passed (pre-built) to
    :func:`~astrogwb.sampling.models.spectral_density_model`.

    For a fiducial proposal, ``proposal_logprob`` can be precomputed with
    :func:`compute_merger_rate_distance_and_logprob`::

        _, _, proposal_logprob = compute_merger_rate_distance_and_logprob(
            fiducials, samples, redshift_grid=redshift_grid
        )

    Parameters
    ----------
    fiducials:
        Fiducial hyperparameters used for the GW/EM ratio correction on the
        proposal side. Must include ``xi_0`` and ``xi_n``.
    redshift_grid:
        Redshift grid used for the cosmology integrals and MD normalization.
        Captured by the returned closure as a constant array.
    proposal_logprob:
        Precomputed proposal redshift log-pdf at the catalog redshifts, shape
        ``(N,)``. It may describe any proposal with support over the target.

    Returns
    -------
    MergerRateAndLogWeightsFn
        Callable ``(params, samples) -> (total_merger_rate, log_weights)``.
        ``total_merger_rate`` is in mergers per second; ``log_weights`` has
        shape ``(N,)``. ``samples`` must include ``redshift`` and
        ``luminosity_distance`` (fiducial EM distances from the catalog).
    """

    def merger_rate_and_log_weights_fn(
        params: Mapping[str, Any],
        samples: Mapping[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        target = bns_population_terms(
            bns_population(params, redshift_grid=redshift_grid), samples
        )
        proposal = _proposal_terms(proposal_logprob, samples, fiducials)
        return target.total_merger_rate, importance_log_weights(target, proposal)

    return merger_rate_and_log_weights_fn
