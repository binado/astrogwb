"""Reference population: BNS + Madau-Dickinson rate + modified propagation.

This module packages one concrete realization of the
:mod:`astrogwb.population` API: a binary-neutron-star population
whose merger-rate density follows the Madau-Dickinson (2017) shape, evolved on
a flat-LambdaCDM cosmology, with a phenomenological GW-to-EM
luminosity-distance ratio (``xi_0``, ``xi_n``) capturing a modified-gravity
propagation effect.

It is a *reference implementation* --
:class:`~astrogwb.importance.estimator.SpectralDensityImportanceEstimator`
accepts any :class:`~astrogwb.population.PopulationFn`, so callers
may substitute their own.

Two routes express the same redshift density here, and
``tests/core/test_distributions.py`` pins them together to rounding:

- :func:`bns_population` builds a
  :class:`~astrogwb.distributions.redshift.madau_dickinson.MadauDickinsonRedshiftDistribution`
  inside a :class:`~astrogwb.population.Population`. Its
  :meth:`~astrogwb.population.Population.compute_population_terms`
  reduces it to the arrays the weights need.
- :func:`compute_merger_rate_distance_and_logprob` is the grid-level formula
  written out by hand. It is kept as the reference the distribution class is
  tested against, and the paper application still calls it directly for the
  fiducial injection spectrum and the proposal density.

Factory contract: :func:`bns_population` takes a ``redshift_grid`` array that
must be concrete rather than traced, so the population it builds is safe to
construct inside ``jax.jit`` during NUTS. Bind it once with
``functools.partial``; sampled hyperparameters arrive through ``params``.

The reference distance the stored polarization power corresponds to is cached
on :class:`~astrogwb.catalog.ImportanceCatalog`, never recomputed from a
cosmology table; :mod:`astrogwb.importance.weights` explains why.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
from astrogwb.distributions.redshift.madau_dickinson import (
    MadauDickinsonRedshiftDistribution,
)
from astrogwb.population import CosmologicalPopulation

AMPLITUDE_PARAMETERS: tuple[str, ...] = ("H0", "local_merger_rate")
"""Parameters this population supports marginalizing analytically."""


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
    moment either copy changed.
    :func:`~astrogwb.importance.weights.importance_log_weights` combines
    these with the reference distance cached on the catalog. Construction of a
    proposal density for a precomputed catalog must call this same function (on
    the grid the catalog was actually *sampled* from).

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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ModifiedPropagationPopulation(CosmologicalPopulation):
    """Cosmological population with the ``xi_0`` / ``xi_n`` propagation law."""

    def luminosity_distance(self, redshift: ArrayLike) -> jax.Array:
        """Effective luminosity distance governing waveform amplitude, in Mpc."""
        return super().luminosity_distance(redshift) * jnp.exp(
            log_gw_em_ratio(redshift, self.params["xi_0"], self.params["xi_n"])
        )


def bns_population(
    params: Mapping[str, ArrayLike], *, redshift_grid: jax.Array
) -> ModifiedPropagationPopulation:
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
    return ModifiedPropagationPopulation(
        distributions={"redshift": redshift}, params=params
    )
