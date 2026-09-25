"""Tests for the time-delayed redshift density.

The delayed merger rate is checked against an independent adaptive quadrature
(:func:`scipy.integrate.quad` in :math:`\\log\\tau`) rather than against a
finer run of its own quantile rule.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy.integrate import quad

from astrogwb.cosmology import lookback_time, redshift_at_lookback_time
from astrogwb.distributions.delay import PowerLawDelayDistribution
from astrogwb.distributions.rates import madau_dickinson_rate
from astrogwb.distributions.redshift import (
    MadauDickinsonRedshiftDistribution,
    TimeDelayedRedshiftDistribution,
    madau_dickinson_time_delayed_redshift_distribution,
)

PARAMS = {"H0": 67.66, "Omega_m": 0.3096, "gamma": 2.7, "kappa": 2.9, "z_peak": 1.9}
N_GRID = 256
MAXIMUM_FORMATION_REDSHIFT = 20.0


def _delayed(
    delay: dist.Distribution, **params: float
) -> TimeDelayedRedshiftDistribution:
    return madau_dickinson_time_delayed_redshift_distribution(
        params={**PARAMS, **params},
        time_delay_distribution=delay,
        maximum_formation_redshift=MAXIMUM_FORMATION_REDSHIFT,
        n_grid=N_GRID,
    )


def _merger_rate_on_grid(distribution: TimeDelayedRedshiftDistribution) -> jax.Array:
    """Undo the (1 + z)^-1 dV_c/dz factor to recover R_m on the grid."""
    return (
        distribution.y
        * (1.0 + distribution.x)
        / distribution.differential_comoving_volume_grid
    )


# --------------------------------------------------------------------------- #
# Time-delayed redshift distribution
# --------------------------------------------------------------------------- #


def test_negligible_delay_reproduces_the_undelayed_distribution() -> None:
    delayed = _delayed(dist.DoublyTruncatedPowerLaw(-1.0, 1e-6, 2e-6))
    undelayed = MadauDickinsonRedshiftDistribution(params=PARAMS, n_grid=N_GRID)
    # Skip z = 0, where both tables vanish with dV_c/dz.
    np.testing.assert_allclose(
        delayed.normalized_y[1:], undelayed.normalized_y[1:], rtol=1e-4
    )


def test_merger_rate_today_is_the_local_merger_rate() -> None:
    """R_m(0) = local_merger_rate. Read just above z = 0, where dV_c/dz > 0."""
    delayed = madau_dickinson_time_delayed_redshift_distribution(
        params={**PARAMS, "local_merger_rate": 320.0},
        time_delay_distribution=dist.DoublyTruncatedPowerLaw(-1.0, 0.02, 13.0),
        minimum_redshift=1e-6,
        n_grid=N_GRID,
    )
    np.testing.assert_allclose(_merger_rate_on_grid(delayed)[0], 320.0, rtol=1e-5)


def test_merger_rate_matches_adaptive_quadrature() -> None:
    delay = dist.DoublyTruncatedPowerLaw(-1.0, 0.02, 13.0)
    distribution = _delayed(delay)
    h0, omega_m = PARAMS["H0"], PARAMS["Omega_m"]
    formation_cutoff = float(lookback_time(MAXIMUM_FORMATION_REDSHIFT, h0, omega_m))

    def reference(merger_redshift: float) -> float:
        merger_time = float(lookback_time(merger_redshift, h0, omega_m))

        def integrand(log_tau: float) -> float:
            tau = np.exp(log_tau)
            if merger_time + tau >= formation_cutoff:
                return 0.0
            formation_redshift = redshift_at_lookback_time(
                merger_time + tau, h0, omega_m
            )
            return (
                tau
                * float(jnp.exp(delay.log_prob(tau)))
                * float(madau_dickinson_rate(formation_redshift, 2.7, 2.9, 1.9))
            )

        points = [np.log(formation_cutoff - merger_time)]
        return quad(
            integrand,
            np.log(0.02),
            np.log(13.0),
            points=[p for p in points if np.log(0.02) < p < np.log(13.0)] or None,
            limit=500,
            epsabs=0.0,
            epsrel=1e-12,
        )[0]

    rate = _merger_rate_on_grid(distribution)
    indices = [26, 51, 128]  # z ~ 1, 2, 5 on the 256-node [0, 10] grid
    expected = [reference(float(distribution.x[i])) / reference(0.0) for i in indices]
    np.testing.assert_allclose(rate[jnp.array(indices)], expected, rtol=1e-8)


def test_log_prob_gradient_in_minimum_delay_is_smooth() -> None:
    """The nodes move with tau_min, so the gradient has no node-scale jumps."""

    def log_prob(minimum_delay: jax.Array) -> jax.Array:
        return _delayed(
            dist.DoublyTruncatedPowerLaw(-1.0, minimum_delay, 13.0)
        ).log_prob(0.3)

    minimum_delays = jnp.linspace(0.02, 0.05, 31)
    gradient = jax.vmap(jax.grad(log_prob))(minimum_delays)
    assert np.all(np.isfinite(gradient))
    # A smooth curve has second differences far below its own scale.
    assert np.max(np.abs(np.diff(gradient, n=2))) < 1e-2 * np.max(np.abs(gradient))


def test_log_prob_is_continuous_across_the_formation_cutoff() -> None:
    """The cut-off is the integral's upper limit, not a mask on fixed nodes.

    As H0 moves, the delay available before z_cut moves with it. Masking fixed
    nodes dropped them one at a time, which showed up as isolated second
    differences of order 1e-3 at z = 10 against a smooth background near 1e-7.
    """
    delay = dist.DoublyTruncatedPowerLaw(-1.0, 0.02, 13.0)
    hubble_constants = jnp.linspace(60.0, 80.0, 2001)
    log_prob = jax.vmap(lambda h0: _delayed(delay, H0=h0).log_prob(10.0))(
        hubble_constants
    )
    second_difference = np.abs(np.diff(np.asarray(log_prob), n=2))
    assert np.max(second_difference) < 10.0 * np.median(second_difference)


def test_distribution_round_trips_as_a_pytree() -> None:
    distribution = _delayed(dist.DoublyTruncatedPowerLaw(-1.0, 0.02, 13.0))
    leaves, treedef = jax.tree_util.tree_flatten(distribution)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is TimeDelayedRedshiftDistribution
    z = jnp.array([0.5, 1.234, 3.7])
    jitted = jax.jit(lambda d, value: d.log_prob(value))(distribution, z)
    np.testing.assert_allclose(jitted, distribution.log_prob(z), rtol=1e-15)
    assert rebuilt.n_delay_nodes == distribution.n_delay_nodes
    np.testing.assert_array_equal(rebuilt.log_prob(z), distribution.log_prob(z))


# --------------------------------------------------------------------------- #
# Power-law delay
# --------------------------------------------------------------------------- #
DELAYS = jnp.array([0.02, 0.1, 1.0, 5.0, 13.0])
QUANTILES = jnp.linspace(0.0, 1.0, 9)


@pytest.mark.parametrize("slope", [-3.0, -1.5, -1.0, -0.5, 0.0, 1.0])
def test_power_law_delay_matches_numpyro_away_from_the_gradient_problem(
    slope: float,
) -> None:
    ours = PowerLawDelayDistribution(slope, 0.02, 13.0)
    reference = dist.DoublyTruncatedPowerLaw(slope, 0.02, 13.0)
    np.testing.assert_allclose(ours.cdf(DELAYS), reference.cdf(DELAYS), atol=1e-14)
    np.testing.assert_allclose(
        ours.icdf(QUANTILES), reference.icdf(QUANTILES), rtol=1e-9
    )
    np.testing.assert_allclose(
        ours.log_prob(DELAYS), reference.log_prob(DELAYS), atol=1e-13
    )


def test_power_law_delay_icdf_inverts_the_cdf() -> None:
    delay = PowerLawDelayDistribution(-1.3, 0.02, 13.0)
    np.testing.assert_allclose(delay.cdf(delay.icdf(QUANTILES)), QUANTILES, atol=1e-14)


@pytest.mark.parametrize("offset", [-1e-6, -1e-9, -1e-12, 0.0, 1e-12, 1e-9, 1e-6])
def test_power_law_delay_slope_gradient_is_smooth_through_minus_one(
    offset: float,
) -> None:
    """numpyro's cdf/icdf slope gradients reach ~1e7 within 1e-9 of -1."""
    interior = QUANTILES[1:-1]

    def gradients(slope: float) -> tuple[jax.Array, ...]:
        return tuple(
            jax.grad(lambda a, f=f: f(PowerLawDelayDistribution(a, 0.02, 13.0)))(
                jnp.asarray(slope)
            )
            for f in (
                lambda d: d.cdf(DELAYS[1:-1]).sum(),
                lambda d: d.icdf(interior).sum(),
                lambda d: d.log_prob(DELAYS).sum(),
            )
        )

    # 1e-4 admits the true O(offset) change at 1e-6 and nothing like the
    # O(1e5) relative error the cancellation produces.
    np.testing.assert_allclose(gradients(-1.0 + offset), gradients(-1.0), rtol=1e-4)


def test_power_law_delay_samples_lie_in_the_support() -> None:
    delay = PowerLawDelayDistribution(-1.0, 0.02, 13.0)
    draws = delay.sample(jax.random.PRNGKey(0), (4096,))
    assert jnp.all((draws >= 0.02) & (draws <= 13.0))
    # Probability integral transform: F(draws) is uniform, mean 1/2 with a
    # standard error of ~0.0045 at this size.
    np.testing.assert_allclose(jnp.mean(delay.cdf(draws)), 0.5, atol=0.02)


def test_power_law_delay_is_numpyros_class_and_survives_a_jit_boundary() -> None:
    """Only the three formulas are overridden; the pytree layout is inherited."""
    delay = PowerLawDelayDistribution(-1.0, 0.02, 13.0)
    assert isinstance(delay, dist.DoublyTruncatedPowerLaw)

    @jax.jit
    def through(d: PowerLawDelayDistribution) -> PowerLawDelayDistribution:
        return d

    returned = through(delay)
    assert type(returned) is PowerLawDelayDistribution
    np.testing.assert_array_equal(returned.icdf(QUANTILES), delay.icdf(QUANTILES))


def test_power_law_delay_cdf_saturates_outside_the_support() -> None:
    delay = PowerLawDelayDistribution(-1.0, 0.02, 13.0)
    np.testing.assert_array_equal(delay.cdf(jnp.array([0.001, 20.0])), [0.0, 1.0])


def test_power_law_delay_rejects_a_zero_floor() -> None:
    """The parent accepts ``low = 0``; the log-space formulas cannot."""
    with pytest.raises(ValueError, match="low"):
        PowerLawDelayDistribution(0.0, 0.0, 1.0, validate_args=True)
