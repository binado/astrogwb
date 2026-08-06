"""Tests for numerical (quadrature) marginalization of the amplitude direction.

Every reference value here is built independently in float64 NumPy -- the
log joint is written out from the Gaussian definition and integrated on a
dense grid -- so the quadrature route is checked against the integral it
claims to evaluate, not against a rearrangement of itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb.sampling.amplitude import (
    amplitude_log_integrand,
    draw_amplitude_posterior,
    draw_marginalized_parameter,
    log_trapezoid,
    make_amplitude_quadrature,
    quadrature_effective_nodes,
)

type _AmplitudePrior = dist.Normal | dist.Uniform

_LOG_TWO_PI = float(np.log(2.0 * np.pi))

TEMPLATE = np.array([1.0, 2.0, 4.0, 3.0])
DATA = np.array([1.3, 1.7, 4.6, 2.8])
SCALE = np.array([0.5, 0.4, 0.8, 0.6])


def _identity_scaling(marginalized_parameter: jax.Array) -> jax.Array:
    return marginalized_parameter


def _ones_scaling(marginalized_parameter: jax.Array) -> jax.Array:
    return jnp.ones_like(marginalized_parameter)


def _statistics() -> tuple[jax.Array, jax.Array]:
    """The two amplitude sufficient statistics, as ``amplitude_marginalized_model`` computes them."""
    template = jnp.asarray(TEMPLATE)
    scale = jnp.asarray(SCALE)
    template_norm = jnp.sum(template**2 / scale**2, axis=-1)
    data_template = jnp.sum(jnp.asarray(DATA) * template / scale**2, axis=-1)
    return data_template / template_norm, jnp.sqrt(template_norm)


def _quadrature_log_evidence(
    grid: jax.Array,
    log_prior: jax.Array,
    *,
    merger_rate_amplitude=_identity_scaling,
    mean_energy_flux_amplitude=_ones_scaling,
) -> float:
    """Assemble the evidence the same way ``amplitude_marginalized_model`` does."""
    amplitude_mle, template_optimal_snr = _statistics()
    quadrature = make_amplitude_quadrature(
        grid=grid,
        log_prior=log_prior,
        merger_rate_amplitude=merger_rate_amplitude,
        mean_energy_flux_amplitude=mean_energy_flux_amplitude,
    )
    log_integrand = amplitude_log_integrand(
        amplitude_mle, template_optimal_snr, quadrature=quadrature
    )
    log_likelihood_at_mle = (
        dist.Normal(amplitude_mle * jnp.asarray(TEMPLATE), jnp.asarray(SCALE))
        .to_event(1)
        .log_prob(jnp.asarray(DATA))
    )
    return float(log_likelihood_at_mle + log_trapezoid(log_integrand, quadrature.grid))


def _prior_support(prior: _AmplitudePrior) -> tuple[float, float]:
    """Integration range covering essentially all of the prior mass."""
    if isinstance(prior, dist.Uniform):
        return float(prior.low), float(prior.high)
    loc, scale = float(prior.loc), float(prior.scale)
    return loc - 40.0 * scale, loc + 40.0 * scale


def _log_prior(amplitudes: np.ndarray, prior: _AmplitudePrior) -> np.ndarray:
    if isinstance(prior, dist.Uniform):
        return np.full_like(amplitudes, -np.log(float(prior.high) - float(prior.low)))
    loc, scale = float(prior.loc), float(prior.scale)
    return -0.5 * ((amplitudes - loc) / scale) ** 2 - np.log(scale) - 0.5 * _LOG_TWO_PI


def _log_joint(amplitudes: np.ndarray, prior: _AmplitudePrior) -> np.ndarray:
    """``log p(d | A) + log pi(A)`` written out from the Gaussian definition."""
    residual = (DATA - amplitudes[:, None] * TEMPLATE) / SCALE
    log_likelihood = np.sum(
        -0.5 * residual**2 - np.log(SCALE) - 0.5 * _LOG_TWO_PI, axis=-1
    )
    return log_likelihood + _log_prior(amplitudes, prior)


def _numerical_log_evidence(prior: _AmplitudePrior, num: int = 200_001) -> float:
    low, high = _prior_support(prior)
    amplitudes = np.linspace(low, high, num)
    joint = np.exp(_log_joint(amplitudes, prior))
    return float(np.log(np.trapezoid(joint, amplitudes)))


# --------------------------------------------------------------------------- #
# Quadrature evidence vs. brute-force numerical integration
# --------------------------------------------------------------------------- #


def test_quadrature_matches_brute_force_integration_uniform_prior() -> None:
    low, high = 0.2, 3.0
    grid = jnp.linspace(low, high, 4001)
    log_prior = jnp.full_like(grid, -jnp.log(high - low))

    np.testing.assert_allclose(
        _quadrature_log_evidence(grid, log_prior),
        _numerical_log_evidence(dist.Uniform(low, high)),
        rtol=1e-4,
        atol=1e-4,
    )


def test_quadrature_matches_brute_force_integration_normal_prior() -> None:
    loc, scale = 1.0, 0.3
    grid = jnp.linspace(loc - 10.0 * scale, loc + 10.0 * scale, 10_001)
    log_prior = dist.Normal(loc, scale).log_prob(grid)

    np.testing.assert_allclose(
        _quadrature_log_evidence(grid, log_prior),
        _numerical_log_evidence(dist.Normal(loc, scale)),
        rtol=1e-4,
        atol=1e-4,
    )


# --------------------------------------------------------------------------- #
# Nonlinear scaling: the real H0 pair, g_R = (H0_fid/H0)**3, g_F = (H0/H0_fid)**2
# --------------------------------------------------------------------------- #


def _h0_log_likelihood(h0: np.ndarray, h0_fid: float) -> np.ndarray:
    """``log p(d | A = f(h0))`` written out from the Gaussian definition.

    ``f(h0) = g_R(h0) * g_F(h0) = (h0_fid/h0)**3 * (h0/h0_fid)**2 = h0_fid/h0``,
    the same net amplitude as the single-scaling formula this test used to
    exercise -- the product of the two real exponents collapses to the
    original inverse relation, so the brute-force reference is unchanged.
    """
    amplitude = h0_fid / h0
    residual = (DATA - amplitude[:, None] * TEMPLATE) / SCALE
    return np.sum(-0.5 * residual**2 - np.log(SCALE) - 0.5 * _LOG_TWO_PI, axis=-1)


def _h0_brute_force_moments(
    low: float, high: float, h0_fid: float, num: int = 200_001
) -> tuple[float, float]:
    """Mean and variance of the H0 posterior under a uniform-in-H0 prior.

    The prior is a constant factor that cancels in the normalized weights, so
    it does not appear explicitly here.
    """
    h0 = np.linspace(low, high, num)
    density = np.exp(_h0_log_likelihood(h0, h0_fid))
    norm = np.trapezoid(density, h0)
    mean = np.trapezoid(density * h0, h0) / norm
    variance = np.trapezoid(density * (h0 - mean) ** 2, h0) / norm
    return float(mean), float(variance)


def test_numerical_h0_marginalization_matches_brute_force_integration() -> None:
    """The only coverage of the nonlinear scaling path, ``f(H0) = H0_fid / H0``.

    ``draw_marginalized_parameter`` materializes an ``(count, K)`` array, so
    this is the one test in the module where fixture size directly drives
    memory, not just wall time -- see the footprint note on that function's
    docstring.
    """
    h0_fid = 70.0
    h0_low, h0_high = 50.0, 90.0
    count = 20_000

    amplitude_mle, template_optimal_snr = _statistics()
    h0_grid = jnp.linspace(h0_low, h0_high, 1001)
    quadrature = make_amplitude_quadrature(
        grid=h0_grid,
        log_prior=jnp.full_like(h0_grid, -jnp.log(h0_high - h0_low)),
        merger_rate_amplitude=lambda marginalized_parameter: (
            (h0_fid / marginalized_parameter) ** 3
        ),
        mean_energy_flux_amplitude=lambda marginalized_parameter: (
            (marginalized_parameter / h0_fid) ** 2
        ),
    )
    h0_draws = draw_marginalized_parameter(
        jnp.broadcast_to(amplitude_mle, (count,)),
        jnp.broadcast_to(template_optimal_snr, (count,)),
        quadrature=quadrature,
        rng_key=jax.random.key(3),
    )
    numeric_mean = float(jnp.mean(h0_draws))
    numeric_variance = float(jnp.var(h0_draws))

    brute_force_mean, brute_force_variance = _h0_brute_force_moments(
        h0_low, h0_high, h0_fid
    )

    np.testing.assert_allclose(numeric_mean, brute_force_mean, atol=1.0)
    np.testing.assert_allclose(numeric_variance, brute_force_variance, rtol=0.25)


# --------------------------------------------------------------------------- #
# draw_marginalized_parameter
# --------------------------------------------------------------------------- #


def test_draw_marginalized_parameter_recovers_conditional_moments() -> None:
    low, high = 0.2, 3.0
    grid = jnp.linspace(low, high, 4001)
    log_prior = jnp.full_like(grid, -jnp.log(high - low))
    quadrature = make_amplitude_quadrature(
        grid=grid,
        log_prior=log_prior,
        merger_rate_amplitude=_identity_scaling,
        mean_energy_flux_amplitude=_ones_scaling,
    )
    amplitude_mle, template_optimal_snr = _statistics()
    count = 20_000

    draws = draw_marginalized_parameter(
        jnp.broadcast_to(amplitude_mle, (count,)),
        jnp.broadcast_to(template_optimal_snr, (count,)),
        quadrature=quadrature,
        rng_key=jax.random.key(1),
    )

    # ``amplitude_mle`` sits ~7 sigma inside [low, high], so the untruncated
    # Normal moments are an adequate reference; numpyro's ``TruncatedNormal``
    # does not implement ``.variance``.
    conditional_mean = amplitude_mle
    conditional_variance = 1.0 / template_optimal_snr**2
    standard_error = float(jnp.sqrt(conditional_variance / count))
    np.testing.assert_allclose(
        float(jnp.mean(draws)), float(conditional_mean), atol=6.0 * standard_error
    )
    assert bool(jnp.all((draws >= low) & (draws <= high)))


def test_draw_marginalized_parameter_broadcasts_and_is_reproducible() -> None:
    low, high = 0.2, 3.0
    grid = jnp.linspace(low, high, 2001)
    log_prior = jnp.full_like(grid, -jnp.log(high - low))
    quadrature = make_amplitude_quadrature(
        grid=grid,
        log_prior=log_prior,
        merger_rate_amplitude=_identity_scaling,
        mean_energy_flux_amplitude=_ones_scaling,
    )
    amplitude_mle, template_optimal_snr = _statistics()
    batched_ml = jnp.broadcast_to(amplitude_mle, (2, 5))
    batched_snr = jnp.broadcast_to(template_optimal_snr, (2, 5))

    draws = draw_marginalized_parameter(
        batched_ml, batched_snr, quadrature=quadrature, rng_key=jax.random.key(0)
    )
    repeat = draw_marginalized_parameter(
        batched_ml, batched_snr, quadrature=quadrature, rng_key=jax.random.key(0)
    )

    assert draws.shape == (2, 5)
    np.testing.assert_array_equal(np.asarray(draws), np.asarray(repeat))
    assert bool(jnp.all((draws >= low) & (draws <= high)))


def test_draw_marginalized_parameter_is_nan_free_far_outside_the_grid() -> None:
    """Deep tails underflow to a flat CDF; the guard must not produce NaNs."""
    low, high = 0.2, 0.4
    grid = jnp.linspace(low, high, 2001)
    log_prior = jnp.full_like(grid, -jnp.log(high - low))
    quadrature = make_amplitude_quadrature(
        grid=grid,
        log_prior=log_prior,
        merger_rate_amplitude=_identity_scaling,
        mean_energy_flux_amplitude=_ones_scaling,
    )
    amplitude_mle, template_optimal_snr = _statistics()

    draws = draw_marginalized_parameter(
        jnp.broadcast_to(amplitude_mle, (100,)),
        jnp.broadcast_to(template_optimal_snr, (100,)),
        quadrature=quadrature,
        rng_key=jax.random.key(2),
    )

    assert bool(jnp.all(jnp.isfinite(draws)))
    assert bool(jnp.all((draws >= low) & (draws <= high)))


# --------------------------------------------------------------------------- #
# quadrature_effective_nodes
# --------------------------------------------------------------------------- #


def test_quadrature_effective_nodes_falls_with_fewer_grid_points() -> None:
    low, high = 0.2, 3.0
    amplitude_mle, template_optimal_snr = _statistics()

    fine_grid = jnp.linspace(low, high, 4001)
    coarse_grid = jnp.linspace(low, high, 21)
    fine = make_amplitude_quadrature(
        grid=fine_grid,
        log_prior=jnp.full_like(fine_grid, -jnp.log(high - low)),
        merger_rate_amplitude=_identity_scaling,
        mean_energy_flux_amplitude=_ones_scaling,
    )
    coarse = make_amplitude_quadrature(
        grid=coarse_grid,
        log_prior=jnp.full_like(coarse_grid, -jnp.log(high - low)),
        merger_rate_amplitude=_identity_scaling,
        mean_energy_flux_amplitude=_ones_scaling,
    )

    fine_nodes = float(
        quadrature_effective_nodes(amplitude_mle, template_optimal_snr, quadrature=fine)
    )
    coarse_nodes = float(
        quadrature_effective_nodes(
            amplitude_mle, template_optimal_snr, quadrature=coarse
        )
    )

    assert coarse_nodes < fine_nodes
    assert coarse_nodes <= coarse_grid.shape[0]


# --------------------------------------------------------------------------- #
# draw_amplitude_posterior
# --------------------------------------------------------------------------- #


def _synthetic_samples(
    n_chain: int, n_draw: int, seed: int = 0
) -> dict[str, jax.Array]:
    rng = np.random.default_rng(seed)
    return {
        "amplitude_mle": jnp.asarray(rng.uniform(0.8, 1.2, size=(n_chain, n_draw))),
        "template_optimal_snr": jnp.asarray(
            rng.uniform(50.0, 150.0, size=(n_chain, n_draw))
        ),
    }


def _toy_quadrature():
    grid = jnp.linspace(0.5, 1.5, 2001)
    log_prior = jnp.zeros_like(grid)
    return make_amplitude_quadrature(
        grid=grid,
        log_prior=log_prior,
        merger_rate_amplitude=lambda marginalized_parameter: marginalized_parameter,
        mean_energy_flux_amplitude=lambda marginalized_parameter: jnp.ones_like(
            marginalized_parameter
        ),
    )


def test_draw_amplitude_posterior_is_chunk_size_invariant() -> None:
    """Chunking is purely a memory knob: results must not depend on chunk_size."""
    samples = _synthetic_samples(4, 37)
    quadrature = _toy_quadrature()
    key = jax.random.key(0)

    phi_unchunked, nodes_unchunked = draw_amplitude_posterior(
        samples, quadrature=quadrature, rng_key=key, chunk_size=10_000
    )
    phi_chunked, nodes_chunked = draw_amplitude_posterior(
        samples, quadrature=quadrature, rng_key=key, chunk_size=7
    )

    assert phi_unchunked.shape == (4, 37)
    np.testing.assert_array_equal(np.asarray(phi_unchunked), np.asarray(phi_chunked))
    # quadrature_effective_nodes reduces over K=2001 grid points; XLA may pick a
    # different summation order for a different vmap batch size, so allow for
    # floating-point reduction-order noise rather than requiring bit-equality.
    np.testing.assert_allclose(
        np.asarray(nodes_unchunked), np.asarray(nodes_chunked), rtol=1e-10
    )


# --------------------------------------------------------------------------- #
# amplitude property
# --------------------------------------------------------------------------- #


def test_amplitude_property_is_the_product_of_the_two_factors() -> None:
    grid = jnp.linspace(0.2, 3.0, 11)
    quadrature = make_amplitude_quadrature(
        grid=grid,
        log_prior=jnp.zeros_like(grid),
        merger_rate_amplitude=lambda marginalized_parameter: marginalized_parameter,
        mean_energy_flux_amplitude=lambda marginalized_parameter: (
            2.0 * marginalized_parameter
        ),
    )
    np.testing.assert_allclose(
        np.asarray(quadrature.amplitude), np.asarray(2.0 * grid**2)
    )


# --------------------------------------------------------------------------- #
# Malformed grids
# --------------------------------------------------------------------------- #


def test_make_amplitude_quadrature_rejects_non_monotonic_grid() -> None:
    grid = jnp.array([0.0, 1.0, 0.5, 2.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        make_amplitude_quadrature(
            grid=grid,
            log_prior=jnp.zeros_like(grid),
            merger_rate_amplitude=_identity_scaling,
            mean_energy_flux_amplitude=_ones_scaling,
        )


def test_make_amplitude_quadrature_rejects_length_one_grid() -> None:
    grid = jnp.array([1.0])
    with pytest.raises(ValueError, match="at least 2 points"):
        make_amplitude_quadrature(
            grid=grid,
            log_prior=jnp.zeros_like(grid),
            merger_rate_amplitude=_identity_scaling,
            mean_energy_flux_amplitude=_ones_scaling,
        )


def test_make_amplitude_quadrature_rejects_shape_mismatch() -> None:
    grid = jnp.linspace(0.0, 1.0, 10)
    log_prior = jnp.zeros(9)
    with pytest.raises(ValueError, match="must match grid shape"):
        make_amplitude_quadrature(
            grid=grid,
            log_prior=log_prior,
            merger_rate_amplitude=_identity_scaling,
            mean_energy_flux_amplitude=_ones_scaling,
        )


def test_make_amplitude_quadrature_rejects_non_1d_grid() -> None:
    grid = jnp.ones((3, 3))
    with pytest.raises(ValueError, match="must be 1D"):
        make_amplitude_quadrature(
            grid=grid,
            log_prior=jnp.zeros_like(grid),
            merger_rate_amplitude=_identity_scaling,
            mean_energy_flux_amplitude=_ones_scaling,
        )
