"""Tests for numerical (quadrature) marginalization of the amplitude direction.

Mirrors ``test_amplitude.py``: the identity-scaling tests reuse
:func:`~astrogwb.sampling.amplitude.amplitude_log_evidence` itself as the
reference (it is already checked there against brute-force quadrature), which
is valid because under ``f(phi) = phi`` the two marginalizers integrate the
exact same integral. The cross-route test is the one that actually exercises
something ``test_amplitude.py`` cannot: it checks that marginalizing the
*physical* parameter numerically agrees with marginalizing the *proxy*
amplitude analytically and correcting with
:func:`~astrogwb.importance.diagnostics.log_prior_reweighting`, which is the
whole reason this module exists.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb.importance.diagnostics import log_prior_reweighting
from astrogwb.sampling.amplitude import (
    AmplitudePrior,
    amplitude_log_evidence,
    amplitude_statistics,
    best_fit_residual,
    draw_amplitude,
    gaussian_log_norm,
)
from astrogwb.sampling.amplitude_quadrature import (
    draw_marginalized_parameter,
    make_amplitude_quadrature,
    quadrature_effective_nodes,
    quadrature_log_evidence,
)
from jax.scipy.special import logsumexp
from numpyro.distributions import transforms

TEMPLATE = np.array([1.0, 2.0, 4.0, 3.0])
DATA = np.array([1.3, 1.7, 4.6, 2.8])
SCALE = np.array([0.5, 0.4, 0.8, 0.6])


def _identity_scaling(marginalized_parameter: jax.Array) -> jax.Array:
    return marginalized_parameter


def _statistics() -> tuple[jax.Array, jax.Array]:
    return amplitude_statistics(
        jnp.asarray(TEMPLATE), jnp.asarray(DATA), jnp.asarray(SCALE)
    )


def _residual_and_log_norm(amplitude_ml: jax.Array) -> tuple[jax.Array, jax.Array]:
    scale = jnp.asarray(SCALE)
    residual = best_fit_residual(
        jnp.asarray(TEMPLATE), jnp.asarray(DATA), scale, amplitude_ml=amplitude_ml
    )
    return residual, gaussian_log_norm(scale)


def _analytic_log_evidence(prior: AmplitudePrior) -> float:
    amplitude_ml, template_optimal_snr = _statistics()
    residual, log_norm = _residual_and_log_norm(amplitude_ml)
    return float(
        amplitude_log_evidence(
            amplitude_ml,
            template_optimal_snr,
            prior=prior,
            residual=residual,
            log_norm=log_norm,
        )
    )


def _quadrature_log_evidence(grid: jax.Array, log_prior: jax.Array) -> float:
    amplitude_ml, template_optimal_snr = _statistics()
    residual, log_norm = _residual_and_log_norm(amplitude_ml)
    quadrature = make_amplitude_quadrature(
        grid=grid, log_prior=log_prior, scaling=_identity_scaling
    )
    return float(
        quadrature_log_evidence(
            amplitude_ml,
            template_optimal_snr,
            quadrature=quadrature,
            residual=residual,
            log_norm=log_norm,
        )
    )


# --------------------------------------------------------------------------- #
# Identity scaling: the numerical route must reproduce the analytic one
# exactly, since both then integrate the same A.
# --------------------------------------------------------------------------- #


def test_quadrature_matches_analytic_uniform_prior() -> None:
    low, high = 0.2, 3.0
    grid = jnp.linspace(low, high, 4001)
    log_prior = jnp.full_like(grid, -jnp.log(high - low))

    np.testing.assert_allclose(
        _quadrature_log_evidence(grid, log_prior),
        _analytic_log_evidence(dist.Uniform(low, high)),
        rtol=1e-4,
        atol=1e-4,
    )


def test_quadrature_matches_analytic_normal_prior() -> None:
    loc, scale = 1.0, 0.3
    grid = jnp.linspace(loc - 10.0 * scale, loc + 10.0 * scale, 10_001)
    log_prior = dist.Normal(loc, scale).log_prob(grid)

    np.testing.assert_allclose(
        _quadrature_log_evidence(grid, log_prior),
        _analytic_log_evidence(dist.Normal(loc, scale)),
        rtol=1e-4,
        atol=1e-4,
    )


def test_unnormalized_log_prior_gives_the_same_evidence() -> None:
    low, high = 0.2, 3.0
    grid = jnp.linspace(low, high, 4001)
    normalized_log_prior = jnp.full_like(grid, -jnp.log(high - low))
    offset = 4.7
    unnormalized_log_prior = normalized_log_prior + offset

    normalized = make_amplitude_quadrature(
        grid=grid, log_prior=normalized_log_prior, scaling=_identity_scaling
    )
    unnormalized = make_amplitude_quadrature(
        grid=grid, log_prior=unnormalized_log_prior, scaling=_identity_scaling
    )

    np.testing.assert_allclose(
        float(unnormalized.log_prior_mass),
        float(normalized.log_prior_mass) + offset,
        rtol=1e-6,
    )

    amplitude_ml, template_optimal_snr = _statistics()
    residual, log_norm = _residual_and_log_norm(amplitude_ml)
    evidence_normalized = quadrature_log_evidence(
        amplitude_ml,
        template_optimal_snr,
        quadrature=normalized,
        residual=residual,
        log_norm=log_norm,
    )
    evidence_unnormalized = quadrature_log_evidence(
        amplitude_ml,
        template_optimal_snr,
        quadrature=unnormalized,
        residual=residual,
        log_norm=log_norm,
    )
    np.testing.assert_allclose(
        float(evidence_normalized), float(evidence_unnormalized), rtol=1e-6
    )


# --------------------------------------------------------------------------- #
# Cross-route: numerically marginalizing the physical parameter must agree
# with analytically marginalizing the proxy amplitude and reweighting.
# --------------------------------------------------------------------------- #


def _h0_pushforward_density(
    low: float, high: float, h0_fid: float
) -> dist.Distribution:
    r"""Distribution of :math:`A = H_{0,\mathrm{fid}}/H_0` under a uniform prior on :math:`H_0`.

    A real ``numpyro`` distribution (rather than a hand-rolled log-density)
    so it type-checks as the ``target: dist.Distribution`` that
    :func:`~astrogwb.importance.diagnostics.log_prior_reweighting` expects;
    ``PowerTransform(-1.0)`` maps :math:`H_0 \mapsto H_0^{-1}` and the
    following ``AffineTransform`` rescales by :math:`H_{0,\mathrm{fid}}`, with
    the Jacobian handled by ``TransformedDistribution`` itself.
    """
    return dist.TransformedDistribution(
        dist.Uniform(low, high),
        [transforms.PowerTransform(-1.0), transforms.AffineTransform(0.0, h0_fid)],
    )


def test_numerical_h0_marginalization_matches_reweighted_amplitude_draws() -> None:
    """Grid and sample counts stay modest deliberately.

    ``draw_marginalized_parameter`` materializes an ``(count, K)`` array, so
    this is the one test in the module where fixture size directly drives
    memory, not just wall time -- see the footprint note on that function's
    docstring.
    """
    h0_fid = 70.0
    h0_low, h0_high = 50.0, 90.0
    count = 20_000

    amplitude_ml, template_optimal_snr = _statistics()
    h0_grid = jnp.linspace(h0_low, h0_high, 1001)
    quadrature = make_amplitude_quadrature(
        grid=h0_grid,
        log_prior=jnp.zeros_like(h0_grid),
        scaling=lambda marginalized_parameter: h0_fid / marginalized_parameter,
    )
    h0_draws = draw_marginalized_parameter(
        jnp.broadcast_to(amplitude_ml, (count,)),
        jnp.broadcast_to(template_optimal_snr, (count,)),
        quadrature=quadrature,
        rng_key=jax.random.key(3),
    )
    numeric_mean = float(jnp.mean(h0_draws))
    numeric_variance = float(jnp.var(h0_draws))

    a_low, a_high = h0_fid / h0_high, h0_fid / h0_low
    used_prior = dist.Uniform(a_low, a_high)
    amplitude_draws = draw_amplitude(
        jnp.broadcast_to(amplitude_ml, (count,)),
        jnp.broadcast_to(template_optimal_snr, (count,)),
        prior=used_prior,
        rng_key=jax.random.key(4),
    )
    target = _h0_pushforward_density(h0_low, h0_high, h0_fid)
    log_weights = log_prior_reweighting(amplitude_draws, used=used_prior, target=target)
    weights = jnp.exp(log_weights - logsumexp(log_weights))
    h0_from_amplitude = h0_fid / amplitude_draws
    reweighted_mean = float(jnp.sum(weights * h0_from_amplitude))
    reweighted_variance = float(
        jnp.sum(weights * (h0_from_amplitude - reweighted_mean) ** 2)
    )

    np.testing.assert_allclose(numeric_mean, reweighted_mean, atol=1.0)
    np.testing.assert_allclose(numeric_variance, reweighted_variance, rtol=0.25)


# --------------------------------------------------------------------------- #
# draw_marginalized_parameter
# --------------------------------------------------------------------------- #


def test_draw_marginalized_parameter_recovers_conditional_moments() -> None:
    low, high = 0.2, 3.0
    grid = jnp.linspace(low, high, 4001)
    log_prior = jnp.full_like(grid, -jnp.log(high - low))
    quadrature = make_amplitude_quadrature(
        grid=grid, log_prior=log_prior, scaling=_identity_scaling
    )
    amplitude_ml, template_optimal_snr = _statistics()
    count = 20_000

    draws = draw_marginalized_parameter(
        jnp.broadcast_to(amplitude_ml, (count,)),
        jnp.broadcast_to(template_optimal_snr, (count,)),
        quadrature=quadrature,
        rng_key=jax.random.key(1),
    )

    # ``amplitude_ml`` sits ~7 sigma inside [low, high], so the untruncated
    # Normal moments are an adequate reference; numpyro's ``TruncatedNormal``
    # does not implement ``.variance``.
    conditional_mean = amplitude_ml
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
        grid=grid, log_prior=log_prior, scaling=_identity_scaling
    )
    amplitude_ml, template_optimal_snr = _statistics()
    batched_ml = jnp.broadcast_to(amplitude_ml, (2, 5))
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
        grid=grid, log_prior=log_prior, scaling=_identity_scaling
    )
    amplitude_ml, template_optimal_snr = _statistics()

    draws = draw_marginalized_parameter(
        jnp.broadcast_to(amplitude_ml, (100,)),
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
    amplitude_ml, template_optimal_snr = _statistics()

    fine_grid = jnp.linspace(low, high, 4001)
    coarse_grid = jnp.linspace(low, high, 21)
    fine = make_amplitude_quadrature(
        grid=fine_grid,
        log_prior=jnp.full_like(fine_grid, -jnp.log(high - low)),
        scaling=_identity_scaling,
    )
    coarse = make_amplitude_quadrature(
        grid=coarse_grid,
        log_prior=jnp.full_like(coarse_grid, -jnp.log(high - low)),
        scaling=_identity_scaling,
    )

    fine_nodes = float(
        quadrature_effective_nodes(amplitude_ml, template_optimal_snr, quadrature=fine)
    )
    coarse_nodes = float(
        quadrature_effective_nodes(
            amplitude_ml, template_optimal_snr, quadrature=coarse
        )
    )

    assert coarse_nodes < fine_nodes
    assert coarse_nodes <= coarse_grid.shape[0]


# --------------------------------------------------------------------------- #
# Malformed grids
# --------------------------------------------------------------------------- #


def test_make_amplitude_quadrature_rejects_non_monotonic_grid() -> None:
    grid = jnp.array([0.0, 1.0, 0.5, 2.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        make_amplitude_quadrature(
            grid=grid, log_prior=jnp.zeros_like(grid), scaling=_identity_scaling
        )


def test_make_amplitude_quadrature_rejects_length_one_grid() -> None:
    grid = jnp.array([1.0])
    with pytest.raises(ValueError, match="at least 2 points"):
        make_amplitude_quadrature(
            grid=grid, log_prior=jnp.zeros_like(grid), scaling=_identity_scaling
        )


def test_make_amplitude_quadrature_rejects_shape_mismatch() -> None:
    grid = jnp.linspace(0.0, 1.0, 10)
    log_prior = jnp.zeros(9)
    with pytest.raises(ValueError, match="must match grid shape"):
        make_amplitude_quadrature(
            grid=grid, log_prior=log_prior, scaling=_identity_scaling
        )


def test_make_amplitude_quadrature_rejects_non_1d_grid() -> None:
    grid = jnp.ones((3, 3))
    with pytest.raises(ValueError, match="must be 1D"):
        make_amplitude_quadrature(
            grid=grid, log_prior=jnp.zeros_like(grid), scaling=_identity_scaling
        )
