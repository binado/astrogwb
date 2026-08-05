"""Tests for the closed-form amplitude marginalization.

Every reference value here is built independently in float64 NumPy -- the
log joint is written out from the Gaussian definition and integrated on a dense
grid -- so the closed form is checked against the integral it claims to
evaluate, not against a rearrangement of itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb.sampling.amplitude import (
    AmplitudePrior,
    amplitude_conditional,
    amplitude_log_evidence,
    amplitude_statistics,
    best_fit_residual,
    draw_amplitude,
    gaussian_log_norm,
    noise_weighted_inner_product,
)

_LOG_TWO_PI = float(np.log(2.0 * np.pi))

TEMPLATE = np.array([1.0, 2.0, 4.0, 3.0])
DATA = np.array([1.3, 1.7, 4.6, 2.8])
SCALE = np.array([0.5, 0.4, 0.8, 0.6])


def _prior_support(prior: AmplitudePrior) -> tuple[float, float]:
    """Integration range covering essentially all of the prior mass."""
    if isinstance(prior, dist.Uniform):
        return float(prior.low), float(prior.high)
    loc, scale = float(prior.loc), float(prior.scale)
    return loc - 40.0 * scale, loc + 40.0 * scale


def _log_prior(amplitudes: np.ndarray, prior: AmplitudePrior) -> np.ndarray:
    if isinstance(prior, dist.Uniform):
        return np.full_like(amplitudes, -np.log(float(prior.high) - float(prior.low)))
    loc, scale = float(prior.loc), float(prior.scale)
    return -0.5 * ((amplitudes - loc) / scale) ** 2 - np.log(scale) - 0.5 * _LOG_TWO_PI


def _log_joint(
    amplitudes: np.ndarray,
    prior: AmplitudePrior,
    *,
    template: np.ndarray = TEMPLATE,
) -> np.ndarray:
    """``log p(d | A) + log pi(A)`` written out from the Gaussian definition."""
    residual = (DATA - amplitudes[:, None] * template) / SCALE
    log_likelihood = np.sum(
        -0.5 * residual**2 - np.log(SCALE) - 0.5 * _LOG_TWO_PI, axis=-1
    )
    return log_likelihood + _log_prior(amplitudes, prior)


def _grid(prior: AmplitudePrior, num: int = 200_001) -> np.ndarray:
    low, high = _prior_support(prior)
    return np.linspace(low, high, num)


def _numerical_log_evidence(prior: AmplitudePrior, **kwargs) -> float:
    amplitudes = _grid(prior)
    joint = np.exp(_log_joint(amplitudes, prior, **kwargs))
    return float(np.log(np.trapezoid(joint, amplitudes)))


def _analytic_log_evidence(
    prior: AmplitudePrior,
    *,
    template: np.ndarray = TEMPLATE,
) -> float:
    model = jnp.asarray(template)
    data = jnp.asarray(DATA)
    scale = jnp.asarray(SCALE)
    amplitude_ml, template_optimal_snr = amplitude_statistics(model, data, scale)
    return float(
        amplitude_log_evidence(
            amplitude_ml,
            template_optimal_snr,
            prior=prior,
            residual=best_fit_residual(model, data, scale, amplitude_ml=amplitude_ml),
            log_norm=gaussian_log_norm(scale),
        )
    )


def test_amplitude_statistics_at_perfect_match() -> None:
    template = jnp.asarray(TEMPLATE)
    scale = jnp.asarray(SCALE)

    amplitude_ml, template_optimal_snr = amplitude_statistics(template, template, scale)

    assert float(amplitude_ml) == 1.0
    np.testing.assert_allclose(
        float(template_optimal_snr),
        np.sqrt(np.sum((TEMPLATE / SCALE) ** 2)),
        rtol=1e-5,
    )


def test_inner_product_matches_explicit_sum() -> None:
    result = noise_weighted_inner_product(
        jnp.asarray(DATA), jnp.asarray(TEMPLATE), jnp.asarray(SCALE)
    )

    np.testing.assert_allclose(
        float(result), np.sum(DATA * TEMPLATE / SCALE**2), rtol=1e-5
    )


@pytest.mark.parametrize(
    "prior",
    [
        dist.Uniform(0.2, 3.0),
        dist.Uniform(0.9, 1.1),
        dist.Uniform(1.5, 4.0),
        dist.Uniform(5.0, 9.0),
        dist.Normal(1.0, 0.3),
        dist.Normal(0.5, 2.0),
    ],
    ids=[
        "uniform_wide",
        "uniform_tight",
        "uniform_offset",
        "uniform_far_outside",
        "normal",
        "normal_broad",
    ],
)
def test_log_evidence_matches_numerical_marginalization(
    prior: AmplitudePrior,
) -> None:
    """The load-bearing check: the closed form equals brute-force quadrature."""
    np.testing.assert_allclose(
        _analytic_log_evidence(prior),
        _numerical_log_evidence(prior),
        rtol=1e-4,
        atol=1e-4,
    )


def test_log_evidence_reduces_to_improper_prior_formula() -> None:
    """Wide Uniform and wide Normal both collapse onto the textbook expression.

    The reference ``-1/2 ln(m|m) + 1/2 (d|m)^2/(m|m)`` drops every additive
    constant, so the comparison is made on the difference between two templates
    -- which is the only part a sampler ever sees.
    """
    other_template = np.array([1.4, 1.9, 3.3, 3.6])

    def reference(template: np.ndarray) -> float:
        template_norm = np.sum(template**2 / SCALE**2)
        data_template = np.sum(DATA * template / SCALE**2)
        return -0.5 * np.log(template_norm) + 0.5 * data_template**2 / template_norm

    expected = reference(TEMPLATE) - reference(other_template)

    for prior in (dist.Uniform(-1e4, 1e4), dist.Normal(0.0, 1e4)):
        delta = _analytic_log_evidence(prior) - _analytic_log_evidence(
            prior, template=other_template
        )
        np.testing.assert_allclose(delta, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "prior",
    [dist.Uniform(0.2, 3.0), dist.Uniform(0.9, 1.1), dist.Normal(1.0, 0.3)],
    ids=["uniform_wide", "uniform_tight", "normal"],
)
def test_conditional_moments_match_numerical_posterior(
    prior: AmplitudePrior,
) -> None:
    amplitude_ml, template_optimal_snr = amplitude_statistics(
        jnp.asarray(TEMPLATE), jnp.asarray(DATA), jnp.asarray(SCALE)
    )
    conditional = amplitude_conditional(amplitude_ml, template_optimal_snr, prior=prior)

    amplitudes = _grid(prior)
    reference_mean, reference_variance = _moments(
        amplitudes, np.exp(_log_joint(amplitudes, prior))
    )
    conditional_mean, conditional_variance = _moments(
        amplitudes,
        np.asarray(jnp.exp(conditional.log_prob(jnp.asarray(amplitudes)))),
    )

    np.testing.assert_allclose(conditional_mean, reference_mean, rtol=1e-4)
    np.testing.assert_allclose(conditional_variance, reference_variance, rtol=1e-3)


def _moments(grid: np.ndarray, density: np.ndarray) -> tuple[float, float]:
    norm = np.trapezoid(density, grid)
    mean = np.trapezoid(density * grid, grid) / norm
    variance = np.trapezoid(density * (grid - mean) ** 2, grid) / norm
    return float(mean), float(variance)


def test_conditional_density_integrates_to_one_over_truncated_support() -> None:
    amplitude_ml, template_optimal_snr = amplitude_statistics(
        jnp.asarray(TEMPLATE), jnp.asarray(DATA), jnp.asarray(SCALE)
    )
    prior = dist.Uniform(0.9, 1.1)
    conditional = amplitude_conditional(amplitude_ml, template_optimal_snr, prior=prior)

    amplitudes = _grid(prior)
    density = np.asarray(jnp.exp(conditional.log_prob(jnp.asarray(amplitudes))))

    np.testing.assert_allclose(np.trapezoid(density, amplitudes), 1.0, rtol=1e-4)


def test_log_evidence_far_outside_support_is_finite_and_differentiable() -> None:
    """Warmup can propose a template whose best fit sits far outside the prior.

    That must cost the step a large log density, not poison the chain with
    ``-inf`` or a NaN gradient.
    """
    data = jnp.asarray(500.0 * TEMPLATE)
    scale = jnp.asarray(SCALE)
    prior = dist.Uniform(0.5, 1.5)

    def log_evidence_of(stretch: jax.Array) -> jax.Array:
        model = stretch * jnp.asarray(TEMPLATE)
        amplitude_ml, template_optimal_snr = amplitude_statistics(model, data, scale)
        return amplitude_log_evidence(
            amplitude_ml,
            template_optimal_snr,
            prior=prior,
            residual=best_fit_residual(model, data, scale, amplitude_ml=amplitude_ml),
            log_norm=gaussian_log_norm(scale),
        )

    value = float(log_evidence_of(jnp.asarray(1.0)))
    gradient = float(jax.grad(log_evidence_of)(jnp.asarray(1.0)))

    assert np.isfinite(value)
    assert value < -1e3
    assert np.isfinite(gradient)


def test_draw_amplitude_broadcasts_and_respects_bounds() -> None:
    amplitude_ml, template_optimal_snr = amplitude_statistics(
        jnp.asarray(TEMPLATE), jnp.asarray(DATA), jnp.asarray(SCALE)
    )
    batched_ml = jnp.broadcast_to(amplitude_ml, (2, 5))
    batched_snr = jnp.broadcast_to(template_optimal_snr, (2, 5))
    prior = dist.Uniform(0.95, 1.05)

    draws = draw_amplitude(
        batched_ml, batched_snr, prior=prior, rng_key=jax.random.key(0)
    )
    repeat = draw_amplitude(
        batched_ml, batched_snr, prior=prior, rng_key=jax.random.key(0)
    )

    assert draws.shape == (2, 5)
    np.testing.assert_array_equal(np.asarray(draws), np.asarray(repeat))
    assert bool(jnp.all((draws >= 0.95) & (draws <= 1.05)))


def test_draw_amplitude_recovers_the_conditional_mean() -> None:
    amplitude_ml, template_optimal_snr = amplitude_statistics(
        jnp.asarray(TEMPLATE), jnp.asarray(DATA), jnp.asarray(SCALE)
    )
    prior = dist.Normal(1.0, 0.5)
    conditional = amplitude_conditional(amplitude_ml, template_optimal_snr, prior=prior)
    count = 40_000

    draws = draw_amplitude(
        jnp.broadcast_to(amplitude_ml, (count,)),
        jnp.broadcast_to(template_optimal_snr, (count,)),
        prior=prior,
        rng_key=jax.random.key(1),
    )

    standard_error = float(jnp.sqrt(conditional.variance / count))
    np.testing.assert_allclose(
        float(jnp.mean(draws)),
        float(jnp.asarray(conditional.mean)),
        atol=4.0 * standard_error,
    )


def test_unsupported_prior_is_rejected() -> None:
    amplitude_ml, template_optimal_snr = amplitude_statistics(
        jnp.asarray(TEMPLATE), jnp.asarray(DATA), jnp.asarray(SCALE)
    )

    with pytest.raises(TypeError, match="Normal or Uniform"):
        amplitude_conditional(
            amplitude_ml, template_optimal_snr, prior=dist.LogNormal(0.0, 1.0)
        )
