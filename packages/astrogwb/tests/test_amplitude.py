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
from astrogwb.sampling import AmplitudeConditional, quadrature_grid

type _AmplitudePrior = dist.Normal | dist.Uniform

_LOG_TWO_PI = float(np.log(2.0 * np.pi))

TEMPLATE = np.array([1.0, 2.0, 4.0, 3.0])
DATA = np.array([1.3, 1.7, 4.6, 2.8])
SCALE = np.array([0.5, 0.4, 0.8, 0.6])


def _identity_scaling(marginalized_parameter: jax.Array) -> jax.Array:
    """With ``fiducial=1.0`` this makes the amplitude the parameter itself."""
    return marginalized_parameter


def _statistics() -> tuple[jax.Array, jax.Array]:
    """The two amplitude sufficient statistics, as ``amplitude_marginalized_model`` computes them."""
    template = jnp.asarray(TEMPLATE)
    scale = jnp.asarray(SCALE)
    template_norm = jnp.sum(template**2 / scale**2, axis=-1)
    data_template = jnp.sum(jnp.asarray(DATA) * template / scale**2, axis=-1)
    return data_template / template_norm, jnp.sqrt(template_norm)


def _uniform_conditional(
    low: float,
    high: float,
    num_nodes: int,
    *,
    batch_shape: tuple[int, ...] = (),
) -> AmplitudeConditional:
    """The conditional under a uniform prior, with the parameter *being* the amplitude.

    ``_identity_scaling`` anchored at ``fiducial=1.0`` gives
    ``A(phi) = phi / 1.0 = phi``, which is what every brute-force reference in
    this module assumes. The grid is left to default, so this also exercises
    :func:`~astrogwb.sampling.amplitude.quadrature_grid` clipping a
    ten-standard-deviation span down to the uniform bounds.
    """
    amplitude_mle, template_optimal_snr = _statistics()
    return AmplitudeConditional(
        jnp.broadcast_to(amplitude_mle, batch_shape),
        jnp.broadcast_to(template_optimal_snr, batch_shape),
        amplitude_fn=_identity_scaling,
        prior=dist.Uniform(low, high),
        fiducial=1.0,
        num_nodes=num_nodes,
    )


def _conditional_log_evidence(
    prior: _AmplitudePrior,
    grid: jax.Array,
) -> float:
    """Assemble the evidence the same way ``amplitude_marginalized_model`` does."""
    amplitude_mle, template_optimal_snr = _statistics()
    conditional = AmplitudeConditional(
        amplitude_mle,
        template_optimal_snr,
        amplitude_fn=_identity_scaling,
        prior=prior,
        fiducial=1.0,
        grid=grid,
    )
    log_likelihood_at_mle = (
        dist.Normal(amplitude_mle * jnp.asarray(TEMPLATE), jnp.asarray(SCALE))
        .to_event(1)
        .log_prob(jnp.asarray(DATA))
    )
    return float(log_likelihood_at_mle + conditional.log_normalizer)


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

    np.testing.assert_allclose(
        _conditional_log_evidence(
            dist.Uniform(low, high), jnp.linspace(low, high, 4001)
        ),
        _numerical_log_evidence(dist.Uniform(low, high)),
        rtol=1e-4,
        atol=1e-4,
    )


def test_quadrature_matches_brute_force_integration_normal_prior() -> None:
    loc, scale = 1.0, 0.3
    prior = dist.Normal(loc, scale)

    np.testing.assert_allclose(
        _conditional_log_evidence(
            prior, jnp.linspace(loc - 10.0 * scale, loc + 10.0 * scale, 10_001)
        ),
        _numerical_log_evidence(prior),
        rtol=1e-4,
        atol=1e-4,
    )


def test_quadrature_grid_reproduces_the_hand_built_grids() -> None:
    """The default grid must be the one the brute-force tests above pass explicitly."""
    np.testing.assert_allclose(
        np.asarray(quadrature_grid(dist.Uniform(20.0, 140.0), num_nodes=101)),
        np.linspace(20.0, 140.0, 101),
    )
    np.testing.assert_allclose(
        np.asarray(quadrature_grid(dist.Normal(1.4, 0.3), num_nodes=101)),
        np.linspace(-1.6, 4.4, 101),
    )


def test_quadrature_grid_rejects_a_prior_without_a_variance() -> None:
    """``TruncatedNormal`` has no ``.variance``; the caller must pass ``grid=``."""
    prior = dist.TruncatedNormal(1.0, 0.3, low=0.0)
    with pytest.raises(TypeError, match="does not implement .variance"):
        quadrature_grid(prior)


# --------------------------------------------------------------------------- #
# Nonlinear scaling: the real H0 pair, g_R = H0**-3, g_F = H0**2
# --------------------------------------------------------------------------- #


def _h0_amplitude(marginalized_parameter: jax.Array) -> jax.Array:
    """``f = g_R * g_F = H0**-3 * H0**2``, left unsimplified on purpose."""
    return marginalized_parameter**-3 * marginalized_parameter**2


def _h0_log_likelihood(h0: np.ndarray, h0_fid: float) -> np.ndarray:
    """``log p(d | A = f(h0)/f(h0_fid))`` written out from the Gaussian definition.

    ``f(h0)/f(h0_fid) = (h0**-3 * h0**2) / (h0_fid**-3 * h0_fid**2) = h0_fid/h0``,
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
    """The only coverage of the nonlinear scaling path, ``A(H0) = H0_fid / H0``.

    ``AmplitudeConditional.icdf`` materializes a ``batch_shape + (K,)`` CDF, so
    this is the one test in the module where fixture size directly drives
    memory, not just wall time.
    """
    h0_fid = 70.0
    h0_low, h0_high = 50.0, 90.0
    count = 20_000

    amplitude_mle, template_optimal_snr = _statistics()
    conditional = AmplitudeConditional(
        jnp.broadcast_to(amplitude_mle, (count,)),
        jnp.broadcast_to(template_optimal_snr, (count,)),
        amplitude_fn=_h0_amplitude,
        prior=dist.Uniform(h0_low, h0_high),
        fiducial=h0_fid,
        grid=jnp.linspace(h0_low, h0_high, 1001),
    )
    h0_draws = conditional.sample(jax.random.key(3))
    numeric_mean = float(jnp.mean(h0_draws))
    numeric_variance = float(jnp.var(h0_draws))

    brute_force_mean, brute_force_variance = _h0_brute_force_moments(
        h0_low, h0_high, h0_fid
    )

    np.testing.assert_allclose(numeric_mean, brute_force_mean, atol=1.0)
    np.testing.assert_allclose(numeric_variance, brute_force_variance, rtol=0.25)


# --------------------------------------------------------------------------- #
# AmplitudeConditional.sample
# --------------------------------------------------------------------------- #


def test_sample_recovers_conditional_moments() -> None:
    low, high = 0.2, 3.0
    count = 20_000
    amplitude_mle, template_optimal_snr = _statistics()

    conditional = _uniform_conditional(low, high, 4001, batch_shape=(count,))
    draws = conditional.sample(jax.random.key(1))

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


def test_sample_broadcasts_is_reproducible_and_honors_sample_shape() -> None:
    low, high = 0.2, 3.0
    conditional = _uniform_conditional(low, high, 2001, batch_shape=(2, 5))

    draws = conditional.sample(jax.random.key(0))
    repeat = conditional.sample(jax.random.key(0))

    assert draws.shape == (2, 5)
    np.testing.assert_array_equal(np.asarray(draws), np.asarray(repeat))
    assert bool(jnp.all((draws >= low) & (draws <= high)))

    with_sample_shape = conditional.sample(jax.random.key(0), sample_shape=(3,))
    assert with_sample_shape.shape == (3, 2, 5)
    assert bool(jnp.all((with_sample_shape >= low) & (with_sample_shape <= high)))

    scalar = _uniform_conditional(low, high, 2001)
    assert scalar.sample(jax.random.key(0), sample_shape=(7,)).shape == (7,)


def test_sample_is_nan_free_far_outside_the_grid() -> None:
    """Deep tails underflow to a flat CDF; the guard must not produce NaNs."""
    low, high = 0.2, 0.4
    conditional = _uniform_conditional(low, high, 2001, batch_shape=(100,))
    draws = conditional.sample(jax.random.key(2))

    assert bool(jnp.all(jnp.isfinite(draws)))
    assert bool(jnp.all((draws >= low) & (draws <= high)))


# --------------------------------------------------------------------------- #
# AmplitudeConditional.log_prob and support
# --------------------------------------------------------------------------- #


def test_log_prob_integrates_to_one_on_the_grid() -> None:
    """The normalizer must be the integral of the density ``log_prob`` exposes.

    Evaluating on a finer sub-grid checks the analytic density between nodes,
    not just at them: the coarse-grid trapezoid normalizer and the fine-grid
    trapezoid of the normalized density must agree, which they do to well
    beyond this tolerance whenever the conditional is resolved (here
    ``sigma/h ~ 4``).
    """
    low, high = 0.2, 3.0
    conditional = _uniform_conditional(low, high, 101)

    fine = jnp.linspace(low, high, 4 * conditional.grid.shape[0] - 3)
    integral = jnp.trapezoid(jnp.exp(conditional.log_prob(fine)), fine)
    np.testing.assert_allclose(float(integral), 1.0, rtol=1e-5, atol=1e-6)


def test_log_prob_is_finite_off_the_grid_inside_the_prior_support() -> None:
    """``log_prob`` is analytic and grid-free; the grid is only the quadrature.

    The tabulated implementation this replaced returned ``-inf`` here.
    """
    amplitude_mle, template_optimal_snr = _statistics()
    conditional = AmplitudeConditional(
        amplitude_mle,
        template_optimal_snr,
        amplitude_fn=_identity_scaling,
        prior=dist.Normal(1.0, 0.3),
        fiducial=1.0,
        grid=jnp.linspace(0.7, 1.3, 101),
    )

    off_grid = jnp.array([0.0, 2.0])
    assert bool(jnp.all(jnp.isfinite(conditional.log_prob(off_grid))))


def test_log_prob_is_minus_inf_outside_the_prior_support() -> None:
    conditional = _uniform_conditional(0.2, 3.0, 101)

    outside = jnp.array([0.2 - 0.5, 3.0 + 0.5])
    assert bool(jnp.all(jnp.isneginf(conditional.log_prob(outside))))


def test_support_matches_the_prior_support() -> None:
    """The support is the prior's, independent of where the grid happens to sit."""
    prior = dist.Uniform(0.2, 3.0)
    amplitude_mle, template_optimal_snr = _statistics()
    conditional = AmplitudeConditional(
        amplitude_mle,
        template_optimal_snr,
        amplitude_fn=_identity_scaling,
        prior=prior,
        fiducial=1.0,
        grid=jnp.linspace(0.5, 1.5, 101),
    )

    support = conditional.support
    np.testing.assert_allclose(float(support.lower_bound), 0.2, rtol=1e-6)
    np.testing.assert_allclose(float(support.upper_bound), 3.0, rtol=1e-6)


# --------------------------------------------------------------------------- #
# AmplitudeConditional as a pytree
# --------------------------------------------------------------------------- #

_MAPPED_FIELDS = ("amplitude_mle", "template_optimal_snr")


def test_distribution_survives_jit_and_vmap_as_a_pytree_argument() -> None:
    """``pytree_data_fields`` must carry the statistics, prior, grid, and fiducial."""
    conditional = _uniform_conditional(0.2, 3.0, 2001, batch_shape=(4,))
    quantiles = 0.5 * jnp.ones(4)

    median = jax.jit(lambda c: c.icdf(quantiles))(conditional)
    assert median.shape == (4,)

    # vmap over the statistics batch while keeping the definition unmapped: the
    # in_axes specimen mirrors the distribution's pytree, with 0 on the two
    # statistics and None covering the prior / grid / fiducial subtrees. The
    # gathered field order is a set iteration order, so build the values by
    # field name rather than positionally.
    _aux = AmplitudeConditional.tree_flatten(conditional)[1]
    _fields = AmplitudeConditional.gather_pytree_data_fields()
    in_axes = AmplitudeConditional.tree_unflatten(
        _aux, tuple(0 if field in _MAPPED_FIELDS else None for field in _fields)
    )
    medians = jax.vmap(lambda c: c.icdf(jnp.array(0.5)), in_axes=(in_axes,))(
        conditional
    )
    assert medians.shape == (4,)
    np.testing.assert_allclose(np.asarray(medians), np.asarray(median), rtol=1e-6)


def test_amplitude_fn_is_aux_data_so_jit_caches_on_it() -> None:
    """A non-hashable scaling would make every construction a fresh cache key."""
    conditional = _uniform_conditional(0.2, 3.0, 101)
    aux = AmplitudeConditional.tree_flatten(conditional)[1]

    assert _identity_scaling in aux
    assert hash(aux) == hash(AmplitudeConditional.tree_flatten(conditional)[1])


# --------------------------------------------------------------------------- #
# AmplitudeConditional.effective_nodes
# --------------------------------------------------------------------------- #


def test_effective_nodes_falls_with_fewer_grid_points() -> None:
    low, high = 0.2, 3.0
    fine = _uniform_conditional(low, high, 4001)
    coarse = _uniform_conditional(low, high, 21)

    fine_nodes = float(fine.effective_nodes)
    coarse_nodes = float(coarse.effective_nodes)

    assert coarse_nodes < fine_nodes
    assert coarse_nodes <= coarse.grid.shape[0]
