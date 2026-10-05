r"""Amplitude marginalization, checked against the full two-parameter likelihood.

The model is a power law, :math:`S_h(f) = A\,(f/f_\mathrm{ref})^\alpha`: the
amplitude :math:`A = T(\varphi)` is marginalized analytically-plus-quadrature by
:func:`~astrogwb.inference.gwb_amplitude_marginalized_model`, and the shape
:math:`\alpha` stays a sampled parameter. Both are small enough that the *joint*
posterior :math:`p(\varphi, \alpha \mid d)` can be evaluated on a dense 2D grid
through the general model, which gives numerical marginals that share no code
with the A-space machinery under test.

Three behaviours are pinned against the general model, one per test:

- the marginalized model's density of :math:`\alpha`, and its gradient, are the
  :math:`\varphi` marginal of the general model's, with no leftover constant;
- the :math:`\varphi` posterior reconstructed as a mixture of
  :class:`~astrogwb.distributions.amplitude.AmplitudeConditional` over
  :math:`\alpha` is the :math:`\alpha` marginal of the general model's;
- at fixed :math:`\alpha`, ``AmplitudeConditional.log_prob`` is the conditional
  slice of the general model.

Both amplitude priors are cases: a ``Uniform`` on :math:`A` whose bounds sit a
few :math:`\sigma_A = 1/\rho` from the maximum-likelihood amplitude, so the bound
genuinely enters, and a ``Normal`` on a rate, so the Jacobian of the map to
:math:`A` does.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.distributions.transforms import Transform
from numpyro.infer import Predictive
from numpyro.infer.util import log_density

from astrogwb.distributions.amplitude import (
    AmplitudeConditional,
    amplitude_prior,
    support_bounds,
)
from astrogwb.inference import (
    LogDensityFn,
    amplitude_H0_transform,
    amplitude_local_merger_rate_transform,
    gwb_amplitude_marginalized_model,
    gwb_spectral_density_model,
)

F_REF = 25.0
ALPHA_TRUE = -0.6
A_TRUE = 1.0
PHI_FID = 2.0
"""Rate at which the Normal case's amplitude is one."""
NUM_PHI = 64001
NUM_ALPHA = 41


@dataclass(frozen=True)
class AmplitudeCase:
    """A physical prior on ``phi``, its map to the amplitude, and a grid over it."""

    phi_prior: dist.Distribution
    transform: Transform
    phi_grid: jax.Array

    @property
    def prior(self) -> dist.Distribution:
        """The amplitude-space pushforward the marginalized model integrates."""
        return amplitude_prior(self.phi_prior, self.transform)


@dataclass(frozen=True)
class PowerLawModel:
    """Both formulations of the power-law likelihood over one dataset."""

    case: AmplitudeCase
    alpha_prior: dist.Uniform
    alpha_grid: jax.Array
    general: Callable[[], None]
    marginalized: Callable[[], None]


def _log_trapezoid(log_y: jax.Array, x: jax.Array, axis: int) -> jax.Array:
    """Stable ``log(trapezoid(exp(log_y), x))`` along ``axis``."""
    shift = jnp.max(log_y, axis=axis, keepdims=True)
    weights = _trapezoid_weights(x)
    integral = jnp.tensordot(jnp.exp(log_y - shift), weights, axes=([axis], [0]))
    return jnp.squeeze(shift, axis=axis) + jnp.log(integral)


def _trapezoid_weights(x: jax.Array) -> jax.Array:
    """Weights ``w`` such that ``jnp.sum(w * y)`` is ``jnp.trapezoid(y, x)``."""
    dx = jnp.diff(x)
    return jnp.zeros_like(x).at[:-1].add(dx / 2).at[1:].add(dx / 2)


@pytest.fixture(scope="module")
def frequencies() -> jax.Array:
    return jnp.geomspace(20.0, 200.0, 16)


@pytest.fixture(scope="module")
def scale() -> jax.Array:
    """Per-bin noise, sized so the template SNR is ~20 and 1024 nodes resolve it."""
    return jnp.linspace(0.08, 0.3, 16)


@pytest.fixture(scope="module")
def observed(frequencies: jax.Array, scale: jax.Array) -> jax.Array:
    noise = np.random.default_rng(7).standard_normal(frequencies.size)
    return A_TRUE * (frequencies / F_REF) ** ALPHA_TRUE + scale * jnp.asarray(noise)


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(("uniform", 0.9, 1.15), id="uniform"),
        pytest.param(("normal-rate", PHI_FID, 0.3), id="normal-rate"),
    ],
)
def case(request: pytest.FixtureRequest) -> AmplitudeCase:
    kind, first, second = request.param
    if kind == "uniform":
        return AmplitudeCase(
            dist.Uniform(first, second),
            amplitude_local_merger_rate_transform(1.0),
            jnp.linspace(first, second, NUM_PHI),
        )
    return AmplitudeCase(
        dist.Normal(first, second),
        amplitude_local_merger_rate_transform(PHI_FID),
        jnp.linspace(first - 10 * second, first + 10 * second, NUM_PHI),
    )


@pytest.fixture(scope="module")
def model(
    case: AmplitudeCase,
    frequencies: jax.Array,
    scale: jax.Array,
    observed: jax.Array,
) -> PowerLawModel:
    alpha_prior = dist.Uniform(ALPHA_TRUE - 1.0, ALPHA_TRUE + 1.0)
    alpha_grid = jnp.linspace(ALPHA_TRUE - 0.4, ALPHA_TRUE + 0.4, NUM_ALPHA)

    def template(params: Mapping[str, ArrayLike]) -> jax.Array:
        return (frequencies / F_REF) ** jnp.asarray(params["alpha"])

    def full_spectrum(
        params: Mapping[str, ArrayLike],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        return jnp.asarray(case.transform(params["phi"])) * template(params), {}

    def template_spectrum(
        params: Mapping[str, ArrayLike],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        return template(params), {}

    def general() -> None:
        gwb_spectral_density_model(
            spectral_density_fn=full_spectrum,
            observed_spectral_density=observed,
            scale=scale,
            priors={"phi": case.phi_prior, "alpha": alpha_prior},
        )

    def marginalized() -> None:
        gwb_amplitude_marginalized_model(
            spectral_density_fn=template_spectrum,
            observed_spectral_density=observed,
            scale=scale,
            priors={"alpha": alpha_prior},
            amplitude_prior=case.prior,
        )

    return PowerLawModel(case, alpha_prior, alpha_grid, general, marginalized)


@pytest.fixture(scope="module")
def log_joint_2d(model: PowerLawModel) -> jax.Array:
    """``log p(phi, alpha, d)`` on the ``(alpha, phi)`` grid, from the general model.

    ``LogDensityFn`` documents axes in insertion order, but the grids cross a
    ``jax.jit`` boundary as a dict, which JAX flattens in sorted key order: the
    axes come out ``("alpha", "phi")`` whatever the order given here.
    """
    return LogDensityFn(model.general, chunk_size=1024)(
        {"phi": model.case.phi_grid, "alpha": model.alpha_grid}
    )


@pytest.fixture(scope="module")
def log_marginalized(model: PowerLawModel) -> jax.Array:
    """``log p(alpha, d)`` on the ``alpha`` grid, from the marginalized model."""
    return LogDensityFn(model.marginalized)({"alpha": model.alpha_grid})


def _statistics(model: PowerLawModel, alpha: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(amplitude_mle, template_optimal_snr)`` at each ``alpha``."""
    sites = Predictive(
        model.marginalized,
        posterior_samples={"alpha": alpha},
        return_sites=["amplitude_mle", "template_optimal_snr"],
    )(jax.random.key(0))
    return sites["amplitude_mle"], sites["template_optimal_snr"]


def test_marginalized_model_alpha_density_and_gradient_match_numerical_marginal(
    model: PowerLawModel,
) -> None:
    phi = model.case.phi_grid

    def numerical(alpha: jax.Array) -> jax.Array:
        """``log p(alpha, d)`` by quadrature of the general model over ``phi``."""
        log_joint = jax.vmap(
            lambda value: log_density(
                model.general, (), {}, {"phi": value, "alpha": alpha}
            )[0]
        )(phi)
        return _log_trapezoid(log_joint, phi, axis=0)

    def marginalized(alpha: jax.Array) -> jax.Array:
        return log_density(model.marginalized, (), {}, {"alpha": alpha})[0]

    # The gradient is what NUTS follows, and it reaches the amplitude's prior
    # bound through the statistics' dependence on alpha.
    value, grad = jax.vmap(jax.value_and_grad(marginalized))(model.alpha_grid)
    expected_value, expected_grad = jax.vmap(jax.value_and_grad(numerical))(
        model.alpha_grid
    )

    # No free constant: the Jacobian is carried by the pushforward prior.
    np.testing.assert_allclose(value, expected_value, rtol=0, atol=1e-9)
    np.testing.assert_allclose(grad, expected_grad, rtol=0, atol=1e-8)


def test_amplitude_conditional_mixture_matches_numerical_amplitude_marginal(
    model: PowerLawModel, log_joint_2d: jax.Array, log_marginalized: jax.Array
) -> None:
    case = model.case
    phi = case.phi_grid
    amplitude = jnp.asarray(case.transform(phi))

    # Numerical: integrate the joint over alpha, normalize over phi.
    log_marginal = _log_trapezoid(log_joint_2d, model.alpha_grid, axis=0)
    numerical = jnp.exp(log_marginal - _log_trapezoid(log_marginal, phi, axis=0))

    # Prediction: the alpha-weighted mixture of conditionals, on the same alpha
    # grid, so only the phi direction is under test.
    mle, snr = _statistics(model, model.alpha_grid)
    conditional = AmplitudeConditional(
        jnp.expand_dims(mle, 1), jnp.expand_dims(snr, 1), prior=case.prior
    )
    log_density_a = conditional.log_prob(amplitude[None, :])
    log_jacobian = case.transform.log_abs_det_jacobian(phi, amplitude)
    log_weight = log_marginalized + jnp.log(_trapezoid_weights(model.alpha_grid))
    log_mixture = jax.scipy.special.logsumexp(
        log_weight[:, None] + log_density_a + log_jacobian, axis=0
    )
    predicted = jnp.exp(log_mixture - _log_trapezoid(log_mixture, phi, axis=0))

    bulk = numerical > 1e-3 * jnp.max(numerical)
    np.testing.assert_allclose(
        predicted[bulk], numerical[bulk], rtol=1e-3, atol=1e-3 * float(numerical.max())
    )


def test_amplitude_conditional_log_prob_matches_conditioned_slice(
    model: PowerLawModel,
) -> None:
    case = model.case
    phi = case.phi_grid
    amplitude = jnp.asarray(case.transform(phi))

    pinned = handlers.condition(model.general, data={"alpha": ALPHA_TRUE})
    log_slice = LogDensityFn(pinned, chunk_size=1024)({"phi": phi})
    numerical = log_slice - _log_trapezoid(log_slice, phi, axis=0)

    mle, snr = _statistics(model, jnp.array([ALPHA_TRUE]))
    conditional = AmplitudeConditional(mle[0], snr[0], prior=case.prior)
    predicted = conditional.log_prob(amplitude) + case.transform.log_abs_det_jacobian(
        phi, amplitude
    )

    np.testing.assert_allclose(predicted, numerical, rtol=0, atol=1e-9)


def _uniform_log_normalizer(
    lower: float, upper: float, mle: float, snr: float
) -> float:
    """Closed-form ``ln Z`` for a ``Uniform(lower, upper)`` prior."""
    lo, hi = snr * (lower - mle), snr * (upper - mle)
    mass = jnp.exp(jax.scipy.special.log_ndtr(hi)) - jnp.exp(
        jax.scipy.special.log_ndtr(lo)
    )
    log_z = 0.5 * jnp.log(2 * jnp.pi) - jnp.log(snr * (upper - lower)) + jnp.log(mass)
    return float(log_z)


@pytest.mark.parametrize("snr", [1e2, 1e4])
def test_log_normalizer_is_independent_of_snr_resolution(snr: float) -> None:
    lower, upper = 1.0, 2.0
    mle = lower + 3.0 / snr  # the bound sits 3 sigma_A from the peak
    conditional = AmplitudeConditional(mle, snr, prior=dist.Uniform(lower, upper))

    expected = _uniform_log_normalizer(lower, upper, mle, snr)

    np.testing.assert_allclose(conditional.log_normalizer, expected, rtol=0, atol=1e-9)


def test_support_bounds_of_a_pushforward_are_ordered() -> None:
    prior = amplitude_prior(dist.Uniform(20.0, 140.0), amplitude_H0_transform(70.0))

    lower, upper = support_bounds(prior)

    # A = 70 / H0 is decreasing, so the base's upper bound gives the lower one.
    np.testing.assert_allclose([lower, upper], [0.5, 3.5])


def test_samples_follow_the_conditioned_slice() -> None:
    lower, upper, snr = 1.0, 2.0, 40.0
    mle = lower + 0.5 / snr  # the bound truncates the likelihood peak
    conditional = AmplitudeConditional(mle, snr, prior=dist.Uniform(lower, upper))

    draws = np.asarray(conditional.sample(jax.random.key(0), (20_000,)))

    grid = jnp.linspace(lower, lower + 12.0 / snr, 20_001)
    density = jnp.exp(conditional.log_prob(grid))
    mean = jnp.trapezoid(grid * density, grid)
    std = jnp.sqrt(jnp.trapezoid((grid - mean) ** 2 * density, grid))

    assert draws.min() >= lower
    assert draws.max() <= upper
    np.testing.assert_allclose(draws.mean(), mean, rtol=0, atol=4 * std / 20_000**0.5)
    np.testing.assert_allclose(draws.std(), std, rtol=2e-2)
