"""The callable sampling boundary, checked against independent likelihoods.

Three layers, in order: the generic model against a hand-written Gaussian
density; the population estimator against the hand-written grid-level formula;
and the amplitude machinery -- statistics, marginalization, reconstruction --
against explicit quadrature.
"""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb_mock_population import (
    FIDUCIALS,
    build_synthetic_estimator,
    make_redshift_grid,
    mock_target_model,
)
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.util import log_density
from reference_population import reference_merger_rate_distance_and_logprob

from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.distributions.amplitude import AmplitudeConditional, quadrature_grid
from astrogwb.gwb import AverageMode, spectral_density
from astrogwb.importance.estimator import SpectralDensityImportanceEstimator
from astrogwb.populations.bns_madau_dickinson import bns_md_modified_propagation
from astrogwb.sampling import (
    SpectralDensityFn,
    amplitude_reconstruction_model,
    gwb_amplitude_marginalized_model,
    gwb_spectral_density_model,
    with_renamed_diagnostics,
)

OBSERVED = jnp.array([1.4, 2.0, 3.2])
SCALE = jnp.array([0.7, 0.9, 1.2])
AMPLITUDE_PRIOR = dist.Uniform(0.2, 4.0)


def _identity(marginalized_parameter: jax.Array) -> jax.Array:
    return marginalized_parameter


def _analytic(
    params: Mapping[str, ArrayLike],
) -> tuple[jax.Array, Mapping[str, ArrayLike]]:
    shape = jnp.array([1.0, 1.5, 2.0]) + params["tilt"] * jnp.array([0.1, -0.2, 0.3])
    return jnp.asarray(params["rate"]) * shape, {}


def _generic_kwargs() -> dict[str, Any]:
    return {
        "spectral_density_fn": _analytic,
        "observed_spectral_density": OBSERVED,
        "scale": SCALE,
        "priors": {"tilt": dist.Normal(0.0, 1.0)},
        "amplitude_parameter": "rate",
        "amplitude_fiducial": 2.0,
        "amplitude_fn": _identity,
        "amplitude_prior": AMPLITUDE_PRIOR,
    }


@pytest.mark.parametrize("empty_priors", [False, True])
def test_analytic_trace_and_gaussian_density(empty_priors: bool) -> None:
    calls = []

    def spectrum(params):
        calls.append(dict(params))
        return OBSERVED + params.get("offset", 0.25), {}

    priors = {} if empty_priors else {"offset": dist.Normal(0.0, 1.0)}
    params = {} if empty_priors else {"offset": jnp.array(0.4)}
    value, trace = log_density(
        gwb_spectral_density_model,
        (),
        {
            "spectral_density_fn": spectrum,
            "observed_spectral_density": OBSERVED,
            "priors": priors,
            "scale": SCALE,
        },
        params,
    )
    assert len(calls) == 1
    assert calls[0].keys() == priors.keys()
    assert set(trace) == set(priors) | {"spectral_density_obs"}
    site = trace["spectral_density_obs"]
    assert site["is_observed"]
    assert site["fn"].event_shape == (3,)
    assert site["fn"].batch_shape == ()
    offset = params.get("offset", 0.25)
    expected = jnp.sum(dist.Normal(OBSERVED + offset, SCALE).log_prob(OBSERVED))
    if not empty_priors:
        expected += priors["offset"].log_prob(offset)
    np.testing.assert_allclose(value, expected, rtol=1e-12)


@pytest.mark.parametrize("marginalized", [False, True])
def test_arbitrary_diagnostics_are_unchanged(marginalized: bool) -> None:
    extras = {"scalar": jnp.array(7.0), "vector": jnp.arange(4.0)}
    kwargs = (
        _generic_kwargs()
        if marginalized
        else {
            "observed_spectral_density": OBSERVED,
            "scale": SCALE,
            "priors": {},
        }
    )
    kwargs["spectral_density_fn"] = lambda params: (OBSERVED, extras)
    model = (
        gwb_amplitude_marginalized_model if marginalized else gwb_spectral_density_model
    )
    trace = handlers.trace(handlers.seed(model, 0)).get_trace(**kwargs)
    for name, value in extras.items():
        assert trace[name]["type"] == "deterministic"
        np.testing.assert_array_equal(trace[name]["value"], value)


@pytest.mark.parametrize(
    ("marginalized", "name"),
    [
        (False, "tilt"),
        (False, "spectral_density_obs"),
        (True, "tilt"),
        (True, "amplitude_mle"),
        (True, "template_optimal_snr"),
        (True, "amplitude_marginalized_log_likelihood"),
    ],
)
def test_diagnostic_collisions_are_rejected(marginalized: bool, name: str) -> None:
    kwargs = _generic_kwargs()
    if not marginalized:
        for key in (
            "amplitude_parameter",
            "amplitude_fiducial",
            "amplitude_fn",
            "amplitude_prior",
        ):
            kwargs.pop(key)
    kwargs["spectral_density_fn"] = lambda params: (OBSERVED, {name: jnp.array(1.0)})
    model = (
        gwb_amplitude_marginalized_model if marginalized else gwb_spectral_density_model
    )
    with pytest.raises(AssertionError, match="unique names"):
        handlers.trace(handlers.seed(model, 0)).get_trace(**kwargs)


@pytest.mark.parametrize("explicit_grid", [False, True])
@pytest.mark.parametrize("empty_priors", [False, True])
def test_generic_marginalization_without_rate_matches_quadrature(
    explicit_grid: bool,
    empty_priors: bool,
) -> None:
    kwargs = _generic_kwargs()
    if explicit_grid:
        kwargs["amplitude_grid"] = jnp.linspace(0.2, 4.0, 4001)
    if empty_priors:
        kwargs["priors"] = {}
    calls = []

    def spectrum(params):
        calls.append(dict(params))
        return _analytic({"tilt": 0.3, **params})

    kwargs["spectral_density_fn"] = spectrum
    value, trace = log_density(
        gwb_amplitude_marginalized_model,
        (),
        kwargs,
        {} if empty_priors else {"tilt": 0.3},
    )
    assert len(calls) == 1
    assert calls[0]["rate"] == 2.0
    assert "rate" not in trace
    assert "spectral_density_obs" not in trace
    assert "template_merger_rate" not in trace
    template, _ = _analytic({"tilt": 0.3, "rate": 2.0})
    norm = jnp.sum((template / SCALE) ** 2)
    mle = jnp.sum(OBSERVED * template / SCALE**2) / norm
    np.testing.assert_allclose(trace["amplitude_mle"]["value"], mle, rtol=1e-12)
    np.testing.assert_allclose(
        trace["template_optimal_snr"]["value"], jnp.sqrt(norm), rtol=1e-12
    )
    grid = kwargs.get("amplitude_grid", quadrature_grid(AMPLITUDE_PRIOR))
    predictions = grid[:, None] / 2.0 * template
    likelihood = dist.Normal(predictions, SCALE).log_prob(OBSERVED).sum(axis=-1)
    expected = jnp.log(
        jnp.trapezoid(jnp.exp(likelihood + AMPLITUDE_PRIOR.log_prob(grid)), grid)
    )
    if not empty_priors:
        expected += kwargs["priors"]["tilt"].log_prob(0.3)
    np.testing.assert_allclose(value, expected, rtol=1e-11, atol=1e-11)
    conditional = AmplitudeConditional(
        mle,
        jnp.sqrt(norm),
        amplitude_fn=_identity,
        prior=AMPLITUDE_PRIOR,
        fiducial=2.0,
        grid=grid,
    )
    assert np.isfinite(conditional.sample(jax.random.key(1))).all()


def test_generic_rejects_sampled_amplitude_before_evaluation() -> None:
    kwargs = _generic_kwargs()
    kwargs["priors"]["rate"] = AMPLITUDE_PRIOR
    with pytest.raises(ValueError, match="cannot also be sampled"):
        handlers.seed(gwb_amplitude_marginalized_model, 0)(**kwargs)


def _importance_estimator(mode: AverageMode) -> SpectralDensityImportanceEstimator:
    """The estimator over a catalog that is its own proposal at ``FIDUCIALS``.

    The target model is wrapped so that only the *sampled* parameters arrive
    through ``params``; everything else is pinned at the fiducials, which is
    what the paper layer's conditioning handlers do.
    """
    estimator, _ = build_synthetic_estimator(
        4, polarization_power=jnp.arange(1.0, 13.0).reshape(3, 4)
    )
    target = mock_target_model()

    def pinned_call(params: Mapping[str, ArrayLike], **settings: object) -> None:
        """Take unsampled hyperparameters from the test's fixed fiducials."""
        bns_md_modified_propagation({**FIDUCIALS, **params}, **settings)

    pinned = replace(target, fn=pinned_call)
    return replace(estimator, model=pinned, average_mode=mode)


def _reference_spectrum(
    estimator: SpectralDensityImportanceEstimator, params: Mapping[str, ArrayLike]
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Rate, weights, and spectrum from the hand-written grid-level formula.

    Deliberately *not* routed through the population model: this restates the
    grid-level formula and the weight ratio in full, so it fails if the model
    path drifts rather than agreeing with it by construction. The two are
    pinned bit-identical by ``tests/core/test_distributions.py``.
    """
    redshift = estimator.source_parameters["redshift"]
    full = {**FIDUCIALS, **params}
    rate, distance, logprob = reference_merger_rate_distance_and_logprob(
        full, redshift, redshift_grid=make_redshift_grid()
    )
    log_target_distance = jnp.log(distance) + log_gw_em_ratio(
        redshift, full["xi_0"], full["xi_n"]
    )
    log_weights = (
        logprob
        - estimator.proposal_log_prob
        - 2.0 * (log_target_distance - estimator.log_reference_distance)
    )
    weights = jnp.exp(log_weights)
    factor = 0.4 if estimator.average_mode == "analytic_inclination" else 1.0
    spectrum = factor * rate * (estimator.polarization_power @ weights) / weights.size
    return rate, log_weights, spectrum


@pytest.mark.parametrize("mode", ["analytic_inclination", "catalog_inclination"])
def test_estimator_likelihood_and_gradient_match_the_grid_formula(
    mode: AverageMode,
) -> None:
    estimator = _importance_estimator(mode)
    fn: SpectralDensityFn = estimator  # Structural protocol conformance is typechecked.
    priors = {"H0": dist.Uniform(50.0, 90.0)}
    params = {"H0": jnp.array(73.0)}
    rate, log_weights, expected = _reference_spectrum(estimator, params)
    weights = jnp.exp(log_weights)
    scale = jnp.full(3, jnp.max(expected) / 3.0)
    observed = expected * 1.1
    kwargs = {
        "spectral_density_fn": fn,
        "observed_spectral_density": observed,
        "scale": scale,
        "priors": priors,
    }
    value, trace = log_density(gwb_spectral_density_model, (), kwargs, params)
    np.testing.assert_allclose(
        trace["spectral_density_obs"]["fn"].base_dist.loc, expected, rtol=1e-12
    )
    np.testing.assert_allclose(trace["total_merger_rate"]["value"], rate, rtol=1e-12)
    ess = jnp.sum(weights) ** 2 / (weights.size * jnp.sum(weights**2))
    np.testing.assert_allclose(
        trace["importance_relative_ess"]["value"], ess, rtol=1e-12
    )
    expected_density = jnp.sum(
        dist.Normal(expected, scale).log_prob(observed)
    ) + priors["H0"].log_prob(params["H0"])
    np.testing.assert_allclose(value, expected_density, rtol=1e-12)

    # The estimator is a pytree, so it can cross a jit boundary as an argument
    # and still differentiate with respect to a sampled hyperparameter.
    def density(e, h0):
        return log_density(
            gwb_spectral_density_model,
            (),
            {**kwargs, "spectral_density_fn": e},
            {"H0": h0},
        )[0]

    compiled = jax.jit(jax.value_and_grad(density, argnums=1))
    actual_value, gradient = compiled(estimator, params["H0"])
    step = 1e-4
    numerical = (
        density(estimator, params["H0"] + step)
        - density(estimator, params["H0"] - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(actual_value, value, rtol=1e-12)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5)


def test_amplitude_adapter_preserves_reconstruction_and_jit() -> None:
    """The rename adapter is what makes the estimator usable as a template."""
    estimator = _importance_estimator("catalog_inclination")
    template = with_renamed_diagnostics(
        estimator, {"total_merger_rate": "template_merger_rate"}
    )

    fiducial = FIDUCIALS["local_merger_rate"]
    prior = dist.Uniform(fiducial * 0.5, fiducial * 1.5)
    observed, _ = estimator({"H0": 70.0})
    scale = jnp.full(3, jnp.max(observed) / 3)
    kwargs = {
        "observed_spectral_density": observed,
        "priors": {"H0": dist.Uniform(50.0, 90.0)},
        "amplitude_parameter": "local_merger_rate",
        "amplitude_fn": _identity,
        "amplitude_prior": prior,
        "spectral_density_fn": template,
        "scale": scale,
        "amplitude_fiducial": fiducial,
    }
    value, trace = log_density(
        gwb_amplitude_marginalized_model, (), kwargs, {"H0": 73.0}
    )
    # The rate reaches the trace under the template name and only that name;
    # every other diagnostic passes through untouched.
    assert "total_merger_rate" not in trace
    _, extras = estimator({"H0": 73.0, "local_merger_rate": fiducial})
    np.testing.assert_allclose(
        trace["template_merger_rate"]["value"],
        extras["total_merger_rate"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        trace["importance_relative_ess"]["value"],
        extras["importance_relative_ess"],
        rtol=1e-12,
    )

    def density(h0):
        return log_density(gwb_amplitude_marginalized_model, (), kwargs, {"H0": h0})[0]

    actual, gradient = jax.jit(jax.value_and_grad(density))(73.0)
    np.testing.assert_allclose(actual, value, rtol=1e-12)
    assert np.isfinite(gradient)

    reconstruction = Predictive(
        partial(
            amplitude_reconstruction_model,
            amplitude_parameter="local_merger_rate",
            amplitude_fn=_identity,
            merger_rate_amplitude_fn=_identity,
            prior=prior,
            fiducial=fiducial,
        ),
        num_samples=4,
    )
    draws = reconstruction(
        jax.random.key(3),
        **{
            name: trace[name]["value"]
            for name in (
                "amplitude_mle",
                "template_optimal_snr",
                "template_merger_rate",
            )
        },
    )
    # The reconstructed physical rate is the template rate scaled by g_R; with
    # the identity scaling that is exactly the ratio to the fiducial.
    np.testing.assert_allclose(
        draws["total_merger_rate"],
        trace["template_merger_rate"]["value"] * draws["local_merger_rate"] / fiducial,
        rtol=1e-12,
    )
    assert np.isfinite(np.asarray(draws["quadrature_effective_nodes"])).all()


def test_renaming_a_missing_or_colliding_diagnostic_is_rejected() -> None:
    """A silent no-op would publish a template rate as the physical one."""
    estimator = _importance_estimator("catalog_inclination")

    missing = with_renamed_diagnostics(estimator, {"absent": "renamed"})
    with pytest.raises(ValueError, match="missing"):
        missing({"H0": 70.0})

    colliding = with_renamed_diagnostics(
        estimator, {"total_merger_rate": "importance_relative_ess"}
    )
    with pytest.raises(ValueError, match="collide"):
        colliding({"H0": 70.0})


@pytest.mark.integration
@pytest.mark.parametrize("marginalized", [False, True])
def test_generic_nuts_with_analytic_spectrum(marginalized: bool) -> None:
    kwargs = _generic_kwargs()
    if marginalized:
        model = gwb_amplitude_marginalized_model
    else:
        model = gwb_spectral_density_model
        for key in (
            "amplitude_parameter",
            "amplitude_fiducial",
            "amplitude_fn",
            "amplitude_prior",
        ):
            kwargs.pop(key)
        kwargs["priors"]["rate"] = AMPLITUDE_PRIOR
    mcmc = MCMC(
        NUTS(partial(model, **kwargs)),
        num_warmup=30,
        num_samples=20,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.key(42))
    posterior = mcmc.get_samples()
    assert posterior["tilt"].shape == (20,)
    assert all(np.isfinite(values).all() for values in posterior.values())
    assert ("rate" in posterior) is not marginalized
    if marginalized:
        assert {"amplitude_mle", "template_optimal_snr"} <= posterior.keys()


# --------------------------------------------------------------------------- #
# The amplitude machinery, on a catalog-free spectrum
#
# Folded here from the retired legacy-model suite: these test the likelihood
# mathematics -- the sufficient statistics, the marginalization integral, and
# the reconstruction that inverts it -- rather than any particular way of
# producing a spectrum. The spectrum below is a real catalog contraction only
# because it must be strictly linear in the marginalized parameter; nothing
# here reweights a population.
# --------------------------------------------------------------------------- #
FIDUCIAL_RATE = 2.0
"""Reference value of the marginalized parameter; the amplitude is its ratio."""

NOISE_SCALE = jnp.array([1.0, 1.4, 0.8, 1.2])
_POWER = jnp.array([[1.0, 2.0], [3.0, 1.5], [2.0, 4.0], [1.0, 1.0]])
_SENTINEL = jnp.array([0.2, -0.4])
_MARGINALIZED_OBSERVED = jnp.array([2.4, 4.1, 5.9, 1.8])

# The marginalized parameter is `local_merger_rate` itself, so the prior and
# the grid live on rates. The fiducial rate is 2.0, so this covers amplitudes
# A = rate / 2 in [0.1, 2.5].
_RATE_PRIOR = dist.Uniform(0.2, 5.0)
_RATE_GRID = jnp.linspace(0.2, 5.0, 2001)


def _linear_spectrum(
    params: Mapping[str, ArrayLike],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """A spectrum strictly linear in ``local_merger_rate``, as the real one is.

    ``tilt`` only reshapes the per-source weights, so it is a genuine shape
    parameter that the amplitude cannot absorb.
    """
    rate = jnp.asarray(params["local_merger_rate"])
    weights = jnp.exp(jnp.asarray(params["tilt"]) * _SENTINEL)
    prediction = spectral_density(
        _POWER, weights, rate, average_mode="catalog_inclination"
    )
    return prediction, {"total_merger_rate": rate}


#: The same spectrum seen as a *template*: at the pinned fiducial amplitude the
#: rate is not the physical one, so it must not reach the trace under that name.
_TEMPLATE_SPECTRUM = with_renamed_diagnostics(
    _linear_spectrum, {"total_merger_rate": "template_merger_rate"}
)

_MARGINALIZED_KWARGS: dict[str, Any] = {
    "spectral_density_fn": _TEMPLATE_SPECTRUM,
    "observed_spectral_density": _MARGINALIZED_OBSERVED,
    "scale": NOISE_SCALE,
    "amplitude_parameter": "local_merger_rate",
    "amplitude_fiducial": FIDUCIAL_RATE,
    "amplitude_fn": _identity,
    "amplitude_prior": _RATE_PRIOR,
    "amplitude_grid": _RATE_GRID,
}

_RECONSTRUCTION_KWARGS: dict[str, Any] = {
    "amplitude_fn": _identity,
    "merger_rate_amplitude_fn": _identity,
    "prior": _RATE_PRIOR,
    "fiducial": FIDUCIAL_RATE,
    "grid": _RATE_GRID,
}
"""The same marginalized direction, spelled the way reconstruction takes it."""


def _condition_without_density(model, fixed_params):
    """Supply fixed sample values without exposing their prior sites."""
    return handlers.block(
        handlers.condition(model, data=fixed_params), hide=list(fixed_params)
    )


def test_amplitude_statistics_match_explicit_sigma_weighted_sums() -> None:
    """The published statistics are the sigma-space sums, not PSD-space ones.

    Also pins, from the same trace, that the amplitude direction reaches the
    potential as a ``numpyro.factor`` and never as an observed likelihood: a
    ``spectral_density_obs`` site here would double-count the data against the
    already-marginalized amplitude.
    """
    tilt = jnp.array(0.3)
    model = _condition_without_density(gwb_amplitude_marginalized_model, {"tilt": tilt})
    trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(
        **_MARGINALIZED_KWARGS, priors={"tilt": dist.Normal(0.0, 1.0)}
    )
    template, _ = _linear_spectrum({"local_merger_rate": FIDUCIAL_RATE, "tilt": tilt})
    template_norm = jnp.sum((template / NOISE_SCALE) ** 2)
    data_template = jnp.sum(_MARGINALIZED_OBSERVED * template / NOISE_SCALE**2)

    np.testing.assert_allclose(
        np.asarray(trace["amplitude_mle"]["value"]),
        np.asarray(data_template / template_norm),
    )
    np.testing.assert_allclose(
        np.asarray(trace["template_optimal_snr"]["value"]),
        np.asarray(jnp.sqrt(template_norm)),
    )
    # The published rate is the template's, at the pinned fiducial amplitude.
    np.testing.assert_allclose(
        np.asarray(trace["template_merger_rate"]["value"]), FIDUCIAL_RATE
    )
    factor_site = trace["amplitude_marginalized_log_likelihood"]
    assert isinstance(factor_site["fn"], dist.Unit)
    assert np.isfinite(float(factor_site["fn"].log_factor))
    assert "spectral_density_obs" not in trace


def _grid_for(prior: dist.Uniform | dist.Normal, num: int = 40_001) -> jax.Array:
    if isinstance(prior, dist.Uniform):
        return jnp.linspace(float(prior.low), float(prior.high), num)
    loc, scale = float(prior.loc), float(prior.scale)
    return jnp.linspace(loc - 40.0 * scale, loc + 40.0 * scale, num)


@pytest.mark.parametrize(
    "rate_prior",
    [dist.Uniform(1.0, 3.0), dist.Normal(2.0, 0.8)],
    ids=["uniform", "normal"],
)
def test_amplitude_marginalized_model_matches_the_general_model(
    rate_prior: dist.Uniform | dist.Normal,
) -> None:
    """Numerically marginalize the general model and compare the log densities.

    Both models are given the *same* prior on the same variable, the rate --
    the marginalized model states it on the marginalized parameter, the
    general model on its sample site. (Equivalently: ``rate_prior`` is the
    pushforward under ``rate = FIDUCIAL_RATE * amplitude`` of a
    ``Uniform(0.5, 1.5)`` / ``Normal(1.0, 0.4)`` prior on the amplitude.)
    Integrating the general model over ``rate`` rather than ``amplitude``
    cancels the Jacobian exactly, so the two log densities must agree without
    any leftover constant -- which is what makes this a joint check on the
    factor term, the normalizations, and the reference injection.
    """
    tilt = 0.35
    general_kwargs = {
        "spectral_density_fn": _linear_spectrum,
        "observed_spectral_density": _MARGINALIZED_OBSERVED,
        "scale": NOISE_SCALE,
        "priors": {"local_merger_rate": rate_prior, "tilt": dist.Normal(0.0, 1.0)},
    }

    def general_log_density(rate: float) -> float:
        value, _ = log_density(
            gwb_spectral_density_model,
            (),
            general_kwargs,
            {"local_merger_rate": jnp.asarray(rate), "tilt": jnp.asarray(tilt)},
        )
        return float(value)

    grid = _grid_for(rate_prior, num=20_001)
    rates = np.asarray(grid)
    densities = np.exp([general_log_density(rate) for rate in rates])
    numerical = float(np.log(np.trapezoid(densities, rates)))

    marginalized, _ = log_density(
        gwb_amplitude_marginalized_model,
        (),
        {
            **_MARGINALIZED_KWARGS,
            "amplitude_prior": rate_prior,
            "amplitude_grid": _grid_for(rate_prior),
            "priors": {"tilt": dist.Normal(0.0, 1.0)},
        },
        {"tilt": jnp.asarray(tilt)},
    )

    np.testing.assert_allclose(float(marginalized), numerical, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- #
# Amplitude reconstruction model
# --------------------------------------------------------------------------- #
def _reconstruction_statistics() -> dict[str, jax.Array]:
    """Statistics shaped ``(chain, draw)``, as ``mcmc.get_samples(group_by_chain=True)``."""
    return {
        "amplitude_mle": jnp.array([[0.9, 1.0, 1.1], [1.2, 0.8, 1.0]]),
        "template_optimal_snr": jnp.array([[25.0, 30.0, 35.0], [40.0, 45.0, 50.0]]),
        "template_merger_rate": jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    }


def _reconstruction_draws(statistics: dict[str, jax.Array]) -> dict[str, jax.Array]:
    draws = Predictive(
        partial(
            amplitude_reconstruction_model,
            amplitude_parameter="local_merger_rate",
            **_RECONSTRUCTION_KWARGS,
        ),
        num_samples=1,
        return_sites=[
            "local_merger_rate",
            "total_merger_rate",
            "quadrature_effective_nodes",
        ],
    )(jax.random.key(0), **statistics)
    return {name: values[0] for name, values in draws.items()}


def test_amplitude_reconstruction_model_returns_chain_draw_sites() -> None:
    draws = _reconstruction_draws(_reconstruction_statistics())

    for name in (
        "local_merger_rate",
        "total_merger_rate",
        "quadrature_effective_nodes",
    ):
        assert draws[name].shape == (2, 3), name
        assert bool(jnp.all(jnp.isfinite(draws[name]))), name

    phi = draws["local_merger_rate"]
    assert bool(jnp.all(phi >= float(_RATE_GRID[0])))
    assert bool(jnp.all(phi <= float(_RATE_GRID[-1])))


def test_amplitude_reconstruction_model_computes_deterministics_from_inputs() -> None:
    statistics = _reconstruction_statistics()
    draws = _reconstruction_draws(statistics)

    # total_merger_rate must be template_merger_rate * g_R(phi) with the
    # input template_merger_rate, elementwise.
    expected_rate = (
        np.asarray(statistics["template_merger_rate"])
        * np.asarray(draws["local_merger_rate"])
        / FIDUCIAL_RATE
    )
    np.testing.assert_allclose(
        np.asarray(draws["total_merger_rate"]), expected_rate, rtol=1e-6
    )

    expected_nodes = np.asarray(
        AmplitudeConditional(
            statistics["amplitude_mle"],
            statistics["template_optimal_snr"],
            amplitude_fn=_identity,
            prior=_RATE_PRIOR,
            fiducial=FIDUCIAL_RATE,
            grid=_RATE_GRID,
        ).effective_nodes
    )
    np.testing.assert_allclose(
        np.asarray(draws["quadrature_effective_nodes"]), expected_nodes, rtol=1e-6
    )


def test_amplitude_reconstruction_model_registers_only_generated_sites() -> None:
    trace = handlers.trace(
        handlers.seed(
            partial(
                amplitude_reconstruction_model,
                amplitude_parameter="local_merger_rate",
                **_RECONSTRUCTION_KWARGS,
            ),
            rng_seed=0,
        )
    ).get_trace(**_reconstruction_statistics())

    for statistic in _reconstruction_statistics():
        assert statistic not in trace
    assert trace["local_merger_rate"]["type"] == "sample"
    assert trace["total_merger_rate"]["type"] == "deterministic"
    assert trace["quadrature_effective_nodes"]["type"] == "deterministic"


def test_amplitude_reconstruction_model_raises_on_a_missing_statistic() -> None:
    statistics = _reconstruction_statistics()
    del statistics["template_optimal_snr"]

    with pytest.raises(TypeError, match="template_optimal_snr"):
        _reconstruction_draws(statistics)
