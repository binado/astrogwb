"""The callable sampling boundary, checked against independent likelihoods."""

from collections.abc import Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb_mock_population import FIDUCIALS, make_redshift_grid
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.util import log_density

from astrogwb.catalog import ImportanceCatalog
from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.distributions.amplitude import AmplitudeConditional, quadrature_grid
from astrogwb.gwb import AverageMode
from astrogwb.importance.estimator import SpectralDensityImportanceEstimator
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    bns_population,
    make_merger_rate_and_log_weights_fn,
)
from astrogwb.sampling import (
    SpectralDensityFn,
    amplitude_marginalized_model,
    amplitude_reconstruction_model,
    gwb_amplitude_marginalized_model,
    gwb_spectral_density_model,
    spectral_density_model,
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


def _importance_inputs(mode: AverageMode):
    grid = make_redshift_grid()
    factory = partial(bns_population, redshift_grid=grid)
    population = factory(FIDUCIALS)
    redshift = jnp.array([0.4, 1.2, 3.7, 7.1])
    samples = {
        "redshift": redshift,
        "luminosity_distance": population.luminosity_distance(redshift),
    }
    power = jnp.arange(1.0, 13.0).reshape(3, 4)
    catalog = ImportanceCatalog.from_population(
        population=population,
        source_parameters=samples,
        polarization_power=power,
        luminosity_distance=samples["luminosity_distance"],
    )

    # The concrete factory binds fixed parameters once; the sampled tilt/rate
    # equivalents here remain dynamic inputs to the population.
    def target(params):
        return factory({**FIDUCIALS, **params})

    estimator = SpectralDensityImportanceEstimator(catalog, target, mode)
    callback = make_merger_rate_and_log_weights_fn(
        fiducials=FIDUCIALS,
        redshift_grid=grid,
        proposal_logprob=catalog.proposal_log_prob,
    )

    def legacy(params, samples):
        return callback({**FIDUCIALS, **params}, samples)

    return estimator, legacy


@pytest.mark.parametrize("mode", ["analytic_inclination", "catalog_inclination"])
def test_estimator_likelihood_and_gradient_match_legacy(mode: AverageMode) -> None:
    estimator, callback = _importance_inputs(mode)
    fn: SpectralDensityFn = estimator  # Structural protocol conformance is typechecked.
    priors = {"H0": dist.Uniform(50.0, 90.0)}
    params = {"H0": jnp.array(73.0)}
    rate, log_weights = callback(params, estimator.catalog.source_parameters)
    weights = jnp.exp(log_weights)
    factor = 0.4 if mode == "analytic_inclination" else 1.0
    expected = (
        factor * rate * (estimator.catalog.polarization_power @ weights) / weights.size
    )
    scale = jnp.full(3, jnp.max(expected) / 3.0)
    observed = expected * 1.1
    generic = {
        "spectral_density_fn": fn,
        "observed_spectral_density": observed,
        "scale": scale,
        "priors": priors,
    }
    legacy = {
        "polarization_power": estimator.catalog.polarization_power,
        "samples": estimator.catalog.source_parameters,
        "observed_spectral_density": observed,
        "effective_psd": scale,
        "observation_time": 1.0 / SECONDS_PER_YEAR,
        "df": 0.5,
        "average_mode": mode,
        "merger_rate_and_log_weights_fn": callback,
        "priors": priors,
    }
    value, trace = log_density(gwb_spectral_density_model, (), generic, params)
    old_value, old_trace = log_density(spectral_density_model, (), legacy, params)
    np.testing.assert_allclose(
        trace["spectral_density_obs"]["fn"].base_dist.loc, expected, rtol=1e-12
    )
    np.testing.assert_allclose(trace["total_merger_rate"]["value"], rate, rtol=1e-12)
    ess = jnp.sum(weights) ** 2 / (weights.size * jnp.sum(weights**2))
    np.testing.assert_allclose(
        trace["importance_relative_ess"]["value"], ess, rtol=1e-12
    )
    np.testing.assert_allclose(value, old_value, rtol=1e-12)
    assert set(trace) == set(old_trace)

    def density(e, h0):
        return log_density(
            gwb_spectral_density_model,
            (),
            {**generic, "spectral_density_fn": e},
            {"H0": h0},
        )[0]

    compiled = jax.jit(jax.value_and_grad(density, argnums=1))
    actual_value, gradient = compiled(estimator, params["H0"])
    old_gradient = jax.grad(
        lambda h0: log_density(spectral_density_model, (), legacy, {"H0": h0})[0]
    )(params["H0"])
    np.testing.assert_allclose(actual_value, value, rtol=1e-12)
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, old_gradient, rtol=1e-10)


def test_amplitude_adapter_preserves_reconstruction_and_jit() -> None:
    estimator, callback = _importance_inputs("catalog_inclination")

    def template(params):
        prediction, extras = estimator(params)
        extras = dict(extras)
        extras["template_merger_rate"] = extras.pop("total_merger_rate")
        return prediction, extras

    fiducial = FIDUCIALS["local_merger_rate"]
    prior = dist.Uniform(fiducial * 0.5, fiducial * 1.5)
    observed, _ = estimator({"H0": 70.0})
    scale = jnp.full(3, jnp.max(observed) / 3)
    shared = {
        "observed_spectral_density": observed,
        "priors": {"H0": dist.Uniform(50.0, 90.0)},
        "amplitude_parameter": "local_merger_rate",
        "amplitude_fn": _identity,
        "amplitude_prior": prior,
    }
    generic = dict(
        **shared, spectral_density_fn=template, scale=scale, amplitude_fiducial=fiducial
    )
    legacy = dict(
        **shared,
        polarization_power=estimator.catalog.polarization_power,
        samples=estimator.catalog.source_parameters,
        effective_psd=scale,
        observation_time=1.0 / SECONDS_PER_YEAR,
        df=0.5,
        average_mode="catalog_inclination",
        merger_rate_and_log_weights_fn=callback,
        fiducials=FIDUCIALS,
    )
    value, trace = log_density(
        gwb_amplitude_marginalized_model, (), generic, {"H0": 73.0}
    )
    old_value, old_trace = log_density(
        amplitude_marginalized_model, (), legacy, {"H0": 73.0}
    )
    np.testing.assert_allclose(value, old_value, rtol=1e-12)
    assert set(trace) == set(old_trace)
    assert "total_merger_rate" not in trace
    for name in (
        "amplitude_mle",
        "template_optimal_snr",
        "template_merger_rate",
        "importance_relative_ess",
    ):
        np.testing.assert_allclose(
            trace[name]["value"], old_trace[name]["value"], rtol=1e-12
        )

    def density(h0):
        return log_density(gwb_amplitude_marginalized_model, (), generic, {"H0": h0})[0]

    actual, gradient = jax.jit(jax.value_and_grad(density))(73.0)
    old_gradient = jax.grad(
        lambda h0: log_density(amplitude_marginalized_model, (), legacy, {"H0": h0})[0]
    )(73.0)
    np.testing.assert_allclose(actual, value, rtol=1e-12)
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, old_gradient, rtol=1e-10)
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

    def reconstruct(sites):
        return reconstruction(
            jax.random.key(3),
            **{
                name: sites[name]["value"]
                for name in (
                    "amplitude_mle",
                    "template_optimal_snr",
                    "template_merger_rate",
                )
            },
        )

    draws, old_draws = reconstruct(trace), reconstruct(old_trace)
    for name in draws:
        np.testing.assert_allclose(draws[name], old_draws[name], rtol=1e-12)


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
