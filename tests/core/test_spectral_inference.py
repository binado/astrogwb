"""The callable inference boundary, checked against independent likelihoods.

Three layers, in order: the generic model against a hand-written Gaussian
density; the importance-sampled spectrum against the hand-written grid-level
formula;
and the amplitude machinery -- statistics and marginalization --
against explicit quadrature.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb_mock_population import (
    FIDUCIALS,
    build_synthetic_importance,
    make_redshift_grid,
    mock_merger_rate_fn,
    mock_target_model,
)
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density
from reference_population import reference_merger_rate_distance_and_logprob

from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.distributions.amplitude import (
    AmplitudeConditional,
    amplitude_prior,
    quadrature_grid,
)
from astrogwb.gwb import spectral_density
from astrogwb.importance.spectral import importance_spectral_density
from astrogwb.inference import (
    SpectralDensityFn,
    amplitude_local_merger_rate_transform,
    gwb_amplitude_marginalized_model,
    gwb_spectral_density_model,
)

OBSERVED = jnp.array([1.4, 2.0, 3.2])
SCALE = jnp.array([0.7, 0.9, 1.2])
AMPLITUDE_PRIOR = dist.Uniform(0.2, 4.0)
"""A prior on the dimensionless amplitude itself."""
TEMPLATE_RATE = 2.0
"""Value of ``rate`` at which the analytic spectrum is the ``A = 1`` template."""


def _analytic(
    params: Mapping[str, ArrayLike],
) -> tuple[jax.Array, Mapping[str, ArrayLike]]:
    shape = jnp.array([1.0, 1.5, 2.0]) + params["tilt"] * jnp.array([0.1, -0.2, 0.3])
    return jnp.asarray(params["rate"]) * shape, {}


def _analytic_template(
    params: Mapping[str, ArrayLike],
) -> tuple[jax.Array, Mapping[str, ArrayLike]]:
    """The analytic spectrum with ``rate`` pinned: the ``A = 1`` template."""
    return _analytic({**params, "rate": TEMPLATE_RATE})


def _generic_kwargs() -> dict[str, Any]:
    return {
        "spectral_density_fn": _analytic_template,
        "observed_spectral_density": OBSERVED,
        "scale": SCALE,
        "priors": {"tilt": dist.Normal(0.0, 1.0)},
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
        kwargs.pop("amplitude_prior")
    kwargs["spectral_density_fn"] = lambda params: (OBSERVED, {name: jnp.array(1.0)})
    model = (
        gwb_amplitude_marginalized_model if marginalized else gwb_spectral_density_model
    )
    with pytest.raises(AssertionError, match="unique names"):
        handlers.trace(handlers.seed(model, 0)).get_trace(**kwargs)


def test_template_rate_is_published_under_its_unit_amplitude_name() -> None:
    kwargs = _generic_kwargs()
    kwargs["spectral_density_fn"] = lambda params: (
        OBSERVED,
        {"total_merger_rate": jnp.array(5.0)},
    )
    trace = handlers.trace(
        handlers.seed(gwb_amplitude_marginalized_model, 0)
    ).get_trace(**kwargs)

    assert "total_merger_rate" not in trace
    assert trace["total_merger_rate_at_unit_amplitude"]["type"] == "deterministic"
    np.testing.assert_allclose(
        trace["total_merger_rate_at_unit_amplitude"]["value"], 5.0
    )


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
        return _analytic({"tilt": 0.3, "rate": TEMPLATE_RATE, **params})

    kwargs["spectral_density_fn"] = spectrum
    value, trace = log_density(
        gwb_amplitude_marginalized_model,
        (),
        kwargs,
        {} if empty_priors else {"tilt": 0.3},
    )
    assert len(calls) == 1
    assert "rate" not in calls[0]
    assert "rate" not in trace
    assert "spectral_density_obs" not in trace
    assert "total_merger_rate_at_unit_amplitude" not in trace
    template, _ = _analytic({"tilt": 0.3, "rate": TEMPLATE_RATE})
    norm = jnp.sum((template / SCALE) ** 2)
    mle = jnp.sum(OBSERVED * template / SCALE**2) / norm
    np.testing.assert_allclose(trace["amplitude_mle"]["value"], mle, rtol=1e-12)
    np.testing.assert_allclose(
        trace["template_optimal_snr"]["value"], jnp.sqrt(norm), rtol=1e-12
    )
    grid = kwargs.get("amplitude_grid", quadrature_grid(AMPLITUDE_PRIOR))
    predictions = grid[:, None] * template
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
        prior=AMPLITUDE_PRIOR,
        grid=grid,
    )
    assert np.isfinite(conditional.sample(jax.random.key(1))).all()


_TARGET_SOURCE = mock_target_model()
_TARGET_RATE = mock_merger_rate_fn()


def pinned_target(params: Mapping[str, ArrayLike]) -> Mapping[str, jax.Array]:
    """The target source model, unsampled hyperparameters pinned at the fiducials."""
    return _TARGET_SOURCE({**FIDUCIALS, **params})


def pinned_rate(params: Mapping[str, ArrayLike]) -> jax.Array:
    """The rate needs the same pinning: it takes ``params`` independently."""
    return _TARGET_RATE({**FIDUCIALS, **params})


def _importance() -> dict[str, Any]:
    """Spectrum keywords over a catalog that is its own proposal at ``FIDUCIALS``.

    The target is pinned so that only the *sampled* parameters arrive through
    ``params``; everything else comes from the fiducials, which is what the
    paper layer's conditioning handlers do.
    """
    importance, _ = build_synthetic_importance(
        4, polarization_power=jnp.arange(1.0, 13.0).reshape(3, 4)
    )
    return {
        **importance,
        "source_model": pinned_target,
        "merger_rate_fn": pinned_rate,
    }


def _importance_estimator() -> SpectralDensityFn:
    """The importance spectrum over :func:`_importance`, as a bound callable."""
    return partial(importance_spectral_density, **_importance())


def _reference_spectrum(
    importance: Mapping[str, Any], params: Mapping[str, ArrayLike]
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Rate, weights, and spectrum from the hand-written grid-level formula.

    Deliberately *not* routed through the source model: this restates the
    grid-level formula and the weight ratio in full, so it fails if the model
    path drifts rather than agreeing with it by construction. The two are
    pinned bit-identical by ``tests/core/test_distributions.py``.
    """
    sources = importance["source_parameters"]
    redshift = sources["redshift"]
    full = {**FIDUCIALS, **params}
    rate, distance, logprob = reference_merger_rate_distance_and_logprob(
        full,
        redshift,
        redshift_grid=make_redshift_grid(),
        source_frame_mass_1=sources["source_frame_mass_1"],
        source_frame_mass_2=sources["source_frame_mass_2"],
    )
    log_target_distance = jnp.log(distance) + log_gw_em_ratio(
        redshift, full["xi_0"], full["xi_n"]
    )
    log_weights = (
        logprob
        - importance["proposal_log_prob"]
        - 2.0 * (log_target_distance - importance["log_reference_distance"])
    )
    weights = jnp.exp(log_weights)
    power = importance["polarization_power"]
    spectrum = 0.4 * rate * (power @ weights) / weights.size
    return rate, log_weights, spectrum


def test_importance_likelihood_and_gradient_match_the_grid_formula() -> None:
    importance = _importance()
    fn: SpectralDensityFn = partial(importance_spectral_density, **importance)
    priors = {"H0": dist.Uniform(50.0, 90.0)}
    params = {"H0": jnp.array(73.0)}
    rate, log_weights, expected = _reference_spectrum(importance, params)
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

    # The bound spectrum closes over the catalog arrays, so it compiles as a
    # constant and still differentiates with respect to a sampled hyperparameter.
    def density(h0):
        return log_density(gwb_spectral_density_model, (), kwargs, {"H0": h0})[0]

    compiled = jax.jit(jax.value_and_grad(density))
    actual_value, gradient = compiled(params["H0"])
    step = 1e-4
    numerical = (density(params["H0"] + step) - density(params["H0"] - step)) / (
        2.0 * step
    )
    np.testing.assert_allclose(actual_value, value, rtol=1e-12)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5)


def test_amplitude_marginalized_importance_spectrum_publishes_template_rate_and_jits() -> (
    None
):
    """The importance spectrum is the template, and its rate says so."""
    estimator = _importance_estimator()

    fiducial = FIDUCIALS["local_merger_rate"]
    transform = amplitude_local_merger_rate_transform(fiducial)
    prior = amplitude_prior(dist.Uniform(fiducial * 0.5, fiducial * 1.5), transform)
    observed, _ = estimator({"H0": 70.0})
    scale = jnp.full(3, jnp.max(observed) / 3)
    kwargs = {
        "observed_spectral_density": observed,
        "priors": {"H0": dist.Uniform(50.0, 90.0)},
        "amplitude_prior": prior,
        "spectral_density_fn": estimator,
        "scale": scale,
    }
    value, trace = log_density(
        gwb_amplitude_marginalized_model, (), kwargs, {"H0": 73.0}
    )
    # The rate reaches the trace under the template name and only that name;
    # every other diagnostic passes through untouched.
    assert "total_merger_rate" not in trace
    _, extras = estimator({"H0": 73.0})
    np.testing.assert_allclose(
        trace["total_merger_rate_at_unit_amplitude"]["value"],
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


@pytest.mark.integration
@pytest.mark.parametrize("marginalized", [False, True])
def test_generic_nuts_with_analytic_spectrum(marginalized: bool) -> None:
    kwargs = _generic_kwargs()
    if marginalized:
        model = gwb_amplitude_marginalized_model
    else:
        model = gwb_spectral_density_model
        kwargs.pop("amplitude_prior")
        kwargs["spectral_density_fn"] = _analytic
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
# rather than any particular way of
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

# The amplitude stands for `local_merger_rate`, so the physical prior lives on
# rates and the A-space prior is its pushforward under A = rate / FIDUCIAL_RATE.
# The fiducial rate is 2.0, so this covers amplitudes in [0.1, 2.5].
_RATE_PRIOR = dist.Uniform(0.2, 5.0)
_RATE_TRANSFORM = amplitude_local_merger_rate_transform(FIDUCIAL_RATE)
_AMPLITUDE_PRIOR = amplitude_prior(_RATE_PRIOR, _RATE_TRANSFORM)
_AMPLITUDE_GRID = quadrature_grid(_AMPLITUDE_PRIOR, num_nodes=2001)


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
        _POWER, weights, rate, source_parameters={"inclination": _SENTINEL}
    )
    return prediction, {"total_merger_rate": rate}


def _pinned(spectrum: SpectralDensityFn, **fixed: float) -> SpectralDensityFn:
    """``spectrum`` with ``fixed`` parameters supplied, i.e. its template."""

    def pinned(
        params: Mapping[str, ArrayLike],
    ) -> tuple[jax.Array, Mapping[str, ArrayLike]]:
        return spectrum({**params, **fixed})

    return pinned


_MARGINALIZED_KWARGS: dict[str, Any] = {
    "spectral_density_fn": _pinned(_linear_spectrum, local_merger_rate=FIDUCIAL_RATE),
    "observed_spectral_density": _MARGINALIZED_OBSERVED,
    "scale": NOISE_SCALE,
    "amplitude_prior": _AMPLITUDE_PRIOR,
    "amplitude_grid": _AMPLITUDE_GRID,
}


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
        np.asarray(trace["total_merger_rate_at_unit_amplitude"]["value"]),
        FIDUCIAL_RATE,
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
    the marginalized model through its A-space pushforward, the general model
    on its sample site. Integrating the general model over ``rate`` while the
    marginalized one integrates over ``amplitude`` under the pushforward prior
    must agree without any leftover constant -- the Jacobian is carried by the
    prior -- which is what makes this a joint check on the factor term, the
    normalizations, the pushforward, and the reference injection.
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
            "amplitude_prior": amplitude_prior(rate_prior, _RATE_TRANSFORM),
            "amplitude_grid": _RATE_TRANSFORM(_grid_for(rate_prior)),
            "priors": {"tilt": dist.Normal(0.0, 1.0)},
        },
        {"tilt": jnp.asarray(tilt)},
    )

    np.testing.assert_allclose(float(marginalized), numerical, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- #
# Frequency masking
#
# The analysis band reaches a likelihood as a traced boolean array instead of
# as arrays compressed to the band. The two must agree exactly -- that is what
# makes the mask a reformulation rather than a second likelihood -- and the
# traced form must let the band vary without a retrace, which is the whole
# reason for it. Both properties are checked here, for both likelihoods.
# --------------------------------------------------------------------------- #

#: A five-bin grid and a *non-contiguous* selection of it: a gappy band is what
#: a detector network with a hole in its coverage actually produces.
_BAND_MASK = jnp.array([False, True, True, False, True])
_BAND_OBSERVED = jnp.array([2.4, 4.1, 5.9, 1.8, 3.3])
_BAND_SCALE = jnp.array([1.0, 1.4, 0.8, 1.2, 0.9])
_BAND_POWER = jnp.array(
    [
        [0.5, 1.1, 0.3],
        [1.7, 0.4, 2.2],
        [0.9, 1.3, 0.6],
        [2.1, 0.8, 1.5],
        [0.7, 1.9, 1.0],
    ]
)
_BAND_TILT = jnp.array([0.1, -0.2, 0.3])
_BAND_FIDUCIAL_RATE = 2.0
_BAND_RATE_PRIOR = dist.Uniform(0.5, 6.0)
_BAND_AMPLITUDE_PRIOR = amplitude_prior(
    _BAND_RATE_PRIOR, amplitude_local_merger_rate_transform(_BAND_FIDUCIAL_RATE)
)


def _band_spectrum(power: jax.Array) -> SpectralDensityFn:
    """A spectrum bound to ``power``, so a compressed twin is one call away."""

    def spectrum(
        params: Mapping[str, ArrayLike],
    ) -> tuple[jax.Array, Mapping[str, ArrayLike]]:
        rate = jnp.asarray(params["rate"])
        weights = jnp.exp(jnp.asarray(params["tilt"]) * _BAND_TILT)
        return rate * (power @ weights), {"total_merger_rate": rate}

    return spectrum


def _band_kwargs(marginalized: bool, *, compressed: bool) -> dict[str, Any]:
    """The same likelihood twice: masked on the full grid, or compressed to it."""
    selection = np.asarray(_BAND_MASK)
    power = _BAND_POWER[selection, :] if compressed else _BAND_POWER
    spectrum = _band_spectrum(power)
    kwargs: dict[str, Any] = {
        "observed_spectral_density": (
            _BAND_OBSERVED[selection] if compressed else _BAND_OBSERVED
        ),
        "scale": _BAND_SCALE[selection] if compressed else _BAND_SCALE,
    }
    if not compressed:
        kwargs["frequency_mask"] = _BAND_MASK
    if marginalized:
        kwargs |= {
            "spectral_density_fn": _pinned(spectrum, rate=_BAND_FIDUCIAL_RATE),
            "priors": {"tilt": dist.Normal(0.0, 1.0)},
            "amplitude_prior": _BAND_AMPLITUDE_PRIOR,
            "amplitude_grid": quadrature_grid(_BAND_AMPLITUDE_PRIOR, num_nodes=2001),
        }
    else:
        kwargs |= {
            "spectral_density_fn": spectrum,
            "priors": {"rate": _BAND_RATE_PRIOR, "tilt": dist.Normal(0.0, 1.0)},
        }
    return kwargs


def _band_model(marginalized: bool) -> Any:
    return (
        gwb_amplitude_marginalized_model if marginalized else gwb_spectral_density_model
    )


def _band_params(marginalized: bool) -> dict[str, jax.Array]:
    params = {"tilt": jnp.array(0.35)}
    if not marginalized:
        params["rate"] = jnp.array(2.6)
    return params


@pytest.mark.parametrize("marginalized", [False, True])
def test_a_frequency_mask_equals_compressing_to_the_selected_bins(
    marginalized: bool,
) -> None:
    """The mask is a reformulation of the band, not a different likelihood.

    Excluded bins must contribute exactly zero -- to the Gaussian site of the
    general model, and to all four sums the marginalized model's sufficient
    statistics and normalization are built from.
    """
    model = _band_model(marginalized)
    params = _band_params(marginalized)

    masked, _ = log_density(
        model, (), _band_kwargs(marginalized, compressed=False), params
    )
    compressed, _ = log_density(
        model, (), _band_kwargs(marginalized, compressed=True), params
    )

    np.testing.assert_allclose(float(masked), float(compressed), rtol=1e-12)
    # The selection is gappy, so an implementation that quietly assumed a
    # contiguous band would not land here by accident.
    assert not bool(jnp.all(_BAND_MASK[1:4]))


def test_masked_amplitude_statistics_are_the_band_restricted_ones() -> None:
    """The published statistics are what the conditional later integrates.

    They carry no frequency axis, so a band that reached the factor but not
    these two would leave the chain self-consistent and the reconstructed
    posterior wrong, with nothing to see.
    """
    params = _band_params(marginalized=True)

    _, masked = log_density(
        gwb_amplitude_marginalized_model,
        (),
        _band_kwargs(True, compressed=False),
        params,
    )
    _, compressed = log_density(
        gwb_amplitude_marginalized_model,
        (),
        _band_kwargs(True, compressed=True),
        params,
    )

    for name in ("amplitude_mle", "template_optimal_snr"):
        np.testing.assert_allclose(
            float(masked[name]["value"]), float(compressed[name]["value"]), rtol=1e-12
        )


@pytest.mark.parametrize("marginalized", [False, True])
def test_an_excluded_bin_may_carry_a_non_finite_scale(marginalized: bool) -> None:
    """A bin with no network coverage has an infinite scale, and is excluded.

    Its log density is ``-inf`` and its inverse variance is zero, so the mask
    has to discard it rather than multiply it: a masked ``inf`` that survived
    into a sum would poison the value, and a masked ``nan`` the gradient.
    """
    kwargs = _band_kwargs(marginalized, compressed=False)
    kwargs["scale"] = jnp.where(_BAND_MASK, kwargs["scale"], jnp.inf)
    params = _band_params(marginalized)

    def density(tilt: jax.Array) -> jax.Array:
        return log_density(
            _band_model(marginalized), (), kwargs, params | {"tilt": tilt}
        )[0]

    value, gradient = jax.value_and_grad(density)(params["tilt"])

    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)
    # And it is still the band's own density: the infinities changed nothing.
    reference, _ = log_density(
        _band_model(marginalized),
        (),
        _band_kwargs(marginalized, compressed=True),
        params,
    )
    np.testing.assert_allclose(float(value), float(reference), rtol=1e-12)


@pytest.mark.parametrize("marginalized", [False, True])
def test_sweeping_a_frequency_mask_reuses_one_compiled_program(
    marginalized: bool,
) -> None:
    """The reason the band is a traced array: a sub-band sweep is free.

    Compressing to the band makes its bin count a shape, so every band costs a
    compilation -- which is what forces a fresh ``MCMC`` per band. Counting
    traces of the spectrum callable is the direct test: it runs once per
    traced program and not once per call.
    """
    traces = 0
    kwargs = _band_kwargs(marginalized, compressed=False)
    wrapped = kwargs["spectral_density_fn"]

    def counted(params: Mapping[str, ArrayLike]) -> Any:
        nonlocal traces
        traces += 1
        return wrapped(params)

    kwargs["spectral_density_fn"] = counted
    params = _band_params(marginalized)

    @jax.jit
    def density(mask: jax.Array) -> jax.Array:
        return log_density(
            _band_model(marginalized), (), kwargs | {"frequency_mask": mask}, params
        )[0]

    wide = density(_BAND_MASK)
    narrow = density(_BAND_MASK & jnp.array([True, True, False, True, True]))

    assert traces == 1
    # A different mask is a different answer, so the single trace is not one
    # program silently ignoring its argument.
    assert not jnp.allclose(wide, narrow)
