"""Tests for source models declared as NumPyro models, and their evaluation.

The physics is checked against
:func:`reference_population.reference_merger_rate_distance_and_logprob`, a
hand-written grid-level restatement kept deliberately independent of the model
declaration, so the model cannot drift without a failure here.

The composition properties get as much attention as the numbers. A source
model is executed *inside* an outer inference model, and the two boundaries
that make that safe -- handler isolation, and conditioning the source values --
fail silently when they are wrong:
sites leak into the outer joint density, or a factor drops out of one side of a
ratio. Neither produces a shape error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from astrogwb_mock_population import (
    FIDUCIALS,
    N_GRID,
    POPULATION_PARAMS,
    Z_MAX,
    Z_MIN,
    derived_columns,
    make_redshift_grid,
    mock_merger_rate_fn,
    mock_population_model,
    mock_target_model,
)
from jax.typing import ArrayLike
from numpyro import handlers
from reference_population import reference_merger_rate_distance_and_logprob

from astrogwb.catalog import REDSHIFT_SITE
from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.populations import (
    DEFAULT_DENSITY_SITES,
    MergerRateFn,
    SourceFn,
    build_merger_rate_fn,
    build_source_model,
    infer_merger_rate_fn,
    known_merger_rate_models,
    known_source_models,
    register_merger_rate_model,
    register_source_model,
)
from astrogwb.populations.bns_madau_dickinson import (
    bns_md_cosmological,
    bns_md_gaussian_cosmological,
    bns_md_gaussian_modified_propagation,
    bns_md_gaussian_uniform_mixture,
    bns_md_modified_propagation,
    bns_md_uniform_mixture,
    madau_dickinson_total_merger_rate,
)
from astrogwb.utils.sampling import evaluate_sources, sample_sources

#: The deterministic output governing waveform amplitude.
LUMINOSITY_DISTANCE_SITE = "luminosity_distance"

#: Interior to the mock grid and not on a node.
SAMPLE_REDSHIFTS = jnp.array([0.5, 1.234, 3.7, 12.0, 19.5])

#: A non-GR, off-fiducial point in every direction the weights depend on.
OFF_FIDUCIALS: dict[str, float] = {
    **FIDUCIALS,
    "H0": 74.0,
    "Omega_m": 0.27,
    "gamma": 2.7,
    "kappa": 2.9,
    "z_peak": 1.9,
    "xi_0": 1.4,
    "xi_n": 2.3,
    "local_merger_rate": 2.5 * FIDUCIALS["local_merger_rate"],
}

STOCHASTIC_SITES = frozenset(
    {
        REDSHIFT_SITE,
        "source_frame_mass_1",
        "source_frame_mass_2",
        "spin_1z",
        "spin_2z",
        "lambda_1",
        "lambda_2",
    }
)
DETERMINISTIC_SITES = frozenset(
    {
        "detector_frame_mass_1",
        "detector_frame_mass_2",
        LUMINOSITY_DISTANCE_SITE,
        "coa_phase",
        "coa_time",
    }
)


def sample_values(redshift: jax.Array = SAMPLE_REDSHIFTS) -> dict[str, jax.Array]:
    """Every stochastic site, with the redshifts under test."""
    ones = jnp.ones_like(redshift)
    return {
        REDSHIFT_SITE: redshift,
        "source_frame_mass_1": 1.4 * ones,
        "source_frame_mass_2": 1.3 * ones,
        "spin_1z": 0.01 * ones,
        "spin_2z": -0.02 * ones,
        "lambda_1": 400.0 * ones,
        "lambda_2": 300.0 * ones,
    }


def evaluate(
    model: SourceFn,
    params: Mapping[str, ArrayLike],
    values: Mapping[str, ArrayLike],
    density_sites: Sequence[str] = DEFAULT_DENSITY_SITES,
) -> tuple[jax.Array, jax.Array]:
    """Selected ``(N,)`` density and recomputed distance, in one isolated pass."""
    log_prob, outputs = evaluate_sources(
        model, params, values, density_sites=density_sites
    )
    return log_prob, outputs[LUMINOSITY_DISTANCE_SITE]


GAUSSIAN_PARAMS: dict[str, float] = {
    **{
        name: value
        for name, value in POPULATION_PARAMS.items()
        if name not in {"minimum_mass", "mass_width"}
    },
    "mass_mean": 1.33,
    "mass_sigma": 0.09,
}
GAUSSIAN_FIDUCIALS: dict[str, float] = {
    **{
        name: value
        for name, value in FIDUCIALS.items()
        if name not in {"minimum_mass", "mass_width"}
    },
    "mass_mean": 1.33,
    "mass_sigma": 0.09,
}
MASS_SITES = ("source_frame_mass_1", "source_frame_mass_2")


def _gaussian_population_model() -> SourceFn:
    return build_source_model(
        "bns_md_gaussian_cosmological",
        settings={"z_min": Z_MIN, "z_max": Z_MAX, "n_grid": N_GRID},
    )


def _gaussian_target_model() -> SourceFn:
    return build_source_model(
        "bns_md_gaussian_modified_propagation",
        settings={"z_min": Z_MIN, "z_max": Z_MAX, "n_grid": N_GRID},
    )


#: The generating models whose draw path runs through the plated replay: the
#: ordered-uniform triangle and the data-dependent truncated Gaussian.
GENERATING_MODELS: dict[str, tuple[Callable[[], SourceFn], dict[str, float]]] = {
    "uniform_mass": (mock_population_model, POPULATION_PARAMS),
    "gaussian_mass": (_gaussian_population_model, GAUSSIAN_PARAMS),
}


def reference(params: dict[str, float]) -> tuple[jax.Array, jax.Array, jax.Array]:
    return reference_merger_rate_distance_and_logprob(
        params,
        SAMPLE_REDSHIFTS,
        redshift_grid=make_redshift_grid(),
        source_frame_mass_1=sample_values()["source_frame_mass_1"],
        source_frame_mass_2=sample_values()["source_frame_mass_2"],
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_shipped_models_are_registered() -> None:
    assert known_source_models() == (
        "bns_md_cosmological",
        "bns_md_gaussian_cosmological",
        "bns_md_gaussian_modified_propagation",
        "bns_md_gaussian_uniform_mixture",
        "bns_md_modified_propagation",
        "bns_md_uniform_mixture",
    )
    assert known_merger_rate_models() == ("madau_dickinson",)
    shipped = {
        "bns_md_cosmological": bns_md_cosmological,
        "bns_md_modified_propagation": bns_md_modified_propagation,
        "bns_md_uniform_mixture": bns_md_uniform_mixture,
        "bns_md_gaussian_cosmological": bns_md_gaussian_cosmological,
        "bns_md_gaussian_modified_propagation": bns_md_gaussian_modified_propagation,
        "bns_md_gaussian_uniform_mixture": bns_md_gaussian_uniform_mixture,
    }
    for name, fn in shipped.items():
        assert build_source_model(name).func is fn  # ty: ignore[unresolved-attribute]
    rate = build_merger_rate_fn()
    assert rate.func is madau_dickinson_total_merger_rate  # ty: ignore[unresolved-attribute]


def test_builders_bind_construction_settings() -> None:
    settings = {"z_min": Z_MIN, "z_max": Z_MAX, "n_grid": N_GRID}
    source = build_source_model(
        "bns_md_uniform_mixture",
        settings={**settings, "uniform_mixing_fraction": 0.2},
    )
    assert source.keywords == {  # ty: ignore[unresolved-attribute]
        **settings,
        "uniform_mixing_fraction": 0.2,
    }
    # ``source_kwargs`` layers on top of ``settings``, overriding a shared key.
    extra = build_source_model(
        "bns_md_uniform_mixture",
        settings={**settings, "uniform_mixing_fraction": 0.2},
        source_kwargs={"uniform_mixing_fraction": 0.3},
    )
    assert extra.keywords == {  # ty: ignore[unresolved-attribute]
        **settings,
        "uniform_mixing_fraction": 0.3,
    }
    # The rate binds only the shared window/grid keys of the same flat mapping.
    rate = build_merger_rate_fn(settings={**settings, "uniform_mixing_fraction": 0.2})
    assert rate.keywords == settings  # ty: ignore[unresolved-attribute]


def test_unknown_model_names_list_the_known_set() -> None:
    with pytest.raises(KeyError, match="bns_md_cosmological"):
        build_source_model("no_such_population")
    with pytest.raises(KeyError, match="madau_dickinson"):
        build_merger_rate_fn("no_such_rate")


def test_registering_a_name_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_source_model("bns_md_cosmological")(bns_md_cosmological)
    with pytest.raises(ValueError, match="already registered"):
        register_merger_rate_model("madau_dickinson")(madau_dickinson_total_merger_rate)


# --------------------------------------------------------------------------- #
# Explicit site metadata and recomputation
# --------------------------------------------------------------------------- #
def test_source_call_returns_every_declared_site() -> None:
    model = mock_population_model()
    sources = handlers.seed(model, 0)(POPULATION_PARAMS)
    assert set(sources) == STOCHASTIC_SITES | DETERMINISTIC_SITES
    for name in sources:
        assert jnp.asarray(sources[name]).shape == ()


def test_source_evaluate_needs_no_physical_rate() -> None:
    """A source model has no notion of a rate at all -- not even an absent one."""
    without_rate = {
        name: value
        for name, value in POPULATION_PARAMS.items()
        if name != "local_merger_rate"
    }
    _, luminosity_distance = evaluate(
        mock_population_model(), without_rate, sample_values()
    )
    assert luminosity_distance.shape == SAMPLE_REDSHIFTS.shape


def test_merger_rate_requires_the_physical_rate_parameter() -> None:
    """The merger-rate function owns this check, not an ``"x" in params`` branch."""
    without_rate = {
        name: value
        for name, value in POPULATION_PARAMS.items()
        if name != "local_merger_rate"
    }
    with pytest.raises(ValueError, match="local_merger_rate"):
        mock_merger_rate_fn()(without_rate)


def test_stored_deterministics_are_recomputed_from_sampled_values() -> None:
    model = mock_population_model()
    stored = {
        **sample_values(),
        LUMINOSITY_DISTANCE_SITE: jnp.ones_like(SAMPLE_REDSHIFTS),
        "detector_frame_mass_1": jnp.zeros_like(SAMPLE_REDSHIFTS),
    }
    clean = derived_columns(model, POPULATION_PARAMS, sample_values())
    tampered = derived_columns(model, POPULATION_PARAMS, stored)
    for name in (LUMINOSITY_DISTANCE_SITE, "detector_frame_mass_1"):
        np.testing.assert_array_equal(tampered[name], clean[name])


# --------------------------------------------------------------------------- #
# One execution supplies density and distance; the rate is a separate call
# --------------------------------------------------------------------------- #
def test_one_execution_supplies_per_sample_density_distance_and_scalar_rate() -> None:
    log_prob, distance = evaluate(
        mock_population_model(), POPULATION_PARAMS, sample_values()
    )
    rate = mock_merger_rate_fn()(POPULATION_PARAMS)
    assert log_prob.shape == SAMPLE_REDSHIFTS.shape
    assert distance.shape == SAMPLE_REDSHIFTS.shape
    assert rate.shape == ()

    expected_rate, expected_distance, expected_logpdf = reference(FIDUCIALS)
    # Bit-exact: the distribution shares the reference's operation order, which
    # is what keeps a catalog that is its own proposal at exactly zero weight.
    np.testing.assert_allclose(
        np.asarray(log_prob), np.asarray(expected_logpdf), rtol=0.0, atol=2e-15
    )
    np.testing.assert_allclose(distance, expected_distance, rtol=1e-14)
    np.testing.assert_allclose(float(rate), float(expected_rate), rtol=1e-15)


def test_derived_columns_match_the_declared_transforms() -> None:
    values = sample_values()
    columns = derived_columns(mock_population_model(), POPULATION_PARAMS, values)
    one_plus_z = 1.0 + SAMPLE_REDSHIFTS
    np.testing.assert_array_equal(
        columns["detector_frame_mass_1"], values["source_frame_mass_1"] * one_plus_z
    )
    np.testing.assert_array_equal(
        columns["detector_frame_mass_2"], values["source_frame_mass_2"] * one_plus_z
    )


# --------------------------------------------------------------------------- #
# Modified propagation
# --------------------------------------------------------------------------- #
def test_modified_propagation_scales_the_distance_by_the_gw_em_ratio() -> None:
    _, distance = evaluate(mock_target_model(), OFF_FIDUCIALS, sample_values())
    _, cosmological_distance, _ = reference(OFF_FIDUCIALS)
    expected = cosmological_distance * jnp.exp(
        log_gw_em_ratio(SAMPLE_REDSHIFTS, OFF_FIDUCIALS["xi_0"], OFF_FIDUCIALS["xi_n"])
    )
    np.testing.assert_allclose(distance, expected, rtol=1e-14)


def test_modified_propagation_reduces_exactly_to_the_cosmological_model() -> None:
    """At xi_0 = 1 the two declarations must agree bit-for-bit.

    Every committed run pins ``xi_0 = 1`` in its fiducials while sampling a
    modified-propagation target, so the catalogs are drawn cosmologically and
    reweighted with the modified model. Anything less than exact here would put
    a floor under the self-proposal log weights.
    """
    values = sample_values()
    cosmological_log_prob, cosmological_distance = evaluate(
        mock_population_model(), POPULATION_PARAMS, values
    )
    modified_log_prob, modified_distance = evaluate(
        mock_target_model(), FIDUCIALS, values
    )
    assert FIDUCIALS["xi_0"] == 1.0
    np.testing.assert_array_equal(cosmological_log_prob, modified_log_prob)
    np.testing.assert_array_equal(cosmological_distance, modified_distance)


# --------------------------------------------------------------------------- #
# Excluded factors
# --------------------------------------------------------------------------- #
def test_density_selection_preserves_supplied_values_and_deterministics() -> None:
    values = sample_values()
    model = mock_population_model()
    log_prob, luminosity_distance = evaluate(model, POPULATION_PARAMS, values)
    selected_log_prob, selected_distance = evaluate(
        model, POPULATION_PARAMS, values, density_sites=("redshift", "spin_1z")
    )
    expected_spin = dist.Uniform(-0.05, 0.05).log_prob(values["spin_1z"])
    expected_mass = jnp.log(2.0) - 2.0 * jnp.log(POPULATION_PARAMS["mass_width"])
    np.testing.assert_allclose(
        selected_log_prob, log_prob + expected_spin - expected_mass
    )
    np.testing.assert_array_equal(luminosity_distance, selected_distance)
    columns = derived_columns(model, POPULATION_PARAMS, values)
    np.testing.assert_array_equal(
        columns["detector_frame_mass_1"],
        values["source_frame_mass_1"] * (1 + values["redshift"]),
    )


def test_empty_density_selection_returns_per_source_zeros() -> None:
    values = sample_values()
    log_prob, _ = evaluate(
        mock_population_model(), POPULATION_PARAMS, values, density_sites=()
    )
    assert log_prob.shape == SAMPLE_REDSHIFTS.shape
    np.testing.assert_array_equal(log_prob, jnp.zeros_like(SAMPLE_REDSHIFTS))


# --------------------------------------------------------------------------- #
# Isolation from an outer inference model
# --------------------------------------------------------------------------- #
def _outer_model(observed: jax.Array) -> None:
    """An inference model that evaluates a source model twice."""
    hubble_constant = numpyro.sample("H0", dist.Uniform(20.0, 140.0))
    params = {**FIDUCIALS, "H0": hubble_constant}
    total = jnp.zeros(())
    for _ in range(2):
        log_prob, luminosity_distance = evaluate(
            mock_target_model(), params, sample_values()
        )
        total = total + jnp.sum(log_prob)
        total = total + jnp.sum(luminosity_distance)
    numpyro.sample("obs", dist.Normal(total * 1e-6, 1.0), obs=observed)


def test_source_sites_stay_out_of_the_outer_trace() -> None:
    with handlers.seed(rng_seed=0):
        trace = handlers.trace(_outer_model).get_trace(jnp.asarray(0.0))
    assert set(trace) == {"H0", "obs"}


def test_repeated_evaluation_inside_one_model_does_not_collide() -> None:
    """Without the handler boundary the second evaluation is a duplicate site."""
    with handlers.seed(rng_seed=0):
        handlers.trace(_outer_model).get_trace(jnp.asarray(0.0))


def test_outer_density_and_gradient_match_a_direct_calculation() -> None:
    from numpyro.infer.util import log_density

    def direct(hubble_constant: jax.Array) -> jax.Array:
        params = {**FIDUCIALS, "H0": hubble_constant}
        total = jnp.zeros(())
        for _ in range(2):
            log_prob, luminosity_distance = evaluate(
                mock_target_model(), params, sample_values()
            )
            total = total + jnp.sum(log_prob)
            total = total + jnp.sum(luminosity_distance)
        prior = dist.Uniform(20.0, 140.0).log_prob(hubble_constant)
        likelihood = dist.Normal(total * 1e-6, 1.0).log_prob(jnp.asarray(0.0))
        return prior + likelihood

    def joint(hubble_constant: jax.Array) -> jax.Array:
        value, _ = log_density(
            _outer_model, (jnp.asarray(0.0),), {}, {"H0": hubble_constant}
        )
        return value

    points = jnp.array([55.0, 67.66, 90.0])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(joint))(points), jax.vmap(direct)(points), rtol=1e-12
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(jax.grad(joint)))(points),
        jax.vmap(jax.grad(direct))(points),
        rtol=1e-8,
    )


@pytest.mark.integration
def test_nuts_samples_only_the_outer_hyperparameter() -> None:
    from numpyro.infer import MCMC, NUTS

    mcmc = MCMC(NUTS(_outer_model), num_warmup=20, num_samples=20, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), jnp.asarray(0.0))
    assert set(mcmc.get_samples()) == {"H0"}


# --------------------------------------------------------------------------- #
# The uniform-guard mixture proposal
# --------------------------------------------------------------------------- #
def _redshift_log_density(
    model: SourceFn,
    params: Mapping[str, ArrayLike],
    redshift: ArrayLike,
) -> jax.Array:
    with handlers.block(), handlers.seed(rng_seed=0):
        trace = handlers.trace(model).get_trace(params)
    site = trace.get(REDSHIFT_SITE)
    if site is None or site["type"] != "sample":
        raise ValueError(
            f"source model declares no {REDSHIFT_SITE!r} sample site; every "
            "source model must draw a redshift"
        )
    return jnp.asarray(site["fn"].log_prob(jnp.asarray(redshift)))


def _uniform_mixture_model(uniform_mixing_fraction: float) -> SourceFn:
    return build_source_model(
        "bns_md_uniform_mixture",
        settings={
            "z_min": Z_MIN,
            "z_max": Z_MAX,
            "n_grid": N_GRID,
            "uniform_mixing_fraction": uniform_mixing_fraction,
        },
    )


def test_uniform_mixture_matches_the_explicit_logaddexp_proposal() -> None:
    epsilon = 0.1
    model = _uniform_mixture_model(epsilon)
    actual = _redshift_log_density(model, POPULATION_PARAMS, SAMPLE_REDSHIFTS)

    _, _, md_logprob = reference_merger_rate_distance_and_logprob(
        FIDUCIALS, SAMPLE_REDSHIFTS, redshift_grid=make_redshift_grid()
    )
    expected = jnp.logaddexp(
        jnp.log1p(-epsilon) + md_logprob,
        jnp.log(epsilon) - jnp.log(Z_MAX - Z_MIN),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13)

    outside = jnp.array([Z_MIN - 0.1, Z_MAX + 0.1])
    assert np.all(
        np.isneginf(
            np.asarray(_redshift_log_density(model, POPULATION_PARAMS, outside))
        )
    )


def test_uniform_mixture_keeps_the_cosmological_distance() -> None:
    """The guard changes which redshifts are drawn, not the physics at one."""
    _, mixture_distance = evaluate(
        _uniform_mixture_model(0.1), POPULATION_PARAMS, sample_values()
    )
    _, plain_distance = evaluate(
        mock_population_model(), POPULATION_PARAMS, sample_values()
    )
    np.testing.assert_array_equal(mixture_distance, plain_distance)


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_uniform_mixture_rejects_an_out_of_range_fraction(fraction: float) -> None:
    model = _uniform_mixture_model(fraction)
    with pytest.raises(ValueError, match="uniform_mixing_fraction"):
        evaluate(model, POPULATION_PARAMS, sample_values())


# --------------------------------------------------------------------------- #
# The merger rate a redshift law already determines
# --------------------------------------------------------------------------- #
#: Source models whose ``redshift`` site is a bare ``RedshiftDistribution``, so
#: the total merger rate is the normalization of a table the model has already
#: built.
#:
#: Each is paired with hyperparameters complete enough to *execute* it, not just
#: to evaluate its redshift law: inference probes the model, so the
#: modified-propagation variants need ``xi_0``/``xi_n`` here even though the
#: rate does not depend on them. That is the same requirement
#: :func:`~astrogwb.sampling.validate_source_model` has.
INFERABLE_MODELS: dict[str, dict[str, float]] = {
    "bns_md_cosmological": POPULATION_PARAMS,
    "bns_md_modified_propagation": FIDUCIALS,
    "bns_md_gaussian_cosmological": GAUSSIAN_PARAMS,
    "bns_md_gaussian_modified_propagation": GAUSSIAN_FIDUCIALS,
}

#: A second point in hyperparameter space, off the probe point in every
#: direction the rate depends on: the cosmology, the rate shape, and the
#: absolute normalization.
MOVED_PARAMS: dict[str, float] = {
    "H0": 70.0,
    "gamma": 2.0,
    "kappa": 5.5,
    "z_peak": 2.2,
    "local_merger_rate": 1000.0,
}


def _settings(**extra: float) -> dict[str, float]:
    return {"z_min": Z_MIN, "z_max": Z_MAX, "n_grid": N_GRID, **extra}


def _registered_rate_fn() -> MergerRateFn:
    return build_merger_rate_fn("madau_dickinson", settings=_settings())


def _inferred_rate_fn(name: str) -> MergerRateFn:
    """The inferred rate, asserted present. Only the mixture tests want ``None``."""
    inferred = infer_merger_rate_fn(
        build_source_model(name, settings=_settings()), INFERABLE_MODELS[name]
    )
    assert inferred is not None
    return inferred


@pytest.mark.parametrize("name", INFERABLE_MODELS)
def test_inferred_rate_is_the_registered_rate(name: str) -> None:
    """The one check the two registries never had.

    ``registry`` is explicit that nothing compares a registered source model
    against the rate it is paired with: "re-pointing a registered key at a
    different density would be invisible here". For a population whose redshift
    law *is* its rate, this closes that hole -- and bit-exactly, since both
    constructions build the same table from the same settings.
    """
    params = INFERABLE_MODELS[name]
    assert _inferred_rate_fn(name)(params) == _registered_rate_fn()(params)


@pytest.mark.parametrize("name", INFERABLE_MODELS)
def test_inferred_rate_tracks_hyperparameters_off_the_probe_point(name: str) -> None:
    """The probe supplies a recipe, not a value.

    Closing over the probed distribution instead would freeze the rate at the
    hyperparameters it was built with -- correct at the probe point and wrong
    everywhere NUTS goes, with no shape error to show it.
    """
    params = INFERABLE_MODELS[name]
    inferred = _inferred_rate_fn(name)
    registered = _registered_rate_fn()
    moved = {**params, **MOVED_PARAMS}

    assert inferred(moved) == registered(moved)
    assert inferred(moved) != inferred(params)
    # And the probe point still evaluates correctly afterwards.
    assert inferred(params) == registered(params)


def test_inferred_rate_is_jittable_without_retracing_per_call() -> None:
    params = POPULATION_PARAMS
    inferred = _inferred_rate_fn("bns_md_cosmological")
    jitted = jax.jit(inferred)
    moved = {**params, **MOVED_PARAMS}

    # Not bit-exact: XLA is free to reassociate the trapezoid sum, so a jitted
    # rate agrees with an eager one to round-off, not to the last bit. The
    # bit-exact claim is between the inferred and registered rates, which run
    # the same code path -- see above.
    for values in (params, moved):
        np.testing.assert_allclose(
            np.asarray(jitted(values)), np.asarray(inferred(values)), rtol=1e-14
        )
    # One signature, so one trace: the captured rate shape is a singleton and
    # the window is closed-over data, not an argument. This is the recompile
    # hazard `registry` warns about, so it is worth reaching for JAX's private
    # cache counter rather than settling for value agreement alone.
    assert cast(Any, jitted)._cache_size() == 1


@pytest.mark.parametrize(
    "name", ["bns_md_uniform_mixture", "bns_md_gaussian_uniform_mixture"]
)
def test_a_mixture_redshift_law_carries_no_rate_to_infer(name: str) -> None:
    """``None``, not an exception: a proposal having no rate is the normal answer.

    These are shipped, registered models whose purpose is to draw importance
    proposals, and a proposal catalog's rate is never read. The component order
    is not a contract either, so unwrapping the first component would silently
    return whichever rate it happened to hold.
    """
    params = GAUSSIAN_PARAMS if "gaussian" in name else POPULATION_PARAMS
    model = build_source_model(name, settings=_settings(uniform_mixing_fraction=0.1))
    assert infer_merger_rate_fn(model, params) is None


def test_a_model_without_a_redshift_site_is_rejected() -> None:
    """Malformed, not rate-free: every population must draw a redshift."""

    def no_redshift(params: Mapping[str, ArrayLike]) -> dict[str, jax.Array]:
        return {"source_frame_mass_1": jnp.asarray(numpyro.sample("m", dist.Uniform()))}

    with pytest.raises(ValueError, match=REDSHIFT_SITE):
        infer_merger_rate_fn(no_redshift, POPULATION_PARAMS)


def test_an_inferred_rate_requires_the_physical_rate_parameter() -> None:
    """The same guard the registered rate applies, from the same function."""
    without_rate = {
        name: value
        for name, value in POPULATION_PARAMS.items()
        if name != "local_merger_rate"
    }
    inferred = _inferred_rate_fn("bns_md_cosmological")
    with pytest.raises(ValueError, match="local_merger_rate"):
        inferred(without_rate)


def test_inference_is_isolated_from_outer_handlers() -> None:
    """The probe executes a whole source model; none of it may reach a trace."""
    with handlers.trace() as outer, handlers.seed(rng_seed=0):
        inferred = infer_merger_rate_fn(mock_population_model(), POPULATION_PARAMS)
        assert inferred is not None
        rate = inferred(POPULATION_PARAMS)
    assert outer == {}
    assert rate == _registered_rate_fn()(POPULATION_PARAMS)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("generating", GENERATING_MODELS)
def test_generation_is_reproducible(generating: str) -> None:
    build, params = GENERATING_MODELS[generating]
    first = sample_sources(build(), jax.random.PRNGKey(7), params, num_samples=32)
    second = sample_sources(build(), jax.random.PRNGKey(7), params, num_samples=32)
    assert set(first) == STOCHASTIC_SITES | DETERMINISTIC_SITES
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_a_smaller_catalog_is_a_prefix_of_a_larger_one() -> None:
    """The property the variable-catalog-size experiment is a series because of.

    ``Predictive`` allocates per-draw keys with ``jax.random.split``, whose
    prefix stability is a property of the installed JAX rather than something
    the API promises, so it is checked here rather than assumed.
    """
    small = sample_sources(
        mock_population_model(),
        jax.random.PRNGKey(7),
        POPULATION_PARAMS,
        num_samples=16,
    )
    large = sample_sources(
        mock_population_model(),
        jax.random.PRNGKey(7),
        POPULATION_PARAMS,
        num_samples=64,
    )
    for name, values in small.items():
        np.testing.assert_array_equal(values, large[name][:16])


@pytest.mark.parametrize("generating", GENERATING_MODELS)
def test_stored_columns_are_bit_identical_to_a_later_recomputation(
    generating: str,
) -> None:
    """What makes both the exact-zero weights and the load check exact.

    ``Predictive`` runs the model under ``vmap``, one draw at a time, while
    every later evaluation runs it batched. Taking the derived columns from a
    batched pass rather than from ``Predictive`` is what removes that
    difference. The Gaussian model is here because its second mass is a
    ``TruncatedNormal`` whose upper bound is the first mass: a data-dependent
    support is where running the replay under a plate could plausibly shift a
    bit.
    """
    build, params = GENERATING_MODELS[generating]
    samples = sample_sources(build(), jax.random.PRNGKey(7), params, num_samples=32)
    stochastic = {name: samples[name] for name in STOCHASTIC_SITES}
    columns = derived_columns(build(), params, stochastic)
    for name in DETERMINISTIC_SITES:
        np.testing.assert_array_equal(samples[name], columns[name])


# --------------------------------------------------------------------------- #
# JAX transformations
# --------------------------------------------------------------------------- #
def test_evaluation_traces_under_jit_and_vmaps_over_hyperparameters() -> None:
    values = sample_values()

    def redshift_density(hubble_constant: jax.Array) -> jax.Array:
        params = {**OFF_FIDUCIALS, "H0": hubble_constant}
        log_prob, _ = evaluate(mock_target_model(), params, values)
        return log_prob

    hubble_constants = jnp.array([60.0, 67.66, 75.0])
    batched = jax.jit(jax.vmap(redshift_density))(hubble_constants)
    assert batched.shape == (3, SAMPLE_REDSHIFTS.shape[0])
    np.testing.assert_allclose(
        batched[1], redshift_density(hubble_constants[1]), rtol=1e-12
    )


def test_sampling_is_jittable_and_isolated_from_outer_handlers() -> None:
    model = mock_population_model()
    sample = jax.jit(partial(sample_sources, model, num_samples=8))
    with handlers.trace() as outer:
        sources = sample(jax.random.PRNGKey(7), POPULATION_PARAMS)
    assert outer == {}
    assert set(sources) == STOCHASTIC_SITES | DETERMINISTIC_SITES
    assert sources["spin_1z"].shape == (8,)
    np.testing.assert_allclose(
        sources["detector_frame_mass_1"],
        sources["source_frame_mass_1"] * (1 + sources["redshift"]),
    )

    def log_prob(
        params: Mapping[str, ArrayLike], values: Mapping[str, ArrayLike]
    ) -> jax.Array:
        return evaluate(model, params, values)[0]

    np.testing.assert_allclose(
        jax.jit(log_prob)(POPULATION_PARAMS, sources),
        log_prob(POPULATION_PARAMS, sources),
        rtol=1e-12,
    )


def test_sampling_and_derivation_are_isolated_without_jit() -> None:
    with handlers.trace() as outer:
        sample_sources(
            mock_population_model(),
            jax.random.PRNGKey(7),
            POPULATION_PARAMS,
            num_samples=8,
        )
    assert outer == {}


# --------------------------------------------------------------------------- #
# Ordered Gaussian masses
# --------------------------------------------------------------------------- #
def _gaussian_mixture_model(uniform_mixing_fraction: float) -> SourceFn:
    return build_source_model(
        "bns_md_gaussian_uniform_mixture",
        settings={
            "z_min": Z_MIN,
            "z_max": Z_MAX,
            "n_grid": N_GRID,
            "uniform_mixing_fraction": uniform_mixing_fraction,
        },
    )


def test_gaussian_mass_density_matches_two_iid_normals_on_the_ordered_half_plane() -> (
    None
):
    values = sample_values()
    actual, _ = evaluate(
        _gaussian_population_model(), GAUSSIAN_PARAMS, values, density_sites=MASS_SITES
    )
    component = dist.Normal(GAUSSIAN_PARAMS["mass_mean"], GAUSSIAN_PARAMS["mass_sigma"])
    expected = (
        jnp.log(2.0)
        + component.log_prob(values["source_frame_mass_1"])
        + component.log_prob(values["source_frame_mass_2"])
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)

    unordered = {
        **values,
        "source_frame_mass_1": values["source_frame_mass_2"],
        "source_frame_mass_2": values["source_frame_mass_1"],
    }

    def unordered_log_prob(sources: Mapping[str, ArrayLike]) -> jax.Array:
        log_prob, _ = evaluate(
            _gaussian_population_model(),
            GAUSSIAN_PARAMS,
            sources,
            density_sites=MASS_SITES,
        )
        return log_prob

    assert np.all(np.isneginf(np.asarray(jax.jit(unordered_log_prob)(unordered))))


def test_gaussian_draws_are_ordered_and_have_finite_self_density() -> None:
    sources = sample_sources(
        _gaussian_population_model(),
        jax.random.PRNGKey(7),
        GAUSSIAN_PARAMS,
        num_samples=64,
    )
    np.testing.assert_array_equal(
        sources["source_frame_mass_1"] >= sources["source_frame_mass_2"],
        jnp.ones(64, dtype=bool),
    )
    log_prob, _ = evaluate(_gaussian_population_model(), GAUSSIAN_PARAMS, sources)
    assert np.all(np.isfinite(np.asarray(log_prob)))


def test_gaussian_mass_density_stays_finite_when_hyperparameters_move() -> None:
    """The NUTS property: moving (mean, sigma) never zeros a catalog sample.

    The ordered-uniform triangle still drops out the moment a sample exits
    the support, which is the reviewer's concern this model exists to answer.
    """
    values = sample_values()
    shifted = {**GAUSSIAN_PARAMS, "mass_mean": 2.0, "mass_sigma": 0.2}
    gaussian_log_prob, _ = evaluate(
        _gaussian_population_model(), shifted, values, density_sites=MASS_SITES
    )
    assert np.all(np.isfinite(np.asarray(gaussian_log_prob)))

    def total(mass_mean: jax.Array, mass_sigma: jax.Array) -> jax.Array:
        params = {
            **GAUSSIAN_PARAMS,
            "mass_mean": mass_mean,
            "mass_sigma": mass_sigma,
        }
        log_prob, _ = evaluate(
            _gaussian_population_model(), params, values, density_sites=MASS_SITES
        )
        return jnp.sum(log_prob)

    d_mean, d_sigma = jax.grad(total, argnums=(0, 1))(
        jnp.asarray(GAUSSIAN_PARAMS["mass_mean"]),
        jnp.asarray(GAUSSIAN_PARAMS["mass_sigma"]),
    )
    assert np.isfinite(float(d_mean))
    assert np.isfinite(float(d_sigma))

    outside_uniform = {**POPULATION_PARAMS, "minimum_mass": 1.35}

    def uniform_log_prob(params: Mapping[str, ArrayLike]) -> jax.Array:
        log_prob, _ = evaluate(
            mock_population_model(), params, values, density_sites=MASS_SITES
        )
        return log_prob

    assert np.all(np.isneginf(np.asarray(jax.jit(uniform_log_prob)(outside_uniform))))


def test_gaussian_modified_propagation_reduces_exactly_to_the_cosmological_model() -> (
    None
):
    values = sample_values()
    cosmological_log_prob, cosmological_distance = evaluate(
        _gaussian_population_model(), GAUSSIAN_PARAMS, values
    )
    modified_log_prob, modified_distance = evaluate(
        _gaussian_target_model(), GAUSSIAN_FIDUCIALS, values
    )
    assert GAUSSIAN_FIDUCIALS["xi_0"] == 1.0
    np.testing.assert_array_equal(cosmological_log_prob, modified_log_prob)
    np.testing.assert_array_equal(cosmological_distance, modified_distance)


def test_gaussian_and_uniform_models_share_distance() -> None:
    values = sample_values()
    _, gaussian_distance = evaluate(
        _gaussian_population_model(), GAUSSIAN_PARAMS, values
    )
    _, uniform_distance = evaluate(mock_population_model(), POPULATION_PARAMS, values)
    np.testing.assert_array_equal(gaussian_distance, uniform_distance)


def test_gaussian_uniform_mixture_matches_the_explicit_logaddexp_proposal() -> None:
    epsilon = 0.1
    actual = _redshift_log_density(
        _gaussian_mixture_model(epsilon), GAUSSIAN_PARAMS, SAMPLE_REDSHIFTS
    )
    _, _, md_logprob = reference_merger_rate_distance_and_logprob(
        FIDUCIALS, SAMPLE_REDSHIFTS, redshift_grid=make_redshift_grid()
    )
    expected = jnp.logaddexp(
        jnp.log1p(-epsilon) + md_logprob,
        jnp.log(epsilon) - jnp.log(Z_MAX - Z_MIN),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13)
