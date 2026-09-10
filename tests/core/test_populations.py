"""Tests for populations declared as NumPyro models, and their evaluation.

The physics is checked against
:func:`reference_population.reference_merger_rate_distance_and_logprob`, a
hand-written grid-level restatement kept deliberately independent of the model
declaration, so the model cannot drift without a failure here.

The composition properties get as much attention as the numbers. A population
model is executed *inside* an outer inference model, and the two boundaries
that make that safe -- handler isolation, and substituting source values inside
the selective block -- fail silently when they are wrong:
sites leak into the outer joint density, or a factor drops out of one side of a
ratio. Neither produces a shape error.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from functools import partial

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
    mock_population_model,
    mock_target_model,
)
from jax.typing import ArrayLike
from numpyro import handlers
from reference_population import reference_merger_rate_distance_and_logprob

from astrogwb.catalog.catalog import REDSHIFT_SITE
from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.populations import (
    Population,
    build_population,
    known_population_models,
    register_population_model,
)
from astrogwb.populations.bns_madau_dickinson import (
    bns_md_cosmological,
    bns_md_gaussian_cosmological,
    bns_md_gaussian_modified_propagation,
    bns_md_gaussian_uniform_mixture,
    bns_md_modified_propagation,
    bns_md_uniform_mixture,
)

#: Deterministic site names, used only where a raw trace (rather than the
#: typed :class:`~astrogwb.populations.PopulationTrace`) is under test.
LUMINOSITY_DISTANCE_SITE = "luminosity_distance"
TOTAL_MERGER_RATE_SITE = "total_merger_rate"

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
        "inclination",
        "coa_phase",
        "coa_time",
        TOTAL_MERGER_RATE_SITE,
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
    assert known_population_models() == (
        "bns_md_cosmological",
        "bns_md_gaussian_cosmological",
        "bns_md_gaussian_modified_propagation",
        "bns_md_gaussian_uniform_mixture",
        "bns_md_modified_propagation",
        "bns_md_uniform_mixture",
    )
    assert build_population("bns_md_cosmological", settings={}).fn is (
        bns_md_cosmological
    )
    assert build_population("bns_md_modified_propagation", settings={}).fn is (
        bns_md_modified_propagation
    )
    assert build_population("bns_md_uniform_mixture", settings={}).fn is (
        bns_md_uniform_mixture
    )
    assert build_population("bns_md_gaussian_cosmological", settings={}).fn is (
        bns_md_gaussian_cosmological
    )
    assert build_population("bns_md_gaussian_modified_propagation", settings={}).fn is (
        bns_md_gaussian_modified_propagation
    )
    assert build_population("bns_md_gaussian_uniform_mixture", settings={}).fn is (
        bns_md_gaussian_uniform_mixture
    )


def test_unknown_model_names_list_the_known_set() -> None:
    with pytest.raises(KeyError, match="bns_md_cosmological"):
        build_population("no_such_population", settings={})


def test_registering_a_name_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_population_model("bns_md_cosmological", source_sites=())(
            bns_md_cosmological
        )


def test_populations_from_reordered_settings_hash_equal_and_compile_once() -> None:
    """The regression guard for keeping ``Population`` a canonical value object.

    A dict's insertion order is not part of its equality, but an unsorted
    ``settings`` tuple would leak that order into hashing and into ``jax.jit``'s
    cache key -- silently retracing the ``(F, N)`` contraction on every
    construction from the same catalog. Sorting in ``__post_init__`` is what
    keeps all three of these genuinely one value.
    """
    kwargs = {"z_min": Z_MIN, "z_max": Z_MAX, "n_grid": N_GRID}
    reordered = {"n_grid": N_GRID, "z_max": Z_MAX, "z_min": Z_MIN}
    first = build_population("bns_md_cosmological", settings=kwargs)
    second = build_population("bns_md_cosmological", settings=kwargs)
    third = build_population("bns_md_cosmological", settings=reordered)
    assert first == second == third
    assert hash(first) == hash(second) == hash(third)

    compiled = jax.jit(
        lambda model, params: model.log_prob(params, sample_values()),
        static_argnums=0,
    )
    for model in (first, second, third):
        compiled(model, POPULATION_PARAMS)
    assert compiled._cache_size() == 1  # ty: ignore[unresolved-attribute]


# --------------------------------------------------------------------------- #
# Explicit site metadata and recomputation
# --------------------------------------------------------------------------- #
def test_population_declares_its_source_outputs_and_density_factors() -> None:
    model = mock_population_model()
    assert set(model.source_sites) == STOCHASTIC_SITES | (
        DETERMINISTIC_SITES - {TOTAL_MERGER_RATE_SITE}
    )
    assert model.density_sites == (
        REDSHIFT_SITE,
        "source_frame_mass_1",
        "source_frame_mass_2",
    )
    with pytest.raises(FrozenInstanceError):
        model.density_sites = ()  # ty: ignore[invalid-assignment]
    assert hash(model) == hash(mock_population_model())


def test_total_merger_rate_is_declared_only_with_a_physical_rate() -> None:
    """A proposal is a density, not an observation, so its rate is optional."""
    without_rate = {
        name: value
        for name, value in POPULATION_PARAMS.items()
        if name != "local_merger_rate"
    }
    trace = mock_population_model().evaluate(without_rate, sample_values())
    assert trace.total_merger_rate is None
    assert trace.luminosity_distance.shape == SAMPLE_REDSHIFTS.shape


def test_stored_deterministics_are_recomputed_from_sampled_values() -> None:
    model = mock_population_model()
    stored = {
        **sample_values(),
        LUMINOSITY_DISTANCE_SITE: jnp.ones_like(SAMPLE_REDSHIFTS),
        "detector_frame_mass_1": jnp.zeros_like(SAMPLE_REDSHIFTS),
    }
    clean = model.trace(POPULATION_PARAMS, sample_values())
    actual = derived_columns(model, POPULATION_PARAMS, stored)
    tampered = model.trace(POPULATION_PARAMS, stored)
    for name in (LUMINOSITY_DISTANCE_SITE, "detector_frame_mass_1"):
        np.testing.assert_array_equal(actual[name], clean[name]["value"])
        np.testing.assert_array_equal(tampered[name]["value"], clean[name]["value"])


# --------------------------------------------------------------------------- #
# One execution supplies density, distance, and rate
# --------------------------------------------------------------------------- #
def test_one_execution_supplies_per_sample_density_distance_and_scalar_rate() -> None:
    trace = mock_population_model().evaluate(POPULATION_PARAMS, sample_values())
    assert trace.log_prob.shape == SAMPLE_REDSHIFTS.shape

    distance = trace.luminosity_distance
    rate = trace.total_merger_rate
    assert distance.shape == SAMPLE_REDSHIFTS.shape
    assert rate is not None
    assert rate.shape == ()

    expected_rate, expected_distance, expected_logpdf = reference(FIDUCIALS)
    # Bit-exact: the distribution shares the reference's operation order, which
    # is what keeps a catalog that is its own proposal at exactly zero weight.
    np.testing.assert_allclose(
        np.asarray(trace.log_prob), np.asarray(expected_logpdf), rtol=0.0, atol=2e-15
    )
    np.testing.assert_allclose(distance, expected_distance, rtol=1e-14)
    np.testing.assert_allclose(float(rate), float(expected_rate), rtol=1e-15)


def test_rate_and_distance_deterministics_add_no_density_factors() -> None:
    """Deterministic sites carry no log density, so they cannot bias a ratio."""
    with_rate = (
        mock_population_model().evaluate(POPULATION_PARAMS, sample_values()).log_prob
    )
    without_rate_params = {
        name: value
        for name, value in POPULATION_PARAMS.items()
        if name != "local_merger_rate"
    }
    without_rate = (
        mock_population_model().evaluate(without_rate_params, sample_values()).log_prob
    )
    # The source table now carries the absolute rate, so renormalizing the
    # differently scaled table can differ by a machine ulp.
    np.testing.assert_allclose(with_rate, without_rate, rtol=0.0, atol=1e-14)


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
    np.testing.assert_array_equal(columns["inclination"], jnp.zeros_like(one_plus_z))
    # The population-level rate has no sample axis and is deliberately not a
    # source column: it is recomputed from the model, where a stored copy would
    # be stale after a redshift window is narrowed.
    assert TOTAL_MERGER_RATE_SITE not in columns


# --------------------------------------------------------------------------- #
# Modified propagation
# --------------------------------------------------------------------------- #
def test_modified_propagation_scales_the_distance_by_the_gw_em_ratio() -> None:
    trace = mock_target_model().evaluate(OFF_FIDUCIALS, sample_values())
    distance = trace.luminosity_distance
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
    cosmological_trace = mock_population_model().evaluate(POPULATION_PARAMS, values)
    modified_trace = mock_target_model().evaluate(FIDUCIALS, values)
    assert FIDUCIALS["xi_0"] == 1.0
    np.testing.assert_array_equal(cosmological_trace.log_prob, modified_trace.log_prob)
    np.testing.assert_array_equal(
        cosmological_trace.luminosity_distance,
        modified_trace.luminosity_distance,
    )


# --------------------------------------------------------------------------- #
# Excluded factors
# --------------------------------------------------------------------------- #
def test_density_selection_preserves_supplied_values_and_deterministics() -> None:
    values = sample_values()
    model = mock_population_model()
    selected = replace(model, density_sites=("redshift", "spin_1z"))
    trace = model.evaluate(POPULATION_PARAMS, values)
    selected_trace = selected.evaluate(POPULATION_PARAMS, values)
    expected_spin = dist.Uniform(-0.05, 0.05).log_prob(values["spin_1z"])
    expected_mass = jnp.log(2.0) - 2.0 * jnp.log(POPULATION_PARAMS["mass_width"])
    np.testing.assert_allclose(
        selected_trace.log_prob, trace.log_prob + expected_spin - expected_mass
    )
    np.testing.assert_array_equal(
        trace.luminosity_distance, selected_trace.luminosity_distance
    )
    assert trace.total_merger_rate is not None
    assert selected_trace.total_merger_rate is not None
    np.testing.assert_array_equal(
        trace.total_merger_rate, selected_trace.total_merger_rate
    )
    raw = selected.trace(POPULATION_PARAMS, values)
    np.testing.assert_array_equal(
        raw["detector_frame_mass_1"]["value"],
        values["source_frame_mass_1"] * (1 + values["redshift"]),
    )


def test_empty_density_selection_returns_scalar_zero() -> None:
    values = sample_values()
    empty = replace(mock_population_model(), density_sites=())
    trace = empty.evaluate(POPULATION_PARAMS, values)
    assert trace.log_prob.shape == ()
    assert float(trace.log_prob) == 0.0


# --------------------------------------------------------------------------- #
# Isolation from an outer inference model
# --------------------------------------------------------------------------- #
def _outer_model(observed: jax.Array) -> None:
    """An inference model that evaluates a population twice."""
    hubble_constant = numpyro.sample("H0", dist.Uniform(20.0, 140.0))
    params = {**FIDUCIALS, "H0": hubble_constant}
    total = jnp.zeros(())
    for _ in range(2):
        trace = mock_target_model().evaluate(params, sample_values())
        total = total + jnp.sum(trace.log_prob)
        total = total + jnp.sum(trace.luminosity_distance)
    numpyro.sample("obs", dist.Normal(total * 1e-6, 1.0), obs=observed)


def test_population_sites_stay_out_of_the_outer_trace() -> None:
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
            trace = mock_target_model().evaluate(params, sample_values())
            total = total + jnp.sum(trace.log_prob)
            total = total + jnp.sum(trace.luminosity_distance)
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
    model: Population,
    params: Mapping[str, ArrayLike],
    redshift: ArrayLike,
) -> jax.Array:
    with handlers.block(), handlers.seed(rng_seed=0):
        trace = handlers.trace(model).get_trace(params)
    site = trace.get(REDSHIFT_SITE)
    if site is None or site["type"] != "sample":
        raise ValueError(
            f"population model declares no {REDSHIFT_SITE!r} sample site; every "
            "population must draw a redshift"
        )
    return jnp.asarray(site["fn"].log_prob(jnp.asarray(redshift)))


def _uniform_mixture_model(uniform_mixing_fraction: float) -> Population:
    return build_population(
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


def test_uniform_mixture_keeps_the_cosmological_distance_and_rate() -> None:
    """The guard changes which redshifts are drawn, not the physics at one."""
    model = _uniform_mixture_model(0.1)
    mixture_trace = model.evaluate(POPULATION_PARAMS, sample_values())
    plain_trace = mock_population_model().evaluate(POPULATION_PARAMS, sample_values())
    np.testing.assert_array_equal(
        mixture_trace.luminosity_distance, plain_trace.luminosity_distance
    )
    assert mixture_trace.total_merger_rate is not None
    assert plain_trace.total_merger_rate is not None
    np.testing.assert_array_equal(
        mixture_trace.total_merger_rate, plain_trace.total_merger_rate
    )


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_uniform_mixture_rejects_an_out_of_range_fraction(fraction: float) -> None:
    model = _uniform_mixture_model(fraction)
    with pytest.raises(ValueError, match="uniform_mixing_fraction"):
        model.evaluate(POPULATION_PARAMS, sample_values())


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def test_generation_is_reproducible() -> None:
    first = mock_population_model().sample(
        jax.random.PRNGKey(7), POPULATION_PARAMS, num_samples=32
    )
    second = mock_population_model().sample(
        jax.random.PRNGKey(7), POPULATION_PARAMS, num_samples=32
    )
    assert set(first) == STOCHASTIC_SITES | (
        DETERMINISTIC_SITES - {TOTAL_MERGER_RATE_SITE}
    )
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_a_smaller_catalog_is_a_prefix_of_a_larger_one() -> None:
    """The property the variable-catalog-size experiment is a series because of.

    ``Predictive`` allocates per-draw keys with ``jax.random.split``, whose
    prefix stability is a property of the installed JAX rather than something
    the API promises, so it is checked here rather than assumed.
    """
    small = mock_population_model().sample(
        jax.random.PRNGKey(7), POPULATION_PARAMS, num_samples=16
    )
    large = mock_population_model().sample(
        jax.random.PRNGKey(7), POPULATION_PARAMS, num_samples=64
    )
    for name, values in small.items():
        np.testing.assert_array_equal(values, large[name][:16])


def test_stored_columns_are_bit_identical_to_a_later_recomputation() -> None:
    """What makes both the exact-zero weights and the load check exact.

    ``Predictive`` runs the model under ``vmap``, one draw at a time, while
    every later evaluation runs it batched. Taking the derived columns from a
    batched pass rather than from ``Predictive`` is what removes that
    difference.
    """
    samples = mock_population_model().sample(
        jax.random.PRNGKey(7), POPULATION_PARAMS, num_samples=32
    )
    stochastic = {name: samples[name] for name in STOCHASTIC_SITES}
    trace = mock_population_model().trace(POPULATION_PARAMS, stochastic)
    for name in DETERMINISTIC_SITES - {TOTAL_MERGER_RATE_SITE}:
        np.testing.assert_array_equal(samples[name], trace[name]["value"])


def test_generation_rejects_a_non_positive_sample_count() -> None:
    with pytest.raises(ValueError, match="num_samples must be positive"):
        mock_population_model().sample(
            jax.random.PRNGKey(7), POPULATION_PARAMS, num_samples=0
        )


# --------------------------------------------------------------------------- #
# JAX transformations
# --------------------------------------------------------------------------- #
def test_evaluation_traces_under_jit_and_vmaps_over_hyperparameters() -> None:
    values = sample_values()

    def redshift_density(hubble_constant: jax.Array) -> jax.Array:
        params = {**OFF_FIDUCIALS, "H0": hubble_constant}
        return mock_target_model().evaluate(params, values).log_prob

    hubble_constants = jnp.array([60.0, 67.66, 75.0])
    batched = jax.jit(jax.vmap(redshift_density))(hubble_constants)
    assert batched.shape == (3, SAMPLE_REDSHIFTS.shape[0])
    np.testing.assert_allclose(
        batched[1], redshift_density(hubble_constants[1]), rtol=1e-12
    )


def test_sampling_is_jittable_and_isolated_from_outer_handlers() -> None:
    model = mock_population_model()
    sample = jax.jit(partial(model.sample, num_samples=8))
    with handlers.trace() as outer:
        sources = sample(jax.random.PRNGKey(7), POPULATION_PARAMS)
    assert outer == {}
    assert set(sources) == set(model.source_sites)
    assert sources["spin_1z"].shape == (8,)
    np.testing.assert_allclose(
        sources["detector_frame_mass_1"],
        sources["source_frame_mass_1"] * (1 + sources["redshift"]),
    )
    np.testing.assert_allclose(
        jax.jit(model.log_prob)(POPULATION_PARAMS, sources),
        model.log_prob(POPULATION_PARAMS, sources),
        rtol=1e-12,
    )


def test_sampling_and_derivation_are_isolated_without_jit() -> None:
    with handlers.trace() as outer:
        mock_population_model().sample(
            jax.random.PRNGKey(7), POPULATION_PARAMS, num_samples=8
        )
    assert outer == {}


# --------------------------------------------------------------------------- #
# Ordered Gaussian masses
# --------------------------------------------------------------------------- #
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


def _gaussian_population_model() -> Population:
    return build_population(
        "bns_md_gaussian_cosmological",
        settings={"z_min": Z_MIN, "z_max": Z_MAX, "n_grid": N_GRID},
    )


def _gaussian_target_model() -> Population:
    return build_population(
        "bns_md_gaussian_modified_propagation",
        settings={"z_min": Z_MIN, "z_max": Z_MAX, "n_grid": N_GRID},
    )


def _gaussian_mixture_model(uniform_mixing_fraction: float) -> Population:
    return build_population(
        "bns_md_gaussian_uniform_mixture",
        settings={
            "z_min": Z_MIN,
            "z_max": Z_MAX,
            "n_grid": N_GRID,
            "uniform_mixing_fraction": uniform_mixing_fraction,
        },
    )


def _mass_only(model: Population) -> Population:
    return replace(model, density_sites=("source_frame_mass_1", "source_frame_mass_2"))


def test_gaussian_mass_density_matches_two_iid_normals_on_the_ordered_half_plane() -> (
    None
):
    values = sample_values()
    actual = (
        _mass_only(_gaussian_population_model())
        .evaluate(GAUSSIAN_PARAMS, values)
        .log_prob
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
        return (
            _mass_only(_gaussian_population_model())
            .evaluate(GAUSSIAN_PARAMS, sources)
            .log_prob
        )

    assert np.all(np.isneginf(np.asarray(jax.jit(unordered_log_prob)(unordered))))


def test_gaussian_draws_are_ordered_and_have_finite_self_density() -> None:
    sources = _gaussian_population_model().sample(
        jax.random.PRNGKey(7), GAUSSIAN_PARAMS, num_samples=64
    )
    np.testing.assert_array_equal(
        sources["source_frame_mass_1"] >= sources["source_frame_mass_2"],
        jnp.ones(64, dtype=bool),
    )
    log_prob = _gaussian_population_model().log_prob(GAUSSIAN_PARAMS, sources)
    assert np.all(np.isfinite(np.asarray(log_prob)))


def test_gaussian_mass_density_stays_finite_when_hyperparameters_move() -> None:
    """The NUTS property: moving (mean, sigma) never zeros a catalog sample.

    The ordered-uniform triangle still drops out the moment a sample exits
    the support, which is the reviewer's concern this model exists to answer.
    """
    values = sample_values()
    shifted = {**GAUSSIAN_PARAMS, "mass_mean": 2.0, "mass_sigma": 0.2}
    gaussian_log_prob = (
        _mass_only(_gaussian_population_model()).evaluate(shifted, values).log_prob
    )
    assert np.all(np.isfinite(np.asarray(gaussian_log_prob)))

    def total(mass_mean: jax.Array, mass_sigma: jax.Array) -> jax.Array:
        params = {
            **GAUSSIAN_PARAMS,
            "mass_mean": mass_mean,
            "mass_sigma": mass_sigma,
        }
        return jnp.sum(
            _mass_only(_gaussian_population_model()).evaluate(params, values).log_prob
        )

    d_mean, d_sigma = jax.grad(total, argnums=(0, 1))(
        jnp.asarray(GAUSSIAN_PARAMS["mass_mean"]),
        jnp.asarray(GAUSSIAN_PARAMS["mass_sigma"]),
    )
    assert np.isfinite(float(d_mean))
    assert np.isfinite(float(d_sigma))

    outside_uniform = {**POPULATION_PARAMS, "minimum_mass": 1.35}

    def uniform_log_prob(params: Mapping[str, ArrayLike]) -> jax.Array:
        return _mass_only(mock_population_model()).evaluate(params, values).log_prob

    assert np.all(np.isneginf(np.asarray(jax.jit(uniform_log_prob)(outside_uniform))))


def test_gaussian_modified_propagation_reduces_exactly_to_the_cosmological_model() -> (
    None
):
    values = sample_values()
    cosmological = _gaussian_population_model().evaluate(GAUSSIAN_PARAMS, values)
    modified = _gaussian_target_model().evaluate(GAUSSIAN_FIDUCIALS, values)
    assert GAUSSIAN_FIDUCIALS["xi_0"] == 1.0
    np.testing.assert_array_equal(cosmological.log_prob, modified.log_prob)
    np.testing.assert_array_equal(
        cosmological.luminosity_distance, modified.luminosity_distance
    )


def test_gaussian_and_uniform_models_share_distance_and_rate() -> None:
    values = sample_values()
    gaussian = _gaussian_population_model().evaluate(GAUSSIAN_PARAMS, values)
    uniform = mock_population_model().evaluate(POPULATION_PARAMS, values)
    np.testing.assert_array_equal(
        gaussian.luminosity_distance, uniform.luminosity_distance
    )
    assert gaussian.total_merger_rate is not None
    assert uniform.total_merger_rate is not None
    np.testing.assert_array_equal(gaussian.total_merger_rate, uniform.total_merger_rate)


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
