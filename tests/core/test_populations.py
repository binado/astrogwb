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
    make_redshift_grid,
    mock_population_model,
    mock_target_model,
)
from numpyro import handlers
from reference_population import reference_merger_rate_distance_and_logprob

from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.populations import (
    LUMINOSITY_DISTANCE_SITE,
    REDSHIFT_SITE,
    TOTAL_MERGER_RATE_SITE,
    bns_md_cosmological,
    bns_md_modified_propagation,
    bns_md_uniform_mixture,
    derive_source_columns,
    draw_population,
    known_population_models,
    population_log_probs,
    population_model,
    population_sites,
    redshift_log_density,
    register_population_model,
    required_deterministic,
    select_stochastic_values,
)

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
        params, SAMPLE_REDSHIFTS, redshift_grid=make_redshift_grid()
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_shipped_models_are_registered() -> None:
    assert known_population_models() == (
        "bns_md_cosmological",
        "bns_md_modified_propagation",
        "bns_md_uniform_mixture",
    )
    assert population_model("bns_md_cosmological") is bns_md_cosmological
    assert population_model("bns_md_modified_propagation") is (
        bns_md_modified_propagation
    )
    assert population_model("bns_md_uniform_mixture") is bns_md_uniform_mixture


def test_unknown_model_names_list_the_known_set() -> None:
    with pytest.raises(KeyError, match="bns_md_cosmological"):
        population_model("no_such_population")


def test_registering_a_name_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_population_model("bns_md_cosmological")(lambda params: None)


# --------------------------------------------------------------------------- #
# Site discovery and selection
# --------------------------------------------------------------------------- #
def test_site_discovery_lists_exactly_the_declared_sites() -> None:
    sites = population_sites(mock_population_model(), POPULATION_PARAMS)
    assert sites.stochastic == STOCHASTIC_SITES
    assert sites.deterministic == DETERMINISTIC_SITES


def test_total_merger_rate_is_declared_only_with_a_physical_rate() -> None:
    """A proposal is a density, not an observation, so its rate is optional."""
    without_rate = {
        name: value
        for name, value in POPULATION_PARAMS.items()
        if name != "local_merger_rate"
    }
    sites = population_sites(mock_population_model(), without_rate)
    assert TOTAL_MERGER_RATE_SITE not in sites.deterministic
    assert LUMINOSITY_DISTANCE_SITE in sites.deterministic


def test_selection_drops_derived_columns_and_demands_stochastic_ones() -> None:
    sites = population_sites(mock_population_model(), POPULATION_PARAMS)
    stored = {
        **sample_values(),
        # A derived column a real catalog also stores. Substituting it would
        # overwrite the value the model recomputes and defeat the check that
        # catches a catalog drifted from its population.
        LUMINOSITY_DISTANCE_SITE: jnp.ones_like(SAMPLE_REDSHIFTS),
    }
    selected = select_stochastic_values(stored, sites, label="catalog")
    assert set(selected) == STOCHASTIC_SITES

    del stored["spin_1z"]
    with pytest.raises(ValueError, match="spin_1z"):
        select_stochastic_values(stored, sites, label="catalog")


# --------------------------------------------------------------------------- #
# One execution supplies density, distance, and rate
# --------------------------------------------------------------------------- #
def test_one_execution_supplies_per_sample_density_distance_and_scalar_rate() -> None:
    site_log_probs, trace = population_log_probs(
        mock_population_model(), POPULATION_PARAMS, sample_values()
    )
    assert set(site_log_probs) == STOCHASTIC_SITES
    for name, value in site_log_probs.items():
        assert value.shape == SAMPLE_REDSHIFTS.shape, name

    distance = required_deterministic(
        trace, LUMINOSITY_DISTANCE_SITE, ndim=1, label="population"
    )
    rate = required_deterministic(
        trace, TOTAL_MERGER_RATE_SITE, ndim=0, label="population"
    )
    assert distance.shape == SAMPLE_REDSHIFTS.shape
    assert rate.shape == ()

    expected_rate, expected_distance, expected_logpdf = reference(FIDUCIALS)
    # Bit-exact: the distribution shares the reference's operation order, which
    # is what keeps a catalog that is its own proposal at exactly zero weight.
    np.testing.assert_array_equal(
        np.asarray(site_log_probs[REDSHIFT_SITE]), np.asarray(expected_logpdf)
    )
    np.testing.assert_allclose(distance, expected_distance, rtol=1e-14)
    np.testing.assert_allclose(float(rate), float(expected_rate), rtol=1e-15)


def test_rate_and_distance_deterministics_add_no_density_factors() -> None:
    """Deterministic sites carry no log density, so they cannot bias a ratio."""
    with_rate, _ = population_log_probs(
        mock_population_model(), POPULATION_PARAMS, sample_values()
    )
    without_rate_params = {
        name: value
        for name, value in POPULATION_PARAMS.items()
        if name != "local_merger_rate"
    }
    without_rate, _ = population_log_probs(
        mock_population_model(), without_rate_params, sample_values()
    )
    assert set(with_rate) == set(without_rate)
    for name in with_rate:
        np.testing.assert_array_equal(with_rate[name], without_rate[name])


def test_missing_required_deterministic_is_rejected() -> None:
    without_rate = {
        name: value
        for name, value in POPULATION_PARAMS.items()
        if name != "local_merger_rate"
    }
    _, trace = population_log_probs(
        mock_population_model(), without_rate, sample_values()
    )
    with pytest.raises(ValueError, match=TOTAL_MERGER_RATE_SITE):
        required_deterministic(
            trace, TOTAL_MERGER_RATE_SITE, ndim=0, label="population"
        )


def test_wrongly_shaped_deterministic_is_rejected() -> None:
    _, trace = population_log_probs(
        mock_population_model(), POPULATION_PARAMS, sample_values()
    )
    with pytest.raises(ValueError, match="1 dimension"):
        required_deterministic(
            trace, TOTAL_MERGER_RATE_SITE, ndim=1, label="population"
        )


def test_derived_columns_match_the_declared_transforms() -> None:
    values = sample_values()
    columns = derive_source_columns(mock_population_model(), POPULATION_PARAMS, values)
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
    _, trace = population_log_probs(mock_target_model(), OFF_FIDUCIALS, sample_values())
    distance = trace[LUMINOSITY_DISTANCE_SITE]["value"]
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
    cosmological, cosmological_trace = population_log_probs(
        mock_population_model(), POPULATION_PARAMS, values
    )
    modified, modified_trace = population_log_probs(
        mock_target_model(), FIDUCIALS, values
    )
    assert FIDUCIALS["xi_0"] == 1.0
    for name in cosmological:
        np.testing.assert_array_equal(cosmological[name], modified[name])
    np.testing.assert_array_equal(
        cosmological_trace[LUMINOSITY_DISTANCE_SITE]["value"],
        modified_trace[LUMINOSITY_DISTANCE_SITE]["value"],
    )


# --------------------------------------------------------------------------- #
# Excluded factors
# --------------------------------------------------------------------------- #
def test_excluding_factors_preserves_supplied_values_and_deterministics() -> None:
    site_log_probs, trace = population_log_probs(
        mock_population_model(), POPULATION_PARAMS, sample_values()
    )
    excluded = frozenset({"source_frame_mass_1", "spin_1z", "lambda_2"})
    filtered_probs, filtered_trace = population_log_probs(
        mock_population_model(),
        POPULATION_PARAMS,
        sample_values(),
        hidden_sites=excluded,
    )
    assert set(filtered_probs) == set(site_log_probs) - excluded
    assert set(filtered_trace) == set(trace) - excluded
    for name, value in filtered_probs.items():
        assert value.shape == SAMPLE_REDSHIFTS.shape
        np.testing.assert_array_equal(value, site_log_probs[name])
    for name, site in filtered_trace.items():
        np.testing.assert_array_equal(site["value"], trace[name]["value"])


# --------------------------------------------------------------------------- #
# Isolation from an outer inference model
# --------------------------------------------------------------------------- #
def _outer_model(observed: jax.Array) -> None:
    """An inference model that evaluates a population twice."""
    hubble_constant = numpyro.sample("H0", dist.Uniform(20.0, 140.0))
    params = {**FIDUCIALS, "H0": hubble_constant}
    total = jnp.zeros(())
    for _ in range(2):
        site_log_probs, trace = population_log_probs(
            mock_target_model(),
            params,
            sample_values(),
            hidden_sites=frozenset({"source_frame_mass_1"}),
        )
        total = total + jnp.sum(site_log_probs[REDSHIFT_SITE])
        total = total + jnp.sum(trace[LUMINOSITY_DISTANCE_SITE]["value"])
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
            site_log_probs, trace = population_log_probs(
                mock_target_model(), params, sample_values()
            )
            total = total + jnp.sum(site_log_probs[REDSHIFT_SITE])
            total = total + jnp.sum(trace[LUMINOSITY_DISTANCE_SITE]["value"])
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
def test_uniform_mixture_matches_the_explicit_logaddexp_proposal() -> None:
    epsilon = 0.1
    model = partial(
        bns_md_uniform_mixture,
        z_min=Z_MIN,
        z_max=Z_MAX,
        n_grid=N_GRID,
        uniform_mixing_fraction=epsilon,
    )
    actual = redshift_log_density(model, POPULATION_PARAMS, SAMPLE_REDSHIFTS)

    _, _, md_logprob = reference(FIDUCIALS)
    expected = jnp.logaddexp(
        jnp.log1p(-epsilon) + md_logprob,
        jnp.log(epsilon) - jnp.log(Z_MAX - Z_MIN),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13)

    outside = jnp.array([Z_MIN - 0.1, Z_MAX + 0.1])
    assert np.all(
        np.isneginf(np.asarray(redshift_log_density(model, POPULATION_PARAMS, outside)))
    )


def test_uniform_mixture_keeps_the_cosmological_distance_and_rate() -> None:
    """The guard changes which redshifts are drawn, not the physics at one."""
    model = partial(
        bns_md_uniform_mixture,
        z_min=Z_MIN,
        z_max=Z_MAX,
        n_grid=N_GRID,
        uniform_mixing_fraction=0.1,
    )
    _, mixture_trace = population_log_probs(model, POPULATION_PARAMS, sample_values())
    _, plain_trace = population_log_probs(
        mock_population_model(), POPULATION_PARAMS, sample_values()
    )
    for name in (LUMINOSITY_DISTANCE_SITE, TOTAL_MERGER_RATE_SITE):
        np.testing.assert_array_equal(
            mixture_trace[name]["value"], plain_trace[name]["value"]
        )


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_uniform_mixture_rejects_an_out_of_range_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="uniform_mixing_fraction"):
        population_sites(
            partial(
                bns_md_uniform_mixture,
                z_min=Z_MIN,
                z_max=Z_MAX,
                n_grid=N_GRID,
                uniform_mixing_fraction=fraction,
            ),
            POPULATION_PARAMS,
        )


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def test_generation_is_reproducible() -> None:
    first = draw_population(
        mock_population_model(), POPULATION_PARAMS, num_samples=32, seed=7
    )
    second = draw_population(
        mock_population_model(), POPULATION_PARAMS, num_samples=32, seed=7
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
    small = draw_population(
        mock_population_model(), POPULATION_PARAMS, num_samples=16, seed=7
    )
    large = draw_population(
        mock_population_model(), POPULATION_PARAMS, num_samples=64, seed=7
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
    samples = draw_population(
        mock_population_model(), POPULATION_PARAMS, num_samples=32, seed=7
    )
    sites = population_sites(mock_population_model(), POPULATION_PARAMS)
    stochastic = {name: samples[name] for name in sites.stochastic}
    _, trace = population_log_probs(
        mock_population_model(), POPULATION_PARAMS, stochastic
    )
    for name in sites.deterministic - {TOTAL_MERGER_RATE_SITE}:
        np.testing.assert_array_equal(samples[name], trace[name]["value"])


def test_generation_rejects_a_non_positive_sample_count() -> None:
    with pytest.raises(ValueError, match="num_samples must be positive"):
        draw_population(
            mock_population_model(), POPULATION_PARAMS, num_samples=0, seed=7
        )


# --------------------------------------------------------------------------- #
# JAX transformations
# --------------------------------------------------------------------------- #
def test_evaluation_traces_under_jit_and_vmaps_over_hyperparameters() -> None:
    values = sample_values()

    def redshift_density(hubble_constant: jax.Array) -> jax.Array:
        params = {**OFF_FIDUCIALS, "H0": hubble_constant}
        site_log_probs, _ = population_log_probs(mock_target_model(), params, values)
        return site_log_probs[REDSHIFT_SITE]

    hubble_constants = jnp.array([60.0, 67.66, 75.0])
    batched = jax.jit(jax.vmap(redshift_density))(hubble_constants)
    assert batched.shape == (3, SAMPLE_REDSHIFTS.shape[0])
    np.testing.assert_allclose(
        batched[1], redshift_density(hubble_constants[1]), rtol=1e-12
    )
