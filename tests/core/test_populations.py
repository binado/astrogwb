"""Tests for source models declared as NumPyro models, and their evaluation.

The physics is checked against
:func:`reference_population.reference_merger_rate_distance_and_logprob`, a
hand-written grid-level restatement independent of the model declaration.

Handler isolation and conditioning are checked too: when wrong, sites leak
into an outer joint density or a factor drops out of a ratio, silently.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    mock_population,
)
from jax.typing import ArrayLike
from numpyro import handlers
from reference_population import reference_merger_rate_distance_and_logprob

from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.distributions.delay import PowerLawDelayDistribution
from astrogwb.distributions.redshift import (
    madau_dickinson_time_delayed_redshift_distribution,
)
from astrogwb.populations import (
    DEFAULT_DENSITY_SITES,
    REDSHIFT_SITE,
    BNSCobaPopulation,
    Population,
    joint_model,
)
from astrogwb.utils.numpyro import sample_and_logprob

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
        "inclination",
        "declination",
        "right_ascension",
        "polarization_angle",
        "coa_phase",
        "arrival_time",
    }
)
DETERMINISTIC_SITES = frozenset(
    {
        LUMINOSITY_DISTANCE_SITE,
        "coa_time",
    }
)


def sample_values(redshift: jax.Array = SAMPLE_REDSHIFTS) -> dict[str, jax.Array]:
    """Every stochastic site, with the redshifts under test."""
    ones = jnp.ones_like(redshift)
    return {
        REDSHIFT_SITE: redshift,
        "declination": 0.1 * ones,
        "right_ascension": 0.2 * ones,
        "polarization_angle": 0.3 * ones,
        "coa_phase": 0.4 * ones,
        "arrival_time": 0.2 * ones,
        "source_frame_mass_1": 1.4 * ones,
        "source_frame_mass_2": 1.3 * ones,
        "spin_1z": 0.01 * ones,
        "spin_2z": -0.02 * ones,
        "lambda_1": 400.0 * ones,
        "lambda_2": 300.0 * ones,
        "inclination": (jnp.pi / 3.0) * ones,
    }


def evaluate(
    population: Population,
    params: Mapping[str, ArrayLike],
    values: Mapping[str, ArrayLike],
    density_sites: Sequence[str] = DEFAULT_DENSITY_SITES,
) -> tuple[jax.Array, jax.Array]:
    """Selected ``(N,)`` density and recomputed distance, in one isolated pass."""
    model = joint_model(*population(params))
    outputs, log_prob = sample_and_logprob(
        model, density_sites=density_sites, values=values
    )
    return log_prob, outputs[LUMINOSITY_DISTANCE_SITE]


def merger_rate(population: Population, params: Mapping[str, ArrayLike]) -> jax.Array:
    """The observer-frame rate, from the same call that builds the model."""
    redshift_distribution, _ = population(params)
    return redshift_distribution.total_merger_rate()


GAUSSIAN_PARAMS: dict[str, float] = {
    **{
        name: value
        for name, value in POPULATION_PARAMS.items()
        if name not in {"minimum_mass", "mass_width"}
    },
    "mass_mean": 1.33,
    "mass_sigma": 0.09,
}
MASS_SITES = ("source_frame_mass_1", "source_frame_mass_2")


def _gaussian_population() -> Population:
    return BNSCobaPopulation(
        mass_model="gaussian",
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        n_grid=N_GRID,
    ).build()


#: The generating populations whose draw path runs through the plated replay:
#: the ordered-uniform triangle and the data-dependent truncated Gaussian.
GENERATING_POPULATIONS: dict[str, tuple[Callable[[], Population], dict[str, float]]] = {
    "uniform_mass": (mock_population, POPULATION_PARAMS),
    "gaussian_mass": (_gaussian_population, GAUSSIAN_PARAMS),
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
# Population records
# --------------------------------------------------------------------------- #
#: The time-delayed population: a 20 Myr floor and formation cut off at
#: z = 20, which also caps the delay.
DELAYED = BNSCobaPopulation(
    mass_model="uniform",
    minimum_redshift=Z_MIN,
    maximum_redshift=Z_MAX,
    n_grid=N_GRID,
    time_delay=True,
    minimum_delay=0.02,
    maximum_formation_redshift=20.0,
    n_delay_nodes=48,
)


# --------------------------------------------------------------------------- #
# Explicit site metadata and recomputation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arrival_time_bounds", [(-0.5, 0.5), (-20.0, 40.0)])
def test_joint_model_extrinsics_are_conditionable_sample_sites(
    arrival_time_bounds: tuple[float, float],
) -> None:
    model = joint_model(
        *mock_population()(POPULATION_PARAMS), arrival_time_bounds=arrival_time_bounds
    )
    fixed: dict[str, ArrayLike] = {
        "declination": 0.1,
        "right_ascension": 0.2,
        "polarization_angle": 0.3,
        "coa_phase": 0.4,
        "arrival_time": 0.2,
    }
    conditioned = handlers.condition(model, data=fixed)
    trace = handlers.trace(handlers.seed(conditioned, 0)).get_trace()
    arrival_prior = trace["arrival_time"]["fn"]
    assert isinstance(arrival_prior, dist.Uniform)
    assert arrival_prior.low == arrival_time_bounds[0]
    assert arrival_prior.high == arrival_time_bounds[1]
    for name, value in fixed.items():
        assert trace[name]["type"] == "sample"
        assert trace[name]["is_observed"]
        assert trace[name]["value"] == value


@pytest.mark.parametrize("arrival_time_bounds", [(-0.5, 0.5), (-20.0, 40.0)])
def test_identical_arrival_priors_cancel_from_importance_ratios(
    arrival_time_bounds: tuple[float, float],
) -> None:
    values = sample_values()
    population = mock_population()
    target, _ = evaluate(population, OFF_FIDUCIALS, values)
    reference, _ = evaluate(population, POPULATION_PARAMS, values)
    sites = (*DEFAULT_DENSITY_SITES, "arrival_time")
    _, target_with_arrival = sample_and_logprob(
        joint_model(
            *population(OFF_FIDUCIALS), arrival_time_bounds=arrival_time_bounds
        ),
        values=values,
        density_sites=sites,
    )
    _, reference_with_arrival = sample_and_logprob(
        joint_model(
            *population(POPULATION_PARAMS), arrival_time_bounds=arrival_time_bounds
        ),
        values=values,
        density_sites=sites,
    )
    np.testing.assert_allclose(
        target_with_arrival - reference_with_arrival, target - reference
    )


def test_source_evaluate_needs_no_physical_rate() -> None:
    """Evaluating a density never reads the rate, so a proposal can omit it."""
    without_rate = {
        name: value
        for name, value in POPULATION_PARAMS.items()
        if name != "local_merger_rate"
    }
    _, luminosity_distance = evaluate(mock_population(), without_rate, sample_values())
    assert luminosity_distance.shape == SAMPLE_REDSHIFTS.shape


# --------------------------------------------------------------------------- #
# One call supplies the density, the distance and the rate
# --------------------------------------------------------------------------- #
def test_one_execution_supplies_per_sample_density_distance_and_scalar_rate() -> None:
    log_prob, distance = evaluate(mock_population(), POPULATION_PARAMS, sample_values())
    rate = merger_rate(mock_population(), POPULATION_PARAMS)
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


def test_source_model_never_sees_redshift() -> None:
    _, source_model = mock_population()(POPULATION_PARAMS)
    trace = handlers.trace(handlers.seed(source_model, 0)).get_trace()

    assert not {REDSHIFT_SITE, LUMINOSITY_DISTANCE_SITE} & set(trace)


# --------------------------------------------------------------------------- #
# Modified propagation
# --------------------------------------------------------------------------- #
def test_modified_propagation_scales_the_distance_by_the_gw_em_ratio() -> None:
    _, distance = evaluate(mock_population(), OFF_FIDUCIALS, sample_values())
    _, cosmological_distance, _ = reference(OFF_FIDUCIALS)
    expected = cosmological_distance * jnp.exp(
        log_gw_em_ratio(SAMPLE_REDSHIFTS, OFF_FIDUCIALS["xi_0"], OFF_FIDUCIALS["xi_n"])
    )
    np.testing.assert_allclose(distance, expected, rtol=1e-14)


def test_modified_propagation_reduces_exactly_to_the_cosmological_model() -> None:
    """At xi_0 = 1 the modified model must agree with the plain one bit-for-bit.

    Catalogs are drawn without ``xi_0`` and reweighted with it at ``xi_0 = 1``,
    so anything less than exact would put a floor under the self-proposal log
    weights.
    """
    values = sample_values()
    cosmological_log_prob, cosmological_distance = evaluate(
        mock_population(), POPULATION_PARAMS, values
    )
    modified_log_prob, modified_distance = evaluate(
        mock_population(), FIDUCIALS, values
    )
    assert FIDUCIALS["xi_0"] == 1.0
    np.testing.assert_array_equal(cosmological_log_prob, modified_log_prob)
    np.testing.assert_array_equal(cosmological_distance, modified_distance)


# --------------------------------------------------------------------------- #
# Excluded factors
# --------------------------------------------------------------------------- #
def test_density_selection_preserves_supplied_values_and_deterministics() -> None:
    values = sample_values()
    model = mock_population()
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
            mock_population(), params, sample_values()
        )
        total = total + jnp.sum(log_prob)
        total = total + jnp.sum(luminosity_distance)
    numpyro.sample("obs", dist.Normal(total * 1e-6, 1.0), obs=observed)


def test_outer_density_and_gradient_match_a_direct_calculation() -> None:
    from numpyro.infer.util import log_density

    def direct(hubble_constant: jax.Array) -> jax.Array:
        params = {**FIDUCIALS, "H0": hubble_constant}
        total = jnp.zeros(())
        for _ in range(2):
            log_prob, luminosity_distance = evaluate(
                mock_population(), params, sample_values()
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


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def draw(
    population: Population,
    params: Mapping[str, ArrayLike],
    key: jax.Array,
    num_samples: int,
) -> dict[str, jax.Array]:
    """``num_samples`` sources from ``population`` at ``params``."""
    model = joint_model(*population(params))
    return sample_and_logprob(model, key, num_samples=num_samples, density_sites=())[0]


@pytest.mark.parametrize("generating", GENERATING_POPULATIONS)
def test_generation_is_reproducible(generating: str) -> None:
    build, params = GENERATING_POPULATIONS[generating]
    first = draw(build(), params, jax.random.PRNGKey(7), 32)
    second = draw(build(), params, jax.random.PRNGKey(7), 32)
    assert set(first) == STOCHASTIC_SITES | DETERMINISTIC_SITES
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


@pytest.mark.parametrize("generating", GENERATING_POPULATIONS)
def test_stored_columns_are_bit_identical_to_a_later_recomputation(
    generating: str,
) -> None:
    """Stored columns equal a batched recomputation bit for bit.

    The Gaussian model's data-dependent ``TruncatedNormal`` support is where a
    plated replay could plausibly shift a bit.
    """
    build, params = GENERATING_POPULATIONS[generating]
    samples = draw(build(), params, jax.random.PRNGKey(7), 32)
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
        log_prob, _ = evaluate(mock_population(), params, values)
        return log_prob

    hubble_constants = jnp.array([60.0, 67.66, 75.0])
    batched = jax.jit(jax.vmap(redshift_density))(hubble_constants)
    assert batched.shape == (3, SAMPLE_REDSHIFTS.shape[0])
    np.testing.assert_allclose(
        batched[1], redshift_density(hubble_constants[1]), rtol=1e-12
    )


def test_sampling_is_jittable_and_isolated_from_outer_handlers() -> None:
    population = mock_population()
    sample = jax.jit(partial(draw, population, num_samples=8))
    with handlers.trace() as outer:
        sources = sample(POPULATION_PARAMS, jax.random.PRNGKey(7))
    assert outer == {}
    assert set(sources) == STOCHASTIC_SITES | DETERMINISTIC_SITES
    assert sources["spin_1z"].shape == (8,)

    def log_prob(
        params: Mapping[str, ArrayLike], values: Mapping[str, ArrayLike]
    ) -> jax.Array:
        return evaluate(population, params, values)[0]

    np.testing.assert_allclose(
        jax.jit(log_prob)(POPULATION_PARAMS, sources),
        log_prob(POPULATION_PARAMS, sources),
        rtol=1e-12,
    )


# --------------------------------------------------------------------------- #
# Ordered Gaussian masses
# --------------------------------------------------------------------------- #
def test_gaussian_mass_density_matches_two_iid_normals_on_the_ordered_half_plane() -> (
    None
):
    values = sample_values()
    actual, _ = evaluate(
        _gaussian_population(), GAUSSIAN_PARAMS, values, density_sites=MASS_SITES
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
            _gaussian_population(),
            GAUSSIAN_PARAMS,
            sources,
            density_sites=MASS_SITES,
        )
        return log_prob

    assert np.all(np.isneginf(np.asarray(jax.jit(unordered_log_prob)(unordered))))


def test_gaussian_draws_are_ordered_and_have_finite_self_density() -> None:
    sources = draw(_gaussian_population(), GAUSSIAN_PARAMS, jax.random.PRNGKey(7), 64)
    np.testing.assert_array_equal(
        sources["source_frame_mass_1"] >= sources["source_frame_mass_2"],
        jnp.ones(64, dtype=bool),
    )
    log_prob, _ = evaluate(_gaussian_population(), GAUSSIAN_PARAMS, sources)
    assert np.all(np.isfinite(np.asarray(log_prob)))


def test_gaussian_mass_density_stays_finite_when_hyperparameters_move() -> None:
    """The NUTS property: moving (mean, sigma) never zeros a catalog sample.

    The ordered-uniform triangle, by contrast, drops a sample as soon as it
    exits the support.
    """
    values = sample_values()
    shifted = {**GAUSSIAN_PARAMS, "mass_mean": 2.0, "mass_sigma": 0.2}
    gaussian_log_prob, _ = evaluate(
        _gaussian_population(), shifted, values, density_sites=MASS_SITES
    )
    assert np.all(np.isfinite(np.asarray(gaussian_log_prob)))

    def total(mass_mean: jax.Array, mass_sigma: jax.Array) -> jax.Array:
        params = {
            **GAUSSIAN_PARAMS,
            "mass_mean": mass_mean,
            "mass_sigma": mass_sigma,
        }
        log_prob, _ = evaluate(
            _gaussian_population(), params, values, density_sites=MASS_SITES
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
            mock_population(), params, values, density_sites=MASS_SITES
        )
        return log_prob

    assert np.all(np.isneginf(np.asarray(jax.jit(uniform_log_prob)(outside_uniform))))


# --------------------------------------------------------------------------- #
# Inclination
# --------------------------------------------------------------------------- #
def test_inclination_is_absent_from_the_default_density() -> None:
    """Inclination has no hyperparameters, so it must not enter the weight."""
    values = sample_values()
    moved = {**values, "inclination": jnp.full_like(SAMPLE_REDSHIFTS, 1.2)}
    log_prob, _ = evaluate(mock_population(), POPULATION_PARAMS, values)
    log_moved, _ = evaluate(mock_population(), POPULATION_PARAMS, moved)
    np.testing.assert_array_equal(np.asarray(log_prob), np.asarray(log_moved))


# --------------------------------------------------------------------------- #
# Time-delayed population
# --------------------------------------------------------------------------- #
DELAYED_PARAMS: dict[str, float] = {**POPULATION_PARAMS, "delay_slope": -1.0}


def _delayed_population() -> Population:
    return DELAYED.build()


def test_time_delayed_draws_evaluate_to_a_finite_density_with_a_slope_gradient() -> (
    None
):
    population = _delayed_population()
    samples = draw(population, DELAYED_PARAMS, jax.random.PRNGKey(0), 64)
    assert jnp.all((samples["redshift"] >= Z_MIN) & (samples["redshift"] <= Z_MAX))

    def total(slope: jax.Array) -> jax.Array:
        log_prob, _ = evaluate(
            population, {**DELAYED_PARAMS, "delay_slope": slope}, samples
        )
        return jnp.sum(log_prob)

    # The fiducial slope sits exactly where numpyro's power law special-cases.
    value, gradient = jax.value_and_grad(total)(jnp.asarray(-1.0))
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)
    nearby = jax.grad(total)(jnp.asarray(-1.0 + 1e-10))
    np.testing.assert_allclose(nearby, gradient, rtol=1e-6)


def test_time_delayed_rate_is_linear_in_the_local_rate_but_not_in_h0() -> None:
    """Why the population declares ``local_merger_rate`` and not ``H0``.

    The local rate only rescales. H0 changes lookback time against a delay
    fixed in Gyr, so it reshapes the redshift law: ``R * H0^3`` is no longer
    constant, which is what the H0 amplitude scaling assumes.
    """
    delayed = _delayed_population()

    def rate(params: Mapping[str, ArrayLike]) -> jax.Array:
        return merger_rate(delayed, params)

    base = rate(DELAYED_PARAMS)
    scaled = rate(
        {
            **DELAYED_PARAMS,
            "local_merger_rate": 3.0 * DELAYED_PARAMS["local_merger_rate"],
        }
    )
    np.testing.assert_allclose(scaled, 3.0 * base, rtol=1e-12)

    h0 = DELAYED_PARAMS["H0"]
    moved = rate({**DELAYED_PARAMS, "H0": 1.2 * h0})
    assert not np.isclose(moved * (1.2 * h0) ** 3, base * h0**3, rtol=1e-3)


@pytest.mark.parametrize("slope", [-1.5, -1.0, 0.5])
def test_time_delayed_ceiling_is_cosmological_not_a_fixed_number(slope: float) -> None:
    """A delay longer than the lookback time to the cut-off is never available.

    So the population's ceiling, t_L(z_cut), must give the same rate as an
    effectively unbounded one: a higher ceiling only rescales the delay CDF,
    which leaves every quadrature node in tau where it was. The H0 gradient
    runs through the ceiling as well as through the cosmology.
    """
    params = {**DELAYED_PARAMS, "delay_slope": slope}
    delayed = _delayed_population()

    def rate(values: Mapping[str, ArrayLike]) -> jax.Array:
        return merger_rate(delayed, values)

    unbounded = madau_dickinson_time_delayed_redshift_distribution(
        params=params,
        time_delay_distribution=PowerLawDelayDistribution(
            slope, DELAYED.minimum_delay, 1.0e3
        ),
        n_delay_nodes=48,
        maximum_formation_redshift=20.0,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        n_grid=N_GRID,
    )
    np.testing.assert_allclose(rate(params), unbounded.total_merger_rate(), rtol=1e-10)

    gradient = jax.grad(lambda h0: rate({**params, "H0": h0}))(params["H0"])
    assert jnp.isfinite(gradient)
