"""Tests for the population-based importance-sampling API.

The generic pieces (:class:`Population`, :class:`PopulationTerms`,
:func:`importance_log_weights`) are checked on toy distributions. The BNS
realization is checked against
:func:`compute_merger_rate_distance_and_logprob`, the hand-written grid-level
formula -- kept deliberately as an independent restatement, so the class-based
path cannot drift without a failure here. The mixture test at the end is the
viability check for expressing an MD + uniform proposal natively.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb_mock_population import FIDUCIALS, N_GRID, Z_MAX, Z_MIN, make_redshift_grid
from jax.typing import ArrayLike
from numpyro.distributions import constraints

from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.distributions.redshift.madau_dickinson import (
    MadauDickinsonRedshiftDistribution,
)
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    bns_population,
    compute_merger_rate_distance_and_logprob,
)
from astrogwb.importance.weights import importance_log_weights
from astrogwb.population import (
    CosmologicalPopulation,
    Population,
    PopulationTerms,
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


# --------------------------------------------------------------------------- #
# Population
# --------------------------------------------------------------------------- #
def _standard_population() -> CosmologicalPopulation:
    modified = bns_population(FIDUCIALS, redshift_grid=make_redshift_grid())
    return CosmologicalPopulation(
        distributions=modified.distributions,
        params={
            key: value
            for key, value in FIDUCIALS.items()
            if key not in {"xi_0", "xi_n"}
        },
    )


def test_population_base_is_abstract() -> None:
    assert inspect.isabstract(Population)


def test_log_prob_sums_the_per_parameter_log_densities() -> None:
    redshift = _standard_population().redshift_distribution
    population = CosmologicalPopulation(
        distributions={"redshift": redshift, "a": dist.Normal(0.0, 1.0)},
        params={},
    )
    source_parameters = {"redshift": SAMPLE_REDSHIFTS, "a": jnp.arange(5.0)}
    expected = redshift.log_prob(SAMPLE_REDSHIFTS) + dist.Normal(0.0, 1.0).log_prob(
        source_parameters["a"]
    )
    np.testing.assert_allclose(population.log_prob(source_parameters), expected)


def test_log_prob_ignores_source_parameters_without_a_distribution() -> None:
    population = _standard_population()
    np.testing.assert_array_equal(
        population.log_prob({"redshift": SAMPLE_REDSHIFTS, "mass": jnp.ones(5)}),
        population.redshift_distribution.log_prob(SAMPLE_REDSHIFTS),
    )


def test_population_requires_redshift() -> None:
    with pytest.raises(ValueError, match="requires a redshift"):
        CosmologicalPopulation(distributions={}, params={})


def test_population_requires_a_redshift_distribution() -> None:
    with pytest.raises(TypeError, match="RedshiftDistribution"):
        CosmologicalPopulation(
            distributions={"redshift": dist.Normal(0.0, 1.0)}, params={}
        )


def test_log_prob_raises_on_a_missing_source_parameter() -> None:
    population = _standard_population()
    with pytest.raises(KeyError, match="redshift"):
        population.log_prob({"mass": jnp.zeros(2)})


def test_standard_population_needs_no_propagation_parameters() -> None:
    population = _standard_population()
    terms = population.compute_population_terms({"redshift": SAMPLE_REDSHIFTS})
    rate, distance, logpdf = _reference(FIDUCIALS)
    np.testing.assert_allclose(terms.total_merger_rate, rate, rtol=1e-15)
    np.testing.assert_array_equal(terms.log_prob, logpdf)
    np.testing.assert_array_equal(terms.log_luminosity_distance, jnp.log(distance))


def test_base_does_not_prescribe_rate_parameters() -> None:
    @jax.tree_util.register_dataclass
    @dataclass(frozen=True)
    class UnitRatePopulation(Population):
        def luminosity_distance(self, redshift: ArrayLike) -> jax.Array:
            return self.redshift_distribution.luminosity_distance(redshift)

        def total_merger_rate(self) -> jax.Array:
            return jnp.asarray(1.0)

    population = UnitRatePopulation(_standard_population().distributions, params={})
    terms = population.compute_population_terms({"redshift": SAMPLE_REDSHIFTS})
    assert float(terms.total_merger_rate) == 1.0


@pytest.mark.parametrize("modified", [False, True])
def test_population_is_a_pytree_of_distributions_and_parameters(modified: bool) -> None:
    population = (
        bns_population(OFF_FIDUCIALS, redshift_grid=make_redshift_grid())
        if modified
        else _standard_population()
    )
    leaves, structure = jax.tree.flatten(population)
    assert len(leaves) == len(jax.tree.leaves(population.distributions)) + len(
        population.params
    )
    rebuilt = jax.tree.unflatten(structure, leaves)
    assert type(rebuilt) is type(population)
    samples = {"redshift": SAMPLE_REDSHIFTS}
    for actual, expected in zip(
        rebuilt.compute_population_terms(samples),
        population.compute_population_terms(samples),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_population_sample_splits_one_key_per_distribution() -> None:
    standard = _standard_population()
    population = CosmologicalPopulation(
        distributions={
            "redshift": standard.redshift_distribution,
            "uniform": dist.Uniform(-2.0, 3.0),
            "normal": dist.Normal(4.0, 0.5),
        },
        params=standard.params,
    )
    key = jax.random.key(19)
    sample_shape = (7,)

    actual = population.sample(key, sample_shape)
    repeated = population.sample(key, sample_shape)
    split_keys = jax.random.split(key, len(population.distributions))
    expected = {
        name: distribution.sample(parameter_key, sample_shape)
        for (name, distribution), parameter_key in zip(
            population.distributions.items(), split_keys, strict=True
        )
    }

    assert list(actual) == ["redshift", "uniform", "normal"]
    for name in actual:
        assert actual[name].shape == sample_shape
        np.testing.assert_array_equal(actual[name], repeated[name])
        np.testing.assert_array_equal(actual[name], expected[name])


def test_population_converts_supported_distributions_to_gwmock_graph() -> None:
    standard = _standard_population()
    population = CosmologicalPopulation(
        distributions={
            "redshift": standard.redshift_distribution,
            "uniform": dist.Uniform(-2.0, 3.0),
            "normal": dist.Normal(4.0, 0.5),
        },
        params=standard.params,
    )

    assert population._to_gwmock_population_graph() == {
        "redshift": {
            "sampler": {
                "function": "madau_dickinson_redshift",
                "arguments": {
                    "z_min": Z_MIN,
                    "z_max": Z_MAX,
                    "gamma": FIDUCIALS["gamma"],
                    "kappa": FIDUCIALS["kappa"],
                    "z_peak": FIDUCIALS["z_peak"],
                    "hubble_constant": FIDUCIALS["H0"],
                    "omega_m": FIDUCIALS["Omega_m"],
                    "n_grid": N_GRID,
                },
            }
        },
        "uniform": {
            "sampler": {
                "function": "uniform",
                "arguments": {"minimum": -2.0, "maximum": 3.0},
            }
        },
        "normal": {
            "sampler": {
                "function": "astrogwb.population._gwmock_normal",
                "arguments": {"loc": 4.0, "scale": 0.5},
            }
        },
    }


def test_population_gwmock_graph_rejects_an_unsupported_distribution() -> None:
    standard = _standard_population()
    population = CosmologicalPopulation(
        distributions={
            "redshift": standard.redshift_distribution,
            "unsupported": dist.Exponential(1.0),
        },
        params=standard.params,
    )

    with pytest.raises(TypeError, match="'unsupported'.*Exponential"):
        population._to_gwmock_population_graph()


@pytest.mark.parametrize(
    "distribution",
    [
        dist.Normal(jnp.zeros(2), jnp.ones(2)),
        dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1),
    ],
    ids=["batch", "event"],
)
def test_population_gwmock_graph_rejects_non_scalar_distributions(
    distribution: dist.Distribution,
) -> None:
    standard = _standard_population()
    population = CosmologicalPopulation(
        distributions={
            "redshift": standard.redshift_distribution,
            "non_scalar": distribution,
        },
        params=standard.params,
    )

    with pytest.raises(ValueError, match="'non_scalar'.*batch_shape.*event_shape"):
        population._to_gwmock_population_graph()


def test_population_gwmock_graph_requires_madau_dickinson_parameters() -> None:
    standard = _standard_population()
    population = CosmologicalPopulation(
        distributions=standard.distributions,
        params={"local_merger_rate": FIDUCIALS["local_merger_rate"]},
    )

    with pytest.raises(ValueError, match=r"Population\.params keys.*H0"):
        population._to_gwmock_population_graph()


def test_population_gwmock_graph_requires_concrete_values() -> None:
    standard = _standard_population()

    @jax.jit
    def convert(low: jax.Array) -> jax.Array:
        population = CosmologicalPopulation(
            distributions={
                "uniform": dist.Uniform(low, 3.0),
                "redshift": standard.redshift_distribution,
            },
            params=standard.params,
        )
        population._to_gwmock_population_graph()
        return low

    with pytest.raises(ValueError, match="'uniform' low must be a concrete scalar"):
        convert(jnp.asarray(-2.0))


@pytest.mark.integration
def test_gwmock_graph_simulates_every_supported_distribution() -> None:
    from gwmock_pop import GraphSimulator

    standard = _standard_population()
    population = CosmologicalPopulation(
        distributions={
            "redshift": standard.redshift_distribution,
            "uniform": dist.Uniform(-2.0, 3.0),
            "normal": dist.Normal(4.0, 0.5),
        },
        params=standard.params,
    )

    simulator = GraphSimulator(
        population._to_gwmock_population_graph(), source_type="bns", seed=29
    )
    samples = simulator.simulate(32)

    assert list(samples) == ["redshift", "uniform", "normal"]
    assert all(sample.shape == (32,) for sample in samples.values())
    assert np.isfinite(np.asarray(samples["normal"])).all()


# --------------------------------------------------------------------------- #
# importance_log_weights
# --------------------------------------------------------------------------- #
def _terms(log_prob: list[float], log_distance: list[float]) -> PopulationTerms:
    return PopulationTerms(
        log_prob=jnp.array(log_prob),
        log_luminosity_distance=jnp.array(log_distance),
        total_merger_rate=jnp.array(1.0),
    )


def test_identical_terms_give_exactly_zero_log_weights() -> None:
    terms = _terms([-1.0, -2.5], [7.0, 7.5])
    np.testing.assert_array_equal(
        np.asarray(
            importance_log_weights(
                terms,
                proposal_log_prob=terms.log_prob,
                log_reference_distance=terms.log_luminosity_distance,
            )
        ),
        np.zeros(2),
    )


def test_log_weights_are_the_density_ratio_minus_twice_the_distance_ratio() -> None:
    target = _terms([-1.0, -2.0], [7.0, 7.5])
    proposal = _terms([-1.5, -1.0], [6.8, 7.7])

    expected = np.array([-1.0 + 1.5, -2.0 + 1.0]) - 2.0 * np.array(
        [7.0 - 6.8, 7.5 - 7.7]
    )
    np.testing.assert_allclose(
        np.asarray(
            importance_log_weights(
                target,
                proposal_log_prob=proposal.log_prob,
                log_reference_distance=proposal.log_luminosity_distance,
            )
        ),
        expected,
        rtol=1e-14,
    )


# --------------------------------------------------------------------------- #
# The BNS realization against the grid-level reference
# --------------------------------------------------------------------------- #
def _reference(params: dict[str, float]) -> tuple[jax.Array, jax.Array, jax.Array]:
    return compute_merger_rate_distance_and_logprob(
        params, {"redshift": SAMPLE_REDSHIFTS}, redshift_grid=make_redshift_grid()
    )


@pytest.mark.parametrize("params", [FIDUCIALS, OFF_FIDUCIALS], ids=["fiducial", "off"])
def test_bns_population_matches_the_reference_formula(
    params: dict[str, float],
) -> None:
    population = bns_population(params, redshift_grid=make_redshift_grid())
    samples = {"redshift": SAMPLE_REDSHIFTS}
    terms = population.compute_population_terms(samples)
    np.testing.assert_allclose(
        population.luminosity_distance(SAMPLE_REDSHIFTS),
        jnp.exp(terms.log_luminosity_distance),
        rtol=1e-14,
    )

    reference_rate, reference_distance, reference_logpdf = _reference(params)
    # Bit-exact, as in `tests/core/test_distributions.py`: the distribution
    # route shares the reference's operation order.
    np.testing.assert_array_equal(
        np.asarray(terms.log_prob), np.asarray(reference_logpdf)
    )
    np.testing.assert_allclose(
        float(terms.total_merger_rate), float(reference_rate), rtol=1e-15
    )
    expected_log_luminosity_distance = jnp.log(reference_distance) + log_gw_em_ratio(
        SAMPLE_REDSHIFTS, params["xi_0"], params["xi_n"]
    )
    np.testing.assert_allclose(
        np.asarray(terms.log_luminosity_distance),
        np.asarray(expected_log_luminosity_distance),
        rtol=1e-14,
    )


def test_bns_population_rebuilds_the_shared_grid_bit_exactly() -> None:
    population = bns_population(FIDUCIALS, redshift_grid=make_redshift_grid())
    redshift = population.distributions["redshift"]
    assert isinstance(redshift, MadauDickinsonRedshiftDistribution)
    np.testing.assert_array_equal(
        np.asarray(redshift.redshift_grid), np.asarray(make_redshift_grid())
    )


# --------------------------------------------------------------------------- #
# Regression: the class-based weights reproduce the explicit formula
# --------------------------------------------------------------------------- #
def test_population_weights_match_the_explicit_formula_off_the_fiducials() -> None:
    """The grid-level formula, restated in full, against the class-based path.

    Written out rather than reusing ``compute_population_terms``: an expected
    value produced by the code under test proves nothing. The catalog here is
    its own proposal at ``FIDUCIALS``, so the same expression also pins the
    exactly-zero fiducial weights the rest of the suite depends on.
    """
    redshift_grid = make_redshift_grid()
    samples = {"redshift": jnp.linspace(Z_MIN, Z_MAX, 16)}
    _, fiducial_distance, proposal_logprob = compute_merger_rate_distance_and_logprob(
        FIDUCIALS, samples, redshift_grid=redshift_grid
    )
    log_reference_distance = jnp.log(fiducial_distance) + log_gw_em_ratio(
        samples["redshift"], FIDUCIALS["xi_0"], FIDUCIALS["xi_n"]
    )

    def weights_at(params: dict[str, float]) -> tuple[jax.Array, jax.Array]:
        terms = bns_population(
            params, redshift_grid=redshift_grid
        ).compute_population_terms(samples)
        return terms.total_merger_rate, importance_log_weights(
            terms,
            proposal_log_prob=proposal_logprob,
            log_reference_distance=log_reference_distance,
        )

    total_rate, actual = weights_at(OFF_FIDUCIALS)

    expected_rate, target_distance, target_logprob = (
        compute_merger_rate_distance_and_logprob(
            OFF_FIDUCIALS, samples, redshift_grid=redshift_grid
        )
    )
    z = samples["redshift"]
    logdiff_distance = (
        jnp.log(target_distance)
        - jnp.log(fiducial_distance)
        + log_gw_em_ratio(z, OFF_FIDUCIALS["xi_0"], OFF_FIDUCIALS["xi_n"])
        - log_gw_em_ratio(z, FIDUCIALS["xi_0"], FIDUCIALS["xi_n"])
    )
    expected = target_logprob - proposal_logprob - 2.0 * logdiff_distance

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12)
    np.testing.assert_allclose(float(total_rate), float(expected_rate), rtol=1e-15)
    # And the premise the whole suite rests on: at the fiducials the catalog
    # is its own proposal and every log-weight is *exactly* zero.
    _, at_fiducials = weights_at(FIDUCIALS)
    np.testing.assert_array_equal(np.asarray(at_fiducials), 0.0)
    # The point is non-trivial: the weights are far from zero here.
    assert np.max(np.abs(np.asarray(actual))) > 0.1


# --------------------------------------------------------------------------- #
# JAX transformations
# --------------------------------------------------------------------------- #
def test_bns_population_compute_terms_traces_under_jit() -> None:
    redshift_grid = make_redshift_grid()
    samples = {"redshift": SAMPLE_REDSHIFTS}

    def terms_at(params: dict[str, jax.Array]) -> PopulationTerms:
        return bns_population(
            params, redshift_grid=redshift_grid
        ).compute_population_terms(samples)

    traced = {name: jnp.asarray(value) for name, value in OFF_FIDUCIALS.items()}
    eager = terms_at(traced)
    jitted = jax.jit(terms_at)(traced)
    for name in PopulationTerms._fields:
        np.testing.assert_allclose(
            np.asarray(getattr(jitted, name)), np.asarray(getattr(eager, name))
        )


def test_bns_population_vmaps_over_hyperparameters() -> None:
    redshift_grid = make_redshift_grid()
    hubble_constants = jnp.array([60.0, 67.66, 75.0])
    samples = {"redshift": SAMPLE_REDSHIFTS}

    def log_prob_at(h0: jax.Array) -> jax.Array:
        params = {**FIDUCIALS, "H0": h0}
        return bns_population(params, redshift_grid=redshift_grid).log_prob(samples)

    batched = jax.vmap(log_prob_at)(hubble_constants)
    assert batched.shape == (3, SAMPLE_REDSHIFTS.shape[0])
    np.testing.assert_allclose(
        np.asarray(batched[1]), np.asarray(log_prob_at(hubble_constants[1]))
    )


# --------------------------------------------------------------------------- #
# Native MD + uniform mixture, for the paper-layer follow-up
# --------------------------------------------------------------------------- #
def test_mixture_general_reproduces_the_md_plus_uniform_proposal() -> None:
    """A native NumPyro mixture equals the hand-rolled `logaddexp` proposal.

    `support=` is required: `Uniform.log_prob` does not mask off-support
    values, and `MixtureGeneral` only masks each component when an explicit
    support is given. Without it the density would be finite outside the
    window, where the target is `-inf`.
    """
    epsilon = 0.1
    md = MadauDickinsonRedshiftDistribution(
        params=FIDUCIALS,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        n_grid=N_GRID,
    )
    mixture = dist.MixtureGeneral(
        dist.Categorical(probs=jnp.array([1.0 - epsilon, epsilon])),
        [md, dist.Uniform(Z_MIN, Z_MAX)],
        support=constraints.interval(Z_MIN, Z_MAX),
    )

    _, _, md_logprob = _reference(FIDUCIALS)
    expected = jnp.logaddexp(
        jnp.log1p(-epsilon) + md_logprob,
        jnp.log(epsilon) - jnp.log(Z_MAX - Z_MIN),
    )
    np.testing.assert_allclose(
        np.asarray(mixture.log_prob(SAMPLE_REDSHIFTS)), np.asarray(expected), rtol=1e-13
    )

    outside = jnp.array([Z_MIN - 0.1, Z_MAX + 0.1])
    assert np.all(np.isneginf(np.asarray(mixture.log_prob(outside))))
