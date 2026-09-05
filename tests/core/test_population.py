"""Tests for the population-based importance-sampling API.

The generic pieces (:class:`Population`, :class:`PopulationTerms`,
:func:`importance_log_weights`) are checked on toy distributions. The BNS
realization is checked against
:func:`compute_merger_rate_distance_and_logprob`, the hand-written grid-level
formula, which is what licenses rebuilding the reference callback on top of
it. The mixture test at the end is the viability check for the paper-layer
follow-up that will express the MD + uniform proposal natively.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb_mock_population import FIDUCIALS, N_GRID, Z_MAX, Z_MIN, make_redshift_grid
from numpyro.distributions import constraints

from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.distributions.redshift.madau_dickinson import (
    MadauDickinsonRedshiftDistribution,
)
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    bns_population,
    bns_population_terms,
    compute_merger_rate_distance_and_logprob,
    make_merger_rate_and_log_weights_fn,
)
from astrogwb.importance.population import (
    Population,
    PopulationTerms,
    importance_log_weights,
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
def test_log_prob_sums_the_per_parameter_log_densities() -> None:
    population = Population(
        distributions={"a": dist.Normal(0.0, 1.0), "b": dist.Uniform(0.0, 2.0)},
        params={},
    )
    source_parameters = {"a": jnp.array([0.0, 1.0]), "b": jnp.array([0.5, 1.5])}

    expected = dist.Normal(0.0, 1.0).log_prob(source_parameters["a"]) + dist.Uniform(
        0.0, 2.0
    ).log_prob(source_parameters["b"])
    np.testing.assert_allclose(
        np.asarray(population.log_prob(source_parameters)), np.asarray(expected)
    )


def test_log_prob_ignores_source_parameters_without_a_distribution() -> None:
    """Parameters the two sides agree on are simply absent from `distributions`."""
    population = Population(distributions={"a": dist.Normal(0.0, 1.0)}, params={})
    values = jnp.array([0.3, -0.7])

    np.testing.assert_allclose(
        np.asarray(population.log_prob({"a": values, "mass": jnp.ones(2)})),
        np.asarray(dist.Normal(0.0, 1.0).log_prob(values)),
    )


def test_log_prob_of_empty_distributions_is_zero() -> None:
    """The empty sum is neutral when only rates or propagation differ."""
    population = Population(
        distributions={},
        params={"local_merger_rate": 1.0, "xi_0": 1.4},
    )

    log_prob = population.log_prob({"mass": jnp.zeros(3)})

    assert log_prob.shape == ()
    np.testing.assert_array_equal(np.asarray(log_prob), np.asarray(0.0))


def test_log_prob_raises_on_a_missing_source_parameter() -> None:
    population = Population(distributions={"a": dist.Normal(0.0, 1.0)}, params={})
    with pytest.raises(KeyError):
        population.log_prob({"b": jnp.zeros(2)})


def test_population_is_a_pytree_of_its_distributions() -> None:
    population = Population(
        distributions={"a": dist.Normal(jnp.array(1.0), jnp.array(2.0))},
        params={"H0": jnp.array(70.0)},
    )
    leaves = jax.tree.leaves(population)
    assert {float(leaf) for leaf in leaves} == {1.0, 2.0, 70.0}


# --------------------------------------------------------------------------- #
# importance_log_weights
# --------------------------------------------------------------------------- #
def _terms(log_prob: list[float], log_distance: list[float]) -> PopulationTerms:
    return PopulationTerms(
        log_prob=jnp.array(log_prob),
        log_gw_distance=jnp.array(log_distance),
        total_merger_rate=jnp.array(1.0),
    )


def test_identical_terms_give_exactly_zero_log_weights() -> None:
    terms = _terms([-1.0, -2.5], [7.0, 7.5])
    np.testing.assert_array_equal(
        np.asarray(importance_log_weights(terms, terms)), np.zeros(2)
    )


def test_log_weights_are_the_density_ratio_minus_twice_the_distance_ratio() -> None:
    target = _terms([-1.0, -2.0], [7.0, 7.5])
    proposal = _terms([-1.5, -1.0], [6.8, 7.7])

    expected = np.array([-1.0 + 1.5, -2.0 + 1.0]) - 2.0 * np.array(
        [7.0 - 6.8, 7.5 - 7.7]
    )
    np.testing.assert_allclose(
        np.asarray(importance_log_weights(target, proposal)), expected, rtol=1e-14
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
    terms = bns_population_terms(population, samples)

    reference_rate, reference_distance, reference_logpdf = _reference(params)
    # Bit-exact, as in `tests/core/test_distributions.py`: the distribution
    # route shares the reference's operation order.
    np.testing.assert_array_equal(
        np.asarray(terms.log_prob), np.asarray(reference_logpdf)
    )
    np.testing.assert_allclose(
        float(terms.total_merger_rate), float(reference_rate), rtol=1e-15
    )
    expected_log_gw_distance = jnp.log(reference_distance) + log_gw_em_ratio(
        SAMPLE_REDSHIFTS, params["xi_0"], params["xi_n"]
    )
    np.testing.assert_allclose(
        np.asarray(terms.log_gw_distance),
        np.asarray(expected_log_gw_distance),
        rtol=1e-14,
    )


def test_bns_population_terms_use_the_given_distance_on_the_proposal_side() -> None:
    """The stored catalog distance wins over the population's own table."""
    population = bns_population(FIDUCIALS, redshift_grid=make_redshift_grid())
    stored = jnp.full(SAMPLE_REDSHIFTS.shape, 1234.5)

    terms = bns_population_terms(
        population, {"redshift": SAMPLE_REDSHIFTS}, luminosity_distance=stored
    )
    expected = jnp.log(stored) + log_gw_em_ratio(
        SAMPLE_REDSHIFTS, FIDUCIALS["xi_0"], FIDUCIALS["xi_n"]
    )
    np.testing.assert_allclose(np.asarray(terms.log_gw_distance), np.asarray(expected))


def test_bns_population_rebuilds_the_shared_grid_bit_exactly() -> None:
    population = bns_population(FIDUCIALS, redshift_grid=make_redshift_grid())
    redshift = population.distributions["redshift"]
    assert isinstance(redshift, MadauDickinsonRedshiftDistribution)
    np.testing.assert_array_equal(
        np.asarray(redshift.redshift_grid), np.asarray(make_redshift_grid())
    )


# --------------------------------------------------------------------------- #
# Regression: the migrated closure reproduces the explicit formula
# --------------------------------------------------------------------------- #
def test_closure_matches_the_explicit_formula_off_the_fiducials() -> None:
    """What licenses rewriting `log_weights` and the closure on the new API."""
    redshift_grid = make_redshift_grid()
    samples = {"redshift": jnp.linspace(Z_MIN, Z_MAX, 16)}
    _, fiducial_distance, proposal_logprob = compute_merger_rate_distance_and_logprob(
        FIDUCIALS, samples, redshift_grid=redshift_grid
    )
    samples["luminosity_distance"] = fiducial_distance

    weights_fn = make_merger_rate_and_log_weights_fn(
        fiducials=FIDUCIALS,
        redshift_grid=redshift_grid,
        proposal_logprob=proposal_logprob,
    )
    total_rate, actual = weights_fn(OFF_FIDUCIALS, samples)

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
    _, at_fiducials = weights_fn(FIDUCIALS, samples)
    np.testing.assert_array_equal(np.asarray(at_fiducials), 0.0)
    # The point is non-trivial: the weights are far from zero here.
    assert np.max(np.abs(np.asarray(actual))) > 0.1


# --------------------------------------------------------------------------- #
# JAX transformations
# --------------------------------------------------------------------------- #
def test_bns_population_terms_trace_under_jit() -> None:
    redshift_grid = make_redshift_grid()
    samples = {"redshift": SAMPLE_REDSHIFTS}

    def terms_at(params: dict[str, jax.Array]) -> PopulationTerms:
        return bns_population_terms(
            bns_population(params, redshift_grid=redshift_grid), samples
        )

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
