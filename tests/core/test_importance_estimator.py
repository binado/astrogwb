"""Catalog-backed spectral estimates against the existing importance callback."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb_mock_population import FIDUCIALS, Z_MAX, Z_MIN, make_redshift_grid
from jax.typing import ArrayLike
from numpyro.distributions import constraints

from astrogwb.catalog import ImportanceCatalog
from astrogwb.gwb.spectral import AverageMode, spectral_density
from astrogwb.importance.diagnostics import relative_ess
from astrogwb.importance.estimator import SpectralDensityImportanceEstimator
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    ModifiedPropagationPopulation,
    bns_population,
    make_merger_rate_and_log_weights_fn,
)
from astrogwb.importance.population import importance_log_weights

REDSHIFTS = jnp.array([0.41, 1.23, 3.77, 7.1])
POWER = jnp.arange(1.0, 13.0).reshape(3, 4)
OFF_FIDUCIALS = {
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


@pytest.fixture
def catalog() -> ImportanceCatalog:
    """A consistent catalog built directly, skipping factory validation."""
    return ImportanceCatalog(
        source_parameters={"redshift": REDSHIFTS},
        polarization_power=POWER,
        proposal_log_prob=jnp.zeros(4),
        log_reference_distance=jnp.log(jnp.full(4, 1234.5)),
    )


def _estimator(
    average_mode: AverageMode = "catalog_inclination",
) -> SpectralDensityImportanceEstimator:
    factory = partial(bns_population, redshift_grid=make_redshift_grid())
    proposal = factory(OFF_FIDUCIALS)
    catalog = ImportanceCatalog.from_population(
        population=proposal,
        source_parameters={"redshift": REDSHIFTS},
        polarization_power=POWER,
        luminosity_distance=proposal.luminosity_distance(REDSHIFTS),
    )
    return SpectralDensityImportanceEstimator(catalog, factory, average_mode)


@pytest.mark.parametrize(
    ("power", "message"),
    [
        (jnp.ones(4), "two-dimensional"),
        (jnp.ones((3, 4), dtype=complex), "real"),
        (jnp.ones((3, 0)), "at least one source"),
    ],
)
def test_from_population_rejects_inconsistent_power(
    power: jax.Array, message: str
) -> None:
    num_samples = power.shape[1] if power.ndim == 2 else REDSHIFTS.size
    with pytest.raises(ValueError, match=message):
        ImportanceCatalog.from_population(
            population=bns_population(FIDUCIALS, redshift_grid=make_redshift_grid()),
            source_parameters={"redshift": REDSHIFTS[:num_samples]},
            polarization_power=power,
            luminosity_distance=jnp.ones(num_samples),
        )


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ({"redshift": jnp.ones((4, 1))}, "shape"),
        ({"redshift": REDSHIFTS, "mass": jnp.ones(3)}, "shape"),
    ],
)
def test_from_population_rejects_inconsistent_source_parameters(
    samples: Mapping[str, jax.Array], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ImportanceCatalog.from_population(
            population=bns_population(FIDUCIALS, redshift_grid=make_redshift_grid()),
            source_parameters=samples,
            polarization_power=POWER,
            luminosity_distance=jnp.ones(4),
        )


@pytest.mark.parametrize("distance", [0.0, -1.0, jnp.inf, jnp.nan, 1.0 + 1.0j])
def test_catalog_rejects_invalid_linear_distances(distance: complex) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        ImportanceCatalog.from_population(
            population=bns_population(FIDUCIALS, redshift_grid=make_redshift_grid()),
            source_parameters={"redshift": REDSHIFTS},
            polarization_power=POWER,
            luminosity_distance=jnp.full(4, distance),
        )


@pytest.mark.parametrize(
    ("samples", "distance", "message"),
    [
        ({"mass": REDSHIFTS}, jnp.ones(4), "redshift"),
        ({"redshift": REDSHIFTS}, jnp.ones(3), "log_reference_distance"),
        ({"redshift": REDSHIFTS}, jnp.asarray(1234.5), "log_reference_distance"),
    ],
)
def test_population_constructor_requires_aligned_redshifts_and_distances(
    samples: Mapping[str, jax.Array], distance: jax.Array, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ImportanceCatalog.from_population(
            population=bns_population(FIDUCIALS, redshift_grid=make_redshift_grid()),
            source_parameters=samples,
            polarization_power=POWER,
            luminosity_distance=distance,
        )


def test_supplied_reference_distance_wins_over_population_cosmology() -> None:
    proposal = bns_population(OFF_FIDUCIALS, redshift_grid=make_redshift_grid())
    distance = jnp.full(4, 1234.5)
    catalog = ImportanceCatalog.from_population(
        population=proposal,
        source_parameters={"redshift": REDSHIFTS},
        polarization_power=POWER,
        luminosity_distance=distance,
    )
    np.testing.assert_array_equal(catalog.log_reference_distance, jnp.log(distance))
    np.testing.assert_array_equal(
        catalog.proposal_log_prob, proposal.log_prob(catalog.source_parameters)
    )
    assert not np.allclose(distance, proposal.luminosity_distance(REDSHIFTS))


def test_catalog_and_estimator_round_trip_as_pytrees() -> None:
    estimator = _estimator()
    leaves, structure = jax.tree.flatten(estimator)
    assert len(leaves) == 4  # one source array, power, proposal density, distance
    rebuilt = jax.tree.unflatten(structure, leaves)
    assert rebuilt.population_fn is estimator.population_fn
    assert rebuilt.average_mode == estimator.average_mode
    for actual, expected in zip(
        jax.tree.leaves(rebuilt(FIDUCIALS)),
        jax.tree.leaves(estimator(FIDUCIALS)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_catalog_reconstruction_supports_tracers_and_batching(
    catalog: ImportanceCatalog,
) -> None:
    leaves, structure = jax.tree.flatten(catalog)
    rebuilt = jax.tree.unflatten(structure, leaves)
    np.testing.assert_array_equal(
        jax.jit(lambda value: value.polarization_power)(rebuilt), POWER
    )
    # vmap reconstructs the pytree with placeholder and batched leaves; the
    # base constructor must accept them without re-validation.
    batched = jax.vmap(lambda scale: jax.tree.map(lambda x: x * scale, catalog))(
        jnp.array([1.0, 2.0])
    )
    np.testing.assert_array_equal(batched.polarization_power[1], 2.0 * POWER)


def test_proposal_density_is_evaluated_only_at_catalog_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = partial(bns_population, redshift_grid=make_redshift_grid())
    proposal = factory(FIDUCIALS)
    original_log_prob = ModifiedPropagationPopulation.log_prob
    evaluations = 0

    def counted_log_prob(
        self: ModifiedPropagationPopulation, samples: Mapping[str, ArrayLike]
    ) -> jax.Array:
        nonlocal evaluations
        if self is proposal:
            evaluations += 1
        return original_log_prob(self, samples)

    monkeypatch.setattr(ModifiedPropagationPopulation, "log_prob", counted_log_prob)
    catalog = ImportanceCatalog.from_population(
        population=proposal,
        source_parameters={"redshift": REDSHIFTS},
        polarization_power=POWER,
        luminosity_distance=proposal.luminosity_distance(REDSHIFTS),
    )
    estimator = SpectralDensityImportanceEstimator(
        catalog, factory, "catalog_inclination"
    )
    estimator(FIDUCIALS)
    estimator(OFF_FIDUCIALS)
    jax.jit(lambda value, params: value(params))(estimator, FIDUCIALS)
    assert evaluations == 1


def test_identical_population_and_catalog_have_exactly_neutral_weights() -> None:
    estimator = _estimator()
    terms = estimator.population_fn(OFF_FIDUCIALS).compute_population_terms(
        estimator.catalog.source_parameters
    )
    weights = importance_log_weights(
        terms,
        proposal_log_prob=estimator.catalog.proposal_log_prob,
        log_reference_distance=estimator.catalog.log_reference_distance,
    )
    np.testing.assert_array_equal(weights, jnp.zeros(4))
    actual, extras = estimator(OFF_FIDUCIALS)
    expected = terms.total_merger_rate * POWER.mean(axis=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    np.testing.assert_array_equal(extras["importance_relative_ess"], 1.0)


def _legacy_fn(
    estimator: SpectralDensityImportanceEstimator,
) -> Callable[[Mapping[str, ArrayLike]], tuple[jax.Array, dict[str, jax.Array]]]:
    proposal = estimator.population_fn(OFF_FIDUCIALS)
    samples = {
        **estimator.catalog.source_parameters,
        "luminosity_distance": proposal.redshift_distribution.luminosity_distance(
            REDSHIFTS
        ),
    }
    callback = make_merger_rate_and_log_weights_fn(
        fiducials=OFF_FIDUCIALS,
        redshift_grid=make_redshift_grid(),
        proposal_logprob=estimator.catalog.proposal_log_prob,
    )

    def evaluate(
        params: Mapping[str, ArrayLike],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        rate, log_weights = callback(params, samples)
        return spectral_density(
            POWER, jnp.exp(log_weights), rate, average_mode=estimator.average_mode
        ), {
            "total_merger_rate": jnp.asarray(rate),
            "importance_relative_ess": relative_ess(log_weights),
        }

    return evaluate


@pytest.mark.parametrize("params", [FIDUCIALS, OFF_FIDUCIALS])
@pytest.mark.parametrize("mode", ["analytic_inclination", "catalog_inclination"])
def test_estimator_matches_legacy_spectrum_rate_and_ess(
    params: dict[str, float], mode: AverageMode
) -> None:
    estimator = _estimator(mode)
    actual = estimator(params)
    expected = _legacy_fn(estimator)(params)
    assert set(actual[1]) == {"total_merger_rate", "importance_relative_ess"}
    for value, reference in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(value, reference, rtol=1e-12)


def test_estimator_jit_vmap_and_gradients() -> None:
    estimator = _estimator()
    compiled = jax.jit(lambda value, params: value(params))
    actual = compiled(estimator, FIDUCIALS)
    for value, expected in zip(
        jax.tree.leaves(actual), jax.tree.leaves(estimator(FIDUCIALS)), strict=True
    ):
        np.testing.assert_allclose(value, expected, rtol=1e-12)
    # Same callable with different dynamic catalog data must use the new power.
    doubled = replace(
        estimator, catalog=replace(estimator.catalog, polarization_power=2.0 * POWER)
    )
    np.testing.assert_allclose(
        compiled(doubled, FIDUCIALS)[0], 2.0 * actual[0], rtol=1e-12
    )

    def at_h0(h0: jax.Array) -> jax.Array:
        return estimator({**FIDUCIALS, "H0": h0})[0]

    hubble_constants = jnp.array([60.0, 67.66, 75.0])
    batched = jax.jit(jax.vmap(at_h0))(hubble_constants)
    np.testing.assert_allclose(
        batched, jnp.stack([at_h0(h0) for h0 in hubble_constants]), rtol=1e-12
    )

    normalization = jnp.sum(estimator(FIDUCIALS)[0])
    params = {name: jnp.asarray(value) for name, value in FIDUCIALS.items()}
    gradient = jax.grad(lambda p: jnp.sum(estimator(p)[0]) / normalization)(params)
    legacy = _legacy_fn(estimator)
    reference = jax.grad(lambda p: jnp.sum(legacy(p)[0]) / normalization)(params)
    for name, value in gradient.items():
        assert np.isfinite(value), name
        np.testing.assert_allclose(
            value, reference[name], rtol=1e-9, atol=1e-12, err_msg=name
        )


def test_precomputed_mixture_density_is_used_in_estimate() -> None:
    estimator = _estimator()
    redshift = estimator.population_fn(OFF_FIDUCIALS).redshift_distribution
    mixture = dist.MixtureGeneral(
        dist.Categorical(probs=jnp.array([0.9, 0.1])),
        [redshift, dist.Uniform(Z_MIN, Z_MAX)],
        support=constraints.interval(Z_MIN, Z_MAX),
    )
    catalog = replace(estimator.catalog, proposal_log_prob=mixture.log_prob(REDSHIFTS))
    estimator = replace(estimator, catalog=catalog)
    terms = estimator.population_fn(FIDUCIALS).compute_population_terms(
        catalog.source_parameters
    )
    manual_log_weights = (
        terms.log_prob
        - mixture.log_prob(REDSHIFTS)
        - 2.0 * (terms.log_luminosity_distance - catalog.log_reference_distance)
    )
    expected = spectral_density(
        POWER,
        jnp.exp(manual_log_weights),
        terms.total_merger_rate,
        average_mode=estimator.average_mode,
    )
    np.testing.assert_allclose(estimator(FIDUCIALS)[0], expected, rtol=1e-14)
