"""Catalog-backed spectral estimates against the hand-written grid formula."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb_mock_population import (
    FIDUCIALS,
    MOCK_POPULATION_SEED,
    N_GRID,
    Z_MAX,
    Z_MIN,
    derived_columns,
    make_redshift_grid,
    mock_target_model,
)
from jax.typing import ArrayLike
from numpyro.distributions import constraints
from reference_population import reference_merger_rate_distance_and_logprob

from astrogwb.catalog import Catalog
from astrogwb.catalog.catalog import REDSHIFT_SITE
from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.gwb.spectral import AverageMode, spectral_density
from astrogwb.importance.diagnostics import relative_ess
from astrogwb.importance.estimator import SpectralDensityImportanceEstimator
from astrogwb.populations import Population, SourceModel, build_population
from astrogwb.populations.bns_madau_dickinson import bns_md_cosmological
from astrogwb.waveform import PolarizationPowerGenerator

#: The catalog column naming the effective distance the stored polarization
#: power was generated at.
LUMINOSITY_DISTANCE_SITE = "luminosity_distance"

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
#: The generating population's own parameters, off the fiducials in every
#: direction the weights depend on. Making the catalog's population the *off*
#: one is deliberate: it proves the proposal density comes from the file rather
#: than from whatever the target happens to be evaluated at.
OFF_POPULATION_PARAMS = {
    name: value for name, value in OFF_FIDUCIALS.items() if name not in {"xi_0", "xi_n"}
}
MODEL_KWARGS = {"z_min": Z_MIN, "z_max": Z_MAX, "n_grid": N_GRID}


def _generating_model() -> Population:
    return build_population("bns_md_cosmological", settings=MODEL_KWARGS)


def _source_parameters(
    params: Mapping[str, float] = OFF_POPULATION_PARAMS,
) -> dict[str, jax.Array]:
    ones = jnp.ones_like(REDSHIFTS)
    return derived_columns(
        _generating_model().source,
        params,
        {
            REDSHIFT_SITE: REDSHIFTS,
            "source_frame_mass_1": 1.4 * ones,
            "source_frame_mass_2": 1.3 * ones,
            "spin_1z": 0.0 * ones,
            "spin_2z": 0.0 * ones,
            "lambda_1": 400.0 * ones,
            "lambda_2": 300.0 * ones,
        },
    )


def _waveform_metadata(num_frequencies: int) -> PolarizationPowerGenerator:
    return PolarizationPowerGenerator(
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=10.0 + 2.0 * (num_frequencies - 1),
        reference_frequency=10.0,
        sampling_frequency=64.0,
        frequency_resolution=2.0,
    )


def _catalog(
    *,
    params: Mapping[str, float] = OFF_POPULATION_PARAMS,
    power: jax.Array = POWER,
) -> Catalog:
    return Catalog(
        source_parameters={
            name: np.asarray(values)
            for name, values in _source_parameters(params).items()
        },
        polarization_power=np.asarray(power),
        frequencies=10.0 + 2.0 * np.arange(power.shape[0]),
        waveform_metadata=_waveform_metadata(power.shape[0]),
        _source_model_name="bns_md_cosmological",
        _rate_model_name="madau_dickinson",
        _model_kwargs=MODEL_KWARGS,
        _fiducials=params,
        _density_sites=("redshift", "source_frame_mass_1", "source_frame_mass_2"),
        seed=MOCK_POPULATION_SEED,
    )


def _estimator(
    average_mode: AverageMode = "catalog_inclination",
) -> SpectralDensityImportanceEstimator:
    return SpectralDensityImportanceEstimator.from_catalog(
        _catalog(),
        model=mock_target_model(),
        average_mode=average_mode,
    )


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def test_preparation_caches_the_catalogs_own_proposal_density() -> None:
    catalog = _catalog()
    estimator = SpectralDensityImportanceEstimator.from_catalog(
        catalog,
        model=mock_target_model(),
        average_mode="catalog_inclination",
    )

    # The proposal is the catalog's population at the catalog's parameters,
    # excluding the recorded constant factors -- so it is the redshift density
    # at OFF_POPULATION_PARAMS, not at the fiducials the target uses.
    _, _, expected_logprob = reference_merger_rate_distance_and_logprob(
        OFF_POPULATION_PARAMS,
        REDSHIFTS,
        redshift_grid=make_redshift_grid(),
        source_frame_mass_1=jnp.full_like(REDSHIFTS, 1.4),
        source_frame_mass_2=jnp.full_like(REDSHIFTS, 1.3),
    )
    np.testing.assert_allclose(
        estimator.proposal_log_prob, expected_logprob, rtol=0.0, atol=2e-15
    )
    np.testing.assert_array_equal(
        estimator.log_reference_distance,
        jnp.log(catalog.source_parameters[LUMINOSITY_DISTANCE_SITE]),
    )
    np.testing.assert_array_equal(estimator.polarization_power, POWER)
    assert set(estimator.source_parameters) == set(catalog.source_parameters)
    assert estimator.model.source.density_sites == (
        "redshift",
        "source_frame_mass_1",
        "source_frame_mass_2",
    )


def test_preparation_reuses_the_stored_reference_distance() -> None:
    """The power on disk corresponds to those exact distances.

    Recomputing them from the target's cosmology table would bias every weight
    by the interpolation difference, so the stored column wins even when the
    target's own distance for the same redshift is different.
    """
    catalog = _catalog()
    estimator = _estimator()
    target_distance = jnp.exp(
        jnp.log(
            reference_merger_rate_distance_and_logprob(
                FIDUCIALS,
                REDSHIFTS,
                redshift_grid=make_redshift_grid(),
                source_frame_mass_1=jnp.full_like(REDSHIFTS, 1.4),
                source_frame_mass_2=jnp.full_like(REDSHIFTS, 1.3),
            )[1]
        )
    )
    np.testing.assert_array_equal(
        estimator.log_reference_distance,
        jnp.log(catalog.source_parameters[LUMINOSITY_DISTANCE_SITE]),
    )
    assert not np.allclose(
        np.asarray(target_distance),
        np.asarray(catalog.source_parameters[LUMINOSITY_DISTANCE_SITE]),
    )


def test_preparation_selects_the_analysis_band_from_the_power_only() -> None:
    estimator = SpectralDensityImportanceEstimator.from_catalog(
        _catalog(),
        model=mock_target_model(),
        average_mode="catalog_inclination",
        frequency_mask=jnp.array([True, False, True]),
    )
    np.testing.assert_array_equal(estimator.polarization_power, POWER[[0, 2], :])
    assert estimator.proposal_log_prob.shape == (4,)
    assert estimator.source_parameters[REDSHIFT_SITE].shape == (4,)


def test_preparation_needs_no_merger_rate_for_the_proposal() -> None:
    """A proposal is a density, not an observation."""
    without_rate = {
        name: value
        for name, value in OFF_POPULATION_PARAMS.items()
        if name != "local_merger_rate"
    }
    estimator = SpectralDensityImportanceEstimator.from_catalog(
        _catalog(params=without_rate),
        model=mock_target_model(),
        average_mode="catalog_inclination",
    )
    assert estimator.proposal_log_prob.shape == (4,)
    # The *target* still needs one, because the spectrum is an observation.
    spectrum, extras = estimator(FIDUCIALS)
    assert spectrum.shape == (3,)
    assert float(jnp.asarray(extras["total_merger_rate"])) > 0.0


def test_empty_density_factors_broadcast_to_source_count() -> None:
    catalog = _catalog()
    catalog = Catalog(
        source_parameters=catalog.source_parameters,
        polarization_power=catalog.polarization_power,
        frequencies=catalog.frequencies,
        waveform_metadata=catalog.waveform_metadata,
        _source_model_name=catalog.population_source_model_name,
        _rate_model_name=catalog.population_rate_model_name,
        _model_kwargs=catalog.population_model_kwargs,
        _fiducials=catalog.fiducials,
        _density_sites=(),
        seed=MOCK_POPULATION_SEED,
    )
    base_target = mock_target_model()
    target = replace(base_target, source=replace(base_target.source, density_sites=()))
    estimator = SpectralDensityImportanceEstimator.from_catalog(
        catalog, model=target, average_mode="catalog_inclination"
    )
    assert estimator.proposal_log_prob.shape == (4,)
    np.testing.assert_array_equal(estimator.proposal_log_prob, jnp.zeros(4))
    spectrum, extras = estimator(FIDUCIALS)
    assert spectrum.shape == (3,)
    assert jnp.asarray(extras["importance_relative_ess"]).shape == ()


def test_mismatched_density_factors_are_rejected() -> None:
    base_target = mock_target_model()
    target = replace(
        base_target,
        source=replace(base_target.source, density_sites=("redshift", "spin_1z")),
    )
    with pytest.raises(ValueError, match="same source density factors"):
        SpectralDensityImportanceEstimator.from_catalog(
            _catalog(), model=target, average_mode="catalog_inclination"
        )


def test_a_catalog_missing_a_stochastic_column_is_rejected() -> None:
    catalog = _catalog()
    trimmed = replace(
        catalog,
        source_parameters={
            name: values
            for name, values in catalog.source_parameters.items()
            if name != "spin_1z"
        },
    )
    with pytest.raises(TypeError, match="PRNG key"):
        # NumPyro cannot draw the missing source without an RNG key.
        SpectralDensityImportanceEstimator.from_catalog(
            trimmed,
            model=mock_target_model(),
            average_mode="catalog_inclination",
        )


# --------------------------------------------------------------------------- #
# Pytree behaviour
# --------------------------------------------------------------------------- #
def test_estimator_round_trips_as_a_pytree() -> None:
    estimator = _estimator()
    leaves, structure = jax.tree.flatten(estimator)
    # All stored source arrays, power, proposal density, and reference distances.
    assert len(leaves) == len(estimator.source_parameters) + 3
    rebuilt = jax.tree.unflatten(structure, leaves)
    assert rebuilt.model is estimator.model
    assert rebuilt.model.source.density_sites == estimator.model.source.density_sites
    assert rebuilt.average_mode == estimator.average_mode
    for actual, expected in zip(
        jax.tree.leaves(rebuilt(FIDUCIALS)),
        jax.tree.leaves(estimator(FIDUCIALS)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_reconstruction_supports_tracers_and_batching() -> None:
    """The constructor must not validate: JAX rebuilds with placeholders."""
    estimator = _estimator()
    np.testing.assert_array_equal(
        jax.jit(lambda value: value.polarization_power)(estimator), POWER
    )
    batched = jax.vmap(lambda scale: jax.tree.map(lambda x: x * scale, estimator))(
        jnp.array([1.0, 2.0])
    )
    np.testing.assert_array_equal(batched.polarization_power[1], 2.0 * POWER)


def test_direct_construction_from_prepared_arrays_is_supported() -> None:
    """The proposal need not be a population at all -- only an ``(N,)`` density."""
    estimator = SpectralDensityImportanceEstimator(
        source_parameters=_source_parameters(),
        polarization_power=POWER,
        proposal_log_prob=jnp.zeros(4),
        log_reference_distance=jnp.log(jnp.full(4, 1234.5)),
        model=mock_target_model(),
        average_mode="catalog_inclination",
    )
    spectrum, extras = estimator(FIDUCIALS)
    assert spectrum.shape == (3,)
    assert np.isfinite(np.asarray(estimator.log_weights(FIDUCIALS))).all()
    assert set(extras) == {"total_merger_rate", "importance_relative_ess"}


def test_proposal_density_is_evaluated_only_during_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proposal's one call happens at ``from_catalog``; the target's, per step.

    The proposal density goes through ``SourceModel.log_prob`` (no rate
    needed); the target goes through ``Population.evaluate``, which calls
    ``self.source.evaluate`` once internally. Counting ``SourceModel.evaluate``
    calls sees both: one for the proposal during preparation, one per later
    ``estimator(...)`` call for the target.
    """
    calls: list[SourceModel] = []
    original = SourceModel.evaluate

    def counted(self, params, sources):
        calls.append(self)
        return original(self, params, sources)

    monkeypatch.setattr(SourceModel, "evaluate", counted)
    catalog = _catalog()
    target = mock_target_model()
    estimator = SpectralDensityImportanceEstimator.from_catalog(
        catalog, model=target, average_mode="catalog_inclination"
    )
    assert len(calls) == 1
    assert calls[0].fn is bns_md_cosmological
    estimator(FIDUCIALS)
    estimator(OFF_FIDUCIALS)
    jax.jit(lambda value, params: value(params))(estimator, FIDUCIALS)
    assert len(calls) == 4
    assert all(model is target.source for model in calls[1:])


# --------------------------------------------------------------------------- #
# The weights themselves
# --------------------------------------------------------------------------- #
def test_a_catalog_reweighted_to_its_own_proposal_has_exactly_zero_log_weights() -> (
    None
):
    """Non-negotiable, and exactly rather than approximately.

    ``xi_0 = 1`` makes the modified-propagation target reduce bit-for-bit to
    the cosmological population that drew the catalog, so target and proposal
    are the same expressions on the same inputs. A tolerance here would hide an
    operation-order change that costs a ulp per weight -- small on its own, but
    this identity is what several other tests build exact expectations on.
    """
    catalog = _catalog()
    estimator = SpectralDensityImportanceEstimator.from_catalog(
        catalog,
        model=mock_target_model(),
        average_mode="catalog_inclination",
    )
    at_generating = {**OFF_POPULATION_PARAMS, "xi_0": 1.0, "xi_n": 2.3}
    np.testing.assert_array_equal(estimator.log_weights(at_generating), jnp.zeros(4))

    spectrum, extras = estimator(at_generating)
    _, _, _ = reference_merger_rate_distance_and_logprob(
        OFF_POPULATION_PARAMS,
        REDSHIFTS,
        redshift_grid=make_redshift_grid(),
        source_frame_mass_1=jnp.full_like(REDSHIFTS, 1.4),
        source_frame_mass_2=jnp.full_like(REDSHIFTS, 1.3),
    )
    np.testing.assert_allclose(
        spectrum,
        float(jnp.asarray(extras["total_merger_rate"])) * POWER.mean(axis=1),
        rtol=1e-14,
    )
    np.testing.assert_array_equal(extras["importance_relative_ess"], 1.0)


def _grid_reference(
    estimator: SpectralDensityImportanceEstimator,
) -> Callable[[Mapping[str, ArrayLike]], tuple[jax.Array, dict[str, jax.Array]]]:
    """The same estimate, written out from the hand-written grid-level formula.

    Deliberately not routed through the population model: an expectation built
    from the code under test would agree by construction. This restates the
    density, the distance, and the weight ratio in full, so a drift in either
    route shows up as a failure rather than as silent agreement -- including
    the case where an identical mistake sits in the numerator and denominator.
    """

    def evaluate(
        params: Mapping[str, ArrayLike],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        rate, distance, logprob = reference_merger_rate_distance_and_logprob(
            params,
            REDSHIFTS,
            redshift_grid=make_redshift_grid(),
            source_frame_mass_1=jnp.full_like(REDSHIFTS, 1.4),
            source_frame_mass_2=jnp.full_like(REDSHIFTS, 1.3),
        )
        log_target_distance = jnp.log(distance) + log_gw_em_ratio(
            REDSHIFTS, params["xi_0"], params["xi_n"]
        )
        log_weights = (
            logprob
            - estimator.proposal_log_prob
            - 2.0 * (log_target_distance - estimator.log_reference_distance)
        )
        return spectral_density(
            estimator.polarization_power,
            jnp.exp(log_weights),
            rate,
            average_mode=estimator.average_mode,
        ), {
            "total_merger_rate": jnp.asarray(rate),
            "importance_relative_ess": relative_ess(log_weights),
        }

    return evaluate


@pytest.mark.parametrize("params", [FIDUCIALS, OFF_FIDUCIALS])
@pytest.mark.parametrize("mode", ["analytic_inclination", "catalog_inclination"])
def test_estimator_matches_the_grid_formula_spectrum_rate_and_ess(
    params: dict[str, float], mode: AverageMode
) -> None:
    estimator = _estimator(mode)
    actual = estimator(params)
    expected = _grid_reference(estimator)(params)
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
    # Same callable with different dynamic data must use the new power.
    doubled = replace(estimator, polarization_power=2.0 * POWER)
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
    grid_formula = _grid_reference(estimator)
    reference = jax.grad(lambda p: jnp.sum(grid_formula(p)[0]) / normalization)(params)
    for name, value in gradient.items():
        assert np.isfinite(value), name
        np.testing.assert_allclose(
            value, reference[name], rtol=1e-9, atol=1e-12, err_msg=name
        )


def test_precomputed_mixture_density_is_used_in_estimate() -> None:
    """A proposal supplied as a bare array need not be a population at all."""
    from astrogwb.distributions.redshift.madau_dickinson import (
        MadauDickinsonRedshiftDistribution,
    )

    estimator = _estimator()
    redshift = MadauDickinsonRedshiftDistribution(
        params=OFF_POPULATION_PARAMS,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        n_grid=N_GRID,
    )
    mixture = dist.MixtureGeneral(
        dist.Categorical(probs=jnp.array([0.9, 0.1])),
        [redshift, dist.Uniform(Z_MIN, Z_MAX)],
        support=constraints.interval(Z_MIN, Z_MAX),
    )
    mixed = replace(estimator, proposal_log_prob=mixture.log_prob(REDSHIFTS))

    rate, distance, logprob = reference_merger_rate_distance_and_logprob(
        FIDUCIALS,
        REDSHIFTS,
        redshift_grid=make_redshift_grid(),
        source_frame_mass_1=jnp.full_like(REDSHIFTS, 1.4),
        source_frame_mass_2=jnp.full_like(REDSHIFTS, 1.3),
    )
    log_target_distance = jnp.log(distance) + log_gw_em_ratio(
        REDSHIFTS, FIDUCIALS["xi_0"], FIDUCIALS["xi_n"]
    )
    manual_log_weights = (
        logprob
        - mixture.log_prob(REDSHIFTS)
        - 2.0 * (log_target_distance - mixed.log_reference_distance)
    )
    expected = spectral_density(
        POWER,
        jnp.exp(manual_log_weights),
        rate,
        average_mode=mixed.average_mode,
    )
    np.testing.assert_allclose(mixed(FIDUCIALS)[0], expected, rtol=1e-13)
