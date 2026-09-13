"""Catalog-backed importance spectra against the hand-written grid formula."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import Any

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
    log_weight_kwargs,
    make_redshift_grid,
    mock_merger_rate_fn,
    mock_target_model,
)
from jax.typing import ArrayLike
from numpyro.distributions import constraints
from reference_population import reference_merger_rate_distance_and_logprob

from astrogwb.catalog import Catalog
from astrogwb.catalog.catalog import REDSHIFT_SITE
from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.gwb.spectral import AverageMode, spectral_density
from astrogwb.importance import spectral
from astrogwb.importance.diagnostics import relative_ess
from astrogwb.importance.spectral import (
    _prepare_importance_arrays,
    build_importance_spectrum,
    evaluate_log_weights,
    importance_spectral_density,
)
from astrogwb.populations import DEFAULT_DENSITY_SITES, SourceFn, build_source_model
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


def _generating_model() -> SourceFn:
    return build_source_model("bns_md_cosmological", settings=MODEL_KWARGS)


def _source_parameters(
    params: Mapping[str, float] = OFF_POPULATION_PARAMS,
) -> dict[str, jax.Array]:
    ones = jnp.ones_like(REDSHIFTS)
    return derived_columns(
        _generating_model(),
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
    density_sites: tuple[str, ...] = DEFAULT_DENSITY_SITES,
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
        _density_sites=density_sites,
        seed=MOCK_POPULATION_SEED,
    )


def _importance(
    average_mode: AverageMode = "catalog_inclination",
    *,
    catalog: Catalog | None = None,
    source_model: SourceFn | None = None,
    frequency_mask: ArrayLike | None = None,
) -> dict[str, Any]:
    """Every keyword of ``importance_spectral_density``, prepared from a catalog."""
    arrays = _prepare_importance_arrays(
        _catalog() if catalog is None else catalog, frequency_mask=frequency_mask
    )
    return {
        **arrays._asdict(),
        "source_model": mock_target_model() if source_model is None else source_model,
        "merger_rate_fn": mock_merger_rate_fn(),
        "average_mode": average_mode,
    }


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def test_preparation_caches_the_catalogs_own_proposal_density() -> None:
    catalog = _catalog()
    arrays = _prepare_importance_arrays(catalog)

    # The proposal is the catalog's source model at the catalog's parameters,
    # with the recorded density factors -- so it is the density at
    # OFF_POPULATION_PARAMS, not at the fiducials the target uses.
    _, _, expected_logprob = reference_merger_rate_distance_and_logprob(
        OFF_POPULATION_PARAMS,
        REDSHIFTS,
        redshift_grid=make_redshift_grid(),
        source_frame_mass_1=jnp.full_like(REDSHIFTS, 1.4),
        source_frame_mass_2=jnp.full_like(REDSHIFTS, 1.3),
    )
    np.testing.assert_allclose(
        arrays.proposal_log_prob, expected_logprob, rtol=0.0, atol=2e-15
    )
    np.testing.assert_array_equal(
        arrays.log_reference_distance,
        jnp.log(catalog.source_parameters[LUMINOSITY_DISTANCE_SITE]),
    )
    np.testing.assert_array_equal(arrays.polarization_power, POWER)
    assert set(arrays.source_parameters) == set(catalog.source_parameters)
    assert arrays.density_sites == catalog.density_sites == DEFAULT_DENSITY_SITES


def test_preparation_reuses_the_stored_reference_distance() -> None:
    """The power on disk corresponds to those exact distances.

    Recomputing them from the target's cosmology table would bias every weight
    by the interpolation difference, so the stored column wins even when the
    target's own distance for the same redshift is different.
    """
    catalog = _catalog()
    arrays = _prepare_importance_arrays(catalog)
    target_distance = reference_merger_rate_distance_and_logprob(
        FIDUCIALS,
        REDSHIFTS,
        redshift_grid=make_redshift_grid(),
        source_frame_mass_1=jnp.full_like(REDSHIFTS, 1.4),
        source_frame_mass_2=jnp.full_like(REDSHIFTS, 1.3),
    )[1]
    np.testing.assert_array_equal(
        arrays.log_reference_distance,
        jnp.log(catalog.source_parameters[LUMINOSITY_DISTANCE_SITE]),
    )
    assert not np.allclose(
        np.asarray(target_distance),
        np.asarray(catalog.source_parameters[LUMINOSITY_DISTANCE_SITE]),
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_preparation_rejects_a_non_physical_stored_distance(bad: float) -> None:
    catalog = _catalog()
    distance = np.array(catalog.source_parameters[LUMINOSITY_DISTANCE_SITE])
    distance[1] = bad
    broken = replace(
        catalog,
        source_parameters={
            **catalog.source_parameters,
            LUMINOSITY_DISTANCE_SITE: distance,
        },
    )
    with pytest.raises(ValueError, match="positive and finite"):
        _prepare_importance_arrays(broken)


def test_preparation_selects_the_analysis_band_from_the_power_only() -> None:
    arrays = _prepare_importance_arrays(
        _catalog(), frequency_mask=jnp.array([True, False, True])
    )
    np.testing.assert_array_equal(arrays.polarization_power, POWER[[0, 2], :])
    assert arrays.proposal_log_prob.shape == (4,)
    assert arrays.source_parameters[REDSHIFT_SITE].shape == (4,)


def test_preparation_needs_no_merger_rate_for_the_proposal() -> None:
    """A proposal is a density, not an observation."""
    without_rate = {
        name: value
        for name, value in OFF_POPULATION_PARAMS.items()
        if name != "local_merger_rate"
    }
    importance = _importance(catalog=_catalog(params=without_rate))
    assert importance["proposal_log_prob"].shape == (4,)
    # The *target* still needs one, because the spectrum is an observation.
    spectrum, extras = importance_spectral_density(FIDUCIALS, **importance)
    assert spectrum.shape == (3,)
    assert float(jnp.asarray(extras["total_merger_rate"])) > 0.0


def test_empty_density_factors_broadcast_to_source_count() -> None:
    importance = _importance(catalog=_catalog(density_sites=()))
    assert importance["density_sites"] == ()
    assert importance["proposal_log_prob"].shape == (4,)
    np.testing.assert_array_equal(importance["proposal_log_prob"], jnp.zeros(4))
    spectrum, extras = importance_spectral_density(FIDUCIALS, **importance)
    assert spectrum.shape == (3,)
    assert jnp.asarray(extras["importance_relative_ess"]).shape == ()


def test_the_catalogs_density_factors_reach_the_target_unchanged() -> None:
    """The structural replacement for a target/proposal factor-set check.

    A catalog records a narrower factor set than the default. Reweighted to
    its own source model at its own parameters, the weights are exactly zero
    only if the target was evaluated with that same set: substituting the
    default anywhere would add the ordered-mass factor to one side alone.
    """
    catalog = _catalog(density_sites=(REDSHIFT_SITE,))
    arrays = _prepare_importance_arrays(catalog)
    assert arrays.density_sites == (REDSHIFT_SITE,)

    source_model = catalog.get_source_model()
    log_weights = evaluate_log_weights(
        catalog.fiducials,
        source_model=source_model,
        source_parameters=arrays.source_parameters,
        proposal_log_prob=arrays.proposal_log_prob,
        log_reference_distance=arrays.log_reference_distance,
        density_sites=arrays.density_sites,
    )
    np.testing.assert_array_equal(log_weights, jnp.zeros(4))

    # The failure this guards against is not subtle: the default factor set
    # gives finite, wrong weights.
    wrong = evaluate_log_weights(
        catalog.fiducials,
        source_model=source_model,
        source_parameters=arrays.source_parameters,
        proposal_log_prob=arrays.proposal_log_prob,
        log_reference_distance=arrays.log_reference_distance,
        density_sites=DEFAULT_DENSITY_SITES,
    )
    assert np.all(np.isfinite(np.asarray(wrong)))
    assert not np.any(np.asarray(wrong) == 0.0)


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
    with pytest.raises(KeyError, match="spin_1z"):
        _prepare_importance_arrays(trimmed)


def test_direct_construction_from_prepared_arrays_is_supported() -> None:
    """The proposal need not be a registered source model -- only an ``(N,)`` density."""
    spectrum = partial(
        importance_spectral_density,
        source_model=mock_target_model(),
        merger_rate_fn=mock_merger_rate_fn(),
        source_parameters=_source_parameters(),
        polarization_power=POWER,
        proposal_log_prob=jnp.zeros(4),
        log_reference_distance=jnp.log(jnp.full(4, 1234.5)),
        density_sites=DEFAULT_DENSITY_SITES,
        average_mode="catalog_inclination",
    )
    prediction, extras = spectrum(FIDUCIALS)
    assert prediction.shape == (3,)
    assert set(extras) == {"total_merger_rate", "importance_relative_ess"}


def test_proposal_density_is_evaluated_only_during_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proposal's one evaluation happens at preparation; the target's, per step.

    Counting ``evaluate_sources`` calls sees both: one for the proposal during
    preparation, one per later spectrum call (and one per trace under
    ``jit``) for the target.
    """
    calls: list[SourceFn] = []
    original = spectral.evaluate_sources

    def counted(source_model, *args, **kwargs):
        calls.append(source_model)
        return original(source_model, *args, **kwargs)

    monkeypatch.setattr(spectral, "evaluate_sources", counted)
    target = mock_target_model()
    importance = _importance(source_model=target)
    assert len(calls) == 1
    assert calls[0].func is bns_md_cosmological  # ty: ignore[unresolved-attribute]

    spectrum = partial(importance_spectral_density, **importance)
    spectrum(FIDUCIALS)
    spectrum(OFF_FIDUCIALS)
    jax.jit(spectrum)(FIDUCIALS)
    assert len(calls) == 4
    assert all(model is target for model in calls[1:])


def test_a_target_without_a_distance_output_is_rejected() -> None:
    def distanceless(params: Mapping[str, ArrayLike]) -> dict[str, jax.Array]:
        outputs = dict(_generating_model()(params))
        del outputs[LUMINOSITY_DISTANCE_SITE]
        return outputs

    weight_kwargs: dict[str, Any] = log_weight_kwargs(_importance())
    weight_kwargs["source_model"] = distanceless
    with pytest.raises(KeyError, match=LUMINOSITY_DISTANCE_SITE):
        evaluate_log_weights(FIDUCIALS, **weight_kwargs)


# --------------------------------------------------------------------------- #
# The builder
# --------------------------------------------------------------------------- #
def _spectrum(
    *,
    catalog: Catalog | None = None,
    frequency_mask: ArrayLike | None = None,
    average_mode: AverageMode = "catalog_inclination",
):
    return build_importance_spectrum(
        _catalog() if catalog is None else catalog,
        source_model=mock_target_model(),
        merger_rate_fn=mock_merger_rate_fn(),
        average_mode=average_mode,
        frequency_mask=frequency_mask,
    )


def test_the_builder_binds_both_callables_from_one_shared_mapping() -> None:
    """The structural version of the shared-``density_sites`` invariant.

    A hand-written dict feeding two partials could drift; here the two
    returned callables must carry the identical array objects and factor set,
    because one internal mapping is splatted into both.
    """
    spectrum = _spectrum()
    density_keywords = spectrum.spectral_density.keywords
    weight_keywords = spectrum.log_weights.keywords
    shared = (
        "source_model",
        "source_parameters",
        "proposal_log_prob",
        "log_reference_distance",
        "density_sites",
    )
    for name in shared:
        assert density_keywords[name] is weight_keywords[name], name


def test_the_builders_frequency_mask_slices_power_but_not_samples() -> None:
    spectrum = _spectrum(frequency_mask=jnp.array([True, False, True]))
    density_keywords = spectrum.spectral_density.keywords
    assert density_keywords["polarization_power"].shape == (2, 4)
    for name, values in density_keywords["source_parameters"].items():
        assert values.shape == (4,), name
    assert density_keywords["proposal_log_prob"].shape == (4,)


def test_the_builders_spectral_density_matches_the_underlying_primitive() -> None:
    catalog = _catalog()
    spectrum = _spectrum(catalog=catalog)
    prediction, extras = spectrum.spectral_density(FIDUCIALS)
    expected_prediction, expected_extras = importance_spectral_density(
        FIDUCIALS, **_importance(catalog=catalog)
    )
    np.testing.assert_array_equal(prediction, expected_prediction)
    assert set(extras) == set(expected_extras)
    log_weights = spectrum.log_weights(FIDUCIALS)
    expected_log_weights = evaluate_log_weights(
        FIDUCIALS, **log_weight_kwargs(_importance(catalog=catalog))
    )
    np.testing.assert_array_equal(log_weights, expected_log_weights)


# --------------------------------------------------------------------------- #
# The weights themselves
# --------------------------------------------------------------------------- #
def test_a_catalog_reweighted_to_its_own_proposal_has_exactly_zero_log_weights() -> (
    None
):
    """Non-negotiable, and exactly rather than approximately.

    ``xi_0 = 1`` makes the modified-propagation target reduce bit-for-bit to
    the cosmological source model that drew the catalog, so target and proposal
    are the same expressions on the same inputs. A tolerance here would hide an
    operation-order change that costs a ulp per weight -- small on its own, but
    this identity is what several other tests build exact expectations on.
    """
    importance = _importance()
    at_generating = {**OFF_POPULATION_PARAMS, "xi_0": 1.0, "xi_n": 2.3}
    np.testing.assert_array_equal(
        evaluate_log_weights(at_generating, **log_weight_kwargs(importance)),
        jnp.zeros(4),
    )

    spectrum, extras = importance_spectral_density(at_generating, **importance)
    np.testing.assert_allclose(
        spectrum,
        float(jnp.asarray(extras["total_merger_rate"])) * POWER.mean(axis=1),
        rtol=1e-14,
    )
    np.testing.assert_array_equal(extras["importance_relative_ess"], 1.0)


def _grid_reference(
    importance: Mapping[str, Any],
) -> Callable[[Mapping[str, ArrayLike]], tuple[jax.Array, dict[str, jax.Array]]]:
    """The same estimate, written out from the hand-written grid-level formula.

    Deliberately not routed through the source model: an expectation built
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
            - importance["proposal_log_prob"]
            - 2.0 * (log_target_distance - importance["log_reference_distance"])
        )
        return spectral_density(
            importance["polarization_power"],
            jnp.exp(log_weights),
            rate,
            average_mode=importance["average_mode"],
        ), {
            "total_merger_rate": jnp.asarray(rate),
            "importance_relative_ess": relative_ess(log_weights),
        }

    return evaluate


@pytest.mark.parametrize("params", [FIDUCIALS, OFF_FIDUCIALS])
@pytest.mark.parametrize("mode", ["analytic_inclination", "catalog_inclination"])
def test_spectrum_matches_the_grid_formula_spectrum_rate_and_ess(
    params: dict[str, float], mode: AverageMode
) -> None:
    importance = _importance(mode)
    actual = importance_spectral_density(params, **importance)
    expected = _grid_reference(importance)(params)
    assert set(actual[1]) == {"total_merger_rate", "importance_relative_ess"}
    for value, reference in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(value, reference, rtol=1e-12)


def test_spectrum_jit_vmap_and_gradients() -> None:
    importance = _importance()
    estimator = partial(importance_spectral_density, **importance)
    actual = jax.jit(estimator)(FIDUCIALS)
    for value, expected in zip(
        jax.tree.leaves(actual), jax.tree.leaves(estimator(FIDUCIALS)), strict=True
    ):
        np.testing.assert_allclose(value, expected, rtol=1e-12)
    # The same compiled function with different array data must use the new power.
    compiled = jax.jit(
        lambda power, params: importance_spectral_density(
            params, **{**importance, "polarization_power": power}
        )
    )
    np.testing.assert_allclose(compiled(POWER, FIDUCIALS)[0], actual[0], rtol=1e-12)
    np.testing.assert_allclose(
        compiled(2.0 * POWER, FIDUCIALS)[0], 2.0 * actual[0], rtol=1e-12
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
    grid_formula = _grid_reference(importance)
    reference = jax.grad(lambda p: jnp.sum(grid_formula(p)[0]) / normalization)(params)
    for name, value in gradient.items():
        assert np.isfinite(value), name
        np.testing.assert_allclose(
            value, reference[name], rtol=1e-9, atol=1e-12, err_msg=name
        )


def test_precomputed_mixture_density_is_used_in_estimate() -> None:
    """A proposal supplied as a bare array need not be a source model at all."""
    from astrogwb.distributions.redshift.madau_dickinson import (
        MadauDickinsonRedshiftDistribution,
    )

    importance = _importance()
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
    mixed = {**importance, "proposal_log_prob": mixture.log_prob(REDSHIFTS)}

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
        - 2.0 * (log_target_distance - mixed["log_reference_distance"])
    )
    expected = spectral_density(
        POWER,
        jnp.exp(manual_log_weights),
        rate,
        average_mode=mixed["average_mode"],
    )
    np.testing.assert_allclose(
        importance_spectral_density(FIDUCIALS, **mixed)[0], expected, rtol=1e-13
    )
