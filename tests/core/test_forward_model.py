"""Exact Poisson-catalog forward model, checked against an explicit power sum."""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb_mock_population import (
    POPULATION_PARAMS,
    mock_merger_rate_fn,
    mock_population_model,
)
from numpyro import handlers
from numpyro.infer import Predictive
from numpyro.infer.util import log_density

from astrogwb.constants import INCLINATION_AVERAGE_TO_FACE_ON_RATIO, ISCO_ALPHA
from astrogwb.sampling import gwb_forward_model
from astrogwb.utils import years_to_seconds
from astrogwb.waveform import AnalyticInspiralGenerator, RippleGenerator

N_EVENTS = 8
BATCH_SIZE = 3
F_MIN = 20.0
F_MAX = 40.0
DF = 10.0


def _generator() -> AnalyticInspiralGenerator:
    generator = AnalyticInspiralGenerator(
        alpha=ISCO_ALPHA,
        approximant="AnalyticInspiral",
        minimum_frequency=F_MIN,
        maximum_frequency=F_MAX,
        reference_frequency=F_MIN,
        sampling_frequency=128.0,
        frequency_resolution=DF,
    )
    return generator


def _ripple_generator(*, chunk_size: int) -> RippleGenerator:
    """TaylorF2 settings shared with :mod:`test_waveform_generator`.

    ``chunk_size`` is at least the forward-model ``batch_size`` so Ripple
    does not chunk again inside each batched generate. One generate warms
    the frequency cache so an empty-catalog spectrum can read it.
    """
    generator = RippleGenerator(
        approximant="TaylorF2",
        sampling_frequency=256.0,
        minimum_frequency=20.0,
        maximum_frequency=100.0,
        reference_frequency=20.0,
        frequency_resolution=4.0,
        chunk_size=chunk_size,
    )
    ones = jnp.ones((1,))
    _ = generator(
        {
            "detector_frame_mass_1": 1.4 * ones,
            "detector_frame_mass_2": 1.3 * ones,
            "inclination": 0.0 * ones,
            "luminosity_distance": 100.0 * ones,
            "lambda_1": 400.0 * ones,
            "lambda_2": 300.0 * ones,
        }
    )
    return generator


def _observation_time_for(expected_events: float) -> float:
    """Years of observation such that ``R * T = expected_events``."""
    rate = mock_merger_rate_fn()(POPULATION_PARAMS)
    return expected_events / (float(rate) * years_to_seconds(1.0))


def _jax_params() -> dict[str, jax.Array]:
    return {name: jnp.asarray(value) for name, value in POPULATION_PARAMS.items()}


def _model_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "source_model": mock_population_model(),
        "merger_rate_fn": mock_merger_rate_fn(),
        "generator": _generator(),
        "observation_time": _observation_time_for(N_EVENTS),
        "batch_size": BATCH_SIZE,
        "num_events": N_EVENTS,
        "average_mode": "catalog_inclination",
    }
    kwargs.update(overrides)
    return kwargs


def _ripple_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs = _model_kwargs(**overrides)
    if "generator" not in overrides:
        kwargs["generator"] = _ripple_generator(
            chunk_size=max(kwargs["batch_size"], kwargs["num_events"])
        )
    return kwargs


def _seeded_trace(model, params, **kwargs):
    return handlers.trace(handlers.seed(model, 0)).get_trace(params, **kwargs)


def _plated_source_site_names(trace) -> list[str]:
    """Every site declared inside the ``events`` plate -- the source columns.

    A structural selector rather than a static name list: any site whose
    ``cond_indep_stack`` is non-empty was declared inside the plate, which is
    exactly the set ``source_model`` returns from one execution.
    """
    return [
        name
        for name, site in trace.items()
        if site["type"] in ("sample", "deterministic") and site["cond_indep_stack"]
    ]


def _plated_sample_site_names(trace) -> list[str]:
    return [
        name
        for name, site in trace.items()
        if site["type"] == "sample" and site["cond_indep_stack"]
    ]


def _expected_spectrum(trace, generator, observation_time, *, average_mode):
    sources = {name: trace[name]["value"] for name in _plated_source_site_names(trace)}
    power = jnp.asarray(generator.generate_batch(sources))
    factor = (
        INCLINATION_AVERAGE_TO_FACE_ON_RATIO
        if average_mode == "analytic_inclination"
        else 1.0
    )
    return factor * power.sum(axis=1) / years_to_seconds(observation_time)


def test_poisson_rate_is_total_merger_rate_times_observation_seconds() -> None:
    kwargs = _model_kwargs()
    trace = _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **kwargs)
    rate = trace["total_merger_rate"]["value"]
    observation_time_sec = years_to_seconds(kwargs["observation_time"])
    np.testing.assert_allclose(
        float(trace["n_events"]["fn"].rate),
        float(rate * observation_time_sec),
        rtol=1e-12,
    )
    assert trace["n_events"]["type"] == "sample"
    assert trace["n_events"]["is_observed"]
    np.testing.assert_array_equal(trace["n_events"]["value"], N_EVENTS)
    assert trace["spectral_density"]["type"] == "deterministic"
    assert "spectral_density_obs" not in trace
    assert trace["redshift"]["value"].shape == (N_EVENTS,)


def test_spectrum_matches_the_sum_of_per_source_power_over_time() -> None:
    generator = _generator()
    kwargs = _model_kwargs(generator=generator)
    trace = _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **kwargs)
    expected = _expected_spectrum(
        trace,
        generator,
        kwargs["observation_time"],
        average_mode="catalog_inclination",
    )
    np.testing.assert_allclose(trace["spectral_density"]["value"], expected, rtol=1e-12)
    np.testing.assert_array_equal(trace["n_events"]["value"], N_EVENTS)


@pytest.mark.parametrize("batch_size", [1, N_EVENTS, N_EVENTS + 5])
def test_batched_power_matches_a_single_generator_call(batch_size: int) -> None:
    generator = _generator()
    kwargs = _model_kwargs(generator=generator, batch_size=batch_size)
    batched = _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **kwargs)
    expected = _expected_spectrum(
        batched,
        generator,
        kwargs["observation_time"],
        average_mode="catalog_inclination",
    )
    np.testing.assert_allclose(
        batched["spectral_density"]["value"], expected, rtol=1e-12
    )


def test_batch_size_does_not_change_the_spectrum() -> None:
    kwargs = _model_kwargs()
    first = _seeded_trace(
        gwb_forward_model, POPULATION_PARAMS, **{**kwargs, "batch_size": 1}
    )
    second = _seeded_trace(
        gwb_forward_model, POPULATION_PARAMS, **{**kwargs, "batch_size": N_EVENTS}
    )
    np.testing.assert_allclose(
        first["spectral_density"]["value"],
        second["spectral_density"]["value"],
        rtol=1e-12,
    )
    for name in _plated_source_site_names(first):
        np.testing.assert_array_equal(first[name]["value"], second[name]["value"])


def test_analytic_inclination_rescales_face_on_power() -> None:
    kwargs = _model_kwargs()
    catalog = _seeded_trace(
        gwb_forward_model,
        POPULATION_PARAMS,
        **{**kwargs, "average_mode": "catalog_inclination"},
    )
    analytic = _seeded_trace(
        gwb_forward_model,
        POPULATION_PARAMS,
        **{**kwargs, "average_mode": "analytic_inclination"},
    )
    np.testing.assert_allclose(
        analytic["spectral_density"]["value"],
        INCLINATION_AVERAGE_TO_FACE_ON_RATIO * catalog["spectral_density"]["value"],
        rtol=1e-12,
    )
    np.testing.assert_array_equal(catalog["inclination"]["value"], jnp.zeros(N_EVENTS))


def test_predictive_stacks_fixed_shape_sites() -> None:
    kwargs = _model_kwargs()
    draws = Predictive(
        partial(gwb_forward_model, **kwargs),
        num_samples=3,
        return_sites=("spectral_density", "n_events", "total_merger_rate"),
    )(jax.random.key(1), POPULATION_PARAMS)
    assert draws["spectral_density"].shape == (3, _generator().frequencies.shape[0])
    assert draws["n_events"].shape == (3,)
    assert draws["total_merger_rate"].shape == (3,)
    assert bool(jnp.all(jnp.isfinite(draws["spectral_density"])))
    assert "redshift" not in draws


def test_jitted_spectrum_matches_eager() -> None:
    kwargs = _model_kwargs()
    model = partial(gwb_forward_model, **kwargs)
    params = _jax_params()

    def spectrum(values: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        trace = handlers.trace(handlers.seed(model, 0)).get_trace(values)
        return trace["spectral_density"]["value"], trace["n_events"]["value"]

    eager_spectrum, eager_n = spectrum(params)
    compiled_spectrum, compiled_n = jax.jit(spectrum)(params)
    np.testing.assert_allclose(compiled_spectrum, eager_spectrum, rtol=1e-12)
    np.testing.assert_array_equal(compiled_n, eager_n)


def test_missing_physical_rate_is_rejected() -> None:
    """The rate model owns this check now, not an ``"x" in params`` branch."""
    params = {
        name: value
        for name, value in POPULATION_PARAMS.items()
        if name != "local_merger_rate"
    }
    with pytest.raises(ValueError, match="local_merger_rate"):
        _seeded_trace(gwb_forward_model, params, **_model_kwargs())


@pytest.mark.parametrize("n_active", [1, 3, 7])
def test_masked_sum_matches_the_truncated_sum_exactly(n_active: int) -> None:
    generator = _generator()
    kwargs = _model_kwargs(generator=generator)
    trace = _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **{**kwargs, "n_active": n_active})
    sources = {name: trace[name]["value"][:n_active] for name in _plated_source_site_names(trace)}
    expected = jnp.asarray(generator.generate_batch(sources)).sum(axis=1) / years_to_seconds(
        kwargs["observation_time"]
    )
    np.testing.assert_allclose(
        np.asarray(trace["spectral_density"]["value"]),
        np.asarray(expected),
        rtol=1e-12,
    )
    np.testing.assert_array_equal(trace["n_events"]["value"], n_active)


def test_n_active_none_preserves_default_behavior() -> None:
    kwargs = _model_kwargs()
    baseline = _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **kwargs)
    explicit_none = _seeded_trace(
        gwb_forward_model, POPULATION_PARAMS, **{**kwargs, "n_active": None}
    )
    np.testing.assert_allclose(
        np.asarray(explicit_none["spectral_density"]["value"]),
        np.asarray(baseline["spectral_density"]["value"]),
        rtol=1e-12,
    )
    np.testing.assert_array_equal(explicit_none["n_events"]["value"], N_EVENTS)


@pytest.mark.parametrize("n_active", [1, 2, 5])
def test_padded_and_unpadded_log_density_match(n_active: int) -> None:
    padded_kwargs = _model_kwargs(num_events=N_EVENTS, n_active=n_active)
    padded_trace = _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **padded_kwargs)
    sample_names = _plated_sample_site_names(padded_trace)
    padded_values = {name: padded_trace[name]["value"] for name in sample_names}
    padded_log_density, _ = log_density(
        gwb_forward_model,
        (POPULATION_PARAMS,),
        padded_kwargs,
        padded_values,
    )

    unpadded_kwargs = _model_kwargs(num_events=n_active)
    unpadded_values = {name: padded_trace[name]["value"][:n_active] for name in sample_names}
    unpadded_log_density, _ = log_density(
        gwb_forward_model,
        (POPULATION_PARAMS,),
        unpadded_kwargs,
        unpadded_values,
    )

    np.testing.assert_allclose(
        np.asarray(padded_log_density),
        np.asarray(unpadded_log_density),
        rtol=1e-10,
    )


def test_jitted_model_compiles_once_for_many_active_counts() -> None:
    calls: list[None] = []
    source_model = mock_population_model()

    def counting_source_model(params):
        calls.append(None)
        return source_model(params)

    kwargs = _model_kwargs(source_model=counting_source_model, num_events=N_EVENTS)

    def spectrum(n_active: jax.Array) -> jax.Array:
        trace = handlers.trace(handlers.seed(gwb_forward_model, 0)).get_trace(
            POPULATION_PARAMS, **{**kwargs, "n_active": n_active}
        )
        return trace["spectral_density"]["value"]

    compiled = jax.jit(spectrum)
    for n_active in (1, 2, 4, 7):
        result = compiled(jnp.asarray(n_active))
        assert result.shape == (_generator().frequencies.shape[0],)
    assert len(calls) == 1


@pytest.mark.integration
def test_ripple_spectrum_matches_the_sum_of_per_source_power_over_time() -> None:
    generator = _ripple_generator(chunk_size=N_EVENTS)
    kwargs = _ripple_kwargs(generator=generator)
    trace = _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **kwargs)
    expected = _expected_spectrum(
        trace,
        generator,
        kwargs["observation_time"],
        average_mode="catalog_inclination",
    )
    np.testing.assert_allclose(trace["spectral_density"]["value"], expected, rtol=1e-12)
    np.testing.assert_array_equal(trace["n_events"]["value"], N_EVENTS)
    assert bool(jnp.all(jnp.isfinite(trace["spectral_density"]["value"])))
    assert bool(jnp.all(trace["spectral_density"]["value"] >= 0.0))


@pytest.mark.integration
@pytest.mark.parametrize("batch_size", [1, N_EVENTS, N_EVENTS + 5])
def test_ripple_batched_power_matches_a_single_generator_call(batch_size: int) -> None:
    kwargs = _ripple_kwargs(batch_size=batch_size)
    generator = kwargs["generator"]
    batched = _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **kwargs)
    expected = _expected_spectrum(
        batched,
        generator,
        kwargs["observation_time"],
        average_mode="catalog_inclination",
    )
    np.testing.assert_allclose(
        batched["spectral_density"]["value"], expected, rtol=1e-12
    )


@pytest.mark.integration
def test_ripple_batch_size_does_not_change_the_spectrum() -> None:
    first = _seeded_trace(
        gwb_forward_model, POPULATION_PARAMS, **_ripple_kwargs(batch_size=1)
    )
    second = _seeded_trace(
        gwb_forward_model, POPULATION_PARAMS, **_ripple_kwargs(batch_size=N_EVENTS)
    )
    np.testing.assert_allclose(
        first["spectral_density"]["value"],
        second["spectral_density"]["value"],
        rtol=1e-12,
    )
    for name in _plated_source_site_names(first):
        np.testing.assert_array_equal(first[name]["value"], second[name]["value"])


@pytest.mark.integration
def test_ripple_predictive_returns_finite_spectrum() -> None:
    """One Predictive draw is eager; ``num_samples>1`` would ``lax.map`` into gwmock."""
    kwargs = _ripple_kwargs()
    draws = Predictive(
        partial(gwb_forward_model, **kwargs),
        num_samples=1,
        return_sites=("spectral_density", "n_events", "total_merger_rate"),
    )(jax.random.key(1), POPULATION_PARAMS)
    assert draws["spectral_density"].shape == (
        1,
        kwargs["generator"].frequencies.shape[0],
    )
    assert draws["n_events"].shape == (1,)
    assert draws["total_merger_rate"].shape == (1,)
    assert bool(jnp.all(jnp.isfinite(draws["spectral_density"])))
    assert bool(jnp.all(draws["spectral_density"] >= 0.0))
    assert "redshift" not in draws
