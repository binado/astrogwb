"""Exact Poisson-catalog forward model, checked against an explicit power sum."""

from collections.abc import Mapping
from functools import partial
from typing import Any, cast
from unittest.mock import Mock

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

from astrogwb.constants import INCLINATION_AVERAGE_TO_FACE_ON_RATIO, ISCO_ALPHA
from astrogwb.gwb.spectral import inclination_averaging_factor
from astrogwb.metadata import WaveformMetadata
from astrogwb.populations import IsotropicInclination
from astrogwb.sampling import gwb_forward_model, validate_source_model
from astrogwb.sampling.forward_model import _sum_polarization_power
from astrogwb.utils import years_to_seconds
from astrogwb.waveform import (
    AnalyticInspiralGenerator,
    PolarizationPowerGenerator,
    RippleGenerator,
)

N_EVENTS = 8
BATCH_SIZE = 3
F_MIN = 20.0
F_MAX = 40.0
DF = 10.0


def _generator() -> AnalyticInspiralGenerator:
    generator = AnalyticInspiralGenerator(
        WaveformMetadata(
            alpha=ISCO_ALPHA,
            approximant="AnalyticInspiral",
            minimum_frequency=F_MIN,
            maximum_frequency=F_MAX,
            reference_frequency=F_MIN,
            sampling_frequency=128.0,
            frequency_resolution=DF,
        )
    )
    return generator


def _ripple_generator() -> RippleGenerator:
    """TaylorF2 settings shared with :mod:`test_waveform_generator`.

    No warm-up call: the generator knows its grid from its configuration, so
    an empty-catalog spectrum can size itself without evaluating a waveform.
    """
    return RippleGenerator(
        WaveformMetadata(
            approximant="TaylorF2",
            sampling_frequency=256.0,
            minimum_frequency=20.0,
            maximum_frequency=100.0,
            reference_frequency=20.0,
            frequency_resolution=4.0,
        )
    )


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
        "max_events": N_EVENTS,
        "observed_num_events": N_EVENTS,
    }
    kwargs.update(overrides)
    return kwargs


def _ripple_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs = _model_kwargs(**overrides)
    if "generator" not in overrides:
        kwargs["generator"] = _ripple_generator()
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


def _expected_spectrum(trace, generator, observation_time):
    sources = {name: trace[name]["value"] for name in _plated_source_site_names(trace)}
    power = jnp.asarray(generator.generate_batch(sources))
    event_mask = jnp.arange(power.shape[-1]) < trace["n_events"]["value"]
    return (
        inclination_averaging_factor(sources)
        * (power * event_mask).sum(axis=1)
        / years_to_seconds(observation_time)
    )


def test_conditioned_poisson_rate_is_total_merger_rate_times_observation_seconds() -> (
    None
):
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


def test_missing_inclination_rescales_face_on_power() -> None:
    kwargs = _model_kwargs()
    analytic = _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **kwargs)
    sources = {
        name: analytic[name]["value"] for name in _plated_source_site_names(analytic)
    }
    power = _generator().generate_batch(sources)
    event_mask = jnp.arange(power.shape[-1]) < analytic["n_events"]["value"]
    np.testing.assert_allclose(
        analytic["spectral_density"]["value"],
        INCLINATION_AVERAGE_TO_FACE_ON_RATIO
        * (power * event_mask).sum(axis=1)
        / years_to_seconds(kwargs["observation_time"]),
        rtol=1e-12,
    )
    assert "inclination" not in sources


def test_returned_inclination_disables_analytic_rescaling() -> None:
    base_model = mock_population_model()

    def inclined_model(params: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        sources = dict(base_model(params))
        return {**sources, "inclination": jnp.zeros_like(sources["redshift"])}

    baseline = _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **_model_kwargs())
    inclined = _seeded_trace(
        gwb_forward_model,
        POPULATION_PARAMS,
        **_model_kwargs(source_model=inclined_model),
    )
    np.testing.assert_allclose(
        inclined["spectral_density"]["value"],
        baseline["spectral_density"]["value"] / INCLINATION_AVERAGE_TO_FACE_ON_RATIO,
        rtol=1e-12,
    )

    model = partial(gwb_forward_model, **_model_kwargs(source_model=inclined_model))

    def spectrum(values: dict[str, jax.Array]) -> jax.Array:
        trace = handlers.trace(handlers.seed(model, 0)).get_trace(values)
        return trace["spectral_density"]["value"]

    np.testing.assert_allclose(
        jax.jit(spectrum)(_jax_params()), spectrum(_jax_params())
    )


def test_isotropic_inclination_messenger_disables_analytic_rescaling() -> None:
    wrapped = IsotropicInclination(mock_population_model())
    trace = _seeded_trace(
        gwb_forward_model,
        POPULATION_PARAMS,
        **_model_kwargs(source_model=wrapped),
    )
    sources = {name: trace[name]["value"] for name in _plated_source_site_names(trace)}
    assert "inclination" in sources
    assert inclination_averaging_factor(sources) == 1.0
    np.testing.assert_allclose(
        trace["spectral_density"]["value"],
        _expected_spectrum(trace, _generator(), _model_kwargs()["observation_time"]),
        rtol=1e-12,
    )


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


def test_unobserved_poisson_count_is_jittable_and_keeps_static_shapes() -> None:
    kwargs = _model_kwargs(observed_num_events=None)

    def outputs(params):
        trace = _seeded_trace(gwb_forward_model, params, **kwargs)
        return (
            trace["spectral_density"]["value"],
            trace["n_events"]["value"],
            tuple(trace[name]["value"] for name in _plated_source_site_names(trace)),
        )

    eager_spectrum, eager_n, eager_sources = outputs(_jax_params())
    compiled_spectrum, compiled_n, compiled_sources = jax.jit(outputs)(_jax_params())

    assert not _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **kwargs)[
        "n_events"
    ]["is_observed"]
    assert np.shape(eager_spectrum) == np.shape(compiled_spectrum)
    assert np.shape(eager_n) == ()
    np.testing.assert_allclose(compiled_spectrum, eager_spectrum, rtol=1e-12)
    np.testing.assert_array_equal(compiled_n, eager_n)
    assert all(source.shape == (kwargs["max_events"],) for source in eager_sources)
    assert all(source.shape == (kwargs["max_events"],) for source in compiled_sources)


def test_unobserved_poisson_predictive_stacks_static_capacity() -> None:
    kwargs = _model_kwargs(observed_num_events=None)
    draws = Predictive(
        partial(gwb_forward_model, **kwargs),
        num_samples=3,
        return_sites=None,
    )(jax.random.key(2), POPULATION_PARAMS)

    assert draws["n_events"].shape == (3,)
    assert draws["spectral_density"].shape == (
        3,
        _generator().frequencies.shape[0],
    )
    for name in ("redshift", "luminosity_distance"):
        assert draws[name].shape == (3, kwargs["max_events"])


def test_zero_conditioned_count_has_zero_power_and_static_sources() -> None:
    kwargs = _model_kwargs(observed_num_events=0)
    trace = _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **kwargs)

    np.testing.assert_array_equal(trace["n_events"]["value"], 0)
    np.testing.assert_array_equal(
        trace["spectral_density"]["value"],
        np.zeros(_generator().frequencies.shape[0]),
    )
    for name in _plated_source_site_names(trace):
        assert trace[name]["value"].shape == (kwargs["max_events"],)


def test_partial_count_masks_power_but_not_source_capacity() -> None:
    active_events = 3
    kwargs = _model_kwargs(observed_num_events=active_events)
    trace = _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **kwargs)
    generator = kwargs["generator"]
    sources = {name: trace[name]["value"] for name in _plated_source_site_names(trace)}
    reference_power = jnp.asarray(generator.generate_batch(sources))[:, :active_events]

    np.testing.assert_allclose(
        trace["spectral_density"]["value"],
        INCLINATION_AVERAGE_TO_FACE_ON_RATIO
        * reference_power.sum(axis=1)
        / years_to_seconds(kwargs["observation_time"]),
        rtol=1e-12,
    )
    assert all(
        trace[name]["value"].shape == (kwargs["max_events"],)
        for name in _plated_source_site_names(trace)
    )


def test_count_above_capacity_is_silently_capped() -> None:
    max_events = 4
    observed_count = max_events + 3
    kwargs = _model_kwargs(
        max_events=max_events,
        observed_num_events=observed_count,
    )
    trace = _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **kwargs)
    sources = {name: trace[name]["value"] for name in _plated_source_site_names(trace)}
    reference = jnp.asarray(kwargs["generator"].generate_batch(sources)).sum(axis=1)

    np.testing.assert_array_equal(trace["n_events"]["value"], observed_count)
    np.testing.assert_allclose(
        trace["spectral_density"]["value"],
        INCLINATION_AVERAGE_TO_FACE_ON_RATIO
        * reference
        / years_to_seconds(kwargs["observation_time"]),
        rtol=1e-12,
    )
    assert all(source.shape == (max_events,) for source in sources.values())


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


@pytest.mark.integration
def test_ripple_spectrum_matches_the_sum_of_per_source_power_over_time() -> None:
    generator = _ripple_generator()
    kwargs = _ripple_kwargs(generator=generator)
    trace = _seeded_trace(gwb_forward_model, POPULATION_PARAMS, **kwargs)
    expected = _expected_spectrum(
        trace,
        generator,
        kwargs["observation_time"],
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


# --------------------------------------------------------------------- #
# The scan replaced an unrolled Python loop
# --------------------------------------------------------------------- #
def _unrolled_power_sum(generator, sources, event_mask, *, batch_size: int):
    """The loop ``_sum_polarization_power`` used to be, kept as the oracle.

    The scan must agree with it for every catalog shape, including the ragged
    and empty ones where the chunking arithmetic is easiest to get wrong.
    """
    n_events = next(iter(sources.values())).shape[0]
    n_full, remainder = divmod(n_events, batch_size)

    def slice_sum(start: int, size: int):
        batch = {name: values[start : start + size] for name, values in sources.items()}
        power = jnp.asarray(generator.generate_batch(batch))
        return (power * event_mask[start : start + size]).sum(axis=1)

    if n_full:
        total = slice_sum(0, batch_size)
        for index in range(1, n_full):
            total = total + slice_sum(index * batch_size, batch_size)
        if not remainder:
            return total
        return total + slice_sum(n_full * batch_size, remainder)
    if remainder:
        return slice_sum(0, remainder)
    return jnp.zeros(jnp.shape(generator.frequencies)[0], dtype=jnp.float64)


def _draw_sources(max_events: int) -> dict[str, jax.Array]:
    trace = _seeded_trace(
        gwb_forward_model,
        _jax_params(),
        **_model_kwargs(max_events=max_events, observed_num_events=max_events),
    )
    return {name: trace[name]["value"] for name in _plated_source_site_names(trace)}


@pytest.mark.parametrize("batch_size", [1, 3, 5, 8, 13])
@pytest.mark.parametrize("n_events", [1, 5, 8, 15])
def test_scan_matches_the_unrolled_loop(n_events: int, batch_size: int) -> None:
    """Full, ragged and single-event catalogs, across chunk sizes."""
    generator = _generator()
    sources = _draw_sources(n_events)
    event_mask = jnp.arange(n_events) < n_events

    np.testing.assert_allclose(
        np.asarray(
            _sum_polarization_power(
                generator, sources, event_mask, batch_size=batch_size
            )
        ),
        np.asarray(
            _unrolled_power_sum(generator, sources, event_mask, batch_size=batch_size)
        ),
        rtol=1e-12,
    )


def test_empty_catalog_reduces_without_calling_the_generator() -> None:
    """Zero events is a static branch, not a zero-length scan.

    A ``lax.scan`` over no chunks still traces its body, which for a Ripple
    generator means building a waveform for a catalog that does not exist.
    """
    real = _generator()

    class _CountingGenerator:
        """Only ``frequencies`` may be read when the catalog is empty."""

        frequencies = real.frequencies
        generate_batch = Mock(side_effect=AssertionError("generator was called"))

    sources = {name: values[:0] for name, values in _draw_sources(4).items()}

    total = _sum_polarization_power(
        cast(PolarizationPowerGenerator, _CountingGenerator()),
        sources,
        jnp.zeros(0, dtype=bool),
        batch_size=BATCH_SIZE,
    )

    _CountingGenerator.generate_batch.assert_not_called()
    np.testing.assert_array_equal(
        np.asarray(total), np.zeros(np.shape(real.frequencies)[0])
    )


@pytest.mark.parametrize("n_events", [6, 7])
def test_jitted_scan_matches_eager_for_full_and_ragged_catalogs(n_events: int) -> None:
    """``n_events=6`` divides ``BATCH_SIZE``; ``7`` leaves a remainder chunk."""
    kwargs = _model_kwargs(max_events=n_events, observed_num_events=n_events)
    eager = _seeded_trace(gwb_forward_model, _jax_params(), **kwargs)

    def spectrum(params):
        trace = _seeded_trace(gwb_forward_model, params, **kwargs)
        return trace["spectral_density"]["value"]

    np.testing.assert_allclose(
        np.asarray(jax.jit(spectrum)(_jax_params())),
        np.asarray(eager["spectral_density"]["value"]),
        rtol=1e-12,
    )


def test_vmap_over_draws_shares_one_static_event_count() -> None:
    """Two hyperparameter draws at one catalog size, mapped rather than looped."""
    kwargs = _model_kwargs()

    def spectrum(params):
        return _seeded_trace(gwb_forward_model, params, **kwargs)["spectral_density"][
            "value"
        ]

    params = _jax_params()
    stacked = {name: jnp.stack([value, value]) for name, value in params.items()}
    mapped = jax.vmap(spectrum)(stacked)

    assert mapped.shape == (2, np.shape(kwargs["generator"].frequencies)[0])
    np.testing.assert_allclose(
        np.asarray(mapped[0]), np.asarray(spectrum(params)), rtol=1e-12
    )


# --------------------------------------------------------------------- #
# Validation the traced model cannot do for itself
# --------------------------------------------------------------------- #
def test_validate_source_model_accepts_a_matched_population() -> None:
    validate_source_model(
        _jax_params(),
        source_model=mock_population_model(),
        generator=_generator(),
        rng_key=jax.random.key(0),
    )


def test_validate_source_model_rejects_a_mismatched_approximant() -> None:
    """A tidal population against an aligned-spin model.

    Nothing downstream would complain: the approximant simply never reads the
    deformabilities, so the spectrum comes out quietly wrong. Catching it is
    the whole reason this helper exists.
    """
    aligned_spin = RippleGenerator(
        WaveformMetadata(
            approximant="IMRPhenomXAS",
            sampling_frequency=256.0,
            minimum_frequency=20.0,
            maximum_frequency=100.0,
            reference_frequency=20.0,
            frequency_resolution=4.0,
        )
    )

    def tidal_population(params):
        sources = dict(mock_population_model()(params))
        sources["lambda_1"] = jnp.full_like(sources["luminosity_distance"], 300.0)
        return sources

    with pytest.raises(ValueError, match="no tidal deformability"):
        validate_source_model(
            _jax_params(),
            source_model=tidal_population,
            generator=aligned_spin,
            rng_key=jax.random.key(0),
        )


def test_validate_source_model_requires_luminosity_distance() -> None:
    def no_distance(params):
        sources = dict(mock_population_model()(params))
        del sources["luminosity_distance"]
        return sources

    with pytest.raises(KeyError, match="luminosity_distance"):
        validate_source_model(
            _jax_params(),
            source_model=no_distance,
            generator=_generator(),
            rng_key=jax.random.key(0),
        )


# --------------------------------------------------------------------- #
# Ripple inside the model, which the eager caveat used to forbid
# --------------------------------------------------------------------- #
@pytest.mark.integration
def test_ripple_forward_model_is_jittable() -> None:
    """The caveat this whole change removes: Ripple under a trace."""
    kwargs = _ripple_kwargs()

    def spectrum(params):
        return _seeded_trace(gwb_forward_model, params, **kwargs)["spectral_density"][
            "value"
        ]

    jitted = np.asarray(jax.jit(spectrum)(_jax_params()))

    assert np.all(np.isfinite(jitted))
    assert np.all(jitted > 0.0)
    np.testing.assert_allclose(jitted, np.asarray(spectrum(_jax_params())), rtol=1e-12)


@pytest.mark.integration
def test_ripple_predictive_stacks_multiple_draws() -> None:
    """``num_samples > 1`` maps the model, which eager Ripple could not survive."""
    kwargs = _ripple_kwargs()
    draws = Predictive(
        partial(gwb_forward_model, **kwargs),
        num_samples=2,
        return_sites=("spectral_density", "n_events"),
    )(jax.random.key(0), _jax_params())

    frequencies = np.shape(kwargs["generator"].frequencies)[0]
    assert draws["spectral_density"].shape == (2, frequencies)
    assert np.all(np.isfinite(draws["spectral_density"]))


@pytest.mark.integration
def test_ripple_waveform_power_is_differentiable() -> None:
    """A prerequisite for gradient-based inference, not a demonstration of it.

    This differentiates deterministic waveform power through the generator.
    Differentiating a NumPyro population draw is a separate question and this
    says nothing about it.
    """
    generator = _ripple_generator()
    sources = _draw_sources(4)
    sources = {
        "detector_frame_mass_1": jnp.full((2,), 1.4),
        "detector_frame_mass_2": jnp.full((2,), 1.3),
        "inclination": jnp.zeros((2,)),
        "luminosity_distance": jnp.array([100.0, 200.0]),
    }

    def total_power(chirp_scale):
        scaled = dict(sources)
        scaled["detector_frame_mass_1"] = sources["detector_frame_mass_1"] * chirp_scale
        return jnp.sum(generator.generate_batch(scaled))

    gradient = float(jax.grad(total_power)(1.0))
    step = 1e-4
    numerical = float(
        (total_power(1.0 + step) - total_power(1.0 - step)) / (2.0 * step)
    )

    assert np.isfinite(gradient)
    assert gradient != 0.0
    np.testing.assert_allclose(gradient, numerical, rtol=1e-4)
