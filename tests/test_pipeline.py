"""The population and background stages, end to end.

Everything is generated for real on the closed-form inspiral: a tiny Poisson
population and a four-bin grid. The Monte Carlo check compares in units of the
sample's own standard error, so it holds for any seed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import jax
import numpy as np
import pytest
from astrogwb_tiny_pipeline import (
    PAIR,
    TRIANGLE,
    WAVEFORM,
    event_pool,
)
from jax.typing import ArrayLike
from tolerances import assert_close

from astrogwb.background.simulation import (
    BackgroundDraw,
    project_background,
    simulate_background,
)
from astrogwb.background.simulation import draw as background_module
from astrogwb.detector import (
    Network,
    isotropic_overlap_reduction_function,
    load_detector,
    network_pairs,
    pair_cross_power_std,
)
from astrogwb.detector.projection import project_polarizations_fd
from astrogwb.statistics import cross_correlation_estimator
from astrogwb.utils import years_to_seconds
from astrogwb.waveform import (
    WaveformConfig,
    WaveformGenerator,
    intensity,
    waveform_generator,
)
from astrogwb.waveform.analytical import AnalyticInspiralGenerator

#: Standard errors a Monte Carlo mean may sit from its expectation.
MAX_STANDARD_ERRORS = 5.0


type Pool = dict[str, np.ndarray]
OBSERVATION_SECONDS = 3000.0


@pytest.fixture(scope="module")
def population() -> Pool:
    return event_pool()


@pytest.fixture(scope="module")
def generator() -> WaveformGenerator:
    return waveform_generator(WAVEFORM)


def _project(
    population: Pool,
    generator: WaveformGenerator,
    network: Network,
    **kwargs,
) -> BackgroundDraw:
    return project_background(
        population,
        generator,
        network,
        observation_seconds=OBSERVATION_SECONDS,
        **kwargs,
    )


def _simulate(
    population: Pool,
    generator: WaveformGenerator,
    network: Network,
    *,
    segment_duration: float,
    seed: int = 0,
    **kwargs,
) -> BackgroundDraw:
    return simulate_background(
        jax.random.key(seed),
        population,
        generator,
        network,
        segment_duration=segment_duration,
        num_segments=int(OBSERVATION_SECONDS // segment_duration),
        poisson_rate=len(population["redshift"]) / (2 * OBSERVATION_SECONDS),
        **kwargs,
    )


@pytest.mark.parametrize(
    "waveform",
    [
        WAVEFORM.model_copy(update={"frequency_resolution": 2.5}),
        WAVEFORM.model_copy(update={"frequency_spacing": "log"}),
    ],
    ids=["fractional-modes", "log-grid"],
)
@pytest.mark.parametrize("noise", [False, True])
def test_segmented_simulation_needs_whole_fourier_modes_per_bin(
    population: Pool, waveform: WaveformConfig, noise: bool
) -> None:
    with pytest.raises(ValueError, match="whole number"):
        _simulate(
            population,
            waveform_generator(waveform),
            TRIANGLE,
            segment_duration=0.5,
            simulate_detector_noise=noise,
        )


@pytest.mark.parametrize(
    "approximant",
    ["AnalyticInspiral", pytest.param("IMRPhenomD", marks=pytest.mark.integration)],
)
def test_coa_time_shifts_both_polarizations_and_preserves_intensity(
    population: Pool, approximant: str
) -> None:
    generator = waveform_generator(
        WaveformConfig.model_validate(
            {
                **WAVEFORM.model_dump(exclude={"alpha"}),
                "approximant": approximant,
                "frequency_resolution": 1.0,
            }
        )
    )
    sources = {
        name: values[:2] for name, values in population.items() if name != "coa_time"
    }
    # Opposite quarter-cycle shifts at the first frequency: fixes the sign
    # and checks independent source times rather than a shared batch phase.
    frequencies = np.asarray(generator.frequencies)
    times = np.array([1.0, -1.0]) / (4 * frequencies[0])
    base = generator.polarizations(sources)
    shifted = jax.jit(generator.polarizations)({**sources, "coa_time": times})
    expected = np.exp(-2j * np.pi * times[:, None] * frequencies)
    for original, timed in zip(base, shifted, strict=True):
        assert_close(timed / original, expected)
    intensity_ratio = intensity(*shifted) / intensity(*base)
    assert_close(intensity_ratio, np.ones(intensity_ratio.shape))
    np.testing.assert_array_equal(
        generator.intensity(sources),
        generator.intensity({**sources, "coa_time": times}),
    )

    # At each physical site the arrival phase multiplies the strain once;
    # the baseline already includes that site's geometric delay.
    projected = []
    for detector in tuple(PAIR[member] for member in sorted(PAIR)):
        coordinates = {
            name: sources[name]
            for name in ("right_ascension", "declination", "polarization_angle")
        }
        args = (
            detector.geometry.response_tensor,
            detector.geometry.location,
            frequencies,
        )
        original = project_polarizations_fd(*args, *base, 0.0, **coordinates)
        timed = project_polarizations_fd(
            *args, shifted[0], shifted[1], 0.0, **coordinates
        )
        assert_close(timed / original, expected)
        projected.append((original, timed))
    first, second = projected
    cross_ratio = (np.conj(first[1]) * second[1]) / (np.conj(first[0]) * second[0])
    assert_close(cross_ratio, np.ones(cross_ratio.shape, dtype=np.complex128))


@pytest.mark.parametrize(
    "approximant",
    ["AnalyticInspiral", pytest.param("IMRPhenomD", marks=pytest.mark.integration)],
)
def test_polarizations_reduce_to_the_generators_intensity(
    population: Pool, approximant: str
) -> None:
    # A 1 Hz grid puts the band's edges on Ripple's FFT grid.
    generator = waveform_generator(
        WaveformConfig.model_validate(
            {
                **WAVEFORM.model_dump(exclude={"alpha"}),
                "approximant": approximant,
                "frequency_resolution": 1.0,
            }
        )
    )
    sources = population

    plus, cross = generator.polarizations(sources)

    generated = np.asarray(generator.intensity(sources))
    assert_close(intensity(plus, cross).T, generated)


def test_pair_spectra_average_to_the_orf_times_the_sky_averaged_spectrum(
    generator: WaveformGenerator,
) -> None:
    r"""Over isotropic sources, :math:`E[Y_{ab}] = \gamma_{ab} S_h`.

    One realization is split into groups; in each, the ratio of every pair
    spectrum to the group's sky-averaged spectrum :math:`\sum_i I_i / T`
    isolates the response, and its mean over groups is :math:`\gamma_{ab}`.
    """
    realization = event_pool(4096, seed=7)
    group = 16
    ratios = []
    for start in range(0, len(realization["redshift"]) - group + 1, group):
        sources = {
            name: values[start : start + group] for name, values in realization.items()
        }
        generated = np.asarray(generator.intensity(sources)).sum(axis=0)
        spectra = _project(
            sources, generator, TRIANGLE, chunk_size=8
        ).pair_spectral_density
        ratios.append(spectra / (generated / OBSERVATION_SECONDS))
    ratios = np.asarray(ratios)
    expected = np.stack(
        [
            isotropic_overlap_reduction_function(
                np.asarray(generator.frequencies), TRIANGLE[first], TRIANGLE[second]
            )
            for first, second in network_pairs(TRIANGLE)
        ]
    )
    errors = ratios.std(axis=0, ddof=1) / np.sqrt(ratios.shape[0])

    assert ratios.shape[0] >= 50
    assert np.all(np.abs(ratios.mean(axis=0) - expected) < MAX_STANDARD_ERRORS * errors)


def test_an_empty_realization_has_zero_spectra(
    empty: Pool, generator: WaveformGenerator
) -> None:
    backgrounds = (
        _project(empty, generator, TRIANGLE),
        _simulate(empty, generator, TRIANGLE, segment_duration=60.0),
    )
    for background in backgrounds:
        pairs = background.pair_spectral_density
        assert pairs.shape == (
            len(network_pairs(TRIANGLE)),
            background.frequencies.size,
        )
        assert not np.any(pairs)


@pytest.fixture(scope="module")
def projected(population: Pool, generator: WaveformGenerator) -> BackgroundDraw:
    return _project(population, generator, PAIR, chunk_size=8)


@pytest.mark.parametrize("network", [PAIR, TRIANGLE], ids=["pair", "triangle"])
def test_combining_pair_spectra_of_a_pure_signal_recovers_its_spectrum(
    generator: WaveformGenerator, network: Network
) -> None:
    frequencies = generator.frequencies
    spectrum = 1e-47 * (1.0 + np.arange(frequencies.size))
    overlap = np.stack(
        [
            isotropic_overlap_reduction_function(
                frequencies, network[first], network[second]
            )
            for first, second in network_pairs(network)
        ]
    )

    combined = cross_correlation_estimator(
        overlap * spectrum,
        frequencies,
        network,
    )

    reordered = dict(reversed(tuple(network.items())))
    assert_close(
        cross_correlation_estimator(overlap * spectrum, frequencies, reordered),
        combined,
    )
    assert_close(combined, spectrum)


def test_custom_detector_geometry_is_used_for_projection(
    population: Pool, generator: WaveformGenerator
) -> None:
    first = load_detector("E1").model_copy(update={"name": "X1"})
    second = load_detector("E2")
    custom = first.model_copy(
        update={
            "geometry": dataclasses.replace(
                first.geometry, longitude_rad=first.geometry.longitude_rad + 0.2
            )
        }
    )
    moved = _project(population, generator, {"X1": custom, "E2": second})
    original = _project(population, generator, {"X1": first, "E2": second})
    assert not np.array_equal(
        moved.pair_spectral_density, original.pair_spectral_density
    )


def test_psd_override_preserves_pair_spectra_and_changes_combination(
    population: Pool, generator: WaveformGenerator
) -> None:
    changed = {
        **TRIANGLE,
        "C1": TRIANGLE["C1"].model_copy(
            update={"psd_reference": "ET_COBA_15km_psd.txt"}
        ),
    }
    baseline = _project(population, generator, TRIANGLE)
    overridden = _project(population, generator, changed)
    np.testing.assert_array_equal(
        baseline.pair_spectral_density, overridden.pair_spectral_density
    )
    frequencies = baseline.frequencies
    assert not np.array_equal(
        cross_correlation_estimator(
            baseline.pair_spectral_density,
            frequencies,
            TRIANGLE,
        ),
        cross_correlation_estimator(
            overridden.pair_spectral_density,
            frequencies,
            changed,
        ),
    )


@pytest.fixture(scope="module")
def empty(population: Pool) -> Pool:
    """A population with no sources, so the background is pure noise."""
    return {k: v[:0] for k, v in population.items()}


def test_pure_noise_matches_the_pair_scale(
    empty: Pool, generator: WaveformGenerator
) -> None:
    segment_duration = 600.0
    residuals = []
    for draw in range(120):
        background = _simulate(
            empty,
            generator,
            TRIANGLE,
            chunk_size=8,
            seed=draw,
            segment_duration=segment_duration,
            simulate_detector_noise=True,
        )
        scale = np.asarray(
            pair_cross_power_std(
                background.frequencies,
                TRIANGLE,
                OBSERVATION_SECONDS / years_to_seconds(1.0),
            )
        )
        residuals.append((background.pair_spectral_density / scale).ravel())
    residuals = np.concatenate(residuals)
    segments = int(OBSERVATION_SECONDS // segment_duration)
    size = residuals.size
    # A mean of unit-variance terms, and a variance whose excess kurtosis is
    # that of a product of Gaussians (6) averaged over the segments.
    mean_bound = MAX_STANDARD_ERRORS / np.sqrt(size)
    variance_bound = MAX_STANDARD_ERRORS * np.sqrt((2.0 + 6.0 / segments) / size)

    assert abs(residuals.mean()) < mean_bound
    assert abs(residuals.var() - 1.0) < variance_bound


@pytest.mark.parametrize("chunk_size,batch_size", [(8, 7), (3, 11), (128, 64)])
@pytest.mark.parametrize("noise", [False, True])
def test_spectra_do_not_depend_on_the_segment_batch_size(
    population: Pool,
    generator: WaveformGenerator,
    chunk_size: int,
    batch_size: int,
    noise: bool,
) -> None:
    reference = _simulate(
        population,
        generator,
        TRIANGLE,
        chunk_size=8,
        segment_duration=60.0,
        segment_batch_size=64,
        simulate_detector_noise=noise,
    )
    batched = _simulate(
        population,
        generator,
        TRIANGLE,
        chunk_size=chunk_size,
        segment_duration=60.0,
        segment_batch_size=batch_size,
        simulate_detector_noise=noise,
    )

    np.testing.assert_array_equal(batched.segment_counts, reference.segment_counts)
    assert_close(batched.pair_spectral_density, reference.pair_spectral_density)


@pytest.mark.parametrize("opposite_phase", [False, True])
def test_source_arrival_times_set_interference_without_detector_noise(
    population: Pool,
    generator: WaveformGenerator,
    monkeypatch: pytest.MonkeyPatch,
    opposite_phase: bool,
) -> None:
    # With detector noise disabled, two identical sources interfere either
    # constructively or destructively at the first frequency.
    # One segment of 0.2 s, two Fourier modes per 10 Hz bin.
    seconds = 0.2
    sources = {name: np.repeat(values[:1], 2) for name, values in population.items()}
    sources["coa_time"] = np.array(
        [0.0, 1 / (2 * WAVEFORM.minimum_frequency) if opposite_phase else 0.0]
    )
    monkeypatch.setattr(
        background_module, "_draw_events", lambda *_: (sources, np.array([2]))
    )
    baseline = project_background(
        sources, generator, TRIANGLE, observation_seconds=seconds
    )
    draw = simulate_background(
        jax.random.key(0),
        sources,
        generator,
        TRIANGLE,
        poisson_rate=10.0,
        num_segments=1,
        segment_duration=seconds,
        chunk_size=1,
    )
    ratio = 1 / np.sqrt(seconds * WAVEFORM.frequency_resolution)
    expected = 1 - ratio if opposite_phase else 1 + ratio
    measured = draw.pair_spectral_density[:, 0] / baseline.pair_spectral_density[:, 0]
    assert_close(measured, np.full(measured.shape, expected))


def test_noisy_background_averages_to_the_incoherent_signal(
    population: Pool,
    projected: BackgroundDraw,
    generator: WaveformGenerator,
) -> None:
    """A half-pool expected count gives half its full incoherent projection."""
    segment_duration = 60.0
    draws = np.stack(
        [
            _simulate(
                population,
                generator,
                PAIR,
                chunk_size=8,
                seed=draw,
                segment_duration=segment_duration,
                simulate_detector_noise=True,
            ).pair_spectral_density
            for draw in range(150)
        ]
    )
    expected = projected.pair_spectral_density / 2
    error = draws.std(axis=0, ddof=1) / np.sqrt(len(draws))

    assert np.all(abs(draws.mean(axis=0) - expected) < MAX_STANDARD_ERRORS * error)


def test_pool_selection_and_capacity() -> None:
    pool = {"right_ascension": np.arange(256)}
    key = jax.random.key(1)
    selected, counts = background_module._draw_events(key, pool, 1.0, 100, 1.0)
    assert counts.shape == (100,)
    assert len(np.unique(selected["right_ascension"])) == counts.sum()
    assert np.all(np.abs(selected["coa_time"]) <= 0.5)
    assert (counts == 0).any()
    repeated, other_counts = background_module._draw_events(key, pool, 1.0, 100, 1.0)
    np.testing.assert_array_equal(counts, other_counts)
    np.testing.assert_array_equal(
        selected["right_ascension"], repeated["right_ascension"]
    )
    with pytest.raises(ValueError, match=f"requires {counts.sum()}"):
        background_module._draw_events(
            key, {"right_ascension": np.arange(1)}, 1.0, 100, 1.0
        )


def test_segment_counts_have_poisson_mean_and_variance() -> None:
    segments = 4096
    rate, seconds = 0.5, 4.0
    _, counts = background_module._draw_events(
        jax.random.key(99),
        {"right_ascension": np.zeros(10000)},
        rate,
        segments,
        seconds,
    )
    mean = rate * seconds
    assert abs(counts.mean() - mean) < MAX_STANDARD_ERRORS * np.sqrt(mean / segments)
    assert abs(counts.var() - mean) < MAX_STANDARD_ERRORS * np.sqrt(
        (mean + 2 * mean**2) / segments
    )


def test_noise_toggle_preserves_the_chosen_sources_and_arrival_times(
    population: Pool,
    generator: WaveformGenerator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Zero PSDs let the two paths be compared at fixed source strain.
    monkeypatch.setattr(
        "astrogwb.detector.sensitivity.Sensitivity.__call__",
        lambda self, frequencies: np.zeros_like(frequencies),
    )
    quiet = _simulate(population, generator, PAIR, segment_duration=60.0, chunk_size=8)
    with_noise = _simulate(
        population,
        generator,
        PAIR,
        segment_duration=60.0,
        chunk_size=8,
        simulate_detector_noise=True,
    )
    np.testing.assert_array_equal(quiet.segment_counts, with_noise.segment_counts)
    assert_close(quiet.pair_spectral_density, with_noise.pair_spectral_density)


def test_simulation_uses_the_supplied_generator_instance(population: Pool) -> None:
    @dataclasses.dataclass(frozen=True)
    class ScaledGenerator(AnalyticInspiralGenerator):
        amplitude: float = dataclasses.field(compare=False)

        def polarizations(
            self, sources: Mapping[str, ArrayLike]
        ) -> tuple[jax.Array, jax.Array]:
            plus, cross = super().polarizations(sources)
            return self.amplitude * plus, self.amplitude * cross

    first = ScaledGenerator(WAVEFORM, amplitude=1.0)
    second = ScaledGenerator(WAVEFORM, amplitude=2.0)
    assert first == second
    reference = _simulate(population, first, PAIR, segment_duration=60.0, chunk_size=8)
    scaled = _simulate(population, second, PAIR, segment_duration=60.0, chunk_size=8)
    assert np.any(reference.pair_spectral_density)
    assert_close(scaled.pair_spectral_density, 4 * reference.pair_spectral_density)
