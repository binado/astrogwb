"""The population and background stages, end to end.

Everything is generated for real on the closed-form inspiral: a tiny Poisson
population and a four-bin grid. The Monte Carlo check compares in units of the
sample's own standard error, so it holds for any seed.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest
from astrogwb_tiny_pipeline import (
    NETWORKS,
    PAIR,
    TRIANGLE,
    WAVEFORM,
    event_pool,
)
from numpy.typing import NDArray
from tolerances import assert_close

from astrogwb.background.simulation import (
    BackgroundDraw,
    project_background,
    simulate_background,
)
from astrogwb.background.simulation import draw as background_module
from astrogwb.detector import (
    Network,
    Networks,
    isotropic_overlap_reduction_function,
    load_detector,
    network_pairs,
    pair_bin_scale,
)
from astrogwb.detector.projection import project_polarizations_fd
from astrogwb.statistics import cross_correlation_estimator
from astrogwb.utils import years_to_seconds
from astrogwb.waveform import WaveformMetadata, intensity, waveform_generator

#: Standard errors a Monte Carlo mean may sit from its expectation.
MAX_STANDARD_ERRORS = 5.0


type Pool = dict[str, np.ndarray]
OBSERVATION_SECONDS = 3000.0


@pytest.fixture(scope="module")
def population() -> Pool:
    return event_pool()


def _project(
    population: Pool,
    waveform: WaveformMetadata,
    networks: Networks,
    **kwargs,
) -> BackgroundDraw:
    return project_background(
        population,
        waveform,
        networks,
        observation_seconds=OBSERVATION_SECONDS,
        **kwargs,
    )


def _simulate(
    population: Pool,
    waveform: WaveformMetadata,
    networks: Networks,
    *,
    segment_duration: float,
    seed: int = 0,
    **kwargs,
) -> BackgroundDraw:
    return simulate_background(
        jax.random.key(seed),
        population,
        waveform,
        networks,
        segment_seconds=segment_duration,
        num_segments=int(OBSERVATION_SECONDS // segment_duration),
        observer_rate_hz=len(population["redshift"]) / (2 * OBSERVATION_SECONDS),
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
def test_segmented_noise_needs_whole_fourier_modes_per_bin(
    population: Pool, waveform: WaveformMetadata
) -> None:
    with pytest.raises(ValueError, match="whole number"):
        _simulate(population, waveform, NETWORKS, segment_duration=0.5)


@pytest.mark.parametrize(
    "approximant",
    ["AnalyticInspiral", pytest.param("IMRPhenomD", marks=pytest.mark.integration)],
)
def test_coa_time_shifts_both_polarizations_and_preserves_intensity(
    population: Pool, approximant: str
) -> None:
    generator = waveform_generator(
        WaveformMetadata.model_validate(
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
    base = generator.generate_polarizations_batch(sources)
    shifted = jax.jit(generator.generate_polarizations_batch)(
        {**sources, "coa_time": times}
    )
    expected = np.exp(-2j * np.pi * times[:, None] * frequencies)
    for original, timed in zip(base, shifted, strict=True):
        assert_close(timed / original, expected)
    intensity_ratio = intensity(*shifted) / intensity(*base)
    assert_close(intensity_ratio, np.ones(intensity_ratio.shape))
    np.testing.assert_array_equal(
        generator.generate_batch(sources),
        generator.generate_batch({**sources, "coa_time": times}),
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
        WaveformMetadata.model_validate(
            {
                **WAVEFORM.model_dump(exclude={"alpha"}),
                "approximant": approximant,
                "frequency_resolution": 1.0,
            }
        )
    )
    sources = population

    plus, cross = generator.generate_polarizations_batch(sources)

    generated = np.asarray(generator.generate_batch(sources))
    assert_close(intensity(plus, cross), generated)


def test_pair_spectra_average_to_the_orf_times_the_sky_averaged_spectrum() -> None:
    r"""Over isotropic sources, :math:`E[Y_{ab}] = \gamma_{ab} S_h`.

    One realization is split into groups; in each, the ratio of every pair
    spectrum to the group's sky-averaged spectrum :math:`\sum_i I_i / T`
    isolates the response, and its mean over groups is :math:`\gamma_{ab}`.
    """
    realization = event_pool(4096, seed=7)
    generator = waveform_generator(WAVEFORM)
    group = 16
    ratios = []
    for start in range(0, len(realization["redshift"]) - group + 1, group):
        sources = {
            name: values[start : start + group] for name, values in realization.items()
        }
        generated = np.asarray(generator.generate_batch(sources)).sum(axis=1)
        spectra = _project(
            sources, WAVEFORM, {"triangle": TRIANGLE}, chunk_size=8
        ).spectra["triangle"]
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


def test_an_empty_realization_has_zero_spectra() -> None:
    empty = event_pool(0)

    background = _project(empty, WAVEFORM, NETWORKS)

    assert all(values.shape == (0,) for values in empty.values())
    for name, network in NETWORKS.items():
        pairs = background.spectra[name]
        assert pairs.shape == (len(network_pairs(network)), background.frequencies.size)
        assert not np.any(pairs)


@pytest.fixture(scope="module")
def projected(population: Pool) -> BackgroundDraw:
    """The tiny population projected onto both networks."""
    return _project(population, WAVEFORM, NETWORKS, chunk_size=8)


@pytest.mark.parametrize("name,network", NETWORKS.items(), ids=NETWORKS)
def test_each_network_holds_its_rows_of_a_single_network_projection(
    projected: BackgroundDraw,
    population: Pool,
    name: str,
    network: Network,
) -> None:
    alone = _project(population, WAVEFORM, {name: network}, chunk_size=8)

    np.testing.assert_array_equal(projected.spectra[name], alone.spectra[name])


@pytest.mark.parametrize("name,network", NETWORKS.items(), ids=NETWORKS)
def test_combining_pair_spectra_of_a_pure_signal_recovers_its_spectrum(
    projected: BackgroundDraw, name: str, network: Network
) -> None:
    spectrum = 1e-47 * (1.0 + np.arange(projected.frequencies.size))
    overlap = np.stack(
        [
            isotropic_overlap_reduction_function(
                projected.frequencies, network[first], network[second]
            )
            for first, second in network_pairs(network)
        ]
    )

    combined = cross_correlation_estimator(
        overlap * spectrum,
        projected.frequencies,
        tuple(network[member] for member in sorted(network)),
    )

    assert_close(combined, spectrum)


def test_custom_detector_geometry_is_used_for_projection(
    population: Pool,
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
    moved = _project(population, WAVEFORM, {"custom": {"X1": custom, "E2": second}})
    original = _project(population, WAVEFORM, {"custom": {"X1": first, "E2": second}})
    assert not np.array_equal(moved.spectra["custom"], original.spectra["custom"])


def test_psd_override_preserves_pair_spectra_and_changes_combination(
    population: Pool,
) -> None:
    changed = {
        **TRIANGLE,
        "C1": TRIANGLE["C1"].model_copy(
            update={"psd_reference": "ET_COBA_15km_psd.txt"}
        ),
    }
    baseline = _project(population, WAVEFORM, {"triangle": TRIANGLE})
    overridden = _project(population, WAVEFORM, {"triangle": changed})
    np.testing.assert_array_equal(
        baseline.spectra["triangle"], overridden.spectra["triangle"]
    )
    frequencies = baseline.frequencies
    assert not np.array_equal(
        cross_correlation_estimator(
            baseline.spectra["triangle"],
            frequencies,
            tuple(TRIANGLE[member] for member in sorted(TRIANGLE)),
        ),
        cross_correlation_estimator(
            overridden.spectra["triangle"],
            frequencies,
            tuple(changed[member] for member in sorted(changed)),
        ),
    )


@pytest.fixture(scope="module")
def empty(population: Pool) -> Pool:
    """A population with no sources, so the background is pure noise."""
    return {k: v[:0] for k, v in population.items()}


def test_pure_noise_matches_the_pair_scale(empty: Pool) -> None:
    segment_duration = 600.0
    residuals = []
    for draw in range(120):
        background = _simulate(
            empty,
            WAVEFORM,
            NETWORKS,
            chunk_size=8,
            seed=draw,
            segment_duration=segment_duration,
        )
        for name, network in NETWORKS.items():
            scale = np.asarray(
                pair_bin_scale(
                    background.frequencies,
                    tuple(network[member] for member in sorted(network)),
                    int(OBSERVATION_SECONDS // segment_duration)
                    * segment_duration
                    / years_to_seconds(1.0),
                )
            )
            residuals.append((background.spectra[name] / scale).ravel())
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
def test_spectra_do_not_depend_on_the_segment_batch_size(
    population: Pool,
    chunk_size: int,
    batch_size: int,
) -> None:
    reference = _simulate(
        population,
        WAVEFORM,
        NETWORKS,
        chunk_size=8,
        segment_duration=60.0,
        segment_batch_size=64,
    )
    batched = _simulate(
        population,
        WAVEFORM,
        NETWORKS,
        chunk_size=chunk_size,
        segment_duration=60.0,
        segment_batch_size=batch_size,
    )

    for name, spectrum in reference.spectra.items():
        assert_close(batched.spectra[name], spectrum)


@pytest.mark.parametrize("opposite_phase", [False, True])
def test_segment_sums_use_sampled_coa_times_without_rescaling(
    population: Pool,
    monkeypatch: pytest.MonkeyPatch,
    opposite_phase: bool,
) -> None:
    # With detector noise disabled, two identical sources interfere either
    # constructively or destructively at the first frequency.
    geometry = background_module._network_geometry

    def without_noise(
        networks: Networks, frequencies: NDArray[np.float64]
    ) -> background_module._Geometry:
        result = geometry(networks, frequencies)
        return result._replace(psds=np.full_like(result.psds, np.inf))

    monkeypatch.setattr(background_module, "_network_geometry", without_noise)
    # One segment of 0.2 s, two Fourier modes per 10 Hz bin.
    seconds = 0.2
    sources = {name: np.repeat(values[:1], 2) for name, values in population.items()}
    sources["coa_time"] = np.array(
        [0.0, 1 / (2 * WAVEFORM.minimum_frequency) if opposite_phase else 0.0]
    )
    baseline = project_background(
        sources, WAVEFORM, NETWORKS, observation_seconds=seconds
    )
    grid, geometry_arrays, kernels = background_module._projection(WAVEFORM, NETWORKS)
    union = background_module._segmented(
        sources,
        np.array([2]),
        jax.random.key(0),
        grid,
        geometry_arrays,
        kernels,
        1,
        seconds,
        1,
    )
    ratio = 1 / np.sqrt(0.2 * WAVEFORM.frequency_resolution)
    for name, spectrum in baseline.spectra.items():
        measured = (union[geometry_arrays.rows[name], 0] / spectrum[:, 0] - 1) / ratio
        assert_close(measured, np.full(measured.shape, -1.0 if opposite_phase else 1.0))


def test_noisy_background_averages_to_the_incoherent_signal(
    population: Pool,
    projected: BackgroundDraw,
) -> None:
    """A half-pool expected count gives half its full incoherent projection."""
    segment_duration = 60.0
    draws = np.stack(
        [
            _simulate(
                population,
                WAVEFORM,
                NETWORKS,
                chunk_size=8,
                seed=draw,
                segment_duration=segment_duration,
            ).spectra["pair"]
            for draw in range(150)
        ]
    )
    expected = projected.spectra["pair"] / 2
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


def test_incoherent_draw_projects_exactly_its_selected_rows(population: Pool) -> None:
    key = jax.random.key(4)
    event_key, _ = jax.random.split(key)
    sources, counts = background_module._draw_events(
        event_key, population, 0.2, 4, 10.0
    )
    expected = project_background(sources, WAVEFORM, NETWORKS, observation_seconds=40.0)
    draw = simulate_background(
        key,
        population,
        WAVEFORM,
        NETWORKS,
        observer_rate_hz=0.2,
        num_segments=4,
        segment_seconds=10.0,
        incoherent=True,
    )
    np.testing.assert_array_equal(draw.segment_counts, counts)
    for name, spectrum in expected.spectra.items():
        np.testing.assert_array_equal(draw.spectra[name], spectrum)
