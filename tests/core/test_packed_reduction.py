"""Packed per-draw power sums and spectrum normalization."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from astrogwb_mock_population import load_mock_population
from numpy.testing import assert_allclose

from astrogwb.constants import INCLINATION_AVERAGE_TO_FACE_ON_RATIO, ISCO_ALPHA
from astrogwb.simulators.spectra.forward import PackedPowerSum, normalize_spectra
from astrogwb.waveform import AnalyticInspiralGenerator, WaveformMetadata

jax.config.update("jax_enable_x64", True)

#: Events per draw; includes an empty draw and counts that no chunk size divides.
COUNTS = [0, 7, 1, 13, 9]
NUM_FREQUENCIES = 3


@pytest.fixture
def generator() -> AnalyticInspiralGenerator:
    return AnalyticInspiralGenerator(
        WaveformMetadata(
            alpha=ISCO_ALPHA,
            approximant="AnalyticInspiral",
            minimum_frequency=20.0,
            maximum_frequency=40.0,
            reference_frequency=20.0,
            sampling_frequency=128.0,
            frequency_resolution=10.0,
        )
    )


@pytest.fixture
def sources() -> dict[str, np.ndarray]:
    return load_mock_population(num_sources=sum(COUNTS))


@pytest.fixture
def segment_ids() -> np.ndarray:
    return np.repeat(np.arange(len(COUNTS)), COUNTS)


def _explicit_sums(
    generator: AnalyticInspiralGenerator, sources: dict[str, np.ndarray]
) -> np.ndarray:
    """One ``generate_batch`` per draw: the unpacked answer."""
    offsets = np.concatenate([[0], np.cumsum(COUNTS)])
    rows = []
    for draw, count in enumerate(COUNTS):
        if count == 0:
            rows.append(np.zeros(NUM_FREQUENCIES))
            continue
        lo, hi = offsets[draw], offsets[draw + 1]
        power = generator.generate_batch({k: v[lo:hi] for k, v in sources.items()})
        rows.append(np.asarray(power).sum(axis=1))
    return np.stack(rows)


@pytest.mark.parametrize("chunk_size", [1, 4, 7, 64])
def test_packed_sums_match_per_draw_sums_for_ragged_counts(
    generator: AnalyticInspiralGenerator,
    sources: dict[str, np.ndarray],
    segment_ids: np.ndarray,
    chunk_size: int,
) -> None:
    packed = PackedPowerSum(generator, chunk_size=chunk_size)(
        sources, segment_ids, len(COUNTS)
    )
    assert packed.shape == (len(COUNTS), NUM_FREQUENCIES)
    assert_allclose(packed, _explicit_sums(generator, sources), rtol=1e-12)


def test_packed_sum_of_an_empty_stream_is_zeros(
    generator: AnalyticInspiralGenerator, sources: dict[str, np.ndarray]
) -> None:
    empty = {name: values[:0] for name, values in sources.items()}
    out = PackedPowerSum(generator, chunk_size=4)(empty, np.array([], dtype=int), 3)
    assert_allclose(out, np.zeros((3, NUM_FREQUENCIES)))


def test_packed_sum_compiles_once_across_stream_lengths_in_one_capacity(
    generator: AnalyticInspiralGenerator, sources: dict[str, np.ndarray]
) -> None:
    reducer = PackedPowerSum(generator, chunk_size=4)
    for total in (10, 11, 12):  # one buffer capacity, three different trip counts
        part = {name: values[:total] for name, values in sources.items()}
        reducer(part, np.zeros(total, dtype=int), 1)
    assert reducer._reduce._cache_size() == 1  # ty: ignore[unresolved-attribute]


def test_packed_sum_rejects_a_segment_outside_the_draws(
    generator: AnalyticInspiralGenerator, sources: dict[str, np.ndarray]
) -> None:
    part = {name: values[:3] for name, values in sources.items()}
    with pytest.raises(ValueError, match="segment_ids must lie"):
        PackedPowerSum(generator, chunk_size=4)(part, np.array([0, 1, 2]), 2)


def test_packed_sum_rejects_columns_of_unequal_length(
    generator: AnalyticInspiralGenerator, sources: dict[str, np.ndarray]
) -> None:
    ragged = {**sources, "redshift": sources["redshift"][:2]}
    with pytest.raises(ValueError, match="one length"):
        PackedPowerSum(generator, chunk_size=4)(
            ragged, np.zeros(sum(COUNTS), dtype=int), 1
        )


def test_packed_sum_requires_a_luminosity_distance(
    generator: AnalyticInspiralGenerator, sources: dict[str, np.ndarray]
) -> None:
    missing = {k: v for k, v in sources.items() if k != "luminosity_distance"}
    with pytest.raises(KeyError, match="luminosity_distance"):
        PackedPowerSum(generator, chunk_size=4)(
            missing, np.zeros(sum(COUNTS), dtype=int), 1
        )


def test_normalize_poisson_divides_the_sum_by_the_observation_time() -> None:
    spectra = normalize_spectra(
        np.ones((2, 3)),
        {"luminosity_distance": np.ones(1)},
        count="poisson",
        total_merger_rate=np.array([2.0, 4.0]),
        observation_seconds=2.0,
        num_events=None,
    )
    assert_allclose(spectra, INCLINATION_AVERAGE_TO_FACE_ON_RATIO * 0.5)


def test_normalize_fixed_scales_the_mean_power_by_the_draw_rate() -> None:
    spectra = normalize_spectra(
        np.ones((2, 3)),
        {"luminosity_distance": np.ones(1)},
        count="fixed",
        total_merger_rate=np.array([2.0, 4.0]),
        observation_seconds=123.0,  # cancels
        num_events=4,
    )
    expected = INCLINATION_AVERAGE_TO_FACE_ON_RATIO * np.array([0.5, 1.0])[:, None]
    assert_allclose(spectra, np.broadcast_to(expected, (2, 3)))


def test_normalize_skips_the_inclination_factor_when_inclination_is_sampled() -> None:
    spectra = normalize_spectra(
        np.ones((1, 2)),
        {"luminosity_distance": np.ones(1), "inclination": np.ones(1)},
        count="poisson",
        total_merger_rate=np.array([1.0]),
        observation_seconds=1.0,
        num_events=None,
    )
    assert_allclose(spectra, np.ones((1, 2)))


def test_normalize_fixed_requires_num_events() -> None:
    with pytest.raises(ValueError, match="num_events"):
        normalize_spectra(
            np.ones((1, 2)),
            {},
            count="fixed",
            total_merger_rate=np.array([1.0]),
            observation_seconds=1.0,
            num_events=None,
        )
