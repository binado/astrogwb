"""Population draws: seed locality, exact counts, flat layout and the cached node."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb_mock_population import POPULATION_PARAMS, mock_population

from astrogwb.distributions.config import DistributionConfig
from astrogwb.populations import PopulationMetadata
from astrogwb.simulators.core import split_seed
from astrogwb.simulators.population import (
    PopulationDrawMetadata,
    PopulationSampler,
    bucket_size,
    population,
)
from astrogwb.utils import years_to_seconds

jax.config.update("jax_enable_x64", True)

POPULATION = PopulationMetadata(
    model_name="bns_md_cosmological",
    model_kwargs={"minimum_redshift": 0.0, "maximum_redshift": 5.0, "n_grid": 64},
)
CHUNK = 8
RATE_PRIOR = {"dist": "Uniform", "kwargs": {"low": 600.0, "high": 900.0}}


def _observation_time_for(expected_events: float) -> float:
    """Years such that ``R * T = expected_events`` at the fiducials."""
    built = POPULATION.build()
    assert built.merger_rate_fn is not None
    rate = float(built.merger_rate_fn(POPULATION_PARAMS))
    return expected_events / (rate * years_to_seconds(1.0))


type SamplerFactory = Callable[..., PopulationSampler]


@pytest.fixture
def make_sampler() -> SamplerFactory:
    """Build a sampler on the mock population; built once per call, reused by tests."""

    def make(
        count: str = "poisson",
        *,
        expected: float = 20.0,
        num_events: int | None = None,
        sample_rate: bool = False,
    ) -> PopulationSampler:
        built = mock_population()
        fixed = {
            name: value
            for name, value in POPULATION_PARAMS.items()
            if not (sample_rate and name == "local_merger_rate")
        }
        priors = (
            {"local_merger_rate": dist.Uniform(600.0, 900.0)} if sample_rate else {}
        )
        return PopulationSampler(
            source_model=built.source_model,
            merger_rate_fn=built.merger_rate_fn,
            fixed=fixed,
            priors=priors,
            observation_time=_observation_time_for(expected),
            count=count,  # ty: ignore[invalid-argument-type]
            num_events=num_events,
        )

    return make


@pytest.mark.parametrize("count", [0, 1, 7, 8, 9, 63, 64, 65, 1000])
def test_bucket_size_is_a_chunk_multiple_that_holds_the_count(count: int) -> None:
    size = bucket_size(count, CHUNK)
    assert size % CHUNK == 0
    assert size >= max(count, CHUNK)
    assert size <= max(count, CHUNK) * 1.25 + CHUNK  # bounded padding


def test_bucket_size_is_non_decreasing_with_few_distinct_sizes() -> None:
    sizes = [bucket_size(count, CHUNK) for count in range(5000)]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) < 40  # a handful of compiles for counts up to 5000


@pytest.mark.parametrize("chunk_size, ratio", [(0, 1.25), (8, 1.0)])
def test_bucket_size_rejects_degenerate_ladders(chunk_size: int, ratio: float) -> None:
    with pytest.raises(ValueError):
        bucket_size(10, chunk_size, ratio)


def test_fixed_counts_are_exact_and_flat(make_sampler: SamplerFactory) -> None:
    draws = make_sampler("fixed", num_events=5)(split_seed(41, 4), chunk_size=CHUNK)
    assert draws.counts.tolist() == [5, 5, 5, 5]
    assert all(v.shape == (20,) for v in draws.source_parameters.values())
    np.testing.assert_array_equal(draws.offsets, [0, 5, 10, 15, 20])
    np.testing.assert_array_equal(draws.segment_ids, np.repeat(np.arange(4), 5))


def test_poisson_counts_have_the_expected_mean(make_sampler: SamplerFactory) -> None:
    expected = 20.0
    draws = make_sampler(expected=expected)(split_seed(41, 200), chunk_size=CHUNK)
    # The mean of 200 Poisson(20) counts has standard error sqrt(20 / 200).
    assert abs(draws.counts.mean() - expected) < 5 * np.sqrt(expected / 200)
    assert draws.source_parameters["redshift"].shape == (int(draws.counts.sum()),)


def test_a_draw_depends_on_its_own_seed_alone(make_sampler: SamplerFactory) -> None:
    sampler = make_sampler(sample_rate=True)
    seeds = split_seed(41, 6)
    batch = sampler(seeds, chunk_size=CHUNK)
    alone = sampler(seeds[3:4], chunk_size=CHUNK)
    lo, hi = batch.offsets[3], batch.offsets[4]
    assert batch.counts[3] == alone.counts[0]
    assert (
        batch.hyperparameters["local_merger_rate"][3]
        == (alone.hyperparameters["local_merger_rate"][0])
    )
    for name, values in alone.source_parameters.items():
        np.testing.assert_array_equal(batch.source_parameters[name][lo:hi], values)


def test_chunk_size_changes_cost_not_the_draw(make_sampler: SamplerFactory) -> None:
    sampler = make_sampler(sample_rate=True)
    seeds = split_seed(41, 5)
    small = sampler(seeds, chunk_size=3)
    large = sampler(seeds, chunk_size=64)
    np.testing.assert_array_equal(small.counts, large.counts)
    for name, values in small.source_parameters.items():
        np.testing.assert_array_equal(values, large.source_parameters[name])


def test_sampled_hyperparameter_follows_its_prior(make_sampler: SamplerFactory) -> None:
    draws = make_sampler(sample_rate=True)(split_seed(41, 50), chunk_size=CHUNK)
    column = draws.hyperparameters["local_merger_rate"]
    assert column.min() >= 600.0 and column.max() <= 900.0
    assert column.std() > 0.0
    # a fixed hyperparameter's column repeats its value
    assert np.all(draws.hyperparameters["H0"] == POPULATION_PARAMS["H0"])


def test_sampler_refuses_a_population_without_a_rate() -> None:
    built = mock_population()
    with pytest.raises(ValueError, match="merger rate"):
        PopulationSampler(
            source_model=built.source_model,
            merger_rate_fn=None,
            fixed=POPULATION_PARAMS,
            priors={},
            observation_time=1.0,
            count="poisson",
        )


def _metadata(**overrides: object) -> PopulationDrawMetadata:
    fields: dict[str, object] = {
        "population": POPULATION,
        "hyperparameters": {
            **{k: v for k, v in POPULATION_PARAMS.items() if k != "local_merger_rate"},
            "local_merger_rate": DistributionConfig.model_validate(RATE_PRIOR),
        },
        "observation_time": _observation_time_for(6.0),
        "count": "poisson",
    }
    fields.update(overrides)
    return PopulationDrawMetadata.model_validate(fields)


def test_metadata_count_modes_are_validated() -> None:
    with pytest.raises(ValueError, match="num_events is required"):
        _metadata(count="fixed")
    with pytest.raises(ValueError, match="only valid for fixed"):
        _metadata(num_events=3)
    assert _metadata(count="fixed", num_events=3).num_events == 3


def test_metadata_key_ignores_int_float_spelling_and_tracks_content() -> None:
    base = _metadata()
    hyperparameters = {**base.hyperparameters, "gamma": 2}  # int, not float
    widened = {**base.hyperparameters, "gamma": 2.0}
    assert (
        _metadata(hyperparameters=hyperparameters).key()
        == _metadata(hyperparameters=widened).key()
    )
    assert base.key() != _metadata(observation_time=2.0).key()
    assert base.key() != _metadata(count="fixed", num_events=4).key()


def test_cached_node_round_trips_and_matches_the_sampler(tmp_path: Path) -> None:
    metadata = _metadata()
    inputs = {"seeds": split_seed(41, 4)}
    fresh = population(inputs, metadata, chunk_size=CHUNK)
    first = population(inputs, metadata, cache_dir=tmp_path, chunk_size=CHUNK)
    assert population.path(inputs, metadata, tmp_path).is_file()
    again = population(
        inputs, metadata, cache_dir=tmp_path, generate=False, chunk_size=CHUNK
    )
    expected = cast(dict[str, Any], fresh["source_parameters"])
    for outputs in (first, again):
        np.testing.assert_array_equal(outputs["counts"], fresh["counts"])
        columns = cast(dict[str, Any], outputs["source_parameters"])
        for name, values in expected.items():
            np.testing.assert_array_equal(columns[name], values)
    assert expected["redshift"].shape == (int(np.asarray(fresh["counts"]).sum()),)


def test_cached_node_rejects_repeated_and_mistyped_seeds() -> None:
    metadata = _metadata()
    seeds = split_seed(41, 2)
    with pytest.raises(ValueError, match="must not repeat"):
        population({"seeds": np.repeat(seeds[:1], 2)}, metadata)
    with pytest.raises(TypeError, match="uint64"):
        population({"seeds": seeds.astype(np.int64)}, metadata)
