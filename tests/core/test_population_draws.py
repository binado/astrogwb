"""Population draws: key locality, exact counts, flat layout and the simulator."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from astrogwb_mock_population import POPULATION_PARAMS

from astrogwb.distributions.config import DistributionConfig
from astrogwb.populations import PopulationMetadata
from astrogwb.simulators.core import batch_keys
from astrogwb.simulators.population import (
    PopulationDrawMetadata,
    PopulationSimulator,
    bucket_size,
    segment_ids,
)
from astrogwb.utils import years_to_seconds

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


type SimulatorFactory = Callable[..., PopulationSimulator]


@pytest.fixture
def make_simulator() -> SimulatorFactory:
    """Build a simulator on the mock population; each call compiles its own."""

    def make(
        count: str = "poisson",
        *,
        expected: float = 20.0,
        num_events: int | None = None,
        sample_rate: bool = False,
        chunk_size: int = CHUNK,
    ) -> PopulationSimulator:
        hyperparameters: dict[str, object] = dict(POPULATION_PARAMS)
        if sample_rate:
            hyperparameters["local_merger_rate"] = DistributionConfig.model_validate(
                RATE_PRIOR
            )
        metadata = _metadata(
            hyperparameters=hyperparameters,
            observation_time=_observation_time_for(expected),
            count=count,
            num_events=num_events,
        )
        return PopulationSimulator(metadata, chunk_size=chunk_size)

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


def test_fixed_counts_are_exact_and_flat(make_simulator: SimulatorFactory) -> None:
    draws = make_simulator("fixed", num_events=5).simulate_batch(batch_keys(41, 4))
    assert draws["counts"].tolist() == [5, 5, 5, 5]
    assert all(v.shape == (20,) for v in draws["source_parameters"].values())
    np.testing.assert_array_equal(
        segment_ids(draws["counts"]), np.repeat(np.arange(4), 5)
    )


def test_poisson_counts_have_the_expected_mean(
    make_simulator: SimulatorFactory,
) -> None:
    expected = 20.0
    draws = make_simulator(expected=expected).simulate_batch(batch_keys(41, 200))
    # The mean of 200 Poisson(20) counts has standard error sqrt(20 / 200).
    assert abs(draws["counts"].mean() - expected) < 5 * np.sqrt(expected / 200)
    assert draws["source_parameters"]["redshift"].shape == (int(draws["counts"].sum()),)


def test_a_draw_depends_on_its_own_seed_alone(
    make_simulator: SimulatorFactory,
) -> None:
    simulator = make_simulator(sample_rate=True)
    keys = batch_keys(41, 6)
    batch = simulator.simulate_batch(keys)
    alone = simulator.simulate_batch(keys[3:4])
    offsets = np.concatenate([[0], np.cumsum(batch["counts"])])
    lo, hi = offsets[3], offsets[4]
    assert batch["counts"][3] == alone["counts"][0]
    assert (
        batch["hyperparameters"]["local_merger_rate"][3]
        == alone["hyperparameters"]["local_merger_rate"][0]
    )
    for name, values in alone["source_parameters"].items():
        np.testing.assert_array_equal(batch["source_parameters"][name][lo:hi], values)


def test_chunk_size_changes_cost_not_the_draw(
    make_simulator: SimulatorFactory,
) -> None:
    keys = batch_keys(41, 5)
    small = make_simulator(sample_rate=True, chunk_size=3).simulate_batch(keys)
    large = make_simulator(sample_rate=True, chunk_size=64).simulate_batch(keys)
    np.testing.assert_array_equal(small["counts"], large["counts"])
    for name, values in small["source_parameters"].items():
        np.testing.assert_array_equal(values, large["source_parameters"][name])


def test_sampled_hyperparameter_follows_its_prior(
    make_simulator: SimulatorFactory,
) -> None:
    draws = make_simulator(sample_rate=True).simulate_batch(batch_keys(41, 50))
    column = draws["hyperparameters"]["local_merger_rate"]
    assert column.min() >= 600.0 and column.max() <= 900.0
    assert column.std() > 0.0
    # a fixed hyperparameter's column repeats its value
    assert np.all(draws["hyperparameters"]["H0"] == POPULATION_PARAMS["H0"])


def test_simulator_refuses_a_population_without_a_rate() -> None:
    rateless = PopulationMetadata(
        model_name="bns_md_uniform_mixture",
        model_kwargs={
            "minimum_redshift": 0.0,
            "maximum_redshift": 5.0,
            "n_grid": 64,
            "uniform_mixing_fraction": 0.5,
        },
    )
    with pytest.raises(ValueError, match="merger rate"):
        PopulationSimulator(_metadata(population=rateless), chunk_size=CHUNK)


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


def test_simulate_batch_is_the_batch_of_simulate() -> None:
    simulator = PopulationSimulator(_metadata(), chunk_size=CHUNK)
    keys = batch_keys(41, 5)
    batch = simulator.simulate_batch(keys)
    offsets = np.concatenate([[0], np.cumsum(batch["counts"])])
    for i in range(keys.shape[0]):
        single = simulator.simulate(keys[i])
        assert single["counts"] == batch["counts"][i]
        # vmap may round a scalar differently than the batch does.
        np.testing.assert_allclose(
            single["total_merger_rate"], batch["total_merger_rate"][i], rtol=1e-12
        )
        for name, column in batch["hyperparameters"].items():
            np.testing.assert_allclose(
                single["hyperparameters"][name], column[i], rtol=1e-12
            )
        for name, column in batch["source_parameters"].items():
            np.testing.assert_array_equal(
                single["source_parameters"][name], column[offsets[i] : offsets[i + 1]]
            )


def test_simulate_batch_rejects_an_unbatched_key() -> None:
    simulator = PopulationSimulator(_metadata(), chunk_size=CHUNK)
    with pytest.raises(ValueError, match="1-d batch"):
        simulator.simulate_batch(batch_keys(41, 1)[0])


def test_simulator_refuses_a_metadata_of_another_version() -> None:
    with pytest.raises(ValueError, match="is installed"):
        PopulationSimulator(_metadata(version="0.0.1"), chunk_size=CHUNK)
