"""The three simulators, end to end on the closed-form inspiral.

Each test generates for real -- a tiny population, a four-bin grid -- so what
is checked is the simulator, not a mock of it: a stack is the stack of its
single calls, a key names its realization, and a file written from one reads
back equal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astrogwb.distributions.config import DistributionConfig
from astrogwb.populations import PopulationMetadata
from astrogwb.simulators.core import Arrays, batch_keys, load, write
from astrogwb.simulators.polarization_power import (
    CatalogMetadata,
    draw_catalog,
)
from astrogwb.simulators.population import PopulationSimulator
from astrogwb.simulators.spectra import (
    BackgroundSpectralDensityMetadata,
    BackgroundSpectralDensitySimulator,
    stack_spectra,
)
from astrogwb.waveform import WaveformMetadata

FIDUCIALS = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
    "minimum_mass": 1.0,
    "mass_width": 1.5,
}
WAVEFORM = WaveformMetadata(
    approximant="AnalyticInspiral",
    minimum_frequency=20.0,
    maximum_frequency=50.0,
    reference_frequency=20.0,
    sampling_frequency=128.0,
    frequency_resolution=10.0,
)
POPULATION = PopulationMetadata(
    model_name="bns_coba",
    model_kwargs={
        "mass_model": "uniform",
        "minimum_redshift": 0.0,
        "maximum_redshift": 5.0,
        "n_grid": 64,
    },
)
CATALOG = CatalogMetadata(
    waveform=WAVEFORM, population=POPULATION, fiducials=FIDUCIALS, num_samples=6
)
SPECTRA = BackgroundSpectralDensityMetadata(
    waveform=WAVEFORM,
    population=POPULATION,
    hyperparameters={
        **{k: v for k, v in FIDUCIALS.items() if k != "local_merger_rate"},
        "local_merger_rate": DistributionConfig(
            dist="Uniform", kwargs={"low": 700.0, "high": 800.0}
        ),
    },
    observation_time=1.0,
    count="fixed",
    num_events=4,
)


def _same(first: Arrays, second: Arrays) -> None:
    assert first.keys() == second.keys()
    for name, value in first.items():
        other = second[name]
        if isinstance(value, dict):
            assert isinstance(other, dict)
            _same(value, other)
        else:
            assert not isinstance(other, dict)
            assert value.dtype == other.dtype
            np.testing.assert_array_equal(value, other)


@pytest.mark.integration
def test_polarization_power_round_trips_through_a_file(tmp_path: Path) -> None:
    fresh = draw_catalog(CATALOG, batch_keys(41, 1)[0])
    path = write(tmp_path / "catalog.h5", fresh, CATALOG, seed=41)

    data, metadata, attrs = load(path, CatalogMetadata)

    _same(dict(fresh), data)  # ty: ignore[invalid-argument-type]
    assert metadata == CATALOG
    assert attrs["seed"] == 41


@pytest.mark.integration
def test_polarization_power_key_picks_the_realization() -> None:
    one = draw_catalog(CATALOG, batch_keys(41, 1)[0])
    again = draw_catalog(CATALOG, batch_keys(41, 1)[0])
    other = draw_catalog(CATALOG, batch_keys(42, 1)[0])

    np.testing.assert_array_equal(
        one["polarization_power"], again["polarization_power"]
    )
    assert not np.array_equal(one["polarization_power"], other["polarization_power"])


@pytest.mark.integration
@pytest.mark.parametrize("chunk_size", [1, 4, 6, 100])
def test_polarization_power_is_independent_of_chunk_size(chunk_size: int) -> None:
    key = batch_keys(41, 1)[0]
    whole = draw_catalog(CATALOG, key, chunk_size=None)
    chunked = draw_catalog(CATALOG, key, chunk_size=chunk_size)

    # Sources are drawn before chunking, so they are exact. The power is the
    # same elementwise computation, but XLA may fuse a differently shaped chunk
    # differently, which moves the last bit.
    _same(whole["source_parameters"], chunked["source_parameters"])  # ty: ignore[invalid-argument-type]
    np.testing.assert_allclose(
        chunked["polarization_power"], whole["polarization_power"], rtol=1e-12, atol=0
    )


@pytest.mark.integration
@pytest.mark.parametrize("num_draws", [1, 3])
def test_spectra_round_trip_through_a_file(tmp_path: Path, num_draws: int) -> None:
    simulator = BackgroundSpectralDensitySimulator(SPECTRA, chunk_size=4)
    parts = [simulator(key) for key in batch_keys(41, num_draws)]
    fresh = parts[0] if num_draws == 1 else stack_spectra(parts)
    path = write(tmp_path / "spectra.h5", fresh, SPECTRA, seed=41)

    data, metadata, attrs = load(path, BackgroundSpectralDensityMetadata)

    _same(dict(fresh), data)  # ty: ignore[invalid-argument-type]
    assert metadata == SPECTRA
    assert attrs["seed"] == 41
    assert fresh["spectral_density"].shape == (num_draws, fresh["frequencies"].size)
    assert np.all(fresh["n_events"] == 4)


@pytest.mark.integration
def test_a_key_picks_the_spectrum() -> None:
    simulator = BackgroundSpectralDensitySimulator(SPECTRA, chunk_size=4)
    keys = batch_keys(41, 3)

    one = simulator(keys[1])
    again = simulator(keys[1])
    other = simulator(keys[2])

    np.testing.assert_array_equal(one["spectral_density"], again["spectral_density"])
    assert one["spectral_density"].shape == (1, one["frequencies"].size)
    assert one["n_events"].shape == one["total_merger_rate"].shape == (1,)
    assert all(value.shape == (1,) for value in one["hyperparameters"].values())
    assert not np.array_equal(one["spectral_density"], other["spectral_density"])
    assert not np.array_equal(
        one["hyperparameters"]["local_merger_rate"],
        other["hyperparameters"]["local_merger_rate"],
    )


@pytest.mark.integration
def test_stacking_a_loop_equals_the_single_calls() -> None:
    simulator = BackgroundSpectralDensitySimulator(SPECTRA, chunk_size=4)
    singles = [simulator(key) for key in batch_keys(41, 3)]

    stacked = stack_spectra(singles)

    assert stacked["spectral_density"].shape == (3, stacked["frequencies"].size)
    for draw, single in enumerate(singles):
        np.testing.assert_array_equal(
            stacked["spectral_density"][draw], single["spectral_density"][0]
        )
        assert stacked["n_events"][draw] == single["n_events"][0]
        assert stacked["total_merger_rate"][draw] == single["total_merger_rate"][0]
        assert (
            stacked["hyperparameters"]["local_merger_rate"][draw]
            == single["hyperparameters"]["local_merger_rate"][0]
        )


@pytest.mark.integration
def test_spectra_chunk_size_changes_cost_not_the_draws() -> None:
    small = BackgroundSpectralDensitySimulator(SPECTRA, chunk_size=1)
    large = BackgroundSpectralDensitySimulator(SPECTRA, chunk_size=64)

    for key in batch_keys(41, 2):
        np.testing.assert_allclose(
            small(key)["spectral_density"], large(key)["spectral_density"], rtol=1e-12
        )


@pytest.mark.integration
def test_population_draw_does_not_depend_on_chunk_size() -> None:
    key = batch_keys(41, 1)[0]

    small = PopulationSimulator(SPECTRA.sources, chunk_size=1)(key)
    large = PopulationSimulator(SPECTRA.sources, chunk_size=64)(key)

    assert int(small["count"]) == int(large["count"]) == 4
    assert small["total_merger_rate"] == large["total_merger_rate"]
    for name, column in small["source_parameters"].items():
        assert column.shape == (4,)
        np.testing.assert_array_equal(column, large["source_parameters"][name])


@pytest.mark.integration
def test_poisson_population_draw_does_not_depend_on_chunk_size() -> None:
    poisson = SPECTRA.model_copy(
        update={"count": "poisson", "num_events": None, "observation_time": 1e-3}
    )
    key = batch_keys(41, 1)[0]

    small = PopulationSimulator(poisson.sources, chunk_size=50)(key)
    large = PopulationSimulator(poisson.sources, chunk_size=1000)(key)

    count = int(small["count"])
    assert count == int(large["count"]) > 50
    for name, column in small["source_parameters"].items():
        assert column.shape == (count,)
        np.testing.assert_array_equal(column, large["source_parameters"][name])


@pytest.mark.integration
def test_reducing_a_population_equals_the_spectrum_simulator() -> None:
    key = batch_keys(41, 1)[0]
    simulator = BackgroundSpectralDensitySimulator(SPECTRA, chunk_size=4)
    population = PopulationSimulator(SPECTRA.sources, chunk_size=4)(key)

    reduced = simulator.reduce(population)
    direct = simulator(key)

    _same(dict(reduced), dict(direct))  # ty: ignore[invalid-argument-type]


def test_population_simulator_takes_one_key() -> None:
    with pytest.raises(ValueError, match="single key"):
        PopulationSimulator(SPECTRA.sources)(batch_keys(41, 2))
