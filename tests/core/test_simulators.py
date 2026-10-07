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
    PolarizationPowerSimulator,
)
from astrogwb.simulators.population import PopulationSimulator
from astrogwb.simulators.spectra import (
    SpectralDensityCatalog,
    SpectraMetadata,
    SpectraSimulator,
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
    model_name="bns_md_cosmological",
    model_kwargs={"minimum_redshift": 0.0, "maximum_redshift": 5.0, "n_grid": 64},
)
CATALOG = CatalogMetadata(
    waveform=WAVEFORM, population=POPULATION, fiducials=FIDUCIALS, num_samples=6
)
SPECTRA = SpectraMetadata(
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
    simulator = PolarizationPowerSimulator(CATALOG)
    fresh = simulator(batch_keys(41, 1)[0])
    path = write(tmp_path / "catalog.h5", fresh, CATALOG, seed=41)

    data, metadata, attrs = load(path, CatalogMetadata)

    _same(dict(fresh), data)  # ty: ignore[invalid-argument-type]
    assert metadata == CATALOG
    assert attrs["seed"] == 41


@pytest.mark.integration
def test_polarization_power_key_picks_the_realization() -> None:
    simulator = PolarizationPowerSimulator(CATALOG)
    one = simulator(batch_keys(41, 1)[0])
    again = simulator(batch_keys(41, 1)[0])
    other = simulator(batch_keys(42, 1)[0])

    np.testing.assert_array_equal(
        one["polarization_power"], again["polarization_power"]
    )
    assert not np.array_equal(one["polarization_power"], other["polarization_power"])


@pytest.mark.integration
def test_spectra_round_trip_through_a_file(tmp_path: Path) -> None:
    simulator = SpectraSimulator(SPECTRA, chunk_size=4)
    fresh = stack_spectra([simulator(key) for key in batch_keys(41, 3)])
    path = write(tmp_path / "spectra.h5", fresh, SPECTRA, seed=41)

    data, metadata, attrs = load(path, SpectraMetadata)

    _same(dict(fresh), data)  # ty: ignore[invalid-argument-type]
    assert attrs["seed"] == 41
    catalog = SpectralDensityCatalog.from_arrays(data, metadata)
    assert catalog.num_draws == 3
    assert np.all(catalog.n_events == 4)


@pytest.mark.integration
def test_a_key_picks_the_spectrum() -> None:
    simulator = SpectraSimulator(SPECTRA, chunk_size=4)
    keys = batch_keys(41, 3)

    one = simulator(keys[1])
    again = simulator(keys[1])
    other = simulator(keys[2])

    np.testing.assert_array_equal(one["spectral_density"], again["spectral_density"])
    assert one["spectral_density"].shape == one["frequencies"].shape
    assert not np.array_equal(one["spectral_density"], other["spectral_density"])
    assert not np.array_equal(
        one["hyperparameters"]["local_merger_rate"],
        other["hyperparameters"]["local_merger_rate"],
    )


@pytest.mark.integration
def test_stacking_a_loop_equals_the_single_calls() -> None:
    simulator = SpectraSimulator(SPECTRA, chunk_size=4)
    singles = [simulator(key) for key in batch_keys(41, 3)]

    stacked = stack_spectra(singles)

    assert stacked["spectral_density"].shape == (3, stacked["frequencies"].size)
    for draw, single in enumerate(singles):
        np.testing.assert_array_equal(
            stacked["spectral_density"][draw], single["spectral_density"]
        )
        assert stacked["n_events"][draw] == single["n_events"]
        assert (
            stacked["hyperparameters"]["local_merger_rate"][draw]
            == single["hyperparameters"]["local_merger_rate"]
        )


@pytest.mark.integration
def test_spectra_chunk_size_changes_cost_not_the_draws() -> None:
    small = SpectraSimulator(SPECTRA, chunk_size=1)
    large = SpectraSimulator(SPECTRA, chunk_size=64)

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
    simulator = SpectraSimulator(SPECTRA, chunk_size=4)
    population = PopulationSimulator(SPECTRA.sources, chunk_size=4)(key)

    reduced = simulator.reduce(population)
    direct = simulator(key)

    np.testing.assert_array_equal(
        reduced["spectral_density"], direct["spectral_density"]
    )
    assert reduced["n_events"] == direct["n_events"]


def test_population_simulator_takes_one_key() -> None:
    with pytest.raises(ValueError, match="single key"):
        PopulationSimulator(SPECTRA.sources)(batch_keys(41, 2))
