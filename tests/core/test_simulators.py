"""The three simulators, end to end on the closed-form inspiral.

Each test generates for real -- a tiny population, a four-bin grid -- so what
is checked is the simulator, not a mock of it: a batch is the batch of its
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
    PolarizationPowerCatalog,
    PolarizationPowerSimulator,
)
from astrogwb.simulators.spectra import (
    SpectralDensityCatalog,
    SpectraMetadata,
    SpectraSimulator,
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
    fresh = simulator.simulate(batch_keys(41, 1)[0])
    path = write(tmp_path / "catalog.h5", fresh, CATALOG, seed=41)

    data, metadata, attrs = load(path, CatalogMetadata)

    _same(dict(fresh), data)  # ty: ignore[invalid-argument-type]
    assert metadata == CATALOG
    assert attrs["seed"] == 41
    catalog = PolarizationPowerCatalog.from_arrays(data, metadata)
    assert catalog.num_samples == CATALOG.num_samples


@pytest.mark.integration
def test_polarization_power_key_picks_the_realization() -> None:
    simulator = PolarizationPowerSimulator(CATALOG)
    one = simulator.simulate(batch_keys(41, 1)[0])
    again = simulator.simulate(batch_keys(41, 1)[0])
    other = simulator.simulate(batch_keys(42, 1)[0])

    np.testing.assert_array_equal(
        one["polarization_power"], again["polarization_power"]
    )
    assert not np.array_equal(one["polarization_power"], other["polarization_power"])


@pytest.mark.integration
def test_polarization_power_batch_is_the_batch_of_simulate() -> None:
    simulator = PolarizationPowerSimulator(CATALOG)
    keys = batch_keys(41, 3)
    batch = simulator.simulate_batch(keys)
    assert batch["polarization_power"].shape[0] == 3
    for i in range(3):
        single = simulator.simulate(keys[i])
        np.testing.assert_allclose(
            single["polarization_power"], batch["polarization_power"][i], rtol=1e-12
        )
        for name, column in single["source_parameters"].items():
            np.testing.assert_array_equal(column, batch["source_parameters"][name][i])


@pytest.mark.integration
def test_spectra_round_trip_through_a_file(tmp_path: Path) -> None:
    simulator = SpectraSimulator(SPECTRA, chunk_size=4)
    fresh = simulator.simulate_batch(batch_keys(41, 3))
    path = write(tmp_path / "spectra.h5", fresh, SPECTRA, seed=41, batch_size=32)

    data, metadata, attrs = load(path, SpectraMetadata)

    _same(dict(fresh), data)  # ty: ignore[invalid-argument-type]
    assert attrs["batch_size"] == 32
    catalog = SpectralDensityCatalog.from_arrays(data, metadata)
    assert catalog.num_draws == 3
    assert np.all(catalog.n_events == 4)


@pytest.mark.integration
def test_a_spectrum_depends_on_its_own_key_alone() -> None:
    simulator = SpectraSimulator(SPECTRA, chunk_size=4)
    keys = batch_keys(41, 3)

    together = simulator.simulate_batch(keys)
    alone = simulator.simulate(keys[1])

    np.testing.assert_allclose(
        together["spectral_density"][1], alone["spectral_density"], rtol=1e-10
    )


@pytest.mark.integration
def test_spectra_chunk_size_changes_cost_not_the_draws() -> None:
    keys = batch_keys(41, 2)

    small = SpectraSimulator(SPECTRA, chunk_size=1).simulate_batch(keys)
    large = SpectraSimulator(SPECTRA, chunk_size=64).simulate_batch(keys)

    np.testing.assert_allclose(
        small["spectral_density"], large["spectral_density"], rtol=1e-12
    )
