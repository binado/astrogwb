"""Plain polarization-power catalogs, end to end on the closed-form inspiral.

Each test generates for real -- a tiny population, a four-bin grid -- so what
is checked is the draw, not a mock of it: a key names its realization, and a
file written from one reads back equal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astrogwb.gwb.catalog import (
    CatalogMetadata,
    read_catalog,
    simulate_catalog,
    validate_matching_frequency_grids,
)
from astrogwb.populations import BNSCobaPopulation
from astrogwb.simulators import realization_key
from astrogwb.utils.io import Arrays, write
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
POPULATION = BNSCobaPopulation(
    mass_model="uniform",
    minimum_redshift=0.0,
    maximum_redshift=5.0,
    n_grid=64,
)
CATALOG = CatalogMetadata(
    waveform=WAVEFORM, population=POPULATION, fiducials=FIDUCIALS, num_samples=6
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
    fresh = simulate_catalog(CATALOG, realization_key(41, 0))
    path = write(tmp_path / "catalog.h5", fresh, CATALOG, seed=41)

    data, metadata, seed = read_catalog(path)

    _same(dict(fresh), dict(data))  # ty: ignore[invalid-argument-type]
    assert metadata == CATALOG
    assert seed == 41


def test_reading_a_missing_catalog_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.h5"):
        read_catalog(tmp_path / "missing.h5")


def test_catalogs_on_different_frequency_grids_are_refused() -> None:
    with pytest.raises(ValueError, match="identical frequency grids"):
        validate_matching_frequency_grids(np.array([2.0, 3.0]), np.array([2.0, 4.0]))


@pytest.mark.integration
def test_polarization_power_key_picks_the_realization() -> None:
    one = simulate_catalog(CATALOG, realization_key(41, 0))
    again = simulate_catalog(CATALOG, realization_key(41, 0))
    other = simulate_catalog(CATALOG, realization_key(42, 0))

    np.testing.assert_array_equal(
        one["polarization_power"], again["polarization_power"]
    )
    assert not np.array_equal(one["polarization_power"], other["polarization_power"])


@pytest.mark.integration
@pytest.mark.parametrize("chunk_size", [4, 100])
def test_polarization_power_is_independent_of_chunk_size(chunk_size: int) -> None:
    key = realization_key(41, 0)
    whole = simulate_catalog(CATALOG, key, chunk_size=None)
    chunked = simulate_catalog(CATALOG, key, chunk_size=chunk_size)

    # Sources are drawn before chunking, so they are exact. The power is the
    # same elementwise computation, but XLA may fuse a differently shaped chunk
    # differently, which moves the last bit.
    _same(whole["source_parameters"], chunked["source_parameters"])  # ty: ignore[invalid-argument-type]
    np.testing.assert_allclose(
        chunked["polarization_power"], whole["polarization_power"], rtol=1e-12, atol=0
    )
