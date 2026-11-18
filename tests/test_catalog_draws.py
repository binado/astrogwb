"""Reference intensity catalogs, end to end on the closed-form inspiral.

Each test generates for real -- a tiny population, a four-bin grid -- so what
is checked is the draw, not a mock of it: a key names its realization, and a
file written from one reads back equal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astrogwb.background.simulation import realization_key
from astrogwb.importance.catalog import (
    CatalogMetadata,
    read_catalog,
)
from astrogwb.importance.spectrum import reference_catalog
from astrogwb.populations import BNSCobaPopulation
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
    minimum_redshift=0.01,
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
def test_intensity_round_trips_through_a_file(tmp_path: Path) -> None:
    fresh = reference_catalog(CATALOG, realization_key(41, 0))
    path = write(tmp_path / "catalog.h5", fresh, CATALOG, seed=41)

    data, metadata, seed = read_catalog(path)

    _same(dict(fresh), dict(data))  # ty: ignore[invalid-argument-type]
    assert metadata == CATALOG
    assert seed == 41


def test_reading_a_missing_catalog_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.h5"):
        read_catalog(tmp_path / "missing.h5")
