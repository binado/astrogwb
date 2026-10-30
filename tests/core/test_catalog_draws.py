"""Plain polarization-power catalogs, end to end on the closed-form inspiral.

Each test generates for real -- a tiny population, a four-bin grid -- so what
is checked is the draw, not a mock of it: a key names its realization, and a
file written from one reads back equal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astrogwb.gwb.catalog import CatalogMetadata, draw_catalog
from astrogwb.populations import PopulationMetadata
from astrogwb.simulators import batch_keys
from astrogwb.utils.io import Arrays, load, write
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


def test_catalog_key_tells_reference_redshifts_apart() -> None:
    keys = {
        CatalogMetadata.model_validate(
            {**CATALOG.model_dump(), "reference_redshift": reference}
        ).key()
        for reference in (None, 0.0, 0.1, 0.2)
    }
    assert len(keys) == 4


def test_catalog_key_tells_sobol_from_random() -> None:
    sobol = CatalogMetadata.model_validate(
        {**CATALOG.model_dump(), "num_samples": 8, "sampling": "sobol"}
    )
    random = CatalogMetadata.model_validate({**CATALOG.model_dump(), "num_samples": 8})
    assert sobol.key() != random.key()


def test_sobol_catalog_needs_a_power_of_two_size() -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        CatalogMetadata.model_validate({**CATALOG.model_dump(), "sampling": "sobol"})


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


@pytest.mark.parametrize("resolution", [None, 0.1, 0.2])
def test_nested_dlogf_round_trips_through_hdf5(
    tmp_path: Path,
    resolution: float | None,
) -> None:
    waveform = WaveformMetadata.model_validate(
        {
            **WAVEFORM.model_dump(),
            "frequency_spacing": "loglinear",
            "turnover_frequency": 30.0,
            "dlogf": resolution,
        }
    )
    metadata = CatalogMetadata.model_validate(
        {**CATALOG.model_dump(), "waveform": waveform.model_dump()}
    )
    path = write(
        tmp_path / "metadata.h5",
        {"frequencies": np.array([20.0, 50.0])},
        metadata,
        seed=41,
    )
    _, restored, _ = load(path, CatalogMetadata)
    assert restored == metadata
    assert restored.waveform.model_dump()["dlogf"] == resolution


def test_dlogf_changes_artifact_keys() -> None:
    keys = []
    for resolution in [None, 0.1, 0.2]:
        waveform = WaveformMetadata.model_validate(
            {
                **WAVEFORM.model_dump(),
                "frequency_spacing": "loglinear",
                "turnover_frequency": 30.0,
                "dlogf": resolution,
            }
        )
        metadata = CatalogMetadata.model_validate(
            {**CATALOG.model_dump(), "waveform": waveform.model_dump()}
        )
        keys.append(metadata.key())
    assert len(set(keys)) == 3
