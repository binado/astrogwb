"""The redshift-node importance catalog: its inclination and its draw."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astrogwb.gwb.importance import (
    EFFECTIVE_INCLINATION,
    ImportanceCatalogMetadata,
    importance_catalog,
)
from astrogwb.populations import PopulationMetadata
from astrogwb.simulators.core import batch_keys, load, write
from astrogwb.waveform import WaveformMetadata
from astrogwb.waveform.generator.analytical import inclination_factor


@pytest.fixture
def metadata() -> ImportanceCatalogMetadata:
    """A tiny closed-form catalog: six draws at three nodes on a four-bin grid."""
    return ImportanceCatalogMetadata(
        waveform=WaveformMetadata(
            approximant="AnalyticInspiral",
            minimum_frequency=20.0,
            maximum_frequency=50.0,
            reference_frequency=20.0,
            sampling_frequency=128.0,
            frequency_resolution=10.0,
        ),
        population=PopulationMetadata(
            model_name="bns_coba",
            model_kwargs={
                "mass_model": "uniform",
                "minimum_redshift": 0.1,
                "maximum_redshift": 5.0,
                "n_grid": 64,
            },
        ),
        fiducials={
            "H0": 67.66,
            "Omega_m": 0.3096,
            "gamma": 1.42,
            "kappa": 4.62,
            "z_peak": 1.84,
            "local_merger_rate": 770.0,
            "minimum_mass": 1.0,
            "mass_width": 1.5,
        },
        num_samples=6,
        num_redshift_nodes=3,
    )


def test_effective_inclination_gives_the_isotropic_mean_factor() -> None:
    assert float(inclination_factor(EFFECTIVE_INCLINATION)) == pytest.approx(
        0.8, rel=1e-14
    )


@pytest.mark.integration
def test_importance_catalog_places_every_draw_at_every_node(
    metadata: ImportanceCatalogMetadata,
) -> None:
    data = importance_catalog(metadata, batch_keys(41, 1)[0], chunk_size=4)

    nodes, _ = metadata.nodes()
    columns = data["source_parameters"]
    assert data["polarization_power"].shape == (4, 18)
    np.testing.assert_array_equal(columns["redshift"], np.repeat(nodes, 6))
    np.testing.assert_array_equal(
        columns["source_frame_mass_1"].reshape(3, 6),
        np.tile(columns["source_frame_mass_1"][:6], (3, 1)),
    )
    np.testing.assert_allclose(
        columns["detector_frame_mass_1"],
        columns["source_frame_mass_1"] * (1.0 + columns["redshift"]),
    )


@pytest.mark.integration
def test_importance_catalog_round_trips_through_a_file(
    metadata: ImportanceCatalogMetadata, tmp_path: Path
) -> None:
    fresh = importance_catalog(metadata, batch_keys(41, 1)[0])
    path = write(tmp_path / "catalog.h5", fresh, metadata, seed=41)

    data, recorded, attrs = load(path, ImportanceCatalogMetadata)

    np.testing.assert_array_equal(
        data["polarization_power"], fresh["polarization_power"]
    )
    assert recorded == metadata
    assert attrs["seed"] == 41
