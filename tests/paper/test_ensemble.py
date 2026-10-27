"""Network ensembles: one cached background per draw, shared by every network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from repo import REPO_ROOT

from astrogwb.paper.config import detector_registry
from astrogwb.paper.ensemble import (
    EnsembleRecord,
    network_configuration,
    network_spectra_ensemble,
)
from astrogwb.populations import PopulationMetadata as PopulationSpec
from astrogwb.simulators import PopulationMetadata
from astrogwb.waveform import WaveformMetadata


def test_an_ensemble_reads_back_its_draws_as_a_prefix(tmp_path: Path) -> None:
    registry = detector_registry(root=REPO_ROOT)
    record = EnsembleRecord(
        population=PopulationMetadata(
            population=PopulationSpec(
                model_name="bns_coba",
                model_kwargs={
                    "mass_model": "uniform",
                    "minimum_redshift": 0.0,
                    "maximum_redshift": 5.0,
                    "n_grid": 64,
                },
            ),
            hyperparameters={
                "H0": 67.66,
                "Omega_m": 0.3096,
                "gamma": 1.42,
                "kappa": 4.62,
                "z_peak": 1.84,
                "minimum_mass": 1.0,
                "mass_width": 1.5,
                "local_merger_rate": 750.0,
            },
            observation_time=1e-4,
        ),
        waveform=WaveformMetadata(
            approximant="AnalyticInspiral",
            minimum_frequency=20.0,
            maximum_frequency=50.0,
            reference_frequency=20.0,
            sampling_frequency=128.0,
            frequency_resolution=10.0,
        ),
        networks={
            name: network_configuration(registry, name)
            for name in ("ET-triangular", "ET-triangular-CE-Hanford")
        },
    )

    drawn = network_spectra_ensemble(record, 41, 2, tmp_path, chunk_size=8)
    cached = network_spectra_ensemble(record, 41, 1, tmp_path, chunk_size=8)

    assert len(list((tmp_path / record.key()).iterdir())) == 2
    triangle, with_ce = drawn
    assert triangle["pair_spectral_density"].shape[:2] == (2, 3)
    assert with_ce["pair_spectral_density"].shape[:2] == (2, 6)
    # The triangle's pairs (E1E2, E1E3, E2E3) are rows of the network adding C1.
    np.testing.assert_array_equal(
        with_ce["pair_spectral_density"][:, [0, 1, 3]],
        triangle["pair_spectral_density"],
    )
    for full, prefix in zip(drawn, cached, strict=True):
        np.testing.assert_array_equal(
            prefix["spectral_density"], full["spectral_density"][:1]
        )
        np.testing.assert_array_equal(prefix["n_events"], full["n_events"][:1])
