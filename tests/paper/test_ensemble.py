"""Ensembles read from an explicit root, and the committed simulation inputs."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest
from repo import REPO_ROOT

from astrogwb.paper.config import detector_registry
from astrogwb.paper.ensemble import (
    ensemble_config_tables,
    network_configuration,
    network_spectra_ensemble,
)
from astrogwb.populations import PopulationMetadata as PopulationSpec
from astrogwb.simulators import (
    PopulationMetadata,
    Realization,
    RealizationSimulator,
    simulate_ensemble,
)
from astrogwb.waveform import WaveformMetadata

SEED = 41
NAMES = ("ET-triangular", "ET-triangular-CE-Hanford")


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two tiny draws of seed 41, recorded by the ET triangle with and without CE."""
    registry = detector_registry(root=REPO_ROOT)
    population = PopulationMetadata(
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
    )
    waveform = WaveformMetadata(
        approximant="AnalyticInspiral",
        minimum_frequency=20.0,
        maximum_frequency=50.0,
        reference_frequency=20.0,
        sampling_frequency=128.0,
        frequency_resolution=10.0,
    )
    simulator = RealizationSimulator(
        population,
        waveform,
        {name: network_configuration(registry, name) for name in NAMES},
        keep_sources=False,
        chunk_size=8,
    )
    directory = tmp_path_factory.mktemp("ensemble")
    simulate_ensemble(simulator, directory, SEED, 2)
    return directory


def test_network_spectra_ensemble_stacks_each_networks_draws(root: Path) -> None:
    triangle, with_ce = network_spectra_ensemble(root, SEED, 2, NAMES)

    assert triangle["pair_spectral_density"].shape[:2] == (2, 3)
    assert with_ce["pair_spectral_density"].shape[:2] == (2, 6)
    # Sorted pairs: (C1, E1), (C1, E2), (C1, E3), then the triangle's three.
    np.testing.assert_array_equal(
        with_ce["pair_spectral_density"][:, 3:], triangle["pair_spectral_density"]
    )


@pytest.mark.parametrize("index", [0, 1])
def test_network_spectra_ensemble_reads_the_stored_network_spectrum(
    root: Path, index: int
) -> None:
    (triangle,) = network_spectra_ensemble(root, SEED, 2, NAMES[:1])

    stored = Realization.open(root / f"{SEED}-{index}").background()
    np.testing.assert_array_equal(
        triangle["spectral_density"][index],
        stored.network(NAMES[0]).spectral_density,
    )
    assert triangle["n_events"][index] == stored.count


def test_network_spectra_ensemble_reads_a_range_of_indices(root: Path) -> None:
    (full,) = network_spectra_ensemble(root, SEED, 2, NAMES[:1])
    (second,) = network_spectra_ensemble(root, SEED, 2, NAMES[:1], start=1)

    np.testing.assert_array_equal(
        second["spectral_density"], full["spectral_density"][1:]
    )


def test_network_spectra_ensemble_of_a_missing_draw_raises(root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        network_spectra_ensemble(root, SEED, 3, NAMES[:1])


@pytest.mark.parametrize(
    ("relative", "table"),
    list(ensemble_config_tables(REPO_ROOT).items()),
    ids=str,
)
def test_committed_ensemble_configs_match_the_paper_config(
    relative: Path, table: dict[str, object]
) -> None:
    """Rerun ``scripts/write_ensemble_configs.py`` when this fails."""
    with (REPO_ROOT / relative).open("rb") as file:
        assert tomllib.load(file) == table
