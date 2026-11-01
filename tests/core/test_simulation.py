"""One-pass simulation: the config, the single HDF5 file and the ensemble reader."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from astrogwb_tiny_pipeline import (
    NETWORKS,
    WAVEFORM,
    assert_same_spectra,
    population_metadata,
)
from pydantic import ValidationError

from astrogwb import __version__
from astrogwb.simulators import (
    BackgroundArtifact,
    BackgroundSimulator,
    PopulationSimulator,
    SimulationConfig,
    network_spectra_ensemble,
    realization_key,
    simulate,
)

SEED = 41
CHUNK = 8


def _config(segment_duration: float | None = None) -> SimulationConfig:
    return SimulationConfig(
        population=population_metadata(),
        waveform=WAVEFORM,
        detectors=NETWORKS,
        segment_duration=segment_duration,
    )


@pytest.fixture(scope="module")
def simulated() -> BackgroundArtifact:
    """The tiny configuration simulated at ``(SEED, 0)``."""
    return simulate(_config(), realization_key(SEED, 0), chunk_size=CHUNK)


@pytest.fixture(scope="module")
def written(
    simulated: BackgroundArtifact, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    return simulated.write(tmp_path_factory.mktemp("one-pass") / f"{SEED}-0.h5")


def test_a_file_round_trips_the_artifact(
    simulated: BackgroundArtifact, written: Path
) -> None:
    reopened = BackgroundArtifact.open(written)

    assert simulated.count > 0
    assert reopened.config == simulated.config
    assert reopened.rng_key == simulated.rng_key
    assert reopened.count == simulated.count
    assert reopened.total_merger_rate == simulated.total_merger_rate
    assert reopened.hyperparameters == simulated.hyperparameters
    assert reopened.pairs == simulated.pairs
    np.testing.assert_array_equal(reopened.frequencies, simulated.frequencies)
    assert_same_spectra(reopened, simulated)


def test_the_file_records_the_config_and_the_version(written: Path) -> None:
    with h5py.File(written, "r") as file:
        assert file.attrs["version"] == __version__
        assert SimulationConfig.model_validate_json(file.attrs["metadata"]) == _config()


@pytest.mark.parametrize("version", ["0.0.0-stale", None])
def test_a_file_of_another_version_is_refused(
    written: Path, tmp_path: Path, version: str | None
) -> None:
    stale = tmp_path / "stale.h5"
    stale.write_bytes(written.read_bytes())
    with h5py.File(stale, "r+") as file:
        if version is None:
            del file.attrs["version"]
        else:
            file.attrs["version"] = version

    with pytest.raises(ValueError, match="regenerate"):
        BackgroundArtifact.open(stale)


def test_simulate_is_the_two_building_blocks_composed(
    simulated: BackgroundArtifact,
) -> None:
    config = _config()
    population = PopulationSimulator(config.population)(realization_key(SEED, 0))

    by_hand = BackgroundSimulator(WAVEFORM, NETWORKS, chunk_size=CHUNK)(population)

    assert_same_spectra(simulated, by_hand)
    assert simulated.rng_key == population.rng_key


def test_a_noisy_config_round_trips_its_segment_duration(tmp_path: Path) -> None:
    written = simulate(_config(60.0), realization_key(SEED, 0), chunk_size=CHUNK)
    reopened = BackgroundArtifact.open(written.write(tmp_path / "noisy.h5"))

    assert reopened.segment_duration == 60.0
    assert reopened.config.segment_duration == 60.0
    noise = reopened.noise()
    assert set(noise) == set(NETWORKS.networks)
    for name, spectrum in reopened.spectra.items():
        assert noise[name].pair_scale.shape == spectrum.shape


def test_a_nonpositive_segment_duration_is_refused() -> None:
    with pytest.raises(ValidationError):
        _config(0.0)


@pytest.fixture(scope="module")
def ensemble(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two draws of ``SEED``, each recorded by every tiny network."""
    root = tmp_path_factory.mktemp("ensemble")
    for index in range(2):
        draw = simulate(_config(), realization_key(SEED, index), chunk_size=CHUNK)
        draw.write(root / f"{SEED}-{index}.h5")
    return root


def test_network_spectra_ensemble_stacks_every_draws_pairs(ensemble: Path) -> None:
    triangle, pair = network_spectra_ensemble(ensemble, SEED, 2, ["triangle", "pair"])

    assert triangle["pair_spectral_density"].shape[:2] == (2, 3)
    # Sorted pairs: the triangle's (C1, S1) is its second, the pair's only one.
    np.testing.assert_array_equal(
        pair["pair_spectral_density"][:, 0], triangle["pair_spectral_density"][:, 1]
    )


@pytest.mark.parametrize("index", [0, 1])
def test_network_spectra_ensemble_reads_each_stored_spectrum(
    ensemble: Path, index: int
) -> None:
    (triangle,) = network_spectra_ensemble(ensemble, SEED, 2, ["triangle"])

    stored = BackgroundArtifact.open(ensemble / f"{SEED}-{index}.h5")
    np.testing.assert_array_equal(
        triangle["pair_spectral_density"][index], stored.spectra["triangle"]
    )


def test_network_spectra_ensemble_from_start_is_a_suffix(ensemble: Path) -> None:
    (full,) = network_spectra_ensemble(ensemble, SEED, 2, ["triangle"])
    (second,) = network_spectra_ensemble(ensemble, SEED, 2, ["triangle"], start=1)

    np.testing.assert_array_equal(
        second["pair_spectral_density"], full["pair_spectral_density"][1:]
    )


def test_network_spectra_ensemble_of_a_missing_draw_raises(ensemble: Path) -> None:
    with pytest.raises(FileNotFoundError):
        network_spectra_ensemble(ensemble, SEED, 3, ["triangle"])
