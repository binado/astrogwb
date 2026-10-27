"""Realization directories: stages recorded, added to, regenerated and refused."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from astrogwb_tiny_pipeline import (
    NETWORKS,
    WAVEFORM,
    assert_same_spectra,
    population_metadata,
)

from astrogwb.simulators import (
    BackgroundSimulator,
    DetectorRegistry,
    Realization,
    RealizationSimulator,
    batch_keys,
    network_spectra_ensemble,
    population_key,
    simulate_ensemble,
)
from astrogwb.simulators._io import (
    BACKGROUND_FILE,
    MANIFEST_FILE,
    POLARIZATION_FILE,
    SOURCES_FILE,
)

SEED = 41
POPULATION = population_metadata()


def _simulator(**settings: Any) -> RealizationSimulator:
    """A tiny-chunk simulator: three chunks or more, whatever the count."""
    return RealizationSimulator(chunk_size=8, source_chunk_size=16, **settings)


def _products(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


@pytest.fixture(scope="module")
def fused(tmp_path_factory: pytest.TempPathFactory) -> Realization:
    """One fused run of every stage, sources kept."""
    directory = tmp_path_factory.mktemp("fused")
    return _simulator(population=POPULATION, waveform=WAVEFORM, background=NETWORKS)(
        directory, seed=SEED
    )


@pytest.mark.parametrize(
    ("keep_sources", "expected"),
    [
        (True, [BACKGROUND_FILE, MANIFEST_FILE, SOURCES_FILE]),
        (False, [BACKGROUND_FILE, MANIFEST_FILE]),
    ],
)
def test_a_fused_run_writes_the_manifest_and_background(
    tmp_path: Path, keep_sources: bool, expected: list[str]
) -> None:
    realization = _simulator(
        population=POPULATION,
        waveform=WAVEFORM,
        background=NETWORKS,
        keep_sources=keep_sources,
    )(tmp_path, seed=SEED, index=2)

    assert _products(tmp_path) == sorted(expected)
    reopened = Realization.open(tmp_path)
    assert reopened.manifest == realization.manifest
    assert (reopened.seed, reopened.index) == (SEED, 2)
    assert reopened.id == population_key(POPULATION, reopened.rng_key)
    assert reopened.background().count == reopened.count


def test_a_background_added_later_equals_the_fused_one(
    tmp_path: Path, fused: Realization
) -> None:
    _simulator(population=POPULATION)(tmp_path, seed=SEED)
    assert _products(tmp_path) == [MANIFEST_FILE, SOURCES_FILE]

    later = _simulator(waveform=WAVEFORM, background=NETWORKS)(tmp_path)

    assert_same_spectra(later.background(), fused.background())
    assert later.manifest["background"] == fused.manifest["background"]


def test_kept_polarizations_give_the_fused_background(
    tmp_path: Path, fused: Realization
) -> None:
    realization = _simulator(
        population=POPULATION,
        waveform=WAVEFORM,
        background=NETWORKS,
        keep_polarizations=True,
    )(tmp_path, seed=SEED)
    (tmp_path / BACKGROUND_FILE).unlink()

    reprojected = realization.background(
        simulator=BackgroundSimulator(NETWORKS, chunk_size=8),
    )

    assert POLARIZATION_FILE in _products(tmp_path)
    assert_same_spectra(reprojected, fused.background())


def test_a_rerun_with_the_same_settings_rewrites_nothing(tmp_path: Path) -> None:
    simulator = _simulator(
        population=POPULATION, waveform=WAVEFORM, background=NETWORKS
    )
    simulator(tmp_path, seed=SEED)
    stamps = {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}

    simulator(tmp_path, seed=SEED)

    assert {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()} == (
        stamps
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"waveform": WAVEFORM.model_copy(update={"maximum_frequency": 40.0})},
        {
            "background": NETWORKS.model_copy(
                update={
                    "networks": {
                        "triangle": ("S1", "R1"),
                        "pair": NETWORKS.networks["pair"],
                    }
                }
            )
        },
        {
            "background": NETWORKS.model_copy(
                update={
                    "detectors": {
                        **NETWORKS.detectors,
                        "C1": NETWORKS.detectors["C1"].model_copy(
                            update={"psd_reference": "ET_COBA_15km_psd.txt"}
                        ),
                    }
                }
            )
        },
        {
            "background": DetectorRegistry(
                detectors=NETWORKS.detectors,
                networks={"triangle": NETWORKS.networks["triangle"]},
            )
        },
        {"population": POPULATION, "seed": SEED + 1},
    ],
    ids=["waveform", "detectors", "noise-curves", "networks", "seed"],
)
def test_conflicting_settings_raise_and_touch_nothing(
    tmp_path: Path, fused: Realization, settings: dict[str, Any]
) -> None:
    for path in fused.directory.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    settings = dict(settings)
    seed = settings.pop("seed", None)

    with pytest.raises(ValueError, match="holds"):
        _simulator(**settings)(tmp_path, seed=seed)

    assert _products(tmp_path) == _products(fused.directory)
    assert Realization.open(tmp_path).manifest == fused.manifest


def test_a_deleted_background_is_regenerated_identically(
    tmp_path: Path, fused: Realization
) -> None:
    simulator = _simulator(
        population=POPULATION,
        waveform=WAVEFORM,
        background=NETWORKS,
        keep_sources=False,
    )
    simulator(tmp_path, seed=SEED)
    (tmp_path / BACKGROUND_FILE).unlink()

    regenerated = simulator(tmp_path, seed=SEED).background()

    assert_same_spectra(regenerated, fused.background())


def test_a_background_without_waveform_settings_raises(tmp_path: Path) -> None:
    _simulator(population=POPULATION)(tmp_path, seed=SEED)

    with pytest.raises(ValueError, match="no waveform settings"):
        _simulator(background=NETWORKS)(tmp_path)


def test_the_manifest_records_the_networks_sorted(fused: Realization) -> None:
    assert fused.networks == ("pair", "triangle")
    assert NETWORKS.pairs("pair") == [("C1", "S1")]
    assert fused.manifest["background"]["metadata"] == {
        "pairs": {
            "pair": [["C1", "S1"]],
            "triangle": [["C1", "R1"], ["C1", "S1"], ["R1", "S1"]],
        }
    }


def test_a_directory_without_a_realization_needs_population_settings(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="no realization"):
        _simulator(waveform=WAVEFORM)(tmp_path)


def test_an_ensemble_is_a_prefix_of_a_larger_one(tmp_path: Path) -> None:
    simulator = _simulator(population=POPULATION, keep_sources=False)

    larger = simulate_ensemble(simulator, tmp_path / "larger", SEED, 3)
    smaller = simulate_ensemble(simulator, tmp_path / "smaller", SEED, 2)

    assert [r.directory.name for r in larger] == [f"{SEED}-{i}" for i in range(3)]
    assert [r.manifest["realization"] for r in smaller] == [
        r.manifest["realization"] for r in larger[:2]
    ]
    keys = np.asarray(jax.random.key_data(batch_keys(SEED, 3)))
    assert [list(r.rng_key) for r in larger] == keys.tolist()


@pytest.fixture(scope="module")
def ensemble(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two draws of ``SEED``, each recorded by every tiny network."""
    root = tmp_path_factory.mktemp("ensemble")
    simulator = _simulator(
        population=POPULATION, waveform=WAVEFORM, background=NETWORKS
    )
    simulate_ensemble(simulator, root, SEED, 2)
    return root


def test_network_spectra_ensemble_stacks_every_draws_pairs(ensemble: Path) -> None:
    triangle, pair = network_spectra_ensemble(ensemble, SEED, 2, ["triangle", "pair"])

    assert triangle["pair_spectral_density"].shape[:2] == (2, 3)
    # Sorted pairs: the triangle's (C1, S1) is its second, the pair's only one.
    np.testing.assert_array_equal(
        pair["pair_spectral_density"][:, 0], triangle["pair_spectral_density"][:, 1]
    )


@pytest.mark.parametrize("index", [0, 1])
def test_network_spectra_ensemble_reads_each_stored_network_spectrum(
    ensemble: Path, index: int
) -> None:
    (triangle,) = network_spectra_ensemble(ensemble, SEED, 2, ["triangle"])

    stored = Realization.open(ensemble / f"{SEED}-{index}").background()
    np.testing.assert_array_equal(
        triangle["spectral_density"][index],
        stored.spectra["triangle"].spectral_density,
    )


def test_network_spectra_ensemble_from_start_is_a_suffix(ensemble: Path) -> None:
    (full,) = network_spectra_ensemble(ensemble, SEED, 2, ["triangle"])
    (second,) = network_spectra_ensemble(ensemble, SEED, 2, ["triangle"], start=1)

    np.testing.assert_array_equal(
        second["spectral_density"], full["spectral_density"][1:]
    )


def test_network_spectra_ensemble_of_a_missing_draw_raises(ensemble: Path) -> None:
    with pytest.raises(FileNotFoundError):
        network_spectra_ensemble(ensemble, SEED, 3, ["triangle"])
