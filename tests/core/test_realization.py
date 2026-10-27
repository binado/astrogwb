"""Realization directories: stages recorded, added to, regenerated and refused."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from astrogwb_tiny_pipeline import DETECTORS, WAVEFORM, population_metadata

from astrogwb.simulators import (
    DetectorConfiguration,
    Realization,
    RealizationSimulator,
    batch_keys,
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
    return _simulator(population=POPULATION, waveform=WAVEFORM, background=DETECTORS)(
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
        background=DETECTORS,
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

    later = _simulator(waveform=WAVEFORM, background=DETECTORS)(tmp_path)

    np.testing.assert_array_equal(
        later.background().pair_spectral_density,
        fused.background().pair_spectral_density,
    )
    assert later.manifest["background"] == fused.manifest["background"]


def test_kept_polarizations_give_the_fused_background(
    tmp_path: Path, fused: Realization
) -> None:
    realization = _simulator(
        population=POPULATION,
        waveform=WAVEFORM,
        background=DETECTORS,
        keep_polarizations=True,
    )(tmp_path, seed=SEED)
    (tmp_path / BACKGROUND_FILE).unlink()

    reprojected = realization.background(_simulator().background_simulator(DETECTORS))

    assert POLARIZATION_FILE in _products(tmp_path)
    np.testing.assert_array_equal(
        reprojected.pair_spectral_density, fused.background().pair_spectral_density
    )


def test_a_rerun_with_the_same_settings_rewrites_nothing(tmp_path: Path) -> None:
    simulator = _simulator(
        population=POPULATION, waveform=WAVEFORM, background=DETECTORS
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
            "background": DetectorConfiguration(
                detectors=["S1", "R1"], psd_references=DETECTORS.psd_references
            )
        },
        {"population": POPULATION, "seed": SEED + 1},
    ],
    ids=["waveform", "background", "seed"],
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
        background=DETECTORS,
        keep_sources=False,
    )
    simulator(tmp_path, seed=SEED)
    (tmp_path / BACKGROUND_FILE).unlink()

    regenerated = simulator(tmp_path, seed=SEED).background()

    np.testing.assert_array_equal(
        regenerated.pair_spectral_density, fused.background().pair_spectral_density
    )


def test_a_background_without_waveform_settings_raises(tmp_path: Path) -> None:
    _simulator(population=POPULATION)(tmp_path, seed=SEED)

    with pytest.raises(ValueError, match="no waveform settings"):
        _simulator(background=DETECTORS)(tmp_path)


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
