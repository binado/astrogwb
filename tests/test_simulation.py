"""One-pass simulation: the config, the single HDF5 file and the ensemble reader."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from astrogwb_tiny_pipeline import (
    NETWORKS,
    PAIR,
    TRIANGLE,
    WAVEFORM,
    event_pool,
    write_population_csv,
)

from astrogwb.background.simulation import (
    BackgroundDraw,
    SimulationConfig,
    network_noise,
    network_spectra_ensemble,
    project_background,
    read_event_pool,
    realization_key,
    simulate_background,
)
from astrogwb.detector import DetectorNetworkConfig

SEED = 41
CHUNK = 8
SEGMENT_SECONDS = 1.0
RATE_HZ = 8.0


def _config(index: int = 0, *, incoherent: bool = True) -> SimulationConfig:
    return SimulationConfig(
        waveform=WAVEFORM,
        networks=NETWORKS,
        observer_rate_hz=RATE_HZ,
        num_segments=2,
        segment_seconds=SEGMENT_SECONDS,
        incoherent=incoherent,
        population="pool.csv",
        seed=SEED,
        index=index,
    )


def _simulate(config: SimulationConfig) -> BackgroundDraw:
    return simulate_background(
        realization_key(config.seed, config.index),
        event_pool(),
        config.waveform,
        config.networks,
        observer_rate_hz=config.observer_rate_hz,
        num_segments=config.num_segments,
        segment_seconds=config.segment_seconds,
        incoherent=config.incoherent,
        chunk_size=CHUNK,
    )


@pytest.fixture(scope="module")
def draw() -> BackgroundDraw:
    """The tiny configuration simulated at ``(SEED, 0)``."""
    return _simulate(_config())


@pytest.fixture(scope="module")
def written(draw: BackgroundDraw, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return draw.write(tmp_path_factory.mktemp("one-pass") / f"{SEED}-0.h5", _config())


def test_a_file_round_trips_the_draw_and_its_config(
    draw: BackgroundDraw, written: Path
) -> None:
    reopened, config = BackgroundDraw.read(written)

    assert draw.segment_counts.sum() > 0
    assert config == _config()
    np.testing.assert_array_equal(reopened.frequencies, draw.frequencies)
    np.testing.assert_array_equal(reopened.segment_counts, draw.segment_counts)
    assert reopened.spectra.keys() == draw.spectra.keys()
    for name, spectrum in draw.spectra.items():
        np.testing.assert_array_equal(reopened.spectra[name], spectrum)


def test_file_keeps_the_legacy_hdf5_and_detector_metadata_layout(written: Path) -> None:
    with h5py.File(written, "r") as file:
        assert set(file) == {"frequencies", "segment_counts", "networks"}
        assert "version" in file.attrs
        assert set(file["networks"]) == {"triangle", "pair"}
        assert set(file["networks/triangle"]) == {"pair_spectral_density"}
        assert set(file["networks/pair"]) == {"pair_spectral_density"}
        metadata = json.loads(file.attrs["metadata"])

    assert "networks" not in metadata
    assert metadata["detectors"] == {
        "detectors": {
            name: detector.model_dump(mode="json")
            for name, detector in TRIANGLE.detectors.items()
        },
        "networks": {"triangle": ["C1", "R1", "S1"], "pair": ["C1", "S1"]},
    }


def _edited(written: Path, tmp_path: Path, **attrs: object) -> Path:
    """A copy of ``written`` with the root attributes set, ``None`` deleting one."""
    copy = tmp_path / "edited.h5"
    copy.write_bytes(written.read_bytes())
    with h5py.File(copy, "r+") as file:
        for name, value in attrs.items():
            if value is None:
                del file.attrs[name]
            else:
                file.attrs[name] = value
    return copy


def test_a_file_records_its_format_version(written: Path) -> None:
    with h5py.File(written, "r") as file:
        assert file.attrs["format_version"] == 1


def test_a_legacy_0_8_file_without_a_format_version_is_read(
    draw: BackgroundDraw, written: Path, tmp_path: Path
) -> None:
    legacy = _edited(written, tmp_path, format_version=None, version="0.8.0")

    reopened, _ = BackgroundDraw.read(legacy)

    np.testing.assert_array_equal(reopened.segment_counts, draw.segment_counts)


def test_a_file_is_read_whatever_astrogwb_version_wrote_it(
    written: Path, tmp_path: Path
) -> None:
    BackgroundDraw.read(_edited(written, tmp_path, version="0.8.1.dev3"))


def test_a_file_of_an_unknown_format_is_refused(written: Path, tmp_path: Path) -> None:
    future = _edited(written, tmp_path, format_version=99, version="9.9.9")

    with pytest.raises(ValueError, match=r"format_version 99.*9\.9\.9"):
        BackgroundDraw.read(future)


def test_an_unversioned_file_of_another_release_is_refused(
    written: Path, tmp_path: Path
) -> None:
    stale = _edited(written, tmp_path, format_version=None, version="0.7.0")

    with pytest.raises(ValueError, match=r"format_version None.*0\.7\.0"):
        BackgroundDraw.read(stale)


def test_extra_attrs_are_written_with_the_file_and_read_back(
    draw: BackgroundDraw, tmp_path: Path
) -> None:
    extra = {"hyperparameters": '{"alpha": 2.7}', "total_merger_rate": 1.5, "n": 3}
    path = draw.write(tmp_path / "attrs.h5", _config(), **extra)

    assert BackgroundDraw.read_attrs(path) == extra
    assert BackgroundDraw.read(path)[1] == _config()


def test_no_attrs_reads_back_empty(written: Path) -> None:
    assert BackgroundDraw.read_attrs(written) == {}


@pytest.mark.parametrize("key", ["metadata", "version", "format_version"])
def test_reserved_attrs_are_refused_and_nothing_is_written(
    draw: BackgroundDraw, tmp_path: Path, key: str
) -> None:
    path = tmp_path / "reserved.h5"

    with pytest.raises(ValueError, match=key):
        draw.write(path, _config(), **{key: "x"})

    assert not path.exists()


def test_observation_seconds_is_the_segments_together() -> None:
    assert _config().observation_seconds == 2 * SEGMENT_SECONDS


def test_network_noise_has_one_row_per_pair_of_the_draw(draw: BackgroundDraw) -> None:
    noise = network_noise(_config(), draw.frequencies)

    assert noise.keys() == draw.spectra.keys()
    for name, spectrum in draw.spectra.items():
        assert noise[name].pair_scale.shape == spectrum.shape
        assert noise[name].pair_overlap.shape == spectrum.shape


def test_coherent_joint_draw_reuses_the_shared_baseline() -> None:
    draw = _simulate(_config(incoherent=False))

    (shared,) = PAIR.pairs()
    pair = draw.spectra[PAIR.name][0]
    assert np.any(pair)
    np.testing.assert_array_equal(
        pair, draw.spectra[TRIANGLE.name][TRIANGLE.pairs().index(shared)]
    )


@pytest.fixture(scope="module")
def ensemble(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two draws of ``SEED``, each recorded by every tiny network."""
    root = tmp_path_factory.mktemp("ensemble")
    for index in range(2):
        config = _config(index)
        _simulate(config).write(root / f"{SEED}-{index}.h5", config)
    return root


def test_network_spectra_ensemble_stacks_every_draws_pairs(ensemble: Path) -> None:
    triangle, pair = network_spectra_ensemble(ensemble, SEED, 2, ["triangle", "pair"])

    assert triangle["pair_spectral_density"].shape[:2] == (2, 3)
    assert triangle["n_events"].shape == (2,)
    # Sorted pairs: the triangle's (C1, S1) is its second, the pair's only one.
    np.testing.assert_array_equal(
        pair["pair_spectral_density"][:, 0], triangle["pair_spectral_density"][:, 1]
    )


def test_network_spectra_ensemble_counts_the_events_of_every_segment(
    ensemble: Path,
) -> None:
    (stacked,) = network_spectra_ensemble(ensemble, SEED, 2, ["pair"])

    expected = [
        BackgroundDraw.read(ensemble / f"{SEED}-{index}.h5")[0].segment_counts.sum()
        for index in range(2)
    ]
    np.testing.assert_array_equal(stacked["n_events"], expected)


def test_network_spectra_ensemble_from_start_is_a_suffix(ensemble: Path) -> None:
    (full,) = network_spectra_ensemble(ensemble, SEED, 2, ["triangle"])
    (second,) = network_spectra_ensemble(ensemble, SEED, 2, ["triangle"], start=1)

    np.testing.assert_array_equal(
        second["pair_spectral_density"], full["pair_spectral_density"][1:]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("num_segments", 0), ("segment_seconds", 0.0), ("observer_rate_hz", -1.0)],
)
def test_config_rejects_a_nonphysical_observation(field: str, value: float) -> None:
    fields = {**_config().model_dump(), field: value}

    with pytest.raises(ValueError, match=field):
        SimulationConfig.model_validate(fields)


@pytest.mark.parametrize(
    ("issue", "message"),
    [
        ("duplicate", "duplicate network name.*pair"),
        ("singleton", "at least two detectors"),
        ("geometry", "conflicting definitions.*C1"),
        ("psd_reference", "conflicting definitions.*C1"),
    ],
)
def test_background_boundaries_reject_invalid_networks(
    issue: str, message: str
) -> None:
    detector = PAIR.detectors["C1"]
    if issue == "duplicate":
        networks = (PAIR, PAIR)
    elif issue == "singleton":
        singleton = DetectorNetworkConfig(name="singleton", detectors={"C1": detector})
        assert singleton.pairs() == ()
        networks = (singleton,)
    else:
        update = (
            {"geometry": replace(detector.geometry, elevation_m=1234.0)}
            if issue == "geometry"
            else {"psd_reference": TRIANGLE.detectors["R1"].psd_reference}
        )
        conflicting = DetectorNetworkConfig(
            name="conflicting",
            detectors={**PAIR.detectors, "C1": detector.model_copy(update=update)},
        )
        networks = (TRIANGLE, conflicting)

    fields = {**_config().model_dump(exclude={"networks"}), "networks": networks}
    with pytest.raises(ValueError, match=message):
        SimulationConfig.model_validate(fields)
    with pytest.raises(ValueError, match=message):
        project_background(
            event_pool(0), WAVEFORM, networks=list(networks), observation_seconds=1.0
        )
    with pytest.raises(ValueError, match=message):
        simulate_background(
            realization_key(SEED, 0),
            event_pool(0),
            WAVEFORM,
            networks=networks,
            observer_rate_hz=0.0,
            num_segments=1,
            segment_seconds=1.0,
        )


@pytest.fixture
def pool() -> dict[str, np.ndarray]:
    return event_pool()


@pytest.fixture
def simulate_pool() -> Callable[[dict[str, np.ndarray]], BackgroundDraw]:
    def run(sources: dict[str, np.ndarray]) -> BackgroundDraw:
        return simulate_background(
            realization_key(SEED, 0),
            sources,
            WAVEFORM,
            NETWORKS,
            observer_rate_hz=RATE_HZ,
            num_segments=2,
            segment_seconds=SEGMENT_SECONDS,
            incoherent=True,
            chunk_size=CHUNK,
        )

    return run


@pytest.mark.parametrize(
    "column", ["right_ascension", "declination", "polarization_angle"]
)
def test_a_pool_without_a_sky_column_is_refused(
    pool: dict[str, np.ndarray],
    simulate_pool: Callable[[dict[str, np.ndarray]], BackgroundDraw],
    column: str,
) -> None:
    del pool[column]

    with pytest.raises(ValueError, match=column):
        simulate_pool(pool)


def test_a_pool_of_ragged_columns_is_refused(
    pool: dict[str, np.ndarray],
    simulate_pool: Callable[[dict[str, np.ndarray]], BackgroundDraw],
) -> None:
    pool["spin_1z"] = pool["spin_1z"][:-1]

    with pytest.raises(ValueError, match="spin_1z"):
        simulate_pool(pool)


def test_a_pool_with_a_nonfinite_value_is_refused(
    pool: dict[str, np.ndarray],
    simulate_pool: Callable[[dict[str, np.ndarray]], BackgroundDraw],
) -> None:
    pool["luminosity_distance"][3] = np.nan

    with pytest.raises(ValueError, match="luminosity_distance"):
        simulate_pool(pool)


def test_a_pool_smaller_than_the_draw_names_the_expected_count(
    pool: dict[str, np.ndarray],
    simulate_pool: Callable[[dict[str, np.ndarray]], BackgroundDraw],
) -> None:
    small = {name: column[:2] for name, column in pool.items()}

    with pytest.raises(ValueError, match="larger pool"):
        simulate_pool(small)


@pytest.mark.integration
def test_an_event_pool_read_from_a_gwmock_catalogue_renames_the_distance(
    tmp_path: Path,
) -> None:
    pytest.importorskip("gwmock_pop")
    path = write_population_csv(tmp_path / "pool.csv")

    pool = read_event_pool(path)

    assert "distance" not in pool
    np.testing.assert_array_equal(
        pool["luminosity_distance"], event_pool()["luminosity_distance"]
    )


@pytest.mark.integration
def test_an_event_pool_without_a_distance_is_refused(tmp_path: Path) -> None:
    pytest.importorskip("gwmock_pop")
    path = tmp_path / "pool.csv"
    path.write_text("detector_frame_mass_1,detector_frame_mass_2\n1.4,1.3\n")

    with pytest.raises(ValueError, match="distance"):
        read_event_pool(path)
