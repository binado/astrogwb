"""One-pass simulation configuration, storage and input validation."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType

import h5py
import jax
import numpy as np
import pytest
from astrogwb_tiny_pipeline import (
    PAIR,
    TRIANGLE,
    WAVEFORM,
    event_pool,
    write_population_csv,
)

from astrogwb.background.simulation import (
    BackgroundDraw,
    SimulationConfig,
    project_background,
    simulate_background,
)
from astrogwb.detector import Network, network_pairs
from astrogwb.populations.io import read_population_file
from astrogwb.waveform import waveform_generator

SEED = 41
CHUNK = 8
SEGMENT_SECONDS = 1.0
RATE_HZ = 8.0


def _config() -> SimulationConfig:
    return SimulationConfig(
        waveform=WAVEFORM,
        network=TRIANGLE,
        poisson_rate=RATE_HZ,
        num_segments=2,
        segment_duration=SEGMENT_SECONDS,
        population="pool.csv",
        seed=SEED,
    )


def _simulate(config: SimulationConfig) -> BackgroundDraw:
    return simulate_background(
        jax.random.key(config.seed),
        event_pool(),
        waveform_generator(config.waveform),
        config.network,
        poisson_rate=config.poisson_rate,
        num_segments=config.num_segments,
        segment_duration=config.segment_duration,
        simulate_detector_noise=config.simulate_detector_noise,
        chunk_size=CHUNK,
    )


@pytest.fixture(scope="module")
def draw() -> BackgroundDraw:
    """The tiny configuration simulated at ``SEED``."""
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
    np.testing.assert_array_equal(
        reopened.pair_spectral_density, draw.pair_spectral_density
    )


def test_file_has_single_network_layout(written: Path) -> None:
    with h5py.File(written, "r") as file:
        assert set(file) == {"frequencies", "segment_counts", "pair_spectral_density"}
        assert "version" in file.attrs
        assert file.attrs["format_version"] == 2
        metadata = json.loads(file.attrs["metadata"])

    assert metadata["network"] == {
        name: detector.model_dump(mode="json") for name, detector in TRIANGLE.items()
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


def test_a_file_is_read_whatever_astrogwb_version_wrote_it(
    written: Path, tmp_path: Path
) -> None:
    BackgroundDraw.read(_edited(written, tmp_path, version="0.8.1.dev3"))


@pytest.mark.parametrize("format_version", [None, 1, 99])
def test_a_file_of_an_unsupported_format_is_refused(
    written: Path, tmp_path: Path, format_version: int | None
) -> None:
    stale = _edited(written, tmp_path, format_version=format_version)

    with pytest.raises(ValueError, match=f"format_version {format_version}"):
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


def test_config_copies_and_sorts_the_network_mapping() -> None:
    fields = _config().model_dump(exclude={"network"})
    members = dict(TRIANGLE)
    config = SimulationConfig.model_validate(
        {**fields, "network": MappingProxyType(members)}
    )

    assert tuple(config.network) == ("C1", "R1", "S1")
    assert config.network["C1"] is members["C1"]
    members.clear()
    assert config.network == TRIANGLE


def test_draw_rows_follow_canonical_detector_order() -> None:
    config = _config()
    draw = _simulate(config)
    pair_draw = _simulate(config.model_copy(update={"network": PAIR}))

    (shared,) = network_pairs(PAIR)
    pair = pair_draw.pair_spectral_density[0]
    assert np.any(pair)
    np.testing.assert_array_equal(
        pair, draw.pair_spectral_density[network_pairs(TRIANGLE).index(shared)]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("num_segments", 0), ("segment_duration", 0.0), ("poisson_rate", -1.0)],
)
def test_config_rejects_a_nonphysical_observation(field: str, value: float) -> None:
    fields = {**_config().model_dump(), field: value}

    with pytest.raises(ValueError, match=field):
        SimulationConfig.model_validate(fields)


@pytest.mark.parametrize(
    ("network", "message"),
    [
        ({}, "at least two detectors"),
        ({"C1": PAIR["C1"]}, "at least two detectors"),
        ({"wrong": PAIR["C1"], "S1": PAIR["S1"]}, "detector.*is keyed by"),
    ],
)
def test_background_boundaries_reject_an_invalid_network(
    network: Network, message: str
) -> None:
    fields = {**_config().model_dump(), "network": network}
    generator = waveform_generator(WAVEFORM)
    with pytest.raises(ValueError, match=message):
        SimulationConfig.model_validate(fields)
    with pytest.raises(ValueError, match=message):
        project_background(event_pool(0), generator, network, observation_seconds=1.0)
    with pytest.raises(ValueError, match=message):
        simulate_background(
            jax.random.key(SEED),
            event_pool(0),
            generator,
            network,
            poisson_rate=0.0,
            num_segments=1,
            segment_duration=1.0,
        )


@pytest.fixture
def pool() -> dict[str, np.ndarray]:
    return event_pool()


@pytest.fixture
def simulate_pool() -> Callable[[dict[str, np.ndarray]], BackgroundDraw]:
    generator = waveform_generator(WAVEFORM)

    def run(sources: dict[str, np.ndarray]) -> BackgroundDraw:
        return simulate_background(
            jax.random.key(SEED),
            sources,
            generator,
            TRIANGLE,
            poisson_rate=RATE_HZ,
            num_segments=2,
            segment_duration=SEGMENT_SECONDS,
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


def test_an_event_pool_read_from_a_gwmock_catalogue_renames_the_distance(
    tmp_path: Path,
) -> None:
    path = write_population_csv(tmp_path / "pool.csv")

    pool = read_population_file(path)

    assert "distance" not in pool
    np.testing.assert_array_equal(
        pool["luminosity_distance"], event_pool()["luminosity_distance"]
    )


def test_an_event_pool_without_a_distance_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "pool.csv"
    path.write_text("detector_frame_mass_1,detector_frame_mass_2\n1.4,1.3\n")

    with pytest.raises(ValueError, match="distance"):
        read_population_file(path)


def test_an_event_pool_read_from_an_hdf5_catalogue_renames_the_distance(
    tmp_path: Path,
) -> None:
    pool = event_pool()
    pool["distance"] = pool.pop("luminosity_distance")
    names = list(pool)
    structured = np.empty(
        next(iter(pool.values())).shape, dtype=[(name, float) for name in names]
    )
    for name in names:
        structured[name] = pool[name]
    path = tmp_path / "pool.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("source_parameters", data=structured)

    read = read_population_file(path)

    assert "distance" not in read
    np.testing.assert_array_equal(
        read["luminosity_distance"], event_pool()["luminosity_distance"]
    )


def test_an_event_pool_read_from_an_hdf5_group_of_datasets(tmp_path: Path) -> None:
    pool = event_pool()
    pool["distance"] = pool.pop("luminosity_distance")
    path = tmp_path / "pool.h5"
    with h5py.File(path, "w") as file:
        group = file.create_group("source_parameters")
        for name, values in pool.items():
            group.create_dataset(name, data=np.asarray(values))

    read = read_population_file(path)

    assert "distance" not in read
    np.testing.assert_array_equal(
        read["luminosity_distance"], event_pool()["luminosity_distance"]
    )


def test_an_event_pool_read_from_a_custom_hdf5_root(tmp_path: Path) -> None:
    pool = event_pool()
    pool["distance"] = pool.pop("luminosity_distance")
    path = tmp_path / "pool.h5"
    with h5py.File(path, "w") as file:
        group = file.create_group("catalogue/events")
        for name, values in pool.items():
            group.create_dataset(name, data=np.asarray(values))

    read = read_population_file(path, root="catalogue/events")

    np.testing.assert_array_equal(read["luminosity_distance"], pool["distance"])


def test_an_event_pool_without_a_catalogue_dataset_is_refused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pool.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("data", data=np.arange(3))

    with pytest.raises(ValueError, match="source_parameters"):
        read_population_file(path)
