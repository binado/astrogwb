"""Batched keys and the HDF5 write/load pair."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import jax
import numpy as np
import pytest
from pydantic import BaseModel

from astrogwb.simulators import batch_keys, realization_key
from astrogwb.simulators._io import (
    MANIFEST_FILE,
    atomic_h5,
    read_realization_manifest,
    write_realization_manifest,
)
from astrogwb.utils.io import content_key, load, write


class Record(BaseModel):
    """A stand-in metadata record: anything with ``key()`` and JSON."""

    name: str
    size: int

    def key(self) -> str:
        return content_key(self.model_dump(mode="json"))


TREE: dict[str, Any] = {
    "frequencies": np.linspace(10.0, 20.0, 4),
    "counts": np.array([3, 5], dtype=np.int64),
    "nested": {"values": np.arange(6.0).reshape(2, 3)},
}


def test_batch_keys_are_distinct_and_prefix_stable() -> None:
    keys = np.asarray(jax.random.key_data(batch_keys(41, 4)))
    assert len({tuple(row) for row in keys.tolist()}) == 4
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(batch_keys(41, 2))), keys[:2]
    )


def test_realization_key_is_the_batch_key_at_its_index() -> None:
    keys = np.asarray(jax.random.key_data(batch_keys(41, 4)))
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(realization_key(41, 3))), keys[3]
    )


def test_batch_keys_reject_a_negative_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        batch_keys(41, -1)


def test_write_then_load_round_trips_data_metadata_and_attrs(tmp_path: Path) -> None:
    record = Record(name="a", size=2)
    path = write(tmp_path / "out" / "file.h5", TREE, record, seed=41, batch_size=8)

    data, metadata, attrs = load(path, Record)

    assert metadata == record
    assert attrs["seed"] == 41 and attrs["batch_size"] == 8
    assert {"version", "created", "host"} <= attrs.keys()
    np.testing.assert_array_equal(data["frequencies"], TREE["frequencies"])
    np.testing.assert_array_equal(data["counts"], TREE["counts"])
    nested = data["nested"]
    assert isinstance(nested, dict)
    np.testing.assert_array_equal(nested["values"], TREE["nested"]["values"])


def test_seed_and_batch_size_are_optional(tmp_path: Path) -> None:
    path = write(tmp_path / "file.h5", TREE, Record(name="a", size=2))
    _, _, attrs = load(path, Record)
    assert "seed" not in attrs and "batch_size" not in attrs


def test_a_failed_write_leaves_no_file_behind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="without '/'"):
        write(tmp_path / "file.h5", {"a/b": np.zeros(1)}, Record(name="a", size=2))
    assert list(tmp_path.iterdir()) == []


def test_a_realization_manifest_round_trips_without_nones(tmp_path: Path) -> None:
    manifest = {
        "realization": {"id": "abc", "seed": 41, "index": None},
        "population": {"key": "abc", "metadata": {"hyperparameters": {"H0": 67.0}}},
    }

    write_realization_manifest(tmp_path, manifest)
    read = read_realization_manifest(tmp_path)

    assert read["layout"]["version"] == 2
    assert read["realization"] == {"id": "abc", "seed": 41}
    assert read["population"] == manifest["population"]
    assert [path.name for path in tmp_path.iterdir()] == [MANIFEST_FILE]


@pytest.mark.parametrize(
    "document", ["[layout]\nversion = 99\n", '[artifact]\nkind = "background"\n']
)
def test_a_manifest_of_another_layout_is_refused(tmp_path: Path, document: str) -> None:
    (tmp_path / MANIFEST_FILE).write_text(document)

    with pytest.raises(ValueError, match="layout version"):
        read_realization_manifest(tmp_path)


def test_a_failed_atomic_h5_leaves_the_previous_file(tmp_path: Path) -> None:
    target = tmp_path / "product.h5"
    with atomic_h5(target) as file:
        file.create_dataset("values", data=np.arange(3))

    with pytest.raises(RuntimeError, match="interrupted"), atomic_h5(target) as file:
        file.create_dataset("values", data=np.zeros(5))
        raise RuntimeError("interrupted")

    assert [path.name for path in tmp_path.iterdir()] == ["product.h5"]
    with h5py.File(target, "r") as file:
        np.testing.assert_array_equal(file["values"][()], np.arange(3))
