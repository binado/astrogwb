"""The simulator protocol's plumbing: batched keys and the HDF5 write/load pair."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from pydantic import BaseModel

from astrogwb.simulators.core import batch_keys, content_key, load, write


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
