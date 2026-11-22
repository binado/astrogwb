"""The HDF5 write/load pair."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from pydantic import BaseModel

from astrogwb.background.importance.catalog import ReferenceCatalog
from astrogwb.utils.io import atomic_h5, load, write


class Record(BaseModel):
    """A stand-in metadata record: anything with JSON."""

    name: str
    size: int


TREE: dict[str, Any] = {
    "frequencies": np.linspace(10.0, 20.0, 4),
    "counts": np.array([3, 5], dtype=np.int64),
    "nested": {"values": np.arange(6.0).reshape(2, 3)},
}


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


def _catalog() -> ReferenceCatalog:
    return ReferenceCatalog(
        frequencies=np.linspace(10.0, 20.0, 4),
        intensity=np.arange(20.0).reshape(4, 5),
        intrinsic={"mass_1": np.arange(5.0), "mass_2": np.arange(5.0) / 2},
        proposal_log_prob=np.linspace(-1.0, -2.0, 5),
        reference_redshift=0.5,
        reference_distance=2800.0,
    )


def test_reference_catalog_round_trips_with_its_metadata(tmp_path: Path) -> None:
    catalog = _catalog()
    record = Record(name="ref", size=5)
    path = catalog.write(tmp_path / "c" / "ref.h5", record, seed=3)

    reopened, metadata = ReferenceCatalog.read(path, Record)

    assert metadata == record
    np.testing.assert_array_equal(reopened.frequencies, catalog.frequencies)
    np.testing.assert_array_equal(reopened.intensity, catalog.intensity)
    assert reopened.intrinsic.keys() == catalog.intrinsic.keys()
    for name, column in catalog.intrinsic.items():
        np.testing.assert_array_equal(reopened.intrinsic[name], column)
    np.testing.assert_array_equal(reopened.proposal_log_prob, catalog.proposal_log_prob)
    assert reopened.reference_redshift == 0.5
    assert reopened.reference_distance == 2800.0


def test_take_keeps_the_first_draws_of_every_per_draw_array() -> None:
    catalog = _catalog()

    head = catalog.take(3)

    np.testing.assert_array_equal(head.intensity, np.arange(20.0).reshape(4, 5)[:, :3])
    np.testing.assert_array_equal(head.intrinsic["mass_2"], np.arange(3.0) / 2)
    np.testing.assert_array_equal(
        head.proposal_log_prob, np.linspace(-1.0, -2.0, 5)[:3]
    )
    np.testing.assert_array_equal(head.frequencies, catalog.frequencies)
    assert head.reference_distance == catalog.reference_distance


@pytest.mark.parametrize("n", [-1, 6])
def test_take_rejects_a_count_outside_the_catalog(n: int) -> None:
    with pytest.raises(ValueError, match="cannot take"):
        _catalog().take(n)
