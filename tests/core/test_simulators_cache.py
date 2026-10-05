"""The cache machinery, on a toy node: no population, waveform or JAX needed."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from astrogwb.simulators.core import (
    Arrays,
    Tree,
    cached,
    content_key,
    digest,
    read,
    split_seed,
)


def _leaf(tree: Arrays, name: str) -> np.ndarray:
    value = tree[name]
    assert not isinstance(value, dict)
    return value


class ToyMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    scale: float

    def key(self) -> str:
        return content_key(self.model_dump(mode="json"))


CALLS: list[int] = []


@cached
def scaled(inputs: Tree, metadata: ToyMetadata, *, offset: float = 0.0) -> Arrays:
    CALLS.append(1)
    x = np.asarray(inputs["x"])
    return {
        "y": metadata.scale * x + offset,
        "nested": {"count": np.asarray(x.size), "label": np.asarray(7, np.uint64)},
    }


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    CALLS.clear()


def test_without_a_cache_dir_the_body_runs_every_time() -> None:
    inputs = {"x": np.arange(3.0)}
    scaled(inputs, ToyMetadata(scale=2.0))
    scaled(inputs, ToyMetadata(scale=2.0))
    assert len(CALLS) == 2


def test_a_hit_returns_what_the_miss_computed(tmp_path: Path) -> None:
    inputs = {"x": np.arange(4.0)}
    metadata = ToyMetadata(scale=3.0)

    first = scaled(inputs, metadata, cache_dir=tmp_path)
    second = scaled(inputs, metadata, cache_dir=tmp_path)

    assert len(CALLS) == 1
    np.testing.assert_array_equal(first["y"], second["y"])
    nested = second["nested"]
    assert isinstance(nested, dict)
    assert _leaf(nested, "label").dtype == np.uint64
    assert _leaf(nested, "count") == 4


def test_the_path_names_the_node_the_metadata_and_the_inputs(tmp_path: Path) -> None:
    inputs = {"x": np.arange(4.0)}
    metadata = ToyMetadata(scale=3.0)

    path = scaled.path(inputs, metadata, tmp_path)

    assert path == tmp_path / f"scaled-{metadata.key()}-{digest(inputs)}.h5"
    scaled(inputs, metadata, cache_dir=tmp_path)
    assert path.is_file()


def test_other_inputs_or_metadata_name_another_file(tmp_path: Path) -> None:
    base = scaled.path({"x": np.arange(4.0)}, ToyMetadata(scale=1.0), tmp_path)

    assert base != scaled.path({"x": np.arange(5.0)}, ToyMetadata(scale=1.0), tmp_path)
    assert base != scaled.path({"x": np.arange(4.0)}, ToyMetadata(scale=2.0), tmp_path)


def test_settings_change_cost_not_the_path(tmp_path: Path) -> None:
    inputs = {"x": np.arange(2.0)}
    metadata = ToyMetadata(scale=1.0)

    scaled(inputs, metadata, cache_dir=tmp_path, offset=0.0)
    scaled(inputs, metadata, cache_dir=tmp_path, offset=99.0)

    assert len(CALLS) == 1


def test_generate_false_names_the_missing_path(tmp_path: Path) -> None:
    inputs = {"x": np.arange(2.0)}
    metadata = ToyMetadata(scale=1.0)

    with pytest.raises(FileNotFoundError, match="scaled-"):
        scaled(inputs, metadata, cache_dir=tmp_path, generate=False)
    assert not CALLS

    scaled(inputs, metadata, cache_dir=tmp_path)
    scaled(inputs, metadata, cache_dir=tmp_path, generate=False)
    assert len(CALLS) == 1


def test_a_file_holds_inputs_outputs_and_provenance(tmp_path: Path) -> None:
    inputs = {"x": np.arange(2.0)}
    metadata = ToyMetadata(scale=2.0)
    scaled(inputs, metadata, cache_dir=tmp_path)

    path = scaled.path(inputs, metadata, tmp_path)
    stored_inputs, outputs, recorded = read(path)

    np.testing.assert_array_equal(stored_inputs["x"], inputs["x"])
    np.testing.assert_array_equal(outputs["y"], [0.0, 2.0])
    assert ToyMetadata.model_validate_json(recorded) == metadata
    with h5py.File(path) as handle:
        assert {"metadata", "node", "version", "created", "host"} <= set(handle.attrs)
        assert handle.attrs["node"] == "scaled"


def test_a_file_filed_under_the_wrong_name_is_refused(tmp_path: Path) -> None:
    inputs = {"x": np.arange(2.0)}
    one, other = ToyMetadata(scale=1.0), ToyMetadata(scale=2.0)
    scaled(inputs, one, cache_dir=tmp_path)
    scaled.path(inputs, one, tmp_path).rename(scaled.path(inputs, other, tmp_path))

    with pytest.raises(ValueError, match="other metadata"):
        scaled(inputs, other, cache_dir=tmp_path)


def test_a_failed_body_leaves_no_file_behind(tmp_path: Path) -> None:
    @cached
    def broken(inputs: Tree, metadata: ToyMetadata) -> Arrays:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        broken({"x": np.zeros(1)}, ToyMetadata(scale=1.0), cache_dir=tmp_path)

    assert not list(tmp_path.iterdir())


def test_a_traced_input_is_refused_with_a_clear_error(tmp_path: Path) -> None:
    jax = pytest.importorskip("jax")

    def under_jit(x: jax.Array) -> jax.Array:
        scaled({"x": x}, ToyMetadata(scale=1.0), cache_dir=tmp_path)
        return x

    with pytest.raises(TypeError, match="tracer"):
        jax.jit(under_jit)(np.arange(2.0))


def test_digest_depends_on_paths_dtypes_shapes_and_values() -> None:
    base = digest({"a": np.arange(4.0)})

    assert base == digest({"a": np.arange(4.0)})
    assert base != digest({"b": np.arange(4.0)})
    assert base != digest({"a": np.arange(4.0, dtype=np.float32)})
    assert base != digest({"a": np.arange(4.0).reshape(2, 2)})
    assert base != digest({"a": np.arange(4.0) + 1})


def test_tree_keys_cannot_collide_with_paths() -> None:
    with pytest.raises(ValueError, match="without '/'"):
        digest({"a/b": np.zeros(1)})


def test_digest_ignores_key_order_and_array_type() -> None:
    assert digest({"a": np.ones(2), "b": np.zeros(2)}) == digest(
        {"b": np.zeros(2), "a": [1.0, 1.0]}
    )


def test_split_seed_children_are_distinct_uint64_and_prefix_stable() -> None:
    seeds = split_seed(41, 50)

    assert seeds.dtype == np.uint64
    assert np.unique(seeds).size == 50
    np.testing.assert_array_equal(seeds[:10], split_seed(41, 10))
    assert not np.array_equal(split_seed(42, 10), seeds[:10])
