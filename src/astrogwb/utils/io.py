"""Content keys and HDF5 array trees for persisted artifacts.

A *tree* is a nested mapping whose leaves are arrays, written as an HDF5 group:
a mapping becomes a subgroup, a leaf a dataset. :func:`write` stores one tree
with the metadata record that produced it as a JSON attribute; :func:`load`
reads both back. :func:`content_key` is the one way a record's canonical JSON
becomes the hash it is cached under. Where a file lives and what it is called
are the caller's -- nothing here addresses by content.

h5py is imported when a file is touched, not when this module is, and nothing
here imports JAX.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb import __version__

__all__ = [
    "Arrays",
    "Keyed",
    "content_key",
    "h5py_module",
    "load",
    "read_tree",
    "write",
    "write_tree",
]

#: Hex characters of the SHA-256 digest kept as a key. 64 bits is far beyond
#: collision range for a cache of tens of files, and short enough to read in a
#: path.
_KEY_LENGTH = 16

#: What a node is handed: nested mappings with array-like leaves.
type Tree = Mapping[str, ArrayLike | Tree]

#: What a reader returns: nested dicts of NumPy arrays.
type Arrays = dict[str, NDArray[Any] | Arrays]


def content_key(payload: dict[str, Any]) -> str:
    """The content address of one artifact's canonical JSON record.

    ``payload`` is a record's ``model_dump(mode="json")``. Sorted keys and fixed
    separators make the JSON canonical, so every artifact type hashes the same
    way and none can drift.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:_KEY_LENGTH]


class Keyed(Protocol):
    """A metadata record: everything that determines an artifact, and its hash."""

    def key(self) -> str: ...

    def model_dump_json(self) -> str: ...


def h5py_module() -> Any:
    """``h5py``, or an error naming the extra that installs it."""
    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "astrogwb artifact files need h5py. Install it with the 'io' "
            "extra: pip install 'astrogwb[io]'"
        ) from error
    return h5py


def _leaves(tree: Mapping[str, object], prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Every ``(path, leaf)`` of ``tree``, depth first in sorted key order."""
    for name in sorted(tree):
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(
                f"tree keys must be non-empty strings without '/': {name!r}"
            )
        value = tree[name]
        path = f"{prefix}{name}"
        if isinstance(value, Mapping):
            yield from _leaves(value, f"{path}/")
        else:
            yield path, value


def write_tree(group: Any, tree: Mapping[str, object]) -> None:
    """Write ``tree`` into the open HDF5 ``group``."""
    for _ in _leaves(tree):  # validates every key before anything is written
        pass
    for name in sorted(tree):
        value = tree[name]
        if isinstance(value, Mapping):
            write_tree(group.create_group(name), value)
        else:
            group.create_dataset(name, data=np.asarray(value))


def read_tree(group: Any) -> Arrays:
    """Read the HDF5 ``group`` back as nested dicts of NumPy arrays."""
    h5py = h5py_module()
    out: Arrays = {}
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            out[name] = read_tree(item)
        else:
            out[name] = np.asarray(item[()])
    return out


def write(
    path: str | Path,
    data: Mapping[str, object],
    metadata: Keyed,
    *,
    seed: int | None = None,
    batch_size: int | None = None,
) -> Path:
    """Write ``data`` and the ``metadata`` that produced it to ``path``.

    The file is written beside ``path`` and renamed into place, so a reader sees
    no file or a complete one. ``seed`` and ``batch_size`` are recorded when
    given. Returns ``path``.
    """
    h5py = h5py_module()
    for _ in _leaves(data):  # validates every key before a file exists
        pass
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.stem}.", suffix=".h5.tmp"
    )
    os.close(handle)
    try:
        with h5py.File(temporary, "w") as file:
            file.attrs["metadata"] = metadata.model_dump_json()
            file.attrs["version"] = __version__
            file.attrs["created"] = datetime.now(UTC).isoformat()
            file.attrs["host"] = platform.node()
            if seed is not None:
                file.attrs["seed"] = np.uint64(seed)
            if batch_size is not None:
                file.attrs["batch_size"] = np.int64(batch_size)
            write_tree(file, data)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def load[M: Keyed](
    path: str | Path, metadata_type: type[M]
) -> tuple[Arrays, M, Mapping[str, Any]]:
    """Read a file :func:`write` made: ``(data, metadata, attrs)``.

    ``metadata_type`` is a Pydantic model that validates the recorded JSON.
    ``attrs`` holds the remaining top-level attributes -- ``version``,
    ``created``, ``host``, and ``seed`` / ``batch_size`` when they were
    recorded -- as Python or NumPy scalars.
    """
    h5py = h5py_module()
    with h5py.File(path, "r") as file:
        attrs = {
            name: value.decode() if isinstance(value, bytes) else value
            for name, value in file.attrs.items()
        }
        recorded = str(attrs.pop("metadata"))
        data = read_tree(file)
    metadata = metadata_type.model_validate_json(recorded)  # ty: ignore[unresolved-attribute]
    return data, metadata, attrs
