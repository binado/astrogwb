"""HDF5 array trees for persisted artifacts.

A *tree* is a nested mapping whose leaves are arrays, written as an HDF5 group:
a mapping becomes a subgroup, a leaf a dataset. :func:`write` stores one tree
with the metadata record that produced it as a JSON attribute; :func:`load`
reads both back. Where a file lives and what it is called are the caller's --
nothing here addresses by content.

Nothing here imports JAX.
"""

from __future__ import annotations

import os
import platform
import tempfile
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel

try:
    import h5py
except ModuleNotFoundError as error:
    if error.name != "h5py":
        raise
    raise ImportError(
        "astrogwb artifact files need h5py. Install it with the 'io' "
        "extra: pip install 'astrogwb[io]'"
    ) from error

from astrogwb import __version__

__all__ = [
    "Arrays",
    "atomic_h5",
    "load",
    "read_tree",
    "write",
    "write_tree",
]

#: What a node is handed: nested mappings with array-like leaves.
type Tree = Mapping[str, ArrayLike | Tree]

#: What a reader returns: nested dicts of NumPy arrays.
type Arrays = dict[str, NDArray[Any] | Arrays]


@contextmanager
def atomic_h5(path: str | Path) -> Generator[Any]:
    """An ``h5py.File`` open for writing beside ``path``, renamed onto it on success.

    A reader sees no file at ``path`` or a complete one: if the body raises,
    the temporary file is removed and ``path`` is left as it was.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.stem}.", suffix=".h5.tmp"
    )
    os.close(handle)
    try:
        with h5py.File(temporary, "w") as file:
            yield file
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


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
    metadata: BaseModel,
    *,
    seed: int | None = None,
    batch_size: int | None = None,
) -> Path:
    """Write ``data`` and the ``metadata`` that produced it to ``path``.

    The file is written beside ``path`` and renamed into place, so a reader sees
    no file or a complete one. ``seed`` and ``batch_size`` are recorded when
    given. Returns ``path``.
    """
    for _ in _leaves(data):  # validates every key before a file exists
        pass
    target = Path(path)
    with atomic_h5(target) as file:
        file.attrs["metadata"] = metadata.model_dump_json()
        file.attrs["version"] = __version__
        file.attrs["created"] = datetime.now(UTC).isoformat()
        file.attrs["host"] = platform.node()
        if seed is not None:
            file.attrs["seed"] = np.uint64(seed)
        if batch_size is not None:
            file.attrs["batch_size"] = np.int64(batch_size)
        write_tree(file, data)
    return target


def load[M: BaseModel](
    path: str | Path, metadata_type: type[M]
) -> tuple[Arrays, M, Mapping[str, Any]]:
    """Read a file :func:`write` made: ``(data, metadata, attrs)``.

    ``metadata_type`` is a Pydantic model that validates the recorded JSON.
    ``attrs`` holds the remaining top-level attributes -- ``version``,
    ``created``, ``host``, and ``seed`` / ``batch_size`` when they were
    recorded -- as Python or NumPy scalars.
    """
    with h5py.File(path, "r") as file:
        attrs = {
            name: value.decode() if isinstance(value, bytes) else value
            for name, value in file.attrs.items()
        }
        recorded = str(attrs.pop("metadata"))
        data = read_tree(file)
    metadata = metadata_type.model_validate_json(recorded)
    return data, metadata, attrs
