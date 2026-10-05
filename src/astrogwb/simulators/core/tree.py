"""Nested array trees: content digests and their HDF5 layout.

A tree is a nested mapping whose leaves are arrays. It is hashed for the cache
path and written as an HDF5 group: a mapping becomes a subgroup, a leaf a
dataset. h5py is imported when a file is touched, not when the package is.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from astrogwb.simulators.core.keys import CATALOG_KEY_LENGTH
from astrogwb.simulators.core.types import Arrays, Tree

__all__ = ["digest", "leaves", "read_tree", "write_tree"]


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "astrogwb cached simulators need h5py. Install it with the 'io' "
            "extra: pip install 'astrogwb[io]'"
        ) from error
    return h5py


def leaves(tree: Tree, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Every ``(path, leaf)`` of ``tree``, depth first in sorted key order."""
    for name in sorted(tree):
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(
                f"tree keys must be non-empty strings without '/': {name!r}"
            )
        value = tree[name]
        path = f"{prefix}{name}"
        if isinstance(value, Mapping):
            yield from leaves(value, f"{path}/")
        else:
            yield path, value


def digest(tree: Tree) -> str:
    """A content hash of ``tree``: its paths, dtypes, shapes and bytes.

    Two trees hash alike exactly when they hold equal arrays at equal paths, so
    it names a node's inputs in the cache path.
    """
    hasher = hashlib.sha256()
    for path, value in leaves(tree):
        array = np.ascontiguousarray(np.asarray(value))
        header = f"{path}\0{array.dtype.str}\0{array.shape}\0"
        hasher.update(header.encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()[:CATALOG_KEY_LENGTH]


def write_tree(group: Any, tree: Tree) -> None:
    """Write ``tree`` into the open HDF5 ``group``."""
    for name in sorted(tree):
        value = tree[name]
        if isinstance(value, Mapping):
            write_tree(group.create_group(name), value)
        else:
            group.create_dataset(name, data=np.asarray(value))


def read_tree(group: Any) -> Arrays:
    """Read the HDF5 ``group`` back as nested dicts of NumPy arrays."""
    h5py = _h5py()
    out: Arrays = {}
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            out[name] = read_tree(item)
        else:
            out[name] = np.asarray(item[()])
    return out
