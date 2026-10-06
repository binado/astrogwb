"""Persist a simulator's output tree as one HDF5 file, and read it back.

The data tree is the file's root group: a mapping becomes a subgroup, a leaf a
dataset. The metadata record travels as JSON in a top-level attribute, with the
seed and batch size the data were drawn at and the package version, host and
time. Where a file lives and what it is called are the caller's -- nothing here
addresses by content.
"""

from __future__ import annotations

import os
import platform
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from astrogwb import __version__
from astrogwb.simulators.core.keys import Keyed
from astrogwb.simulators.core.tree import _h5py, leaves, read_tree, write_tree
from astrogwb.simulators.core.types import Arrays

__all__ = ["load", "write"]


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
    h5py = _h5py()
    for _ in leaves(data):  # validates every key before a file exists
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
    h5py = _h5py()
    with h5py.File(path, "r") as file:
        attrs = {
            name: value.decode() if isinstance(value, bytes) else value
            for name, value in file.attrs.items()
        }
        recorded = str(attrs.pop("metadata"))
        data = read_tree(file)
    metadata = metadata_type.model_validate_json(recorded)  # ty: ignore[unresolved-attribute]
    return data, metadata, attrs
