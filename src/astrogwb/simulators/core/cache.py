"""Cache a simulator function's outputs on disk, addressed by its inputs.

A *node* is a function ``fn(inputs, metadata, **settings) -> outputs``. Its
``inputs`` and ``outputs`` are array trees (:mod:`astrogwb.simulators.core.types`);
``metadata`` is a validated record that says everything else the result depends
on and names itself with ``key()``. Seeds are inputs, not metadata: a seed picks
one realization out of the distribution the metadata describes.

:func:`cached` wraps a node so that, given a ``cache_dir``, the result lives at::

    <cache_dir>/<fn.__name__>-<metadata.key()>-<digest(inputs)>.h5

and the body runs only on a miss. Because the path is a function of the inputs'
*content*, a node whose inputs are another node's outputs composes by content.
``settings`` -- chunk sizes and other knobs that change cost but not the result
-- are passed to the body and kept out of the path.

The cache hits without importing JAX: only generating needs it.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import platform
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from astrogwb import __version__
from astrogwb.simulators.core.keys import Keyed
from astrogwb.simulators.core.tree import (
    _h5py,
    digest,
    leaves,
    read_tree,
    write_tree,
)
from astrogwb.simulators.core.types import Arrays, Tree

__all__ = ["Cached", "cached", "read"]

logger = logging.getLogger(__name__)


class _Node(Protocol):
    """A plain function: ``fn(inputs, metadata, **settings) -> outputs``."""

    __name__: str

    def __call__(self, *args: Any, **kwargs: Any) -> Arrays: ...


def read(path: str | Path) -> tuple[Arrays, Arrays, str]:
    """Read one cached file: ``(inputs, outputs, metadata_json)``.

    The by-path half of the cache, for callers that were handed a file rather
    than a request. The metadata comes back as the JSON the node ran with;
    validating it into a record is the caller's, since this package does not
    know the record types.
    """
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        metadata_json = handle.attrs["metadata"]
        if isinstance(metadata_json, bytes):
            metadata_json = metadata_json.decode()
        return (
            read_tree(handle["inputs"]),
            read_tree(handle["outputs"]),
            str(metadata_json),
        )


def _reject_tracers(inputs: Tree, name: str) -> None:
    """Refuse a traced input: a cache key needs concrete values."""
    if "jax" not in sys.modules:
        return
    from jax.core import Tracer

    for path, leaf in leaves(inputs):
        if isinstance(leaf, Tracer):
            raise TypeError(
                f"{name}: input {path!r} is a JAX tracer. A cached simulator "
                "hashes its inputs, so it cannot run under jit/vmap/grad; call "
                "it eagerly, or call the undecorated body "
                f"({name}.__wrapped__) inside the transformation."
            )


def _save_atomically(
    path: Path,
    *,
    name: str,
    inputs: Tree,
    outputs: Arrays,
    metadata: Keyed,
) -> None:
    """Write the file beside ``path`` and rename it into place.

    A reader -- or a second writer racing on the same path -- sees either no
    file or a complete one. Two racing writers produce equivalent files, so the
    last rename winning is harmless.
    """
    h5py = _h5py()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".h5.tmp"
    )
    os.close(handle)
    try:
        with h5py.File(temporary, "w") as file:
            file.attrs["metadata"] = metadata.model_dump_json()
            file.attrs["node"] = name
            file.attrs["version"] = __version__
            file.attrs["created"] = datetime.now(UTC).isoformat()
            file.attrs["host"] = platform.node()
            write_tree(file.create_group("inputs"), inputs)
            write_tree(file.create_group("outputs"), outputs)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _to_numpy(tree: Any) -> Arrays:
    return {
        name: _to_numpy(value) if isinstance(value, dict) else np.asarray(value)
        for name, value in tree.items()
    }


class Cached:
    """A simulator node with a cache in front of it; build one with :func:`cached`.

    ``__call__`` is ``(inputs, metadata, *, cache_dir=None, generate=True,
    **settings)``. Without a ``cache_dir`` it calls the body directly. With one
    it serves a hit from the file, raises ``FileNotFoundError`` on a miss when
    ``generate`` is false, and otherwise runs the body and saves its outputs.
    The undecorated body stays reachable as ``__wrapped__``.
    """

    def __init__(self, fn: _Node) -> None:
        functools.update_wrapper(self, fn)
        self.__wrapped__ = fn
        self.__name__ = fn.__name__

    def path(self, inputs: Tree, metadata: Keyed, cache_dir: str | Path) -> Path:
        """Where ``(inputs, metadata)``'s result lives in ``cache_dir``."""
        return Path(cache_dir) / f"{self.__name__}-{metadata.key()}-{digest(inputs)}.h5"

    def __call__(
        self,
        inputs: Tree,
        metadata: Keyed,
        *,
        cache_dir: str | Path | None = None,
        generate: bool = True,
        **settings: Any,
    ) -> Arrays:
        name = self.__name__
        if cache_dir is None:
            if not generate:
                raise ValueError("generate=False needs a cache_dir to serve hits from")
            return _to_numpy(self.__wrapped__(inputs, metadata, **settings))

        _reject_tracers(inputs, name)
        target = self.path(inputs, metadata, cache_dir)
        if target.is_file():
            _, outputs, recorded = read(target)
            if json.loads(recorded) != json.loads(metadata.model_dump_json()):
                raise ValueError(
                    f"{target} records other metadata than the request:\n"
                    f"  recorded:  {recorded}\n"
                    f"  requested: {metadata.model_dump_json()}"
                )
            logger.info("%s: cache hit at %s", name, target)
            return outputs

        if not generate:
            raise FileNotFoundError(
                f"{name}: no cached result at {target}, and generation is disabled"
            )
        logger.info("%s: cache miss, generating into %s", name, target)
        outputs = _to_numpy(self.__wrapped__(inputs, metadata, **settings))
        _save_atomically(
            target, name=name, inputs=inputs, outputs=outputs, metadata=metadata
        )
        return outputs


def cached(fn: _Node) -> Cached:
    """Turn ``fn(inputs, metadata, **settings)`` into a cached node."""
    return Cached(fn)
