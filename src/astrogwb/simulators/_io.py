"""Stage manifests, parent references and batched keys.

Every stage artifact is a directory holding ``metadata.toml``, its manifest,
and ``data.h5``, its arrays. A manifest is an input, not a sidecar: the next
stage reads it to find its parent and to check it is the parent it was made
from. It has four tables:

- ``[artifact]`` -- ``kind``, ``schema_version``, the ``astrogwb_version`` that
  wrote it, the scientific ``key`` and the ``data`` file name.
- ``[metadata]`` -- the stage's scientific settings, the record ``key`` hashes.
- ``[parent]`` -- for a downstream stage: the parent's ``key`` and, when it was
  read from or written to disk, its ``manifest`` path relative to this file.
- ``[execution]`` -- cost-only settings such as ``chunk_size``, recorded for
  provenance and never hashed.

TOML has no null, so ``None`` values are left out on write and come back as
the field defaults.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from astrogwb import __version__

__all__ = [
    "DATA_FILE",
    "MANIFEST_FILE",
    "batch_keys",
    "read_manifest",
    "relative_reference",
    "resolve_reference",
    "write_manifest",
]

MANIFEST_FILE = "metadata.toml"
DATA_FILE = "data.h5"


def batch_keys(seed: int | np.integer, n: int) -> Any:
    """``n`` keys of ``seed``: ``fold_in(key(seed), arange(n))``.

    Key ``i`` depends on ``(seed, i)`` alone, so asking for more keys extends
    the batch without changing the ones already there (prefix-stable): draw
    ``i`` of an ensemble is the same whatever the ensemble size.

    Seeds are 64-bit, so this needs ``jax_enable_x64``: without it JAX would
    silently truncate the seed to 32 bits, and two different seeds could share a
    realization.
    """
    import jax

    if n < 0:
        raise ValueError("n must be non-negative")
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "batch_keys needs jax_enable_x64; JAX would truncate a 64-bit seed"
        )
    base = jax.random.key(np.uint64(seed))
    return jax.vmap(lambda i: jax.random.fold_in(base, i))(
        jax.numpy.arange(n, dtype=jax.numpy.uint32)
    )


def _without_none(value: Any) -> Any:
    """``value`` with every ``None`` dropped from its mappings, as TOML needs."""
    if isinstance(value, Mapping):
        return {k: _without_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list | tuple):
        return [_without_none(v) for v in value]
    return value


def write_manifest(
    directory: Path,
    *,
    kind: str,
    schema_version: int,
    key: str,
    tables: Mapping[str, Any],
) -> Path:
    """Write ``directory/metadata.toml`` atomically and return its path.

    ``tables`` are the manifest's tables after ``[artifact]``, which this
    writes from the other arguments.
    """
    import tomli_w

    manifest = {
        "artifact": {
            "kind": kind,
            "schema_version": schema_version,
            "astrogwb_version": __version__,
            "key": key,
            "data": DATA_FILE,
        },
        **_without_none(dict(tables)),
    }
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MANIFEST_FILE
    handle, temporary = tempfile.mkstemp(
        dir=directory, prefix=".metadata.", suffix=".toml.tmp"
    )
    try:
        with os.fdopen(handle, "wb") as file:
            tomli_w.dump(manifest, file)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def read_manifest(
    path: str | Path, *, kind: str, schema_version: int
) -> dict[str, Any]:
    """Read a manifest, checking it is a ``kind`` artifact of ``schema_version``.

    ``path`` is the manifest itself or the artifact directory holding it.

    Raises
    ------
    FileNotFoundError
        If there is no manifest at ``path``.
    ValueError
        If the manifest is of another kind or schema version.
    """
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no {kind} manifest at {path}")
    with path.open("rb") as file:
        manifest = tomllib.load(file)
    artifact = manifest.get("artifact", {})
    if artifact.get("kind") != kind:
        raise ValueError(f"{path} is a {artifact.get('kind')!r} manifest, not {kind!r}")
    if artifact.get("schema_version") != schema_version:
        raise ValueError(
            f"{path} has {kind} schema version {artifact.get('schema_version')}, "
            f"but this astrogwb reads version {schema_version}"
        )
    manifest["artifact"]["path"] = path.resolve()
    return manifest


def relative_reference(target: Path, start: Path) -> str:
    """``target`` relative to the directory ``start``, as a manifest records it."""
    return Path(os.path.relpath(target.resolve(), start.resolve())).as_posix()


def resolve_reference(manifest: Mapping[str, Any], reference: str) -> Path:
    """A path a manifest records, resolved against the manifest's own directory."""
    return (Path(manifest["artifact"]["path"]).parent / reference).resolve()
