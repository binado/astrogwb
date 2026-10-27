"""The realization manifest, atomic product files and batched keys.

A realization is one directory holding ``metadata.toml``, its manifest, and at
most one product file per stage: ``source_parameters.h5``,
``polarization.h5`` and ``background.h5``. The manifest has a table per
configured stage; a product file is written beside its final name and renamed
into place, so an existing product is a complete one. Its tables:

- ``[layout]`` -- the directory ``version`` and the ``astrogwb_version`` that
  last wrote it.
- ``[realization]`` -- its ``id`` (the population key), ``seed``, ``index``,
  ``rng_key`` and the realized count, rate, observation time and
  hyperparameters.
- ``[population]``, ``[waveforms]``, ``[background]`` -- each stage's
  scientific ``metadata`` and ``key``, with the waveform conventions and the
  pair-spectrum normalization.
- ``[execution]`` -- cost-only settings such as chunk sizes, recorded for
  provenance and never hashed.

TOML has no null, so ``None`` values are left out on write and come back as
the field defaults.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from astrogwb import __version__
from astrogwb.utils.io import h5py_module

__all__ = [
    "BACKGROUND_FILE",
    "LAYOUT_VERSION",
    "MANIFEST_FILE",
    "POLARIZATION_FILE",
    "SOURCES_FILE",
    "atomic_h5",
    "batch_keys",
    "read_realization_manifest",
    "realization_key",
    "write_realization_manifest",
]

MANIFEST_FILE = "metadata.toml"
SOURCES_FILE = "source_parameters.h5"
POLARIZATION_FILE = "polarization.h5"
BACKGROUND_FILE = "background.h5"

#: The version of the realization directory layout and its manifest.
LAYOUT_VERSION = 1

#: The order manifest tables are written in; any others follow.
_TABLE_ORDER = ("realization", "population", "waveforms", "background", "execution")


def _require_x64() -> None:
    import jax

    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "realization keys need jax_enable_x64; JAX would truncate a 64-bit seed"
        )


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
    _require_x64()
    base = jax.random.key(np.uint64(seed))
    return jax.vmap(lambda i: jax.random.fold_in(base, i))(
        jax.numpy.arange(n, dtype=jax.numpy.uint32)
    )


def realization_key(seed: int | np.integer, index: int) -> Any:
    """Key ``index`` of :func:`batch_keys` of ``seed``, without the batch."""
    import jax

    if index < 0:
        raise ValueError("index must be non-negative")
    _require_x64()
    return jax.random.fold_in(jax.random.key(np.uint64(seed)), np.uint32(index))


def _without_none(value: Any) -> Any:
    """``value`` with every ``None`` dropped from its mappings, as TOML needs."""
    if isinstance(value, Mapping):
        return {k: _without_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list | tuple):
        return [_without_none(v) for v in value]
    return value


def write_realization_manifest(
    directory: str | Path, manifest: Mapping[str, Any]
) -> Path:
    """Write ``directory/metadata.toml`` atomically and return its path.

    ``manifest`` holds the tables after ``[layout]``, which this writes; a
    ``[layout]`` table in ``manifest`` is replaced.
    """
    import tomli_w

    names = [name for name in _TABLE_ORDER if name in manifest]
    names += [name for name in manifest if name not in {*names, "layout"}]
    tables = {name: manifest[name] for name in names}
    document = {
        "layout": {"version": LAYOUT_VERSION, "astrogwb_version": __version__},
        **_without_none(tables),
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MANIFEST_FILE
    handle, temporary = tempfile.mkstemp(
        dir=directory, prefix=".metadata.", suffix=".toml.tmp"
    )
    try:
        with os.fdopen(handle, "wb") as file:
            tomli_w.dump(document, file)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def read_realization_manifest(directory: str | Path) -> dict[str, Any]:
    """Read the manifest of the realization at ``directory``.

    Raises
    ------
    FileNotFoundError
        If ``directory`` holds no manifest.
    ValueError
        If the manifest is of another layout version, or has none.
    """
    path = Path(directory) / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no realization manifest at {path}")
    with path.open("rb") as file:
        manifest = tomllib.load(file)
    version = manifest.get("layout", {}).get("version")
    if version != LAYOUT_VERSION:
        raise ValueError(
            f"{path} has realization layout version {version}, but this astrogwb "
            f"reads version {LAYOUT_VERSION}"
            + (
                "; it is a per-stage artifact of an earlier astrogwb, see "
                "scripts/migrate_ensemble_cache.py"
                if "artifact" in manifest
                else ""
            )
        )
    return manifest


@contextmanager
def atomic_h5(path: str | Path) -> Generator[Any]:
    """An ``h5py.File`` open for writing beside ``path``, renamed onto it on success.

    A reader sees no file at ``path`` or a complete one: if the body raises,
    the temporary file is removed and ``path`` is left as it was.
    """
    h5py = h5py_module()
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
