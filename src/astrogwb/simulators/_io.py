"""Atomic HDF5 files and batched realization keys."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from astrogwb.utils.io import h5py_module

__all__ = ["atomic_h5", "batch_keys", "realization_key"]


def _require_x64() -> None:
    import jax

    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "realization keys need jax_enable_x64; JAX would truncate a 64-bit seed"
        )


def batch_keys(seed: int | np.integer, n: int) -> Any:
    """``n`` keys of ``seed``: ``fold_in(key(seed), arange(n))``.

    Key ``i`` depends on ``(seed, i)`` alone, so the batch is prefix-stable.
    Seeds are 64-bit, so this needs ``jax_enable_x64``; JAX would otherwise
    truncate the seed and two seeds could share a realization.
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
