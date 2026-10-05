"""Derive many seeds from one, without JAX."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["split_seed", "validate_seeds"]


def split_seed(seed: int, n: int) -> NDArray[np.uint64]:
    """``n`` child seeds of ``seed``, as a ``uint64`` array.

    Child ``i`` is ``SeedSequence(seed, spawn_key=(i,))``'s first 64-bit word,
    so it depends on ``(seed, i)`` alone: asking for more children extends the
    array without changing the ones already there (prefix-stable), and the
    result is the same on every platform.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    return np.array(
        [
            np.random.SeedSequence(seed, spawn_key=(i,)).generate_state(1, np.uint64)[0]
            for i in range(n)
        ],
        dtype=np.uint64,
    )


def validate_seeds(inputs: Mapping[str, Any]) -> NDArray[np.uint64]:
    """``inputs["seeds"]`` as a non-empty, duplicate-free 1-d ``uint64`` array.

    A repeated seed is a duplicate draw, so it raises rather than being kept.
    """
    seeds = np.asarray(inputs["seeds"])
    if seeds.ndim != 1 or seeds.size == 0 or seeds.dtype != np.uint64:
        raise TypeError(
            "inputs['seeds'] must be a non-empty 1-d uint64 array, e.g. "
            f"split_seed(41, 8); got dtype={seeds.dtype} shape={seeds.shape}"
        )
    if np.unique(seeds).size != seeds.size:
        raise ValueError(
            "inputs['seeds'] must not repeat: a repeat is a duplicate draw"
        )
    return seeds
