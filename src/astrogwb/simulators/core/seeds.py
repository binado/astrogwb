"""Derive many seeds from one, without JAX."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["split_seed"]


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
