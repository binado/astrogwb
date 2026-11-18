"""Fixed-size source chunks: the one shape waveform evaluation compiles for.

A Poisson count differs per realization, and every distinct array shape is a
compile. So waveform evaluation works through its sources in pieces of one static
``chunk_size``, the last piece padded: a realization of any count reuses the
same compiled body, and ``chunk_size`` changes cost, never the result.

Padding rows repeat the chunk's first source -- a physical source, so a waveform
stays finite -- and the polarizations of padding rows are zero, so they add
nothing to any cross-power.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "iter_rows",
    "pad_rows",
]


def pad_rows(
    columns: Mapping[str, NDArray[Any]], start: int, size: int
) -> tuple[dict[str, NDArray[Any]], int]:
    """Rows ``start`` to ``start + size`` of ``columns``, padded, and how many are real.

    Padding rows repeat row ``start``.
    """
    rows = {
        name: np.asarray(values)[start : start + size]
        for name, values in columns.items()
    }
    valid = next(iter(rows.values())).shape[0]
    if valid < size:
        rows = {
            name: np.concatenate([values, np.repeat(values[:1], size - valid, axis=0)])
            for name, values in rows.items()
        }
    return rows, valid


def iter_rows(count: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """``(start, stop)`` of every chunk of ``count`` rows; none for no rows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, count, chunk_size):
        yield start, min(start + chunk_size, count)
