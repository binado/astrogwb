"""The two tree types every cached node trades in."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from numpy.typing import ArrayLike, NDArray

__all__ = ["Arrays", "Tree"]

#: What a node is handed: nested mappings with array-like leaves (a Python
#: scalar, a NumPy or JAX array).
type Tree = Mapping[str, ArrayLike | Tree]

#: What a node returns and the cache stores: nested dicts of NumPy arrays.
type Arrays = dict[str, NDArray[Any] | Arrays]
