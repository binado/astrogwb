"""Decoding and encoding for the scalar attributes an HDF5 artifact carries.

h5py hands back NumPy scalars and, for older files, bytes; HDF5 attributes
themselves hold only scalars and small arrays. Every persisted artifact in this
package therefore reduces its metadata to ``str``/``int``/``float`` and encodes
anything structured as JSON. This module is that codec, and the
``{name: column} <-> (row, column) matrix`` pattern both formats use for their
named parameter arrays.

It touches no h5py: everything here operates on values a reader has already
pulled out of a file, or values a writer is about to put in. That is what lets
:mod:`astrogwb.populations.record` serialize itself without the population
layer taking on an optional dependency.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DOMAIN_FREQUENCY",
    "FORMAT_NAME_ATTR",
    "int_attr",
    "json_array_attr",
    "json_object_attr",
    "require_attrs",
    "require_format",
    "scalar_attr",
    "stack_columns",
    "unstack_columns",
]

#: The two attributes every artifact stamps: what the file is, and what its
#: leading axis means. Read before anything else, so a foreign file fails with
#: one clear message instead of a cascade of missing names.
FORMAT_NAME_ATTR = "format_name"
DOMAIN_FREQUENCY = "frequency"


def scalar_attr(value: Any, *, name: str) -> str | int | float:
    """Reduce one raw attribute value to a plain Python scalar.

    Booleans are rejected rather than widened to ``int``: an attribute that
    reads back as ``True`` is a writer bug, and silently accepting it would let
    a seed of ``True`` round-trip as ``1``.
    """
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise TypeError(
            f"attribute {name!r} must be a str, non-boolean int, or float scalar"
        )
    return value


def int_attr(value: Any, *, label: str, name: str) -> int:
    """Read one attribute that must be a non-boolean integer."""
    scalar = scalar_attr(value, name=name)
    if isinstance(scalar, bool) or not isinstance(scalar, int):
        raise TypeError(f"{label}: {name} must be an int")
    return scalar


def json_object_attr(value: Any, *, label: str, name: str) -> dict[str, Any]:
    """Decode one attribute that must hold a JSON object."""
    decoded = _json(value, label=label, name=name)
    if not isinstance(decoded, dict):
        raise TypeError(f"{label}: attribute {name!r} must decode to a JSON object")
    return decoded


def json_array_attr(value: Any, *, label: str, name: str) -> list[Any]:
    """Decode one attribute that must hold a JSON array."""
    decoded = _json(value, label=label, name=name)
    if not isinstance(decoded, list):
        raise TypeError(f"{label}: attribute {name!r} must decode to a JSON array")
    return decoded


def _json(value: Any, *, label: str, name: str) -> Any:
    try:
        return json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{label}: attribute {name!r} is not valid JSON: {error}"
        ) from error


def require_format(
    attrs: Mapping[Any, Any], *, label: str, format_name: str, domain: str
) -> None:
    """Reject a file that is not this exact format and domain.

    Each artifact passes its own ``format_name``; there is no shared version
    space and no compatibility reader. A file written in an older format is an
    error, not something to migrate on read.
    """
    if attrs.get(FORMAT_NAME_ATTR) != format_name:
        raise ValueError(
            f"{label}: format_name is {attrs.get(FORMAT_NAME_ATTR)!r}, "
            f"expected {format_name!r}"
        )
    if attrs.get("domain") != domain:
        raise ValueError(
            f"{label}: domain is {attrs.get('domain')!r}, expected {domain!r}"
        )


def require_attrs(
    attrs: Mapping[Any, Any],
    names: Sequence[str],
    *,
    label: str,
    kind: str = "",
    hint: str = "",
) -> None:
    """Raise naming every required attribute the file is missing, at once."""
    missing = [name for name in names if name not in attrs]
    if missing:
        described = f"{kind} " if kind else ""
        raise ValueError(
            f"{label}: missing {described}attribute(s): {', '.join(missing)}{hint}"
        )


def stack_columns(
    columns: Mapping[str, ArrayLike], *, rows: int
) -> tuple[list[str], NDArray[np.float64]]:
    """Stack named 1-D columns into one ``(rows, column)`` float64 matrix.

    Returns the column order alongside the matrix: the names are persisted as
    a separate JSON attribute, and the two are only meaningful together. An
    artifact with no columns still yields a correctly shaped empty matrix, so
    the row count survives the round trip.
    """
    names = list(columns)
    if not names:
        return names, np.empty((rows, 0), dtype=np.float64)
    return names, np.stack(
        [np.asarray(columns[name], dtype=np.float64) for name in names], axis=1
    )


def unstack_columns(
    values: NDArray[Any], names: Sequence[str]
) -> dict[str, NDArray[Any]]:
    """Split a ``(rows, column)`` matrix back into named 1-D columns."""
    return {str(name): values[:, index] for index, name in enumerate(names)}
