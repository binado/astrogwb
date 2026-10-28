"""Layered TOML configuration shared by the detector registry and the paper.

Layers are TOML files merged left to right with ``knf``, the engine behind the
``knf`` CLI, so the shell and Python spell one merge rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import knf

__all__ = ["MERGE_KEY", "merge_layers", "require_toml"]

#: The inheritance key: ``extends = "${a.b}"`` in a table starts it from the
#: table ``a.b`` and lets its own fields win. It is removed from the result.
MERGE_KEY = "extends"


def require_toml(path: Path) -> None:
    """Reject a config layer that is not a ``.toml`` file.

    Every layer in ``config/`` is TOML. ``knf`` would merge an all-JSON layer
    list just as happily, which is exactly why the format is pinned here: a
    second format should arrive as one deliberate migration, not a stray file.
    """
    if path.suffix.lower() != ".toml":
        raise ValueError(f"config layers are TOML; got {path.suffix!r} for {path}")


def merge_layers(
    paths: Sequence[Path], *, shallow: str | None = None
) -> dict[str, Any]:
    """Fold TOML layer files into one raw mapping, in the order given.

    A deep merge, left to right -- arrays and scalars replace -- except at the
    ``knf`` key-path glob ``shallow``, followed by resolving every ``${...}``
    reference against the merged result and every :data:`MERGE_KEY`
    inheritance.

    Raises
    ------
    ValueError
        If a path is not a ``.toml`` file.
    """
    for path in paths:
        require_toml(path)
    return knf.load(paths, interpolate=True, shallow=shallow, merge_key=MERGE_KEY)
