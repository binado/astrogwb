"""Layered TOML configuration for external simulation and catalog inputs.

Layers are TOML files merged left to right with ``knf``, the engine behind the
``knf`` CLI, so the shell and Python spell one merge rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import knf

from astrogwb.detector.config import DETECTOR_DEFAULT_PATHS

__all__ = [
    "MERGE_KEY",
    "merge_config",
    "merge_layers",
    "require_toml",
]

#: The inheritance key: ``extends = "${a.b}"`` in a table starts it from the
#: table ``a.b`` and lets its own fields win. It is removed from the result.
MERGE_KEY = "extends"


def require_toml(path: Path) -> None:
    """Reject a config layer that is not a ``.toml`` file; ``knf`` would accept JSON.

    Parameters
    ----------
    path
        File path.
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

    Parameters
    ----------
    paths
        Configuration file paths, in merge order.
    shallow
        Key-path glob to merge shallowly.

    Raises
    ------
    ValueError
        If a path is not a ``.toml`` file.
    """
    for path in paths:
        require_toml(path)
    return knf.load(paths, interpolate=True, shallow=shallow, merge_key=MERGE_KEY)


def merge_config(
    config_dir: Path = Path("config"),
    *,
    overrides: Sequence[Path] = (),
    shallow: str | None = "priors.*",
) -> dict[str, Any]:
    """Merge the shared configuration and optional overrides, unvalidated.

    The detector defaults, then every ``*.toml`` in ``config_dir`` in name
    order, then ``overrides``. A caller validates the tables it reads, e.g.
    ``CatalogMetadata.model_validate(merged["catalogs"][name])``.

    Parameters
    ----------
    config_dir
        Directory containing the base TOML files.
    overrides
        Additional TOML layers, in merge order.
    shallow
        Key-path glob for shallow merging.

    Raises
    ------
    ValueError
        If ``config_dir`` holds no TOML file.
    """
    paths = sorted(config_dir.glob("*.toml"))
    if not paths:
        raise ValueError(f"no config TOML files found in {config_dir}")
    return merge_layers([*DETECTOR_DEFAULT_PATHS, *paths, *overrides], shallow=shallow)
