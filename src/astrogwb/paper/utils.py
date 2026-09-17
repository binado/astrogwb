"""Small shared config-I/O helpers: merge semantics and mapping file loading.

stdlib only, so every consumer -- the config layer, the CLI entrypoints, and
the ``Snakefile`` -- can import this without paying for pydantic, xarray, or
JAX.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``.

    Nested mappings are merged; all other values (including lists) replace.
    Neither input mapping is mutated.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_mapping(path: Path) -> dict[str, Any]:
    """Parse one JSON config file into a plain dict.

    JSON only, and that is the point rather than a limitation: every layer in
    ``config/`` is JSON, so ``jq`` can fold any of them in the shell exactly as
    this function folds them in Python. A second accepted format would make
    that true only by convention. ``astrogwb.detector``'s packaged
    ``geometry.toml`` and ``sensitivity.toml`` are detector *data*, not config
    layers, and are read with ``tomllib`` where they are used.
    """
    if path.suffix.lower() != ".json":
        raise ValueError(f"config layers are JSON; got {path.suffix!r} for {path}")
    with path.open("rb") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path} must contain a mapping")
    return dict(raw)
