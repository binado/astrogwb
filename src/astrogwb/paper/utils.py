"""Small shared config-I/O helpers: merge semantics and mapping file loading.

Light on imports, so every consumer can use this without paying for pydantic,
xarray, or JAX.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from astrogwb.config import require_toml


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
    """Parse one TOML config file into a plain dict."""
    require_toml(path)
    with path.open("rb") as handle:
        return tomllib.load(handle)
