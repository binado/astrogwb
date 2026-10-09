"""Small shared config-I/O helpers: merge semantics and mapping file loading.

stdlib only, so every consumer can import this without paying for pydantic,
xarray, or JAX.
"""

from __future__ import annotations

import tomllib
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


def require_toml(path: Path) -> None:
    """Reject a config layer that is not a ``.toml`` file.

    Every layer in ``config/`` is TOML. ``knf`` would merge an all-JSON layer
    list just as happily, which is exactly why the format is pinned here: a
    second format should arrive as one deliberate migration, not a stray file.
    ``astrogwb.detector``'s packaged ``geometry.toml`` and ``sensitivity.toml``
    are also TOML: core reads them with ``tomllib`` and the paper application
    merges their registry tables with its config layers through ``knf``.
    """
    if path.suffix.lower() != ".toml":
        raise ValueError(f"config layers are TOML; got {path.suffix!r} for {path}")


def load_mapping(path: Path) -> dict[str, Any]:
    """Parse one TOML config file into a plain dict."""
    require_toml(path)
    with path.open("rb") as handle:
        return tomllib.load(handle)
