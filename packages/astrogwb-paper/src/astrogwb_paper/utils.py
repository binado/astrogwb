"""Small shared config-I/O helpers: merge semantics and mapping file loading.

stdlib + PyYAML only, so every consumer -- the config layer, the CLI
entrypoints, and the ``Snakefile`` -- can import this without paying for
pydantic, xarray, or JAX.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


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
    """Parse a YAML, TOML, or JSON config file into a plain dict."""
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        if suffix == ".toml":
            raw = tomllib.load(handle)
        elif suffix == ".json":
            raw = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            raw = yaml.safe_load(handle)
        else:
            raise ValueError(f"unsupported config extension: {path.suffix!r}")
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path} must contain a mapping")
    return dict(raw)
