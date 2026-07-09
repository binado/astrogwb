"""Shared config loading helpers (stdlib + pydantic only).

Load TOML/JSON into a dict, deep-merge optional overrides, then validate with a
caller-supplied pydantic model. Kept free of JAX so scripts can parse configs
before runtime initialization.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


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
    """Parse a TOML or JSON config file into a plain dict."""
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        if suffix == ".toml":
            return tomllib.load(handle)
        if suffix == ".json":
            return json.load(handle)
    raise ValueError(f"unsupported config extension: {path.suffix!r}")


def load_config_model(path: Path, model: type[T], **overrides: Any) -> T:
    """Load ``path``, deep-merge top-level ``**overrides``, and validate as ``model``."""
    raw = deep_merge(load_mapping(path), overrides)
    return model.model_validate(raw)
