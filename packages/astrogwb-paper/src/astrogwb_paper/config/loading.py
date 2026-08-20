"""Shared config loading helpers.

Load YAML/TOML/JSON into a dict and deep-merge optional overrides. Kept free of
JAX so scripts can parse configs before runtime initialization.
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


def merge_run_overlay(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge a run overlay, replacing named prior tables wholesale.

    ``deep_merge`` key-merges nested mappings, which leaves stale ``low`` /
    ``high`` behind when a uniform prior is replaced by a normal one. Each
    ``[priors.<param>]`` table in ``override`` replaces the base spec instead.
    """
    overlay_priors = override.get("priors")
    merged = deep_merge(
        base, {key: value for key, value in override.items() if key != "priors"}
    )
    if not isinstance(overlay_priors, Mapping):
        return merged
    priors = dict(merged.get("priors") or {})
    for name, spec in overlay_priors.items():
        priors[name] = dict(spec) if isinstance(spec, Mapping) else spec
    merged["priors"] = priors
    return merged


def load_inventory(
    path: Path, *, required: tuple[str, ...], optional: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Load a committed inventory file and validate its top-level sections.

    Unknown top-level keys are rejected so a typo fails at load instead of
    being silently ignored, and every ``required`` section must be a non-empty
    mapping.
    """
    raw = load_mapping(path)
    unknown = sorted(set(raw) - set(required) - set(optional))
    if unknown:
        raise ValueError(f"{path} has unknown top-level keys: {', '.join(unknown)}")
    for key in required:
        section = raw.get(key)
        if not isinstance(section, Mapping) or not section:
            raise ValueError(f"{path} must define a non-empty {key} mapping")
    return raw


def load_mapping(path: Path | str) -> dict[str, Any]:
    """Parse a YAML, TOML, or JSON config file into a plain dict."""
    path = Path(path)
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        if suffix == ".toml":
            return tomllib.load(handle)
        if suffix == ".json":
            return json.load(handle)
        if suffix in {".yaml", ".yml"}:
            raw = yaml.safe_load(handle)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path} must contain a mapping")
            return dict(raw)
    raise ValueError(f"unsupported config extension: {path.suffix!r}")
