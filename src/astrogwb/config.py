"""Layered TOML configuration shared by the detector registry and the paper.

Layers are TOML files merged left to right with ``knf``, the engine behind the
``knf`` CLI, so the shell and Python spell one merge rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import knf
from pydantic import BaseModel, model_validator

from astrogwb.detector.config import DETECTOR_DEFAULT_PATHS, DetectorRegistry
from astrogwb.distributions.config import DistributionConfig
from astrogwb.importance.catalog import CatalogMetadata
from astrogwb.waveform import WaveformMetadata

__all__ = [
    "MERGE_KEY",
    "PaperConfig",
    "load_config",
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


class PaperConfig(BaseModel):
    """Validated shared settings for the paper application."""

    fiducials: dict[str, float]
    priors: dict[str, DistributionConfig]
    registry: DetectorRegistry
    catalog: CatalogMetadata
    waveforms: dict[str, WaveformMetadata]
    figure_format: str
    figure_dpi: int
    labels: dict[str, str]

    @model_validator(mode="before")
    @classmethod
    def _registry(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                **value,
                "registry": {
                    "detectors": value.get("detectors"),
                    "networks": value.get("networks", {}),
                },
            }
        return value


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


def load_config(
    config_dir: Path = Path("config"),
    *,
    overrides: Sequence[Path] = (),
    shallow: str | None = "priors.*",
) -> PaperConfig:
    """Merge and validate the shared configuration and optional overrides.

    Parameters
    ----------
    config_dir
        Directory containing the base TOML files.
    overrides
        Additional TOML layers, in merge order.
    shallow
        Key-path glob for shallow merging.
    """
    merged = merge_config(config_dir, overrides=overrides, shallow=shallow)
    return PaperConfig.model_validate(merged, extra="ignore")
