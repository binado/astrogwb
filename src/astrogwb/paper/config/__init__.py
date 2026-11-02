"""Load the paper's shared TOML configuration."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, model_validator

from astrogwb.config import merge_layers
from astrogwb.detector.config import DETECTOR_DEFAULT_PATHS, DetectorRegistry
from astrogwb.distributions.config import DistributionConfig
from astrogwb.gwb.catalog import CatalogMetadata
from astrogwb.waveform import WaveformMetadata


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
    paths = sorted(config_dir.glob("*.toml"))
    if not paths:
        raise ValueError(f"no config TOML files found in {config_dir}")
    merged = merge_layers(
        [*DETECTOR_DEFAULT_PATHS, *paths, *overrides], shallow=shallow
    )
    return PaperConfig.model_validate(merged, extra="ignore")
