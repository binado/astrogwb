"""Pydantic models and I/O for headless MCMC run configs.

This module imports only stdlib and pydantic so callers can parse and validate
configs before JAX initializes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb_paper.config.hashing import canonical_sha256
from astrogwb_paper.config.loading import deep_merge

_STRICT = ConfigDict(frozen=True, extra="forbid")


class AnalysisConfig(BaseModel):
    model_config = _STRICT

    detectors: tuple[str, ...]
    f_min: float
    f_max: float


class CosmoConfig(BaseModel):
    model_config = _STRICT

    z_min: float
    z_max: float
    n_grid: int


class SamplerConfig(BaseModel):
    model_config = _STRICT

    num_warmup: Annotated[int, Field(gt=0)]
    num_samples: Annotated[int, Field(gt=0)]
    num_chains: Annotated[int, Field(gt=0)] = 1
    target_accept: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.9
    dense_mass: bool = True
    max_tree_depth: Annotated[int, Field(gt=0, le=20)] = 10
    forward_mode_differentiation: bool = False
    progress_bar: bool = False
    jit_model_args: bool = True


class OutputConfig(BaseModel):
    model_config = _STRICT

    outdir: Path = Path("chains")
    label: str = ""


class RunConfig(BaseModel):
    model_config = _STRICT

    seed: int = 42
    observation_time: float = 1.0
    fiducials: dict[str, float]
    priors: dict[str, dict[str, Any]]  # prior name -> spec table
    # Unset (empty) -> default to the keys present in [priors]; resolved below.
    sampled_params: tuple[str, ...] = ()
    analysis: AnalysisConfig
    cosmology: CosmoConfig
    sampler: SamplerConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    # Derived in the validator (every fiducial not sampled).
    constants: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _resolve_sampled_and_constants(self) -> RunConfig:
        if not self.fiducials:
            raise ValueError("config must define a non-empty [fiducials] table")
        if not self.priors:
            raise ValueError("config must define at least one [priors.<param>] table")

        sampled = self.sampled_params or tuple(self.priors)
        missing_priors = [p for p in sampled if p not in self.priors]
        if missing_priors:
            raise ValueError(
                f"sampled_params without a [priors.*] table: {missing_priors}"
            )
        missing_fid = [p for p in sampled if p not in self.fiducials]
        if missing_fid:
            raise ValueError(f"sampled_params missing from [fiducials]: {missing_fid}")

        aligned_priors = {name: self.priors[name] for name in sampled}
        constants = {k: v for k, v in self.fiducials.items() if k not in sampled}

        object.__setattr__(self, "sampled_params", sampled)
        object.__setattr__(self, "priors", aligned_priors)
        object.__setattr__(self, "constants", constants)
        return self

    @property
    def outdir(self) -> Path:
        return self.output.outdir

    @property
    def label(self) -> str:
        return self.output.label


def config_sha256(config: RunConfig) -> str:
    """Digest inference settings, excluding output routing."""
    return canonical_sha256(config.model_dump(mode="json", exclude={"output"}))


def save_config(config: RunConfig, path: Path) -> None:
    """Write a validated run config as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def build_run_config(
    raw: dict[str, Any],
    *,
    seed: int | None = None,
    outdir: Path | None = None,
    label: str | None = None,
    **overrides: Any,
) -> RunConfig:
    """Apply optional overrides and validate a raw config mapping into a RunConfig.

    Named ``seed`` / ``outdir`` / ``label`` remain for CLI compatibility. Extra
    ``**overrides`` are deep-merged into the raw mapping (nested dicts merge;
    other values replace) before validation.
    """
    cli_overrides: dict[str, Any] = {}
    if seed is not None:
        cli_overrides["seed"] = seed
    if outdir is not None or label is not None:
        output: dict[str, Any] = {}
        if outdir is not None:
            output["outdir"] = str(outdir)
        if label is not None:
            output["label"] = label
        cli_overrides["output"] = output

    merged = deep_merge(raw, deep_merge(overrides, cli_overrides))
    return RunConfig.model_validate(merged)
