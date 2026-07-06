"""Pydantic models and I/O for headless MCMC run configs.

This module imports only stdlib and pydantic so callers can parse and validate
configs before JAX initializes.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

_STRICT = ConfigDict(frozen=True, extra="forbid")


class RuntimeConfig(BaseModel):
    """Device/thread control for :func:`scripts.run_mcmc.configure_runtime`."""

    model_config = _STRICT

    platform: str = "auto"  # auto | cpu | gpu
    host_device_count: int | None = None  # CPU devices for parallel chains
    cpu_threads: int = 0  # 0 -> leave XLA/OMP default
    chain_method: str = "auto"  # auto | parallel | sequential | vectorized


class CatalogConfig(BaseModel):
    model_config = _STRICT

    path: Path
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
    forward_mode_differentiation: bool = True
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
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    catalog: CatalogConfig
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


def load_config(path: Path) -> dict[str, Any]:
    """Parse a TOML or JSON config file into a plain dict."""
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        if suffix == ".toml":
            return tomllib.load(handle)
        if suffix == ".json":
            return json.load(handle)
    raise ValueError(f"unsupported config extension: {path.suffix!r}")


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
) -> RunConfig:
    """Apply optional overrides and validate a raw config mapping into a RunConfig."""
    raw = dict(raw)
    if seed is not None:
        raw["seed"] = seed
    if outdir is not None or label is not None:
        output = dict(raw.get("output", {}))
        if outdir is not None:
            output["outdir"] = str(outdir)
        if label is not None:
            output["label"] = label
        raw["output"] = output

    return RunConfig.model_validate(raw)
