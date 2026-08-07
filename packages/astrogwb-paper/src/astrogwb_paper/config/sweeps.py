"""Generate MCMC JSON configs for the three sweep campaigns.

Run from the repository root::

    uv run astrogwb-generate-mcmc-configs --force

Writes into ``packages/astrogwb-paper/configs/mcmc/``.
The declarative sweep spec names a base template plus networks, observations,
analyses, and prior variants. Each campaign expands its selected components as
``networks x analyses x observations`` before every point is validated as a
``RunConfig``.

Pass ``--write-manifests`` to also (re)generate the Snakemake batch manifest
for each campaign (under
``packages/astrogwb-paper/configs/mcmc/manifests/mcmc.batch.{campaign}.json``),
listing every selected config for that campaign for use with
``packages/astrogwb-paper/workflow/mcmc.smk``. A manifest is catalog-free: it
holds only ``{"chains_dir": ..., "runs": [...]}``. ``workflow/mcmc.smk``
sources the catalog separately from the paper project's
``configs/workflow.yaml``, so the same manifest can be run against any
catalog without regenerating it. Like the JSON configs, generated manifests
are gitignored -- they are fully reproducible from
``configs/mcmc.sweeps.toml``, so there is nothing to commit::

    uv run astrogwb-generate-mcmc-configs --write-manifests --force
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.config.mcmc import (
    AmplitudeParameter,
    RunConfig,
    build_run_config,
    save_config,
)
from astrogwb_paper.paths import paper_project_root

logger = logging.getLogger("generate_mcmc_configs")

PAPER_ROOT = paper_project_root()
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "configs/mcmc"
DEFAULT_SWEEP_SPEC = PAPER_ROOT / "configs/mcmc.sweeps.toml"
DEFAULT_MANIFEST_DIR = PAPER_ROOT / "configs/mcmc/manifests"
DEFAULT_CHAINS_DIR = "chains"

_STRICT = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
_NonEmptyStrings = Annotated[tuple[str, ...], Field(min_length=1)]
_PriorLibrary = dict[str, dict[str, dict[str, Any]]]


def _duplicates(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


class NetworkSpec(BaseModel):
    """A named detector network used by a sweep."""

    model_config = _STRICT

    detectors: _NonEmptyStrings

    @model_validator(mode="after")
    def _detectors_are_unique(self) -> NetworkSpec:
        duplicates = _duplicates(self.detectors)
        if duplicates:
            raise ValueError(f"duplicate detectors: {duplicates}")
        return self


class ObservationSpec(BaseModel):
    """Likelihood exposure and frequency mask for a sweep point."""

    model_config = _STRICT

    observation_time: Annotated[float, Field(gt=0.0)]
    f_min: Annotated[float, Field(gt=0.0)]
    f_max: Annotated[float, Field(gt=0.0)]

    @model_validator(mode="after")
    def _frequency_bounds_are_ordered(self) -> ObservationSpec:
        if self.f_min >= self.f_max:
            raise ValueError("f_min must be less than f_max")
        return self


class AnalysisSpec(BaseModel):
    """Sampled parameters, named priors, and optional fiducial overrides."""

    model_config = _STRICT

    sampled_params: _NonEmptyStrings
    priors: dict[str, str]
    fiducials: dict[str, float] = Field(default_factory=dict)
    likelihood: Literal["default", "amplitude_marginalized"] = "default"
    amplitude_parameter: AmplitudeParameter | None = None
    amplitude_num_nodes: Annotated[int, Field(gt=1)] | None = None
    amplitude_prior_span_sigma: Annotated[float, Field(gt=0.0)] | None = None

    @model_validator(mode="after")
    def _priors_match_sampled_params(self) -> AnalysisSpec:
        duplicates = _duplicates(self.sampled_params)
        if duplicates:
            raise ValueError(f"duplicate sampled_params: {duplicates}")

        marginalized = self.likelihood == "amplitude_marginalized"
        if marginalized and self.amplitude_parameter is None:
            raise ValueError(
                "amplitude_parameter is required when "
                "likelihood == 'amplitude_marginalized'"
            )
        if not marginalized and self.amplitude_parameter is not None:
            raise ValueError(
                "amplitude_parameter is only valid when "
                "likelihood == 'amplitude_marginalized'"
            )

        sampled = set(self.sampled_params)
        if self.amplitude_parameter is not None and (
            self.amplitude_parameter in sampled
        ):
            raise ValueError(
                f"amplitude_parameter {self.amplitude_parameter!r} cannot also "
                "appear in sampled_params"
            )

        expected = sampled | (
            {self.amplitude_parameter}
            if self.amplitude_parameter is not None
            else set()
        )
        prior_parameters = set(self.priors)
        if expected != prior_parameters:
            missing = sorted(expected - prior_parameters)
            extra = sorted(prior_parameters - expected)
            raise ValueError(
                "priors must exactly match sampled_params "
                f"(missing={missing}, extra={extra})"
            )
        return self


class RunSpec(BaseModel):
    """Named component selections whose Cartesian product forms a campaign."""

    model_config = _STRICT

    networks: _NonEmptyStrings
    analyses: _NonEmptyStrings
    observations: _NonEmptyStrings

    @model_validator(mode="after")
    def _selections_are_unique(self) -> RunSpec:
        for field_name in ("networks", "analyses", "observations"):
            values = getattr(self, field_name)
            duplicates = _duplicates(values)
            if duplicates:
                raise ValueError(f"duplicate {field_name}: {duplicates}")
        return self


class SweepConfig(BaseModel):
    """Complete declarative specification for MCMC sweep campaigns."""

    model_config = _STRICT

    base_config: Path
    networks: Annotated[dict[str, NetworkSpec], Field(min_length=1)]
    observations: Annotated[dict[str, ObservationSpec], Field(min_length=1)]
    analyses: Annotated[dict[str, AnalysisSpec], Field(min_length=1)]
    priors: Annotated[_PriorLibrary, Field(min_length=1)]
    runs: Annotated[dict[str, RunSpec], Field(min_length=1)]

    @model_validator(mode="after")
    def _references_exist(self) -> SweepConfig:
        for analysis_name, analysis in self.analyses.items():
            for parameter, prior_name in analysis.priors.items():
                parameter_priors = self.priors.get(parameter)
                if parameter_priors is None:
                    raise ValueError(
                        f"analysis {analysis_name!r} references priors for unknown "
                        f"parameter {parameter!r}"
                    )
                if prior_name not in parameter_priors:
                    raise ValueError(
                        f"analysis {analysis_name!r} references unknown prior "
                        f"{parameter}.{prior_name}"
                    )

        component_maps = {
            "networks": self.networks,
            "analyses": self.analyses,
            "observations": self.observations,
        }
        for run_name, run in self.runs.items():
            for field_name, components in component_maps.items():
                missing = sorted(set(getattr(run, field_name)) - set(components))
                if missing:
                    raise ValueError(
                        f"run {run_name!r} references unknown {field_name}: {missing}"
                    )
        return self


@dataclass(frozen=True)
class SweepPoint:
    """One named point in a campaign's Cartesian product."""

    campaign: str
    network: str
    analysis: str
    observation: str

    @property
    def filename(self) -> str:
        """Return the collision-resistant generated configuration filename."""
        return f"{self.network}__{self.analysis}__{self.observation}.json"


def load_sweep_config(path: Path) -> SweepConfig:
    """Load a sweep TOML/JSON file and resolve its base path relative to it."""
    resolved_path = path.resolve()
    sweep = SweepConfig.model_validate(load_mapping(resolved_path))
    base_config = sweep.base_config
    if not base_config.is_absolute():
        base_config = resolved_path.parent / base_config
    return sweep.model_copy(update={"base_config": base_config.resolve()})


def iter_sweep_points(sweep: SweepConfig) -> Iterator[SweepPoint]:
    """Yield sweep points in network/analysis/observation declaration order."""
    for campaign, run in sweep.runs.items():
        for network in run.networks:
            for analysis in run.analyses:
                for observation in run.observations:
                    yield SweepPoint(
                        campaign=campaign,
                        network=network,
                        analysis=analysis,
                        observation=observation,
                    )


def load_sweep_base(sweep: SweepConfig) -> dict[str, Any]:
    """Load and validate the invariant base mapping for a sweep."""
    base = load_mapping(sweep.base_config)
    fiducials = base.get("fiducials")
    if not isinstance(fiducials, dict) or not fiducials:
        raise ValueError(
            f"sweep base {sweep.base_config} must define a non-empty [fiducials] table"
        )

    for name, value in fiducials.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            # Configuration validation consistently surfaces schema errors as
            # ValueError, including incorrect scalar types.
            raise ValueError(  # noqa: TRY004
                f"base fiducial {name!r} must be numeric"
            )
        if not math.isfinite(value):
            raise ValueError(f"base fiducial {name!r} must be finite")

    base_names = set(fiducials)
    for analysis_name, analysis in sweep.analyses.items():
        unknown = sorted(set(analysis.fiducials) - base_names)
        if unknown:
            raise ValueError(
                f"analysis {analysis_name!r} overrides unknown fiducials: {unknown}"
            )
    return base


def materialize_run_config(
    base: dict[str, Any],
    sweep: SweepConfig,
    point: SweepPoint,
) -> RunConfig:
    """Merge one sweep point with its base and return a validated run config."""
    network = sweep.networks[point.network]
    observation = sweep.observations[point.observation]
    analysis = sweep.analyses[point.analysis]

    base_fiducials = base.get("fiducials")
    if not isinstance(base_fiducials, dict) or not base_fiducials:
        raise ValueError("sweep base must define a non-empty [fiducials] table")
    unknown_fiducials = sorted(set(analysis.fiducials) - set(base_fiducials))
    if unknown_fiducials:
        raise ValueError(
            f"analysis {point.analysis!r} overrides unknown fiducials: "
            f"{unknown_fiducials}"
        )

    raw = deepcopy(base)
    raw["fiducials"] = {
        **deepcopy(base_fiducials),
        **analysis.fiducials,
    }
    raw["observation_time"] = observation.observation_time
    analysis_raw: dict[str, Any] = {
        "detectors": list(network.detectors),
        "f_min": observation.f_min,
        "f_max": observation.f_max,
        "likelihood": analysis.likelihood,
    }
    if analysis.likelihood == "amplitude_marginalized":
        analysis_raw["amplitude_parameter"] = analysis.amplitude_parameter
        if analysis.amplitude_num_nodes is not None:
            analysis_raw["amplitude_num_nodes"] = analysis.amplitude_num_nodes
        if analysis.amplitude_prior_span_sigma is not None:
            analysis_raw["amplitude_prior_span_sigma"] = (
                analysis.amplitude_prior_span_sigma
            )
    raw["analysis"] = analysis_raw
    raw["sampled_params"] = list(analysis.sampled_params)
    raw["priors"] = {
        parameter: deepcopy(sweep.priors[parameter][prior_name])
        for parameter, prior_name in analysis.priors.items()
    }
    return build_run_config(raw)


def _resolve_paper_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PAPER_ROOT / resolved
    return resolved.resolve()


def render_manifest(
    campaign: str,
    run_configs: list[Path],
    *,
    chains_dir: str,
) -> str:
    """Render a Snakemake batch manifest (``workflow/mcmc.smk`` schema) as JSON.

    Snakemake's ``--configfile`` loader (``snakemake.common.configfile``) tries
    JSON before YAML regardless of file extension, so JSON needs no hand-rolled
    escaping and no extra dependency (``pyyaml`` is not part of the ``mcmc``
    extra this script is documented to run under).

    The manifest deliberately carries no ``catalog`` field: which data file to
    reweight is not a sweep concern, so ``workflow/mcmc.smk`` sources it
    separately from ``configs/workflow.yaml``. This lets the same manifest run
    against any catalog without regenerating it.

    ``jax_platforms`` is deliberately not part of this schema either: it is a
    runtime concern (which JAX backend to init), not an MCMC-campaign concern,
    and ``workflow/mcmc.smk`` already defaults it to ``"cuda"``. The Snakemake
    profile you run with (``profiles/local``, ``profiles/slurm-cpu``, ...)
    is what actually decides it.
    """
    manifest = {
        "chains_dir": chains_dir,
        "runs": [
            {"campaign": campaign, "config": path.as_posix()} for path in run_configs
        ],
    }
    return json.dumps(manifest, indent=2) + "\n"


def generate_configs(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    sweep_spec: SweepConfig | None = None,
    skip_existing: bool = True,
    write_manifests: bool = False,
    manifest_dir: str | Path = DEFAULT_MANIFEST_DIR,
    chains_dir: str = DEFAULT_CHAINS_DIR,
) -> tuple[Path, list[Path], list[Path], list[Path], list[Path]]:
    """Write campaign JSON configs and, if requested, their batch manifests.

    Returns ``(output_dir, written, skipped, written_manifests, skipped_manifests)``.
    """
    resolved_output_dir = _resolve_paper_path(output_dir)
    resolved_manifest_dir = _resolve_paper_path(manifest_dir)
    sweep_spec = sweep_spec or load_sweep_config(DEFAULT_SWEEP_SPEC)
    base = load_sweep_base(sweep_spec)

    planned: list[tuple[SweepPoint, Path, RunConfig]] = []
    campaign_runs: dict[str, list[Path]] = {
        campaign: [] for campaign in sweep_spec.runs
    }
    generated_paths: set[Path] = set()
    for point in iter_sweep_points(sweep_spec):
        path = resolved_output_dir / point.campaign / point.filename
        if path in generated_paths:
            raise ValueError(f"duplicate generated config path: {path}")
        generated_paths.add(path)
        relative_path = Path(os.path.relpath(path, PAPER_ROOT))
        campaign_runs[point.campaign].append(relative_path)
        config = materialize_run_config(base, sweep_spec, point)
        planned.append((point, path, config))

    # All points are validated before the first directory or file is created.
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    skipped: list[Path] = []
    written_manifests: list[Path] = []
    skipped_manifests: list[Path] = []

    for point, path, config in planned:
        if skip_existing and path.exists():
            skipped.append(path)
            logger.info("skipping existing config %s", path)
            continue

        save_config(config, path)
        written.append(path)
        logger.info(
            "wrote config %s campaign=%s network=%s analysis=%s observation=%s",
            path,
            point.campaign,
            point.network,
            point.analysis,
            point.observation,
        )

    if write_manifests:
        for campaign, run_configs in campaign_runs.items():
            resolved_manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = resolved_manifest_dir / f"mcmc.batch.{campaign}.json"
            if skip_existing and manifest_path.exists():
                skipped_manifests.append(manifest_path)
                logger.info("skipping existing manifest %s", manifest_path)
                continue
            manifest_text = render_manifest(
                campaign,
                run_configs,
                chains_dir=chains_dir,
            )
            manifest_path.write_text(manifest_text, encoding="utf-8")
            written_manifests.append(manifest_path)
            logger.info(
                "wrote manifest %s campaign=%s runs=%d",
                manifest_path,
                campaign,
                len(run_configs),
            )

    logger.info(
        "done output_dir=%s written=%d skipped=%d written_manifests=%d skipped_manifests=%d",
        resolved_output_dir,
        len(written),
        len(skipped),
        len(written_manifests),
        len(skipped_manifests),
    )
    return resolved_output_dir, written, skipped, written_manifests, skipped_manifests


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MCMC JSON configs for the cosmology, "
            "modified-propagation, and astrophysical sweep campaigns."
        )
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help=f"MCMC configs root directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sweep-spec",
        type=Path,
        default=DEFAULT_SWEEP_SPEC,
        help=f"Sweep campaign TOML (default: {DEFAULT_SWEEP_SPEC})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config files instead of skipping them.",
    )
    parser.add_argument(
        "--write-manifests",
        action="store_true",
        help=(
            "Also (re)generate a Snakemake batch manifest per campaign "
            "(packages/astrogwb-paper/configs/mcmc/manifests/"
            "mcmc.batch.{campaign}.json), holding only chains_dir and the "
            "campaign's runs -- no catalog field. "
            "packages/astrogwb-paper/workflow/mcmc.smk sources the catalog "
            "separately from packages/astrogwb-paper/configs/workflow.yaml. "
            "Gitignored, like the JSON configs: fully reproducible from "
            "mcmc.sweeps.toml."
        ),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help=(
            "Directory for generated batch manifests "
            f"(default: {DEFAULT_MANIFEST_DIR}). Only used with --write-manifests."
        ),
    )
    parser.add_argument(
        "--chains-dir",
        default=DEFAULT_CHAINS_DIR,
        help=f"chains_dir for generated manifests (default: {DEFAULT_CHAINS_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    generate_configs(
        args.output_dir,
        sweep_spec=load_sweep_config(args.sweep_spec),
        skip_existing=not args.force,
        write_manifests=args.write_manifests,
        manifest_dir=args.manifest_dir,
        chains_dir=args.chains_dir,
    )


if __name__ == "__main__":
    main()
