"""Expand declarative MCMC sweep campaigns into per-run fragment lists.

A campaign is a Cartesian product of named networks, analyses, and
observations (``configs/mcmc.sweeps.toml``). Each name refers to a *fragment*
under ``configs/mcmc/fragments/`` -- a partial :class:`RunConfig` in TOML.
Merging a point's fragments left to right yields that run's config::

    base.toml -> priors.toml -> networks/<n>.toml
              -> observations/<o>.toml -> analyses/<a>.toml

This module only decides *which* fragments a point needs and in what order.
The merge itself is ``knf`` and the validation is
``astrogwb-validate-config``, both driven by ``workflow/mcmc.smk``'s
``mcmc_config`` rule -- so a name with no matching fragment surfaces as a
missing Snakemake input, and a merged config that violates ``RunConfig``
fails before any sampling is scheduled.

Layer order is the whole contract here: broadest first, narrowest last. An
analysis fragment is the last word on its own priors and sampled parameters.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb_paper.config.loading import load_mapping

_STRICT = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
_NonEmptyStrings = Annotated[tuple[str, ...], Field(min_length=1)]

FRAGMENTS_DIR = Path("configs/mcmc/fragments")
SWEEP_SPEC = Path("configs/mcmc.sweeps.toml")


def _duplicates(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


class RunSpec(BaseModel):
    """Named component selections whose Cartesian product forms a campaign."""

    model_config = _STRICT

    networks: _NonEmptyStrings
    analyses: _NonEmptyStrings
    observations: _NonEmptyStrings

    @model_validator(mode="after")
    def _selections_are_unique(self) -> RunSpec:
        for field_name in ("networks", "analyses", "observations"):
            values: tuple[str, ...] = getattr(self, field_name)
            duplicates = _duplicates(values)
            if duplicates:
                raise ValueError(f"duplicate {field_name}: {duplicates}")
        return self


class SweepConfig(BaseModel):
    """The campaign spec: nothing but product structure."""

    model_config = _STRICT

    runs: Annotated[dict[str, RunSpec], Field(min_length=1)]


@dataclass(frozen=True)
class SweepPoint:
    """One named point in a campaign's Cartesian product."""

    campaign: str
    network: str
    analysis: str
    observation: str

    @property
    def run(self) -> str:
        """Return the collision-resistant run name used for output paths."""
        return f"{self.network}__{self.analysis}__{self.observation}"

    def fragments(self, fragments_dir: Path = FRAGMENTS_DIR) -> list[Path]:
        """Return this point's fragment layers, broadest first."""
        return [
            fragments_dir / "base.toml",
            fragments_dir / "priors.toml",
            fragments_dir / "networks" / f"{self.network}.toml",
            fragments_dir / "observations" / f"{self.observation}.toml",
            fragments_dir / "analyses" / f"{self.analysis}.toml",
        ]


def load_sweep_config(path: Path = SWEEP_SPEC) -> SweepConfig:
    """Load and validate a sweep campaign spec."""
    return SweepConfig.model_validate(load_mapping(Path(path)))


def iter_sweep_points(sweep: SweepConfig) -> Iterator[SweepPoint]:
    """Yield sweep points in network/analysis/observation declaration order."""
    for campaign, run in sweep.runs.items():
        for network, analysis, observation in product(
            run.networks, run.analyses, run.observations
        ):
            yield SweepPoint(
                campaign=campaign,
                network=network,
                analysis=analysis,
                observation=observation,
            )


def run_fragments(
    sweep: SweepConfig, fragments_dir: Path = FRAGMENTS_DIR
) -> dict[tuple[str, str], list[Path]]:
    """Map every ``(campaign, run)`` to the fragment layers that build it.

    This is what ``workflow/mcmc.smk`` reads at parse time to drive both the
    ``mcmc_config`` rule's inputs and the set of chains the campaign targets.
    """
    mapping: dict[tuple[str, str], list[Path]] = {}
    for point in iter_sweep_points(sweep):
        key = (point.campaign, point.run)
        if key in mapping:
            raise ValueError(f"duplicate MCMC run {point.campaign}/{point.run}")
        mapping[key] = point.fragments(fragments_dir)
    return mapping
