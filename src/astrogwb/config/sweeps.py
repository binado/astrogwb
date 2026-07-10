"""Load the declarative MCMC sweep specification."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrogwb.config.loading import load_mapping
from astrogwb.utils import repo_root

DEFAULT_SWEEP_SPEC = repo_root() / "configs" / "mcmc.sweeps.toml"


@dataclass(frozen=True)
class SweepSpec:
    """Networks, priors, and campaign runs loaded from a TOML file."""

    networks: dict[str, tuple[str, ...]]
    priors: dict[str, dict[str, Any]]
    campaigns: dict[str, dict[str, tuple[tuple[str, ...], dict[str, dict[str, Any]]]]]

    def filenames(self) -> list[str]:
        """Return ``{campaign}/{network}__{sample}.json`` paths."""
        return [
            f"{campaign}/{network_label}__{sample_label}.json"
            for campaign, sample_sets in self.campaigns.items()
            for network_label in self.networks
            for sample_label in sample_sets
        ]

    def campaign_runs(self, campaign: str) -> list[str]:
        """Return sorted run stems for one generated campaign."""
        return sorted(
            filename.removeprefix(f"{campaign}/").removesuffix(".json")
            for filename in self.filenames()
            if filename.startswith(f"{campaign}/")
        )


def load_sweep_spec(path: Path = DEFAULT_SWEEP_SPEC) -> SweepSpec:
    """Parse a sweep TOML file into the data required to generate its configs."""
    raw = load_mapping(path)
    networks = {name: tuple(detectors) for name, detectors in raw["networks"].items()}
    priors = deepcopy(raw["priors"])
    campaigns = {
        campaign: {
            label: (
                tuple(run["sampled_params"]),
                deepcopy(run.get("priors", {})),
            )
            for label, run in runs.items()
        }
        for campaign, runs in raw["campaigns"].items()
    }
    return SweepSpec(networks=networks, priors=priors, campaigns=campaigns)
