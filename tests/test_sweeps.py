from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.generate_mcmc_configs import (
    AnalysisSpec,
    ObservationSpec,
    RunSpec,
    SweepConfig,
    iter_sweep_points,
    load_sweep_base,
    load_sweep_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_CONFIG = REPO_ROOT / "configs" / "mcmc.sweeps.toml"


def _raw_sweep() -> dict[str, object]:
    return {
        "base_config": REPO_ROOT / "configs" / "mcmc.base.toml",
        "networks": {"network": {"detectors": ["S1", "R1"]}},
        "observations": {
            "baseline": {
                "observation_time": 1.0,
                "f_min": 2.0,
                "f_max": 4096.0,
            }
        },
        "analyses": {
            "H0": {
                "sampled_params": ["H0"],
                "priors": {"H0": "uniform"},
            }
        },
        "priors": {"H0": {"uniform": {"type": "uniform", "low": 20.0, "high": 140.0}}},
        "runs": {
            "campaign": {
                "networks": ["network"],
                "analyses": ["H0"],
                "observations": ["baseline"],
            }
        },
    }


def test_canonical_sweep_preserves_campaign_coverage() -> None:
    sweep = load_sweep_config(SWEEP_CONFIG)
    points = list(iter_sweep_points(sweep))

    assert sweep.base_config == (REPO_ROOT / "configs" / "mcmc.base.toml").resolve()
    assert len(points) == 66
    assert Counter(point.campaign for point in points) == {
        "cosmology": 8,
        "cosmology-all-detectors": 24,
        "modified-propagation-all-detectors": 18,
        "astrophysical": 4,
        "astrophysical-all-detectors": 12,
    }
    assert {point.observation for point in points} == {"baseline"}


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (
            {"observation_time": 0.0, "f_min": 2.0, "f_max": 10.0},
            "greater than 0",
        ),
        (
            {"observation_time": 1.0, "f_min": 10.0, "f_max": 2.0},
            "f_min must be less than f_max",
        ),
        (
            {"observation_time": float("nan"), "f_min": 2.0, "f_max": 10.0},
            "finite number",
        ),
    ],
)
def test_observation_validation(observation: dict[str, float], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ObservationSpec.model_validate(observation)


def test_analysis_priors_must_exactly_match_sampled_params() -> None:
    with pytest.raises(ValidationError, match="exactly match sampled_params"):
        AnalysisSpec(
            sampled_params=("H0", "Omega_m"),
            priors={"H0": "uniform"},
        )


@pytest.mark.parametrize(
    ("section", "replacement", "message"),
    [
        (
            "analyses",
            {
                "H0": {
                    "sampled_params": ["H0"],
                    "priors": {"H0": "missing"},
                }
            },
            "unknown prior H0.missing",
        ),
        (
            "runs",
            {
                "campaign": {
                    "networks": ["missing"],
                    "analyses": ["H0"],
                    "observations": ["baseline"],
                }
            },
            "unknown networks",
        ),
    ],
)
def test_sweep_rejects_unknown_references(
    section: str, replacement: dict[str, object], message: str
) -> None:
    raw = _raw_sweep()
    raw[section] = replacement

    with pytest.raises(ValidationError, match=message):
        SweepConfig.model_validate(raw)


def test_run_selections_must_be_nonempty_and_unique() -> None:
    with pytest.raises(ValidationError, match="duplicate networks"):
        RunSpec(
            networks=("network", "network"),
            analyses=("H0",),
            observations=("baseline",),
        )
    with pytest.raises(ValidationError, match="at least 1 item"):
        RunSpec(networks=(), analyses=("H0",), observations=("baseline",))


def test_base_rejects_unknown_analysis_fiducial_override() -> None:
    raw = _raw_sweep()
    raw["analyses"] = {
        "H0": {
            "sampled_params": ["H0"],
            "priors": {"H0": "uniform"},
            "fiducials": {"misspelled": 1.0},
        }
    }
    sweep = SweepConfig.model_validate(raw)

    with pytest.raises(ValueError, match="unknown fiducials.*misspelled"):
        load_sweep_base(sweep)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_analysis_fiducial_overrides_must_be_finite(value: float) -> None:
    with pytest.raises(ValidationError, match="finite number"):
        AnalysisSpec(
            sampled_params=("H0",),
            priors={"H0": "uniform"},
            fiducials={"H0": value},
        )
