from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from astrogwb_paper.config.loading import deep_merge, load_mapping
from astrogwb_paper.config.mcmc import build_run_config
from astrogwb_paper.config.sweeps import (
    FRAGMENTS_DIR,
    SWEEP_SPEC,
    RunSpec,
    SweepConfig,
    SweepPoint,
    iter_sweep_points,
    load_sweep_config,
    run_fragments,
)
from astrogwb_paper.paths import paper_project_root
from pydantic import ValidationError

PAPER_ROOT = paper_project_root()
SWEEP_CONFIG = PAPER_ROOT / SWEEP_SPEC
FRAGMENTS = PAPER_ROOT / FRAGMENTS_DIR


def _merge(paths: list[Path]) -> dict:
    """Mirror knf's merge: mappings merge key by key, everything else replaces."""
    merged: dict = {}
    for path in paths:
        merged = deep_merge(merged, load_mapping(path))
    return merged


def test_canonical_sweep_preserves_campaign_coverage() -> None:
    sweep = load_sweep_config(SWEEP_CONFIG)
    points = list(iter_sweep_points(sweep))

    assert len(points) == 58
    assert Counter(point.campaign for point in points) == {
        "cosmology": 6,
        "cosmology-all-detectors": 18,
        "modified-propagation-all-detectors": 18,
        "astrophysical": 4,
        "astrophysical-all-detectors": 12,
    }
    assert {point.observation for point in points} == {"baseline"}


def test_run_names_carry_every_product_dimension() -> None:
    point = SweepPoint(
        campaign="cosmology",
        network="ET-2L-aligned",
        analysis="H0-Omega_m",
        observation="baseline",
    )

    assert point.run == "ET-2L-aligned__H0-Omega_m__baseline"


def test_fragment_layers_are_ordered_broadest_first() -> None:
    """Layer order is the semantic contract; knf merges left to right."""
    point = SweepPoint(
        campaign="cosmology",
        network="ET-2L-aligned",
        analysis="H0-Omega_m",
        observation="baseline",
    )

    assert [path.name for path in point.fragments()] == [
        "base.toml",
        "priors.toml",
        "ET-2L-aligned.toml",
        "baseline.toml",
        "H0-Omega_m.toml",
    ]


def test_every_referenced_fragment_exists() -> None:
    """A dangling name must be a missing Snakemake input, not a silent skip."""
    sweep = load_sweep_config(SWEEP_CONFIG)

    for point in iter_sweep_points(sweep):
        for fragment in point.fragments(FRAGMENTS):
            assert fragment.is_file(), f"{point.campaign}/{point.run}: {fragment}"


def test_run_fragments_keys_every_campaign_run_pair() -> None:
    sweep = load_sweep_config(SWEEP_CONFIG)

    mapping = run_fragments(sweep, FRAGMENTS)

    assert len(mapping) == 58
    assert ("cosmology", "ET-2L-aligned__H0__baseline") in mapping


def test_run_fragments_keeps_same_point_in_distinct_campaigns() -> None:
    """Campaigns overlap by design; the key is (campaign, run), not run."""
    sweep = SweepConfig.model_validate(
        {
            "runs": {
                "quick": {
                    "networks": ["ET-2L-aligned"],
                    "analyses": ["H0"],
                    "observations": ["baseline"],
                },
                "all-detectors": {
                    "networks": ["ET-2L-aligned"],
                    "analyses": ["H0"],
                    "observations": ["baseline"],
                },
            }
        }
    )

    assert len(run_fragments(sweep)) == 2


def test_run_selections_must_be_nonempty_and_unique() -> None:
    with pytest.raises(ValidationError, match="duplicate networks"):
        RunSpec(
            networks=("network", "network"),
            analyses=("H0",),
            observations=("baseline",),
        )
    with pytest.raises(ValidationError, match="at least 1 item"):
        RunSpec(networks=(), analyses=("H0",), observations=("baseline",))


def test_sweep_spec_rejects_unknown_sections() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SweepConfig.model_validate(
            {
                "priors": {"H0": {"type": "uniform", "low": 20.0, "high": 140.0}},
                "runs": {
                    "campaign": {
                        "networks": ["a"],
                        "analyses": ["b"],
                        "observations": ["c"],
                    }
                },
            }
        )


def test_sweep_spec_carries_no_settings_beyond_product_structure() -> None:
    """Everything scientific lives in fragments; the spec is only structure."""
    assert set(load_mapping(SWEEP_CONFIG)) == {"runs"}


def test_every_sweep_point_merges_into_a_valid_run_config() -> None:
    """The knf | astrogwb-validate-config pipeline, exercised in-process.

    Guards the invariants that used to live in the sweep models: priors must
    match sampled params, marginalized analyses need an amplitude parameter,
    and every sampled parameter needs a fiducial.
    """
    sweep = load_sweep_config(SWEEP_CONFIG)

    for point in iter_sweep_points(sweep):
        config = build_run_config(_merge(point.fragments(FRAGMENTS)))

        assert set(config.priors) == set(config.sampled_params)
        assert set(config.sampled_params) <= set(config.fiducials)
        if config.analysis.likelihood == "amplitude_marginalized":
            assert config.analysis.amplitude_parameter is not None
            assert config.amplitude_prior is not None


def test_base_priors_are_overridden_by_an_analysis_fragment() -> None:
    """Xi_0-H0-gauss replaces the shared uniform H0 prior with a normal one.

    Cross-type overrides leave the uniform's `low`/`high` in the merged table;
    `PriorSpec` drops them, so the run records the prior it actually samples.
    """
    point = SweepPoint(
        campaign="modified-propagation-all-detectors",
        network="ET-2L-aligned",
        analysis="Xi_0-H0-gauss",
        observation="baseline",
    )
    merged = _merge(point.fragments(FRAGMENTS))
    assert merged["priors"]["H0"]["low"] == 20.0  # the stale key knf leaves

    config = build_run_config(merged)

    assert config.priors["H0"].model_dump() == {
        "type": "normal",
        "loc": 67.66,
        "scale": 0.6766,
    }


def test_unused_base_priors_are_dropped_not_carried() -> None:
    """priors.toml defines every parameter; a run keeps only what it samples."""
    base_priors = load_mapping(FRAGMENTS / "priors.toml")["priors"]
    point = SweepPoint(
        campaign="cosmology",
        network="ET-2L-aligned",
        analysis="H0",
        observation="baseline",
    )

    config = build_run_config(_merge(point.fragments(FRAGMENTS)))

    assert len(base_priors) > len(config.priors)
    assert set(config.priors) == {"H0"}
