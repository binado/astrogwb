"""Tests for shared config loading and deep-merge overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from astrogwb.config.loading import deep_merge, load_mapping
from astrogwb.config.mcmc import build_run_config, config_sha256
from astrogwb.utils import repo_root

REPO_ROOT = repo_root()


def test_deep_merge_nested_dicts_and_list_replacement() -> None:
    base = {
        "seed": 1,
        "figures": {"compare": {"var_name": "H0", "figure_dpi": 300}},
        "networks": ["A", "B"],
    }
    override = {
        "figures": {"compare": {"var_name": "Omega_m"}},
        "networks": ["C"],
    }

    merged = deep_merge(base, override)

    assert merged == {
        "seed": 1,
        "figures": {"compare": {"var_name": "Omega_m", "figure_dpi": 300}},
        "networks": ["C"],
    }
    # Inputs are not mutated.
    figures = base["figures"]
    assert isinstance(figures, dict)
    compare = figures["compare"]
    assert isinstance(compare, dict)
    assert compare["var_name"] == "H0"
    assert base["networks"] == ["A", "B"]


def test_load_mapping_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported config extension"):
        load_mapping(path)


def test_build_run_config_deep_merges_extra_overrides() -> None:
    raw = load_mapping(REPO_ROOT / "configs/mcmc.example.toml")
    config = build_run_config(
        raw,
        seed=99,
        sampler={"num_warmup": 11, "num_samples": 13},
    )

    assert config.seed == 99
    assert config.sampler.num_warmup == 11
    assert config.sampler.num_samples == 13
    # Unrelated sampler fields keep their file values.
    assert config.sampler.target_accept == raw["sampler"]["target_accept"]


def test_analysis_settings_round_trip() -> None:
    raw = load_mapping(REPO_ROOT / "configs/mcmc.example.toml")
    config = build_run_config(raw)

    assert config.analysis.detectors == ("S1", "R1", "C1")
    assert config.analysis.f_min == 2.0
    assert config.model_dump(mode="json")["analysis"]["f_max"] == 4096.0


def test_curated_configs_are_catalog_independent() -> None:
    config_paths = list((REPO_ROOT / "configs/mcmc/curated").glob("*/*.json"))

    assert config_paths
    for path in config_paths:
        raw = load_mapping(path)
        assert "catalog" not in raw
        assert build_run_config(raw).analysis.detectors


def test_all_committed_mcmc_configs_validate_without_runtime() -> None:
    config_paths = [
        REPO_ROOT / "configs/mcmc.example.toml",
        REPO_ROOT / "configs/mcmc.cosmology.toml",
        *(REPO_ROOT / "configs/mcmc/curated").glob("*/*.json"),
    ]

    for path in config_paths:
        raw = load_mapping(path)
        assert "runtime" not in raw
        build_run_config(raw)


def test_build_run_config_rejects_legacy_runtime_section() -> None:
    raw = load_mapping(REPO_ROOT / "configs/mcmc.example.toml")
    raw["runtime"] = {"platform": "cpu"}

    with pytest.raises(ValidationError, match="runtime"):
        build_run_config(raw)


def test_config_sha256_stable_across_key_order_and_sensitive_to_values() -> None:
    raw = load_mapping(REPO_ROOT / "configs/mcmc.example.toml")
    reordered = dict(reversed(list(raw.items())))

    assert config_sha256(build_run_config(raw)) == config_sha256(
        build_run_config(reordered)
    )
    assert config_sha256(build_run_config(raw)) != config_sha256(
        build_run_config(raw, seed=99)
    )


def test_config_sha256_excludes_output_routing() -> None:
    raw = load_mapping(REPO_ROOT / "configs/mcmc.example.toml")

    baseline = build_run_config(raw)
    routed = build_run_config(
        raw,
        outdir=Path("chains/another-catalog/campaign"),
        label="another-run",
    )

    assert config_sha256(baseline) == config_sha256(routed)
