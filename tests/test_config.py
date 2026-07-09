"""Tests for shared config loading and deep-merge overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from astrogwb.config import deep_merge, load_mapping
from astrogwb.sampling.config import build_run_config, load_config
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
    assert base["figures"]["compare"]["var_name"] == "H0"
    assert base["networks"] == ["A", "B"]


def test_load_mapping_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported config extension"):
        load_mapping(path)


def test_build_run_config_deep_merges_extra_overrides() -> None:
    raw = load_config(REPO_ROOT / "configs/mcmc.example.toml")
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
