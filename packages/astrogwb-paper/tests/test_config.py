"""Tests for shared config loading and deep-merge overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
from astrogwb_paper.config.loading import deep_merge, load_mapping
from astrogwb_paper.config.mcmc import build_run_config, config_sha256, save_config
from astrogwb_paper.paths import paper_project_root
from pydantic import ValidationError

PAPER_ROOT = paper_project_root()


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
    raw = load_mapping(PAPER_ROOT / "configs/mcmc.example.toml")
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
    raw = load_mapping(PAPER_ROOT / "configs/mcmc.example.toml")
    config = build_run_config(raw)

    assert config.analysis.detectors == ("S1", "R1", "C1")
    assert config.analysis.f_min == 2.0
    assert config.model_dump(mode="json")["analysis"]["f_max"] == 4096.0


def test_all_committed_mcmc_configs_validate_without_runtime() -> None:
    config_paths = [
        PAPER_ROOT / "configs/mcmc.example.toml",
        PAPER_ROOT / "configs/mcmc.cosmology.toml",
        *(PAPER_ROOT / "configs/mcmc/curated").glob("*/*.json"),
    ]

    for path in config_paths:
        raw = load_mapping(path)
        assert "runtime" not in raw
        build_run_config(raw)


def test_build_run_config_rejects_legacy_runtime_section() -> None:
    raw = load_mapping(PAPER_ROOT / "configs/mcmc.example.toml")
    raw["runtime"] = {"platform": "cpu"}

    with pytest.raises(ValidationError, match="runtime"):
        build_run_config(raw)


def test_config_sha256_stable_across_key_order_and_sensitive_to_values() -> None:
    raw = load_mapping(PAPER_ROOT / "configs/mcmc.example.toml")
    reordered = dict(reversed(list(raw.items())))

    assert config_sha256(build_run_config(raw)) == config_sha256(
        build_run_config(reordered)
    )
    assert config_sha256(build_run_config(raw)) != config_sha256(
        build_run_config(raw, seed=99)
    )


def test_config_sha256_excludes_output_routing() -> None:
    raw = load_mapping(PAPER_ROOT / "configs/mcmc.example.toml")

    baseline = build_run_config(raw)
    routed = build_run_config(
        raw,
        outdir=Path("chains/another-catalog/campaign"),
        label="another-run",
    )

    assert config_sha256(baseline) == config_sha256(routed)


# --------------------------------------------------------------------------- #
# Amplitude-marginalized likelihood
# --------------------------------------------------------------------------- #


def _marginalized_raw() -> dict:
    """The committed example config, switched to marginalize H0 out entirely."""
    raw = load_mapping(PAPER_ROOT / "configs/mcmc.example.toml")
    raw["analysis"] = {
        **raw["analysis"],
        "likelihood": "amplitude_marginalized",
        "amplitude_parameter": "H0",
    }
    raw["sampled_params"] = []
    return raw


def test_default_likelihood_configs_still_validate() -> None:
    raw = load_mapping(PAPER_ROOT / "configs/mcmc.example.toml")
    config = build_run_config(raw)

    assert config.analysis.likelihood == "default"
    assert config.analysis.amplitude_parameter is None
    assert config.amplitude_prior is None


def test_marginalized_config_extracts_amplitude_prior_and_preserves_invariant() -> None:
    config = build_run_config(_marginalized_raw())

    assert config.analysis.amplitude_parameter == "H0"
    assert config.amplitude_prior == {"type": "uniform", "low": 20.0, "high": 140.0}
    assert "H0" not in config.priors
    assert set(config.priors) == set(config.sampled_params)
    # H0 is not sampled, but it is still a fiducial constant the model pins to.
    assert config.constants["H0"] == 67.66


def test_marginalized_config_round_trips_through_save_config(tmp_path) -> None:
    """generate_configs() writes normalized configs; run_mcmc must reload them.

    Regression guard: the validator pops the amplitude prior out of ``priors``,
    so a saved JSON carries it only under ``amplitude_prior`` and used to be
    rejected on reload as "needs a [priors.*] table".
    """
    config = build_run_config(_marginalized_raw())
    path = tmp_path / "run.json"
    save_config(config, path)

    reloaded = build_run_config(load_mapping(path))

    assert reloaded.amplitude_prior == config.amplitude_prior
    assert reloaded.priors == config.priors
    assert reloaded.sampled_params == config.sampled_params
    assert reloaded.constants == config.constants
    assert config_sha256(reloaded) == config_sha256(config)


def test_reloaded_marginalized_config_still_rejects_amplitude_parameter_sampled(
    tmp_path,
) -> None:
    config = build_run_config(_marginalized_raw())
    path = tmp_path / "run.json"
    save_config(config, path)
    raw = load_mapping(path)
    raw["sampled_params"] = [*raw["sampled_params"], "H0"]

    with pytest.raises(ValidationError, match="cannot also appear in sampled_params"):
        build_run_config(raw)


def test_marginalized_config_rejects_amplitude_parameter_also_sampled() -> None:
    raw = _marginalized_raw()
    raw["sampled_params"] = ["H0"]

    with pytest.raises(ValidationError, match="cannot also appear in sampled_params"):
        build_run_config(raw)


def test_marginalized_config_rejects_amplitude_parameter_without_prior_table() -> None:
    raw = _marginalized_raw()
    raw["analysis"]["amplitude_parameter"] = "local_merger_rate"

    with pytest.raises(ValidationError, match=r"needs a \[priors\.\*\] table"):
        build_run_config(raw)


def test_marginalized_config_rejects_amplitude_parameter_missing_fiducial() -> None:
    raw = _marginalized_raw()
    raw["analysis"]["amplitude_parameter"] = "local_merger_rate"
    raw["priors"] = {
        **raw["priors"],
        "local_merger_rate": {"type": "uniform", "low": 50.0, "high": 300.0},
    }
    del raw["fiducials"]["local_merger_rate"]

    with pytest.raises(ValidationError, match=r"missing from \[fiducials\]"):
        build_run_config(raw)


def test_marginalized_config_rejects_unsupported_amplitude_parameter_name() -> None:
    raw = _marginalized_raw()
    raw["analysis"]["amplitude_parameter"] = "Omega_m"
    raw["priors"] = {
        **raw["priors"],
        "Omega_m": {"type": "uniform", "low": 0.05, "high": 0.95},
    }

    with pytest.raises(ValidationError):
        build_run_config(raw)


def test_default_likelihood_rejects_amplitude_parameter() -> None:
    raw = load_mapping(PAPER_ROOT / "configs/mcmc.example.toml")
    raw["analysis"] = {**raw["analysis"], "amplitude_parameter": "H0"}

    with pytest.raises(ValidationError, match="only valid when"):
        build_run_config(raw)
