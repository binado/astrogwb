"""Tests for shared config loading and deep-merge overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
from astrogwb_paper.config.experiments import load_base, load_experiments, overlay_for
from astrogwb_paper.config.loading import deep_merge, load_mapping, merge_run_overlay
from astrogwb_paper.config.mcmc import (
    build_run_config,
    prior_to_spec,
    save_config,
)
from astrogwb_paper.paths import paper_project_root
from config_fixtures import example_raw
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


def test_load_mapping_resolves_yaml_aliases(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "shared: &shared\n  detectors: [S1, R1]\nrun:\n  analysis: *shared\n",
        encoding="utf-8",
    )

    assert load_mapping(path) == {
        "shared": {"detectors": ["S1", "R1"]},
        "run": {"analysis": {"detectors": ["S1", "R1"]}},
    }


def test_load_mapping_accepts_str_paths(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"seed": 7}', encoding="utf-8")

    assert load_mapping(str(path)) == {"seed": 7}


def test_build_run_config_deep_merges_extra_overrides() -> None:
    raw = example_raw()
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
    raw = example_raw()
    config = build_run_config(raw)

    assert config.analysis.detectors == ("S1", "R1", "C1")
    assert config.analysis.f_min == 2.0
    assert config.model_dump(mode="json")["analysis"]["f_max"] == 4096.0


def test_analysis_grid_mirrors_the_config() -> None:
    config = build_run_config(example_raw())
    grid = config.analysis_grid

    assert grid.observation_time == config.observation_time
    assert (grid.f_min, grid.f_max) == (config.analysis.f_min, config.analysis.f_max)
    assert (grid.z_min, grid.z_max, grid.n_grid) == (
        config.cosmology.z_min,
        config.cosmology.z_max,
        config.cosmology.n_grid,
    )


def test_analysis_grid_is_not_serialized(tmp_path) -> None:
    """`analysis_grid` is a plain property, never a computed field.

    A computed field would be written into every ``outputs/configs/*.json``
    that ``save_config`` produces, and ``extra="forbid"`` would then reject
    those files on reload -- breaking every workflow job. ``constants`` needs
    an explicit strip in ``build_run_config`` for exactly that reason; this
    guards against `analysis_grid` acquiring the same problem.
    """
    config = build_run_config(example_raw())
    assert "analysis_grid" not in config.model_dump(mode="json")

    path = tmp_path / "run.json"
    save_config(config, path)
    assert "analysis_grid" not in load_mapping(path)

    reloaded = build_run_config(load_mapping(path))
    assert reloaded.analysis_grid == config.analysis_grid


def test_every_experiment_run_assembles_into_a_valid_config() -> None:
    """Every run the workflow can build must validate without a runtime.

    This replaces a check over two committed example configs. Assembling each
    experiment run is both wider coverage and the thing that actually ships.
    """
    base = load_base()
    assert "runtime" not in base

    specs = load_experiments()
    assert specs, "no experiments discovered"
    for spec in specs.values():
        for run in spec.runs:
            raw = overlay_for(spec, run, base=base)
            assert "runtime" not in raw, f"{spec.name}/{run} declares a runtime section"
            build_run_config(raw)


def test_build_run_config_rejects_legacy_runtime_section() -> None:
    raw = example_raw()
    raw["runtime"] = {"platform": "cpu"}

    with pytest.raises(ValidationError, match="runtime"):
        build_run_config(raw)


# --------------------------------------------------------------------------- #
# Amplitude-marginalized likelihood
# --------------------------------------------------------------------------- #


def _marginalized_raw() -> dict:
    """The assembled run config, switched to marginalize H0 out entirely."""
    raw = example_raw()
    raw["analysis"] = {
        **raw["analysis"],
        "likelihood": "amplitude_marginalized",
        "amplitude_parameter": "H0",
    }
    raw["sampled_params"] = []
    return raw


def test_default_likelihood_configs_still_validate() -> None:
    raw = example_raw()
    config = build_run_config(raw)

    assert config.analysis.likelihood == "default"
    assert config.analysis.amplitude_parameter is None


def test_marginalized_config_keeps_amplitude_prior_in_priors() -> None:
    config = build_run_config(_marginalized_raw())

    assert config.analysis.amplitude_parameter == "H0"
    assert "H0" not in config.sampled_params
    assert prior_to_spec(config.priors["H0"]) == {
        "type": "uniform",
        "low": 20.0,
        "high": 140.0,
    }
    # Invariant: priors = sampled params + the marginalized amplitude parameter.
    assert set(config.priors) == set(config.sampled_params) | {"H0"}
    # H0 is not sampled, but it is still a fiducial constant the model pins to.
    assert config.constants["H0"] == 67.66


def test_marginalized_config_round_trips_through_save_config(tmp_path) -> None:
    """assemble_config writes normalized configs; run_mcmc must reload them.

    The amplitude prior lives in ``priors``, so the round trip is symmetric:
    a saved JSON reloads unchanged without any dedicated amplitude field.
    """
    config = build_run_config(_marginalized_raw())
    path = tmp_path / "run.json"
    save_config(config, path)

    # `constants` is a computed field: present in the dump (keep save_config
    # canonical), stripped on input by build_run_config.
    assert "constants" in load_mapping(path)

    reloaded = build_run_config(load_mapping(path))

    # Distributions have no value equality; compare their wire-format specs.
    assert {name: prior_to_spec(prior) for name, prior in reloaded.priors.items()} == {
        name: prior_to_spec(prior) for name, prior in config.priors.items()
    }
    assert reloaded.sampled_params == config.sampled_params
    assert reloaded.constants == config.constants
    assert reloaded.model_dump(mode="json") == config.model_dump(mode="json")


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
    # The assembled base gives every fiducial a prior; drop one to create the
    # amplitude-parameter-without-a-prior case this test is about.
    del raw["priors"]["local_merger_rate"]
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
    raw = example_raw()
    raw["analysis"] = {**raw["analysis"], "amplitude_parameter": "H0"}

    with pytest.raises(ValidationError, match="only valid when"):
        build_run_config(raw)


# --------------------------------------------------------------------------- #
# Prior specs
# --------------------------------------------------------------------------- #
def test_merge_run_overlay_replaces_named_priors_wholesale() -> None:
    raw = example_raw()
    merged = merge_run_overlay(
        raw,
        {"priors": {"H0": {"type": "normal", "loc": 67.66, "scale": 0.6766}}},
    )

    assert merged["priors"]["H0"] == {
        "type": "normal",
        "loc": 67.66,
        "scale": 0.6766,
    }
    config = build_run_config(merged)
    assert prior_to_spec(config.priors["H0"]) == merged["priors"]["H0"]


def test_prior_spec_rejects_stale_keys_from_a_cross_type_override() -> None:
    raw = example_raw()
    polluted = deep_merge(
        raw,
        {"priors": {"H0": {"type": "normal", "loc": 67.66, "scale": 0.6766}}},
    )
    assert polluted["priors"]["H0"]["low"] == 20.0

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        build_run_config(polluted)


def test_prior_spec_rejects_unsupported_type_before_jax_starts() -> None:
    raw = example_raw()
    raw["priors"]["H0"] = {"type": "lognormal", "loc": 1.0, "scale": 1.0}

    with pytest.raises(ValidationError, match="does not match any of the expected"):
        build_run_config(raw)


def test_prior_spec_rejects_missing_required_key() -> None:
    raw = example_raw()
    raw["priors"]["H0"] = {"type": "normal", "loc": 67.66, "scal": 0.6766}

    with pytest.raises(ValidationError, match="scale"):
        build_run_config(raw)
