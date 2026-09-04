"""Tests for shared config loading and deep-merge overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
from config_fixtures import example_raw
from pydantic import ValidationError
from repo import REPO_ROOT

from astrogwb.paper.config.mcmc import build_run_config, prior_to_spec
from astrogwb.paper.config.runs import (
    assemble_run,
    discover_runs,
    load_base,
)
from astrogwb.paper.utils import deep_merge, load_mapping

PAPER_ROOT = REPO_ROOT


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
    assert (grid.minimum_redshift, grid.maximum_redshift, grid.n_grid) == (
        config.cosmology.minimum_redshift,
        config.cosmology.maximum_redshift,
        config.cosmology.n_grid,
    )


def test_run_config_carries_no_proposal_density() -> None:
    """The density is derived from the catalog file's provenance, not an input.

    Window equality with [cosmology] used to need a validator; it now holds by
    construction, because `resolve_proposal` is handed the run's own window.
    """
    config = build_run_config(example_raw())

    assert not hasattr(config, "proposal")
    assert "proposal" not in config.model_dump(mode="json")
    # `catalog.proposal` is the catalog, not the density -- it stays.
    assert config.catalog.proposal


def test_analysis_grid_is_not_serialized(tmp_path) -> None:
    """Derived properties stay out of normalized workflow configurations."""
    config = build_run_config(example_raw())
    assert "analysis_grid" not in config.model_dump(mode="json")
    assert "fixed_params" not in config.model_dump(mode="json")

    path = tmp_path / "run.json"
    config.save(path)
    assert "analysis_grid" not in load_mapping(path)

    reloaded = build_run_config(load_mapping(path))
    assert reloaded.analysis_grid == config.analysis_grid


def test_every_experiment_run_assembles_into_a_valid_config() -> None:
    """Every run the workflow can build must validate without a runtime.

    This replaces a check over two committed example configs. Assembling each
    experiment run is both wider coverage and the thing that actually ships.
    """
    assert "runtime" not in load_base()

    runs = discover_runs()
    assert runs, "no experiments discovered"
    for experiment, names in runs.items():
        for run in names:
            raw = assemble_run(experiment, run)
            assert "runtime" not in raw, (
                f"{experiment}/{run} declares a runtime section"
            )
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
    assert set(config.priors) == set(config.fiducials)
    # H0 is not sampled, but the model still pins its template to the fiducial.
    assert config.fixed_params["H0"] == 67.66


def test_marginalized_config_round_trips_through_save(tmp_path) -> None:
    """assemble_config writes normalized configs; run_mcmc must reload them.

    The amplitude prior lives in ``priors``, so the round trip is symmetric:
    a saved JSON reloads unchanged without any dedicated amplitude field.
    """
    config = build_run_config(_marginalized_raw())
    path = tmp_path / "run.json"
    config.save(path)

    assert "fixed_params" not in load_mapping(path)

    reloaded = build_run_config(load_mapping(path))

    # Distributions have no value equality; compare their wire-format specs.
    assert {name: prior_to_spec(prior) for name, prior in reloaded.priors.items()} == {
        name: prior_to_spec(prior) for name, prior in config.priors.items()
    }
    assert reloaded.sampled_params == config.sampled_params
    assert reloaded.fixed_params == config.fixed_params
    assert reloaded.model_dump(mode="json") == config.model_dump(mode="json")


def test_reloaded_marginalized_config_still_rejects_amplitude_parameter_sampled(
    tmp_path,
) -> None:
    config = build_run_config(_marginalized_raw())
    path = tmp_path / "run.json"
    config.save(path)
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
def test_config_rejects_a_fiducial_without_a_prior() -> None:
    raw = example_raw()
    del raw["priors"]["gamma"]

    with pytest.raises(ValidationError, match="fiducials without"):
        build_run_config(raw)


def test_config_rejects_a_prior_without_a_fiducial() -> None:
    raw = example_raw()
    raw["priors"]["unused"] = {"type": "normal", "loc": 0.0, "scale": 1.0}

    with pytest.raises(ValidationError, match="priors missing from"):
        build_run_config(raw)


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
