"""Tests for shared config loading and deep-merge overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
from config_fixtures import example_raw
from pydantic import ValidationError
from repo import REPO_ROOT

from astrogwb.constants import ISCO_ALPHA
from astrogwb.paper.config import waveform_generator
from astrogwb.paper.config.mcmc import build_run_config, prior_to_spec
from astrogwb.paper.config.runs import (
    assemble_run,
    discover_runs,
    load_base,
)
from astrogwb.paper.utils import deep_merge, load_mapping
from astrogwb.waveform import AnalyticInspiralGenerator, RippleGenerator

PAPER_ROOT = REPO_ROOT


def test_deep_merge_nested_dicts_and_list_replacement() -> None:
    # `detector_ids` is a stand-in list key, deliberately not named `networks`:
    # that is a real config section now, and it is a *mapping*, so reusing the
    # name here would advertise the wrong merge rule for it.
    base = {
        "seed": 1,
        "figures": {"compare": {"var_name": "H0", "figure_dpi": 300}},
        "detector_ids": ["A", "B"],
    }
    override = {
        "figures": {"compare": {"var_name": "Omega_m"}},
        "detector_ids": ["C"],
    }

    merged = deep_merge(base, override)

    assert merged == {
        "seed": 1,
        "figures": {"compare": {"var_name": "Omega_m", "figure_dpi": 300}},
        "detector_ids": ["C"],
    }
    # Inputs are not mutated.
    figures = base["figures"]
    assert isinstance(figures, dict)
    compare = figures["compare"]
    assert isinstance(compare, dict)
    assert compare["var_name"] == "H0"
    assert base["detector_ids"] == ["A", "B"]


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
    """The density is the proposal catalog's own record, not a config input.

    Window equality with [cosmology] used to need a validator; it now holds by
    construction, because `PolarizationPowerCatalog.restrict_redshift` is
    handed the run's own
    window and moves the samples and the recorded density together.
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
        "local_merger_rate": {
            "dist": "Uniform",
            "kwargs": {"low": 50.0, "high": 300.0},
        },
    }
    del raw["fiducials"]["local_merger_rate"]

    with pytest.raises(ValidationError, match=r"missing from \[fiducials\]"):
        build_run_config(raw)


def test_marginalized_config_rejects_unsupported_amplitude_parameter_name() -> None:
    raw = _marginalized_raw()
    raw["analysis"]["amplitude_parameter"] = "Omega_m"
    raw["priors"] = {
        **raw["priors"],
        "Omega_m": {"dist": "Uniform", "kwargs": {"low": 0.05, "high": 0.95}},
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
    raw["priors"]["unused"] = {"dist": "Normal", "kwargs": {"loc": 0.0, "scale": 1.0}}

    with pytest.raises(ValidationError, match="priors missing from"):
        build_run_config(raw)


def test_prior_spec_rejects_stale_keys_from_a_cross_type_override() -> None:
    raw = example_raw()
    polluted = deep_merge(
        raw,
        {
            "priors": {
                "H0": {"dist": "Normal", "kwargs": {"loc": 67.66, "scale": 0.6766}}
            }
        },
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        build_run_config(polluted)


def test_prior_spec_rejects_an_unknown_distribution_name() -> None:
    raw = example_raw()
    raw["priors"]["H0"] = {"dist": "Lognormal", "kwargs": {"loc": 1.0, "scale": 1.0}}

    with pytest.raises(ValidationError, match="not a numpyro distribution"):
        build_run_config(raw)


def test_prior_spec_rejects_a_name_that_is_not_a_distribution() -> None:
    """The guard on the `getattr`, not just the lookup.

    `dist` is a config-supplied string indexed into a live module, so a name
    that resolves to something other than a Distribution subclass -- here the
    `constraints` submodule -- must be refused rather than called.
    """
    raw = example_raw()
    raw["priors"]["H0"] = {"dist": "constraints", "kwargs": {}}

    with pytest.raises(ValidationError, match="not a numpyro distribution"):
        build_run_config(raw)


def test_prior_spec_rejects_missing_required_key() -> None:
    raw = example_raw()
    raw["priors"]["H0"] = {"dist": "Normal", "kwargs": {"loc": 67.66, "scal": 0.6766}}

    with pytest.raises(ValidationError, match="scale"):
        build_run_config(raw)


def test_waveform_generator_defaults_to_the_committed_ripple() -> None:
    generator = waveform_generator(REPO_ROOT)

    assert isinstance(generator, RippleGenerator)
    assert generator.approximant == "IMRPhenomXAS_NRTidalv3"
    assert generator.minimum_frequency == 2.0
    assert generator.maximum_frequency == 2048.0


def test_waveform_generator_kwargs_select_the_analytical_inspiral() -> None:
    generator = waveform_generator(REPO_ROOT, approximant="analytical")

    assert isinstance(generator, AnalyticInspiralGenerator)
    assert generator.approximant == "analytical"
    assert generator.alpha == ISCO_ALPHA
    assert generator.minimum_frequency == 2.0
