from __future__ import annotations

from pathlib import Path

from astrogwb_paper.config.experiments import (
    DEFAULT_CATALOG,
    EXPERIMENTS,
    chain_path,
    config_path,
    merged_config_path,
    sidecar_path,
)
from astrogwb_paper.config.loading import deep_merge, load_mapping
from astrogwb_paper.config.mcmc import build_run_config
from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()
BASE_CONFIG = PAPER_ROOT / "inputs/mcmc.base.toml"


def declared_runs() -> set[tuple[str, str]]:
    return {
        (experiment.name, run)
        for experiment in EXPERIMENTS.values()
        for run in experiment.runs
    }


def test_inventory_contains_seven_experiments_and_22_runs() -> None:
    assert list(EXPERIMENTS) == [
        "H0-all-detectors",
        "modified-propagation-all-detectors",
        "H0-merger-rate",
        "H0-omega-m",
        "astrophysical-parameters",
        "star-formation-peak",
        "variable-injection-size",
    ]
    assert len(declared_runs()) == 22


def test_declared_run_files_exactly_match_the_inventory() -> None:
    discovered = {
        (
            path.parent.name,
            path.name.removeprefix("mcmc.").removesuffix(".toml"),
        )
        for path in (PAPER_ROOT / "experiments").glob("*/mcmc.*.toml")
    }

    assert discovered == declared_runs()


def test_every_base_and_run_merge_is_a_valid_run_config() -> None:
    base = load_mapping(BASE_CONFIG)

    for experiment_name, run in declared_runs():
        path = PAPER_ROOT / config_path(experiment_name, run)
        config = build_run_config(deep_merge(base, load_mapping(path)))

        assert set(config.priors) == set(config.sampled_params)
        assert set(config.sampled_params) <= set(config.fiducials)
        if config.analysis.likelihood == "amplitude_marginalized":
            assert config.analysis.amplitude_parameter is not None
            assert config.amplitude_prior is not None


def test_run_paths_are_one_to_one_with_the_source_toml() -> None:
    assert config_path("H0-all-detectors", "ET-triangular") == Path(
        "experiments/H0-all-detectors/mcmc.ET-triangular.toml"
    )
    assert merged_config_path("H0-all-detectors", "ET-triangular") == Path(
        "outputs/configs/H0-all-detectors/ET-triangular.json"
    )
    assert chain_path("H0-all-detectors", "ET-triangular") == Path(
        "outputs/chains/H0-all-detectors/ET-triangular.nc"
    )
    assert sidecar_path("H0-all-detectors", "ET-triangular") == Path(
        "outputs/chains/H0-all-detectors/ET-triangular.json"
    )


def test_catalog_selection_is_fixed_except_for_injection_size() -> None:
    for name, experiment in EXPERIMENTS.items():
        if name == "variable-injection-size":
            continue
        assert {experiment.catalog_for(run) for run in experiment.runs} == {
            DEFAULT_CATALOG
        }

    injection = EXPERIMENTS["variable-injection-size"]
    assert {run: injection.catalog_for(run).name for run in injection.runs} == {
        "n8192": "bns-n8192-df1.h5",
        "n16384": "bns-n16384-df1.h5",
        "n32768": "bns-n32768-df1.h5",
    }
