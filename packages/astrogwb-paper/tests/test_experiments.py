from __future__ import annotations

from pathlib import Path

from astrogwb_paper.config.experiments import (
    DEFAULT_CATALOG,
    chain_path,
    config_path,
    load_experiments,
    merged_config_path,
    overlay_for,
    sidecar_path,
)
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.config.mcmc import build_run_config
from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()
BASE_CONFIG = PAPER_ROOT / "inputs/mcmc.base.toml"
EXPERIMENTS_DIR = PAPER_ROOT / "experiments"


def declared_runs() -> set[tuple[str, str]]:
    return {
        (name, run)
        for name, specification in load_experiments().items()
        for run in specification.runs
    }


def test_inventory_contains_seven_experiments_and_22_runs() -> None:
    assert set(load_experiments()) == {
        "H0-all-detectors",
        "modified-propagation-all-detectors",
        "H0-merger-rate",
        "H0-omega-m",
        "astrophysical-parameters",
        "star-formation-peak",
        "variable-injection-size",
    }
    assert len(declared_runs()) == 22


def test_declared_run_files_exactly_match_the_inventory() -> None:
    discovered = {path.stem for path in EXPERIMENTS_DIR.glob("*.toml")}

    assert discovered == set(load_experiments())
    assert list(EXPERIMENTS_DIR.glob("*/*.toml")) == []


def test_every_base_and_run_merge_is_a_valid_run_config() -> None:
    base = load_mapping(BASE_CONFIG)

    for specification in load_experiments().values():
        for run in specification.runs:
            config = build_run_config(overlay_for(specification, run, base=base))

            assert set(config.priors) == set(config.sampled_params)
            assert set(config.sampled_params) <= set(config.fiducials)
            if config.analysis.likelihood == "amplitude_marginalized":
                assert config.analysis.amplitude_parameter is not None
                assert config.amplitude_prior is not None


def test_run_paths_are_one_to_one_with_the_source_toml() -> None:
    assert config_path("H0-all-detectors", "ET-triangular") == Path(
        "experiments/H0-all-detectors.toml"
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
    experiments = load_experiments()
    for name, specification in experiments.items():
        if name == "variable-injection-size":
            continue
        assert {specification.catalog_for(run) for run in specification.runs} == {
            DEFAULT_CATALOG
        }

    injection = experiments["variable-injection-size"]
    assert {run: injection.catalog_for(run).name for run in injection.runs} == {
        "n8192": "bns-n8192-df1.h5",
        "n16384": "bns-n16384-df1.h5",
        "n32768": "bns-n32768-df1.h5",
    }


def test_cross_type_prior_override_replaces_the_table() -> None:
    spec = load_experiments()["modified-propagation-all-detectors"]
    base = load_mapping(BASE_CONFIG)
    merged = overlay_for(spec, "Xi_0-H0", base=base)

    assert merged["priors"]["H0"] == {
        "type": "normal",
        "loc": 67.66,
        "scale": 0.6766,
    }
    assert "low" not in merged["priors"]["H0"]
    assert "high" not in merged["priors"]["H0"]
