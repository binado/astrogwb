from __future__ import annotations

from pathlib import Path

from astrogwb_paper.cli.validate_config import main as assemble_configs
from astrogwb_paper.config.experiments import (
    DEFAULT_CATALOG,
    EXPERIMENTS_PATH,
    chain_path,
    config_path,
    load_base,
    load_experiments,
    merged_config_path,
    overlay_for,
    sidecar_path,
)
from astrogwb_paper.config.mcmc import build_run_config
from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()


def declared_runs() -> set[tuple[str, str]]:
    return {
        (name, run)
        for name, specification in load_experiments().items()
        for run in specification.runs
    }


def test_inventory_contains_four_experiments_and_21_runs() -> None:
    assert set(load_experiments()) == {
        "cosmological-parameters",
        "astrophysical-parameters",
        "modified-propagation",
        "variable-injection-size",
    }
    assert len(declared_runs()) == 21


def test_single_yaml_is_the_only_mcmc_inventory() -> None:
    assert (PAPER_ROOT / EXPERIMENTS_PATH).is_file()
    assert not (PAPER_ROOT / "inputs/config.yaml").exists()
    assert not (PAPER_ROOT / "inputs/mcmc.base.toml").exists()
    assert list((PAPER_ROOT / "experiments").glob("**/*.toml")) == []


def test_every_base_and_run_merge_is_a_valid_run_config() -> None:
    base = load_base()

    for specification in load_experiments().values():
        for run in specification.runs:
            config = build_run_config(overlay_for(specification, run, base=base))

            amplitude_parameter = config.analysis.amplitude_parameter
            amplitude_priors = {amplitude_parameter} if amplitude_parameter else set()
            assert set(config.priors) == set(config.sampled_params) | amplitude_priors
            assert set(config.sampled_params) <= set(config.fiducials)
            if config.analysis.likelihood == "amplitude_marginalized":
                assert amplitude_parameter is not None


def test_all_run_configs_are_assembled_together(tmp_path: Path) -> None:
    assemble_configs(
        [
            str(PAPER_ROOT / EXPERIMENTS_PATH),
            "--output-dir",
            str(tmp_path),
        ]
    )

    generated = list(tmp_path.glob("*/*.json"))
    assert len(generated) == 21
    assert (tmp_path / "cosmological-parameters/H0-Omega_m.json").is_file()
    assert (tmp_path / "cosmological-parameters/H0-merger-rate.json").is_file()
    assert (tmp_path / "astrophysical-parameters/z_peak.json").is_file()
    assert (tmp_path / "modified-propagation/Xi_0-H0.json").is_file()
    assert (tmp_path / "variable-injection-size/n32768.json").is_file()


def test_run_paths_are_one_to_one_with_the_source_yaml() -> None:
    assert config_path("cosmological-parameters", "ET-triangular") == EXPERIMENTS_PATH
    assert merged_config_path("cosmological-parameters", "ET-triangular") == Path(
        "outputs/configs/cosmological-parameters/ET-triangular.json"
    )
    assert chain_path("cosmological-parameters", "ET-triangular") == Path(
        "outputs/chains/cosmological-parameters/ET-triangular.nc"
    )
    assert sidecar_path("cosmological-parameters", "ET-triangular") == Path(
        "outputs/chains/cosmological-parameters/ET-triangular.json"
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
    spec = load_experiments()["modified-propagation"]
    base = load_base()
    merged = overlay_for(spec, "Xi_0-H0", base=base)

    assert merged["priors"]["H0"] == {
        "type": "normal",
        "loc": 67.66,
        "scale": 0.6766,
    }
    assert "low" not in merged["priors"]["H0"]
    assert "high" not in merged["priors"]["H0"]
