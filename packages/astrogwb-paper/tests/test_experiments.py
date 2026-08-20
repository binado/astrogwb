from __future__ import annotations

from pathlib import Path

import astrogwb_paper.config.experiments as experiments_module
import pytest
from astrogwb_paper.cli.validate_config import main as assemble_configs
from astrogwb_paper.config.experiments import (
    DEFAULT_CATALOG,
    EXPERIMENTS_PATH,
    load_base,
    load_experiments,
    overlay_for,
)
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.config.mcmc import build_run_config
from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()


def declared_runs() -> set[tuple[str, str]]:
    return {
        (name, run)
        for name, specification in load_experiments().items()
        for run in specification.runs
    }


def test_inventory_contains_five_experiments_and_24_runs() -> None:
    assert set(load_experiments()) == {
        "cosmological-parameters",
        "astrophysical-parameters",
        "modified-propagation",
        "variable-proposal-size",
        "variable-proposal-guard",
    }
    assert len(declared_runs()) == 24


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
    assert len(generated) == 24
    assert (tmp_path / "cosmological-parameters/H0-Omega_m.json").is_file()
    assert (tmp_path / "cosmological-parameters/H0-merger-rate.json").is_file()
    assert (tmp_path / "astrophysical-parameters/z_peak.json").is_file()
    assert (tmp_path / "modified-propagation/Xi_0-H0.json").is_file()
    assert (tmp_path / "variable-proposal-size/n32768.json").is_file()
    assert (tmp_path / "variable-proposal-guard/eps1e-3.json").is_file()
    guard = load_mapping(tmp_path / "variable-proposal-guard/eps1e-3.json")
    assert guard["proposal"]["uniform_mixing_fraction"] == 0.001
    assert guard["proposal"]["n_grid"] == 4096


def test_run_paths_are_one_to_one_with_the_source_yaml() -> None:
    spec = load_experiments()["cosmological-parameters"]

    assert spec.merged_config_path("ET-triangular") == Path(
        "outputs/configs/cosmological-parameters/ET-triangular.json"
    )
    assert spec.chain_path("ET-triangular") == Path(
        "outputs/chains/cosmological-parameters/ET-triangular.nc"
    )
    assert spec.chain_paths() == [
        f"outputs/chains/cosmological-parameters/{run}.nc" for run in spec.runs
    ]
    with pytest.raises(ValueError, match="unknown run cosmological-parameters/nope"):
        spec.chain_path("nope")


def test_catalog_selection_is_fixed_except_for_proposal_sweeps() -> None:
    experiments = load_experiments()
    for name, specification in experiments.items():
        if name in {"variable-proposal-size", "variable-proposal-guard"}:
            continue
        assert {specification.catalog_for(run) for run in specification.runs} == {
            DEFAULT_CATALOG
        }

    proposal = experiments["variable-proposal-size"]
    assert {run: proposal.catalog_for(run) for run in proposal.runs} == {
        "n8192": "bns-n8192-eps=0.1-df1",
        "n16384": "bns-n16384-eps=0.1-df1",
        "n32768": "bns-n32768-eps=0.1-df1",
    }

    guard = experiments["variable-proposal-guard"]
    assert {run: guard.catalog_for(run) for run in guard.runs} == {
        "eps1e-1": "bns-n16384-eps=0.1-df1",
        "eps1e-2": "bns-n16384-eps=0.01-df1",
        "eps1e-3": "bns-n16384-eps=0.001-df1",
    }


def test_a_run_naming_an_undeclared_catalog_is_rejected_at_load(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "experiments.yaml"
    inventory.write_text(
        "base: {seed: 1}\n"
        "experiments:\n"
        "  demo:\n"
        "    runs:\n"
        "      only: {catalog: bns-does-not-exist}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "run demo/only names unknown catalog 'bns-does-not-exist'; "
            "choose from injection-bns-n32768-eps=0-df1"
        ),
    ):
        load_experiments(inventory)


def test_run_config_rejects_proposal_that_disagrees_with_fiducials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = experiments_module.proposal_config

    def mismatched(recipe):
        return {**original(recipe), "H0": 70.0}

    monkeypatch.setattr(experiments_module, "proposal_config", mismatched)
    spec = load_experiments()["cosmological-parameters"]

    with pytest.raises(ValueError, match="proposal does not match run settings: H0"):
        overlay_for(spec, "ET-triangular", base=load_base())


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
