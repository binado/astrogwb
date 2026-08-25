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


def test_inventory_contains_six_experiments_and_26_runs() -> None:
    assert set(load_experiments()) == {
        "cosmological-parameters",
        "astrophysical-parameters",
        "modified-propagation",
        "variable-catalog-size",
        "variable-proposal-guard",
        "waveform-approximant",
    }
    assert len(declared_runs()) == 26


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


def test_1d_runs_use_diagonal_mass_matrix_and_short_warmup() -> None:
    """1D problems (one latent, marginalized amplitude included) need no dense
    mass matrix and converge faster: NUTS is configured accordingly."""
    base = load_base()

    for specification in load_experiments().values():
        for run in specification.runs:
            config = build_run_config(overlay_for(specification, run, base=base))
            if len(config.sampled_params) != 1:
                continue
            assert config.sampler.num_warmup == 500, f"{specification.name}/{run}"
            assert config.sampler.dense_mass is False, f"{specification.name}/{run}"


def test_all_run_configs_are_assembled_together(tmp_path: Path) -> None:
    assemble_configs(
        [
            str(PAPER_ROOT / EXPERIMENTS_PATH),
            "--output-dir",
            str(tmp_path),
        ]
    )

    generated = list(tmp_path.glob("*/*.json"))
    assert len(generated) == 26
    assert (tmp_path / "cosmological-parameters/H0-Omega_m.json").is_file()
    assert (tmp_path / "cosmological-parameters/H0-merger-rate.json").is_file()
    assert (tmp_path / "astrophysical-parameters/z_peak.json").is_file()
    assert (tmp_path / "modified-propagation/Xi_0-H0.json").is_file()
    assert (tmp_path / "variable-catalog-size/n32768.json").is_file()
    assert (tmp_path / "variable-proposal-guard/eps1e-3.json").is_file()
    assert (tmp_path / "waveform-approximant/IMRPhenom.json").is_file()
    assert (tmp_path / "waveform-approximant/TaylorF2.json").is_file()
    guard = load_mapping(tmp_path / "variable-proposal-guard/eps1e-3.json")
    assert guard["proposal"]["uniform_mixing_fraction"] == 0.001
    assert guard["proposal"]["n_grid"] == 4096
    assert guard["sampled_params"] == ["gamma", "kappa", "z_peak"]
    assert guard["analysis"]["likelihood"] == "amplitude_marginalized"
    assert guard["analysis"]["amplitude_parameter"] == "H0"


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


def test_catalog_selection_is_fixed_except_for_proposal_comparisons() -> None:
    experiments = load_experiments()
    for name in ("cosmological-parameters", "modified-propagation"):
        specification = experiments[name]
        assert {specification.catalog_for(run) for run in specification.runs} == {
            DEFAULT_CATALOG
        }

    astrophysical = experiments["astrophysical-parameters"]
    assert {astrophysical.catalog_for(run) for run in astrophysical.runs} == {
        "bns-n16384-eps=0.1-df1"
    }

    catalog_size = experiments["variable-catalog-size"]
    assert {run: catalog_size.catalog_for(run) for run in catalog_size.runs} == {
        "n8192": "bns-n8192-eps=0-df1",
        "n16384": "bns-n16384-eps=0-df1",
        "n32768": "bns-n32768-eps=0-df1",
    }

    guard = experiments["variable-proposal-guard"]
    assert {run: guard.catalog_for(run) for run in guard.runs} == {
        "eps1e-1": "bns-n16384-eps=0.1-df1",
        "eps1e-2": "bns-n16384-eps=0.01-df1",
        "eps1e-3": "bns-n16384-eps=0.001-df1",
    }

    approximants = experiments["waveform-approximant"]
    assert {run: approximants.catalog_for(run) for run in approximants.runs} == {
        "IMRPhenom": "injection-bns-n32768-eps=0-df1",
        "TaylorF2": "bns-n32768-eps=0-df1-taylorf2",
    }


def test_variable_proposal_guard_samples_h0_md() -> None:
    """Guard sweeps epsilon on the H0-marginalized Madau–Dickinson problem."""
    base = load_base()
    spec = load_experiments()["variable-proposal-guard"]

    for run in spec.runs:
        config = build_run_config(overlay_for(spec, run, base=base))
        assert config.sampled_params == ("gamma", "kappa", "z_peak"), run
        assert config.analysis.likelihood == "amplitude_marginalized"
        assert config.analysis.amplitude_parameter == "H0"
        assert config.posterior_params == ("gamma", "kappa", "z_peak", "H0")
        assert config.sampler.dense_mass is True
        assert config.sampler.num_warmup == 1000


def test_only_astrophysical_and_guard_runs_use_a_mixed_proposal() -> None:
    base = load_base()
    mixed = {"astrophysical-parameters", "variable-proposal-guard"}

    for specification in load_experiments().values():
        for run in specification.runs:
            config = build_run_config(overlay_for(specification, run, base=base))
            epsilon = config.proposal.uniform_mixing_fraction
            if specification.name in mixed:
                assert epsilon > 0.0, f"{specification.name}/{run}"
            else:
                assert epsilon == 0.0, f"{specification.name}/{run}"


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
    original = experiments_module.analysis_proposal_config

    def mismatched(recipe, *, minimum_redshift, maximum_redshift):
        return {
            **original(
                recipe,
                minimum_redshift=minimum_redshift,
                maximum_redshift=maximum_redshift,
            ),
            "H0": 70.0,
        }

    monkeypatch.setattr(experiments_module, "analysis_proposal_config", mismatched)
    spec = load_experiments()["cosmological-parameters"]

    with pytest.raises(ValueError, match="proposal does not match run settings: H0"):
        overlay_for(spec, "ET-triangular", base=load_base())


def test_analysis_window_outside_generation_support_is_rejected() -> None:
    spec = load_experiments()["cosmological-parameters"]

    below = load_base()
    below["cosmology"]["minimum_redshift"] = -0.1
    with pytest.raises(ValueError, match="must lie within the catalog generation"):
        overlay_for(spec, "ET-triangular", base=below)

    above = load_base()
    above["cosmology"]["maximum_redshift"] = 21.0
    with pytest.raises(ValueError, match="must lie within the catalog generation"):
        overlay_for(spec, "ET-triangular", base=above)


def test_analysis_window_equal_to_generation_support_is_accepted() -> None:
    base = load_base()
    base["cosmology"]["minimum_redshift"] = 0.0
    spec = load_experiments()["cosmological-parameters"]

    raw = overlay_for(spec, "ET-triangular", base=base)

    assert raw["cosmology"]["minimum_redshift"] == 0.0
    assert raw["proposal"]["minimum_redshift"] == 0.0


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
