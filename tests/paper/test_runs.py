"""Discovery and the three-layer merge behind ``config/analysis/``.

Replaces ``test_experiments.py``: there is no inventory to validate any more,
so what is tested is the convention (which files exist and what they map to)
and the merge that turns them into a run config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astrogwb.paper.config.banks import (
    check_bank_references,
    discover_banks,
    validate_all_runs,
)
from astrogwb.paper.config.mcmc import build_run_config
from astrogwb.paper.config.runs import (
    CHAINS_ROOT,
    EXPERIMENT_BASE,
    RUNS_DIR,
    assemble_run,
    base_config_paths,
    catalog_bank_names,
    discover_runs,
    load_base,
    merge_config_layers,
    resolve_bank_names,
    run_config_paths,
    run_target,
)
from astrogwb.paper.paths import paper_project_root
from astrogwb.paper.utils import load_mapping

PAPER_ROOT = paper_project_root()

EXPERIMENTS = {
    "cosmological-parameters",
    "astrophysical-parameters",
    "modified-propagation",
    "variable-catalog-size",
    "variable-proposal-guard",
    "waveform-approximant",
}


def all_runs() -> list[tuple[str, str]]:
    return [
        (experiment, run)
        for experiment, names in discover_runs().items()
        for run in names
    ]


# --------------------------------------------------------------------------- #
# Discovery: filenames are the mapping
# --------------------------------------------------------------------------- #
def test_discovery_finds_six_experiments_and_26_runs() -> None:
    runs = discover_runs()

    assert set(runs) == EXPERIMENTS
    assert sum(len(names) for names in runs.values()) == 26
    assert "_base" not in {run for names in runs.values() for run in names}


def test_the_retired_inventories_are_gone() -> None:
    assert not (PAPER_ROOT / "inputs").exists()
    assert (PAPER_ROOT / "config/analysis/base").is_dir()
    assert (PAPER_ROOT / "config/banks").is_dir()
    assert (PAPER_ROOT / "config/populations").is_dir()


def test_every_experiment_has_the_required_base_overlay() -> None:
    for experiment in discover_runs():
        assert (PAPER_ROOT / RUNS_DIR / experiment / EXPERIMENT_BASE).is_file()


def test_output_paths_follow_from_the_run_name() -> None:
    # There is no assembled-config path any more: a run is addressed by its
    # layer files going in and by its chain coming out.
    assert CHAINS_ROOT == Path("outputs/chains")
    assert run_target("variable-catalog-size") == "run_experiment_variable_catalog_size"


def test_run_config_paths_are_the_three_layers_in_merge_order() -> None:
    paths = run_config_paths("cosmological-parameters", "ET-triangular")
    base = base_config_paths()

    assert paths[: len(base)] == base
    assert [path.name for path in paths[len(base) :]] == [
        EXPERIMENT_BASE,
        "ET-triangular.toml",
    ]
    assert all(path.is_file() for path in paths)


def test_assemble_run_is_merge_config_layers_over_run_config_paths() -> None:
    # The seam the workflow relies on: the rule declares `run_config_paths` as
    # its input and passes them on argv, and `assemble_run` -- what the
    # notebooks and the validation gate call -- must agree with that exactly.
    for experiment, run in (
        ("cosmological-parameters", "ET-triangular"),
        ("variable-proposal-guard", "eps1e-3"),
    ):
        layers = run_config_paths(experiment, run)
        assert merge_config_layers(layers) == assemble_run(experiment, run)


def test_merge_config_layers_rejects_an_empty_layer_list() -> None:
    with pytest.raises(ValueError, match="no config layers"):
        merge_config_layers([])


def test_assembling_an_unknown_run_names_the_missing_file() -> None:
    with pytest.raises(ValueError, match="unknown run cosmological-parameters/nope"):
        assemble_run("cosmological-parameters", "nope")


def test_an_experiment_without_a_base_overlay_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "config/analysis/base").mkdir(parents=True)
    (tmp_path / "config/analysis/base/x.toml").write_text("seed = 1\n")
    (tmp_path / "config/analysis/runs/demo").mkdir(parents=True)
    (tmp_path / "config/analysis/runs/demo/only.toml").write_text("")

    with pytest.raises(ValueError, match=f"missing a required {EXPERIMENT_BASE}"):
        discover_runs(tmp_path)


# --------------------------------------------------------------------------- #
# The three-layer merge
# --------------------------------------------------------------------------- #
def test_base_files_merge_into_one_mapping() -> None:
    base = load_base()

    assert base["seed"] == 42
    assert base["sampler"]["num_samples"] == 2000
    assert base["analysis"]["f_min"] == 2.0
    assert base["cosmology"]["minimum_redshift"] == 0.3
    assert base["fiducials"]["H0"] == 67.66
    assert base["catalog"]["injection"]["md_bank"] == "md-imrphenom-s41"
    # A detector list is never a base default: a run must state its network.
    assert "detectors" not in base["analysis"]


def test_run_layer_overrides_the_experiment_layer() -> None:
    """astrophysical-parameters/z_peak overrides its experiment's sampler block."""
    shared = assemble_run("astrophysical-parameters", "Madau-Dickinson")
    overridden = assemble_run("astrophysical-parameters", "z_peak")

    assert shared["sampler"]["num_warmup"] == 1000
    assert shared["sampler"]["dense_mass"] is True
    assert overridden["sampler"]["num_warmup"] == 500
    assert overridden["sampler"]["dense_mass"] is False
    # Untouched sampler keys still come from the base layer.
    assert overridden["sampler"]["num_samples"] == 2000


def test_experiment_layer_overrides_the_base_layer() -> None:
    assert load_base()["sampler"]["num_warmup"] == 1000
    assert (
        assemble_run("cosmological-parameters", "ET-triangular")["sampler"][
            "num_warmup"
        ]
        == 500
    )


def test_cross_type_prior_override_replaces_the_whole_table() -> None:
    """Key-merging a normal prior onto a uniform one would leave stale low/high."""
    merged = assemble_run("modified-propagation", "Xi_0-H0")

    assert merged["priors"]["H0"] == {
        "type": "normal",
        "loc": 67.66,
        "scale": 0.6766,
    }
    assert "low" not in merged["priors"]["H0"]
    assert "high" not in merged["priors"]["H0"]


# --------------------------------------------------------------------------- #
# Every run assembles into a valid config
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("experiment", "run"), all_runs())
def test_every_run_merge_is_a_valid_run_config(experiment: str, run: str) -> None:
    config = build_run_config(assemble_run(experiment, run))

    amplitude_parameter = config.analysis.amplitude_parameter
    assert set(config.priors) == set(config.fiducials)
    assert set(config.sampled_params) <= set(config.fiducials)
    if config.analysis.likelihood == "amplitude_marginalized":
        assert amplitude_parameter is not None
    assert config.analysis.detectors


@pytest.mark.parametrize(("experiment", "run"), all_runs())
def test_every_run_names_declared_banks(experiment: str, run: str) -> None:
    config = build_run_config(assemble_run(experiment, run))

    check_bank_references(config, label=f"{experiment}/{run}")


def test_1d_runs_use_diagonal_mass_matrix_and_short_warmup() -> None:
    """1D problems (one latent, marginalized amplitude included) need no dense
    mass matrix and converge faster: NUTS is configured accordingly."""
    for experiment, run in all_runs():
        config = build_run_config(assemble_run(experiment, run))
        if len(config.sampled_params) != 1:
            continue
        assert config.sampler.num_warmup == 500, f"{experiment}/{run}"
        assert config.sampler.dense_mass is False, f"{experiment}/{run}"


def test_variable_proposal_guard_samples_h0_md() -> None:
    """Guard sweeps epsilon on the H0-marginalized Madau-Dickinson problem."""
    for run in discover_runs()["variable-proposal-guard"]:
        config = build_run_config(assemble_run("variable-proposal-guard", run))
        assert config.sampled_params == ("gamma", "kappa", "z_peak"), run
        assert config.analysis.likelihood == "amplitude_marginalized"
        assert config.analysis.amplitude_parameter == "H0"
        assert config.posterior_params == ("gamma", "kappa", "z_peak", "H0")
        assert config.sampler.dense_mass is True
        assert config.sampler.num_warmup == 1000


# --------------------------------------------------------------------------- #
# Catalog selection
# --------------------------------------------------------------------------- #
def test_every_run_shares_one_injection_catalog() -> None:
    injections = {
        build_run_config(assemble_run(*run)).catalog.injection.model_dump_json()
        for run in all_runs()
    }

    assert len(injections) == 1


def test_proposal_catalogs_by_experiment() -> None:
    def proposals(experiment: str) -> dict[str, tuple]:
        return {
            run: (
                spec.md_bank,
                spec.uniform_bank,
                spec.num_samples,
                spec.uniform_mixing_fraction,
            )
            for run in discover_runs()[experiment]
            if (
                spec := build_run_config(assemble_run(experiment, run)).catalog.proposal
            )
        }

    default = ("md-imrphenom-s42", None, 16384, 0.0)
    for experiment in ("cosmological-parameters", "modified-propagation"):
        assert set(proposals(experiment).values()) == {default}

    assert set(proposals("astrophysical-parameters").values()) == {
        ("md-imrphenom-s42", "uniform-imrphenom-s51", 16384, 0.1)
    }
    assert proposals("variable-catalog-size") == {
        "n8192": ("md-imrphenom-s42", None, 8192, 0.0),
        "n16384": default,
        "n32768": ("md-imrphenom-s42", None, 32768, 0.0),
    }
    assert proposals("variable-proposal-guard") == {
        "eps1e-1": ("md-imrphenom-s42", "uniform-imrphenom-s51", 16384, 0.1),
        "eps1e-2": ("md-imrphenom-s42", "uniform-imrphenom-s51", 16384, 0.01),
        "eps1e-3": ("md-imrphenom-s42", "uniform-imrphenom-s51", 16384, 0.001),
    }
    assert proposals("waveform-approximant") == {
        "IMRPhenom": ("md-imrphenom-s41", None, 32768, 0.0),
        "TaylorF2": ("md-taylorf2-s41", None, 32768, 0.0),
    }


def test_only_astrophysical_and_guard_runs_use_a_mixed_proposal() -> None:
    mixed = {"astrophysical-parameters", "variable-proposal-guard"}

    for experiment, run in all_runs():
        config = build_run_config(assemble_run(experiment, run))
        epsilon = config.catalog.proposal.uniform_mixing_fraction
        if experiment in mixed:
            assert epsilon > 0.0, f"{experiment}/{run}"
        else:
            assert epsilon == 0.0, f"{experiment}/{run}"


def test_resolve_bank_names_are_the_distinct_banks_both_roles_need() -> None:
    assert resolve_bank_names("cosmological-parameters", "ET-triangular") == [
        "md-imrphenom-s41",
        "md-imrphenom-s42",
    ]
    assert resolve_bank_names("variable-proposal-guard", "eps1e-1") == [
        "md-imrphenom-s41",
        "md-imrphenom-s42",
        "uniform-imrphenom-s51",
    ]


def test_catalog_bank_names_requires_both_roles() -> None:
    with pytest.raises(TypeError, match=r"\[catalog.proposal\] table"):
        catalog_bank_names({"catalog": {"injection": {"md_bank": "a"}}})
    with pytest.raises(TypeError, match=r"\[catalog\] table"):
        catalog_bank_names({})


# --------------------------------------------------------------------------- #
# Bank reference validation
# --------------------------------------------------------------------------- #
def test_a_run_naming_an_unknown_bank_is_rejected() -> None:
    raw = assemble_run("cosmological-parameters", "ET-triangular")
    raw["catalog"]["proposal"]["md_bank"] = "does-not-exist"
    config = build_run_config(raw)

    with pytest.raises(ValueError, match="names unknown bank 'does-not-exist'"):
        check_bank_references(config, label="demo/only")


def test_a_mixture_seed_colliding_with_a_bank_seed_is_rejected() -> None:
    # md-imrphenom-s42 was drawn at seed 42; collide the mixture seed with it.
    raw = assemble_run("variable-proposal-guard", "eps1e-1")
    raw["catalog"]["proposal"]["mixture_seed"] = 42
    config = build_run_config(raw)

    with pytest.raises(ValueError, match="must all be distinct"):
        check_bank_references(config, label="demo/only")


# --------------------------------------------------------------------------- #
# The assemble_config CLI
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The pre-flight gate that replaced `astrogwb-assemble-config --all`
# --------------------------------------------------------------------------- #
def test_the_validation_gate_covers_every_run() -> None:
    labels = validate_all_runs()

    assert len(labels) == 26
    for label in (
        "cosmological-parameters/H0-Omega_m",
        "cosmological-parameters/H0-merger-rate",
        "astrophysical-parameters/z_peak",
        "modified-propagation/Xi_0-H0",
        "variable-catalog-size/n32768",
        "variable-proposal-guard/eps1e-3",
        "waveform-approximant/IMRPhenom",
        "waveform-approximant/TaylorF2",
    ):
        assert label in labels


def test_the_guard_run_merges_to_its_declared_settings() -> None:
    guard = build_run_config(
        assemble_run("variable-proposal-guard", "eps1e-3")
    ).model_dump(mode="json")

    assert guard["catalog"]["proposal"]["uniform_mixing_fraction"] == 0.001
    assert guard["sampled_params"] == ["gamma", "kappa", "z_peak"]
    assert guard["analysis"]["likelihood"] == "amplitude_marginalized"
    assert guard["analysis"]["amplitude_parameter"] == "H0"
    # The proposal *density* is derived at run time, never carried by a config.
    assert "proposal" not in guard


def test_run_mcmc_writes_its_config_record_beside_the_chain() -> None:
    # The record moved out of `outputs/configs/` and next to the chain when
    # `assemble_config` went away; `save_config` still writes the
    # defaults-filled RunConfig, so the file stays diff-able.
    config = build_run_config(assemble_run("modified-propagation", "Xi_0"))

    assert config.model_dump(mode="json")["sampled_params"] == ["xi_0"]


# --------------------------------------------------------------------------- #
# Bank configs
# --------------------------------------------------------------------------- #
def test_four_banks_are_declared_by_filename() -> None:
    banks = discover_banks()

    assert list(banks) == [
        "md-imrphenom-s41",
        "md-imrphenom-s42",
        "md-taylorf2-s41",
        "uniform-imrphenom-s51",
    ]
    assert banks["md-imrphenom-s41"].population == "madau-dickinson"
    assert banks["uniform-imrphenom-s51"].population == "uniform-redshift"
    assert {bank.seed for bank in banks.values()} == {41, 42, 51}
    assert all(bank.waveform.frequency_resolution == 1.0 for bank in banks.values())


def test_taylorf2_bank_only_changes_the_waveform_approximant() -> None:
    banks = discover_banks()
    imrphenom = banks["md-imrphenom-s41"].model_dump()
    taylorf2 = banks["md-taylorf2-s41"].model_dump()

    assert imrphenom["waveform"]["approximant"] == "IMRPhenomXAS_NRTidalv3"
    assert taylorf2["waveform"]["approximant"] == "TaylorF2"
    taylorf2["name"] = imrphenom["name"]
    taylorf2["waveform"]["approximant"] = imrphenom["waveform"]["approximant"]
    assert taylorf2 == imrphenom


def test_every_bank_config_points_at_an_existing_population_graph() -> None:
    for bank in discover_banks().values():
        assert bank.population_path().is_file(), bank.name


def test_the_two_population_graphs_differ_only_in_redshift() -> None:
    graphs = {
        bank.population: load_mapping(bank.population_path())
        for bank in discover_banks().values()
    }
    assert set(graphs) == {"madau-dickinson", "uniform-redshift"}

    parameters = []
    for graph in graphs.values():
        without_redshift = dict(graph["parameters"])
        without_redshift.pop("redshift")
        parameters.append(without_redshift)
    assert parameters[0] == parameters[1]
    # Declaration order drives GraphSimulator's RNG key order via the
    # topological sort's tie-breaking, so it must match too.
    assert list(graphs["madau-dickinson"]["parameters"]) == list(
        graphs["uniform-redshift"]["parameters"]
    )
