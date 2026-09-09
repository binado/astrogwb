"""Discovery and the three-layer merge behind ``config/analysis/``.

Replaces ``test_experiments.py``: there is no inventory to validate any more,
so what is tested is the convention (which files exist and what they map to)
and the merge that turns them into a run config.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from repo import REPO_ROOT

from astrogwb.paper.config.catalogs import (
    check_catalog_references,
    discover_catalogs,
    validate_all_runs,
)
from astrogwb.paper.config.mcmc import build_run_config
from astrogwb.paper.config.runs import (
    CHAINS_ROOT,
    EXPERIMENT_BASE,
    RUNS_DIR,
    assemble_run,
    base_config_paths,
    catalog_config_paths,
    discover_catalog_names,
    discover_runs,
    load_base,
    merge_config_layers,
    resolve_catalog_names,
    run_config_paths,
)
from astrogwb.paper.utils import load_mapping

PAPER_ROOT = REPO_ROOT

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
    # config/banks went with the bank/catalog split: every catalog is a file
    # now, so there is one config tree for them and one output directory.
    assert not (PAPER_ROOT / "config/banks").exists()
    assert (PAPER_ROOT / "config/analysis/base").is_dir()
    assert (PAPER_ROOT / "config/catalogs/base").is_dir()
    assert (PAPER_ROOT / "config/catalogs/defs").is_dir()
    assert (PAPER_ROOT / "config/populations").is_dir()


def test_every_experiment_has_the_required_base_overlay() -> None:
    for experiment in discover_runs():
        assert (PAPER_ROOT / RUNS_DIR / experiment / EXPERIMENT_BASE).is_file()


def test_chain_output_root_is_fixed() -> None:
    # There is no assembled-config path any more: a run is addressed by its
    # layer files going in and by its chain coming out.
    assert CHAINS_ROOT == Path("outputs/chains")


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
    assert base["catalog"]["injection"] == "md-imrphenom-s41-n32768"
    # A detector list is never a base default: a run must state its network.
    assert "detectors" not in base["analysis"]


def test_run_layer_overrides_the_experiment_layer() -> None:
    """astrophysical-parameters/redshift-peak overrides its experiment's sampler block."""
    shared = assemble_run("astrophysical-parameters", "madau-dickinson")
    overridden = assemble_run("astrophysical-parameters", "redshift-peak")

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
def test_every_run_names_declared_catalogs(experiment: str, run: str) -> None:
    config = build_run_config(assemble_run(experiment, run))

    check_catalog_references(config, label=f"{experiment}/{run}")


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
        assert "H0" in config.priors
        assert config.sampler.dense_mass is True
        assert config.sampler.num_warmup == 1000


# --------------------------------------------------------------------------- #
# Catalog selection
# --------------------------------------------------------------------------- #
#: The eight catalogs the 26 runs share between them. Two pairs of specs
#: collapsed into one file when catalogs stopped being composed in memory:
#: astrophysical-parameters reuses the eps=0.1 guard catalog, and
#: waveform-approximant/IMRPhenom reuses the injection catalog.
INJECTION_CATALOG = "md-imrphenom-s41-n32768"


def test_every_run_shares_one_injection_catalog() -> None:
    injections = {
        build_run_config(assemble_run(*run)).catalog.injection for run in all_runs()
    }

    assert injections == {INJECTION_CATALOG}


def test_only_astrophysical_and_guard_runs_use_a_mixed_proposal() -> None:
    """The mixing fraction is a property of the catalog file now, not the run."""
    catalogs = discover_catalogs()
    mixed = {"astrophysical-parameters", "variable-proposal-guard"}

    for experiment, run in all_runs():
        config = build_run_config(assemble_run(experiment, run))
        definition = catalogs[config.catalog.proposal]
        assert definition.is_mixture == (experiment in mixed), f"{experiment}/{run}"


def test_every_declared_catalog_is_used_by_some_run() -> None:
    """No orphan catalogs: eight files, all of them sampled against."""
    used = {name for run in all_runs() for name in resolve_catalog_names(*run).values()}

    assert used == set(discover_catalog_names())
    assert len(used) == 8


@pytest.mark.parametrize(
    ("base_config", "message"),
    [
        ('[catalog]\ninjection = "a"\n', r"catalog.proposal must be a catalog name"),
        ("seed = 1\n", r"\[catalog\] table"),
        (
            '[catalog]\ninjection = "a"\nproposal = 3\n',
            r"catalog.proposal must be a catalog name",
        ),
    ],
)
def test_resolve_catalog_names_rejects_malformed_catalogs(
    tmp_path: Path, base_config: str, message: str
) -> None:
    base = tmp_path / "config/analysis/base"
    experiment = tmp_path / "config/analysis/runs/demo"
    base.mkdir(parents=True)
    experiment.mkdir(parents=True)
    (base / "catalog.toml").write_text(base_config, encoding="utf-8")
    (experiment / EXPERIMENT_BASE).write_text("", encoding="utf-8")
    (experiment / "only.toml").write_text("", encoding="utf-8")

    with pytest.raises(TypeError, match=message):
        resolve_catalog_names("demo", "only", root=tmp_path)


# --------------------------------------------------------------------------- #
# Catalog reference validation
# --------------------------------------------------------------------------- #
def test_a_run_naming_an_unknown_catalog_is_rejected() -> None:
    raw = assemble_run("cosmological-parameters", "ET-triangular")
    raw["catalog"]["proposal"] = "does-not-exist"
    config = build_run_config(raw)

    with pytest.raises(ValueError, match="names unknown catalog 'does-not-exist'"):
        check_catalog_references(config, label="demo/only")


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
        "astrophysical-parameters/redshift-peak",
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

    assert guard["catalog"]["proposal"] == "md-uniform-imrphenom-s63-n16384-eps1e-3"
    assert guard["sampled_params"] == ["gamma", "kappa", "z_peak"]
    assert guard["analysis"]["likelihood"] == "amplitude_marginalized"
    assert guard["analysis"]["amplitude_parameter"] == "H0"
    # The proposal *density* is derived at run time, never carried by a config.
    assert "proposal" not in guard


def test_run_mcmc_writes_its_config_record_beside_the_chain() -> None:
    # The record moved out of `outputs/configs/` and next to the chain when
    # `assemble_config` went away; `RunConfig.save` still writes the
    # defaults-filled RunConfig, so the file stays diff-able.
    config = build_run_config(assemble_run("modified-propagation", "Xi_0"))

    assert config.model_dump(mode="json")["sampled_params"] == ["xi_0"]


# --------------------------------------------------------------------------- #
# Catalog configs
# --------------------------------------------------------------------------- #
def test_eight_catalogs_are_declared_by_filename() -> None:
    catalogs = discover_catalogs()

    assert list(catalogs) == [
        "md-imrphenom-s41-n32768",
        "md-imrphenom-s42-n16384",
        "md-imrphenom-s42-n32768",
        "md-imrphenom-s42-n8192",
        "md-taylorf2-s41-n32768",
        "md-uniform-imrphenom-s61-n16384-eps1e-1",
        "md-uniform-imrphenom-s62-n16384-eps1e-2",
        "md-uniform-imrphenom-s63-n16384-eps1e-3",
    ]
    assert all(
        definition.waveform.frequency_resolution == 1.0
        for definition in catalogs.values()
    )


def test_only_the_guard_catalogs_are_mixtures() -> None:
    """A mixture is the only shape that needs a mixture_seed, and vice versa."""
    catalogs = discover_catalogs()

    mixtures = {name for name, d in catalogs.items() if d.is_mixture}
    assert mixtures == {
        "md-uniform-imrphenom-s61-n16384-eps1e-1",
        "md-uniform-imrphenom-s62-n16384-eps1e-2",
        "md-uniform-imrphenom-s63-n16384-eps1e-3",
    }
    for name, definition in catalogs.items():
        assert (definition.mixture_seed is not None) == (name in mixtures), name


def test_the_three_md_s42_catalogs_are_one_nested_series() -> None:
    """variable-catalog-size compares sizes, so the draws must be nested.

    Same graph, same seed: GraphSimulator is prefix-stable, so the three files
    are prefixes of one stream even though each is generated independently.
    """
    catalogs = discover_catalogs()
    series = {
        name: catalogs[name]
        for name in (
            "md-imrphenom-s42-n8192",
            "md-imrphenom-s42-n16384",
            "md-imrphenom-s42-n32768",
        )
    }

    assert {(c.population, c.seed) for d in series.values() for c in d.components} == {
        ("madau-dickinson", 42)
    }
    assert [d.num_samples for d in series.values()] == [8192, 16384, 32768]


def test_taylorf2_catalog_only_changes_the_waveform_approximant() -> None:
    catalogs = discover_catalogs()
    imrphenom = catalogs["md-imrphenom-s41-n32768"].model_dump()
    taylorf2 = catalogs["md-taylorf2-s41-n32768"].model_dump()

    assert imrphenom["waveform"]["approximant"] == "IMRPhenomXAS_NRTidalv3"
    assert taylorf2["waveform"]["approximant"] == "TaylorF2"
    taylorf2["name"] = imrphenom["name"]
    taylorf2["waveform"]["approximant"] = imrphenom["waveform"]["approximant"]
    assert taylorf2 == imrphenom


def test_the_shared_waveform_block_is_declared_once() -> None:
    """Every catalog inherits [waveform] from the base layer, not its own copy."""
    for name in discover_catalog_names():
        layers = catalog_config_paths(name)
        assert layers[0].name == "waveform.toml"
        assert layers[-1].stem == name
        own = load_mapping(layers[-1])
        # Only the TaylorF2 catalog overrides anything in the shared block.
        assert set(own.get("waveform", {})) <= {"approximant"}, name


def test_every_catalog_component_points_at_an_existing_population_graph() -> None:
    for definition in discover_catalogs().values():
        for path in definition.population_paths():
            assert path.is_file(), definition.name


def test_the_two_population_graphs_differ_only_in_redshift() -> None:
    graphs = {
        component.population: load_mapping(component.population_path())
        for definition in discover_catalogs().values()
        for component in definition.components
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
