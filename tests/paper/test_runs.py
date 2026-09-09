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

    # A detector list is never a base default: a run must state its network.
    assert "detectors" not in base["analysis"]


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


# --------------------------------------------------------------------------- #
# Catalog selection
# --------------------------------------------------------------------------- #
#: The eight catalogs the 26 runs share between them. Two pairs of specs
#: collapsed into one file when catalogs stopped being composed in memory:
#: astrophysical-parameters reuses the eps=0.1 guard catalog, and
#: waveform-approximant/IMRPhenom reuses the injection catalog.
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


# --------------------------------------------------------------------------- #
# Catalog configs
# --------------------------------------------------------------------------- #
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
