"""Discovery and the layered merge behind ``config/``.

Replaces ``test_experiments.py``: there is no inventory to validate any more,
so what is tested is the convention (which files exist and what they map to)
and the merge that turns them into a run config.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from config_fixtures import write_root_layers
from repo import REPO_ROOT

from astrogwb.paper.config.catalogs import (
    check_catalog_references,
    validate_all_runs,
)
from astrogwb.paper.config.mcmc import build_run_config
from astrogwb.paper.config.runs import (
    BASE_OUT_DIR,
    BLOCK_FOLDS,
    CATALOGS_ROOT,
    CHAINS_ROOT,
    EXPERIMENT_BASE,
    FIGURES_DIR,
    RUNS_DIR,
    assemble_run,
    base_config_paths,
    catalog_config_paths,
    discover_catalog_names,
    discover_runs,
    load_base,
    load_config_blocks,
    merge_config_layers,
    resolve_catalog_names,
    run_config_paths,
)
from astrogwb.paper.utils import load_mapping


def _import_script(name: str):
    """Import one ``scripts/*.py``, which are not installed modules."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    # config/analysis/ went the same way as config/catalogs/base: every
    # shared run layer is one config/*.json named after the block it declares,
    # so there is no base/ to glob and no runs/ level to nest under.
    assert not (PAPER_ROOT / "config/analysis").exists()
    assert (PAPER_ROOT / "config/analysis.json").is_file()
    assert (PAPER_ROOT / "config/runs").is_dir()
    assert (PAPER_ROOT / "config/catalogs").is_dir()
    # The catalog tree is one flat directory of defs: the shared layers sit in
    # config/ with the run tables, where `jq` reads them, so there is no
    # base/ to glob and no defs/ to distinguish it from.
    assert not (PAPER_ROOT / "config/catalogs/base").exists()
    assert not (PAPER_ROOT / "config/catalogs/defs").exists()
    # config/populations went with the gwmock graph path: a population is a
    # registered NumPyro model now, named by config/population.json.
    assert not (PAPER_ROOT / "config/populations").exists()


def test_every_experiment_has_the_required_base_overlay() -> None:
    for experiment in discover_runs():
        assert (PAPER_ROOT / RUNS_DIR / experiment / EXPERIMENT_BASE).is_file()


def test_output_roots_are_derived_from_one_base() -> None:
    # There is no assembled-config path any more: a run is addressed by its
    # layer files going in and by its chain coming out. The three output roots
    # share one base so a second `outputs` literal cannot drift from the first.
    assert BASE_OUT_DIR == Path("outputs")
    assert CHAINS_ROOT == BASE_OUT_DIR / "chains"
    assert CATALOGS_ROOT == BASE_OUT_DIR / "catalogs"
    assert FIGURES_DIR == BASE_OUT_DIR / "figures"


def test_run_config_paths_are_the_layers_in_merge_order() -> None:
    """Every shared layer first, then the experiment, then the run.

    The shared names are asserted literally rather than against
    `base_config_paths()`: a self-consistent comparison against the helper
    would pass whatever list the helper happened to return, and what matters
    here is that there is one file per top-level block and that the run's own
    file is last.
    """
    paths = run_config_paths("cosmological-parameters", "ET-triangular")
    base = base_config_paths()

    assert [path.name for path in base] == [
        "analysis.json",
        "fiducials.json",
        "networks.json",
        "priors.json",
        "sampler.json",
    ]
    assert paths[: len(base)] == base
    assert [path.name for path in paths[len(base) :]] == [
        EXPERIMENT_BASE,
        "ET-triangular.json",
    ]
    assert all(path.is_file() for path in paths)


def test_every_shared_layer_is_one_block_named_after_its_stem() -> None:
    """The layer-0 convention, which the per-block CLI and `jq` folds rely on.

    A shared layer that declared two blocks, or a block whose name did not
    match its filename, would make "one flag per block, one file per block"
    false -- and `astrogwb.paper.config`'s accessors look the table up by stem.
    """
    for path in base_config_paths(REPO_ROOT):
        declared = json.loads(path.read_text(encoding="utf-8"))
        assert list(declared) == [path.stem], path


def test_plotting_settings_are_not_a_run_layer() -> None:
    """`config/plotting.json` is presentation and must not reach a RunConfig.

    It sits in the same directory as the three shared layers, so a glob would
    sweep it in and `extra="forbid"` would then reject every run.
    """
    names = {
        path.name
        for path in run_config_paths("cosmological-parameters", "ET-triangular")
    }

    assert "plotting.json" not in names


def test_waveform_settings_are_not_a_run_layer() -> None:
    """`config/waveform.json` is catalog generation and must not reach a RunConfig.

    It sits in the same directory as the three shared run layers, so a glob
    would sweep it in and `extra="forbid"` would then reject every run.
    """
    names = {
        path.name
        for path in run_config_paths("cosmological-parameters", "ET-triangular")
    }

    assert "waveform.json" not in names


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
    write_root_layers(tmp_path)
    (tmp_path / "config/runs/demo").mkdir(parents=True)
    # `{}` rather than an empty file: a layer is JSON now, and "" is not.
    (tmp_path / "config/runs/demo/only.json").write_text("{}")

    with pytest.raises(ValueError, match=f"missing a required {EXPERIMENT_BASE}"):
        discover_runs(tmp_path)


# --------------------------------------------------------------------------- #
# The layered merge
# --------------------------------------------------------------------------- #
def test_base_files_merge_into_one_mapping() -> None:
    base = load_base()

    # Neither a network nor a detector list is ever a base default: a run must
    # state its own, or inherit someone else's silently.
    assert "network" not in base["analysis"]
    assert "detectors" not in base["analysis"]
    # Every shared layer is in this merge, one block each.
    assert set(base) == {"analysis", "fiducials", "networks", "priors", "sampler"}


def test_no_run_layer_redeclares_the_fiducials_or_priors() -> None:
    """The shared values have one home, not two.

    `config/analysis/base/parameters.toml` used to declare both. It is gone,
    and no run layer may quietly reintroduce either table -- a second
    declaration would win the merge and the JSON the notebooks read would
    silently stop describing what the runs sample. A *run* overriding one
    named prior is a different thing and is allowed; a whole table is not.
    """
    for path in sorted((REPO_ROOT / "config/runs").rglob("*.json")):
        raw = load_mapping(path)
        assert "fiducials" not in raw, path


def test_no_run_declares_a_raw_detector_list() -> None:
    """Read unmerged, so a reintroduced list is caught where it is written.

    A merged-config test cannot see this: `_resolve_network` would either
    resolve the name or cross-check the list, and both pass when the list
    happens to agree.
    """
    layers = [
        REPO_ROOT / "config/analysis.json",
        *(REPO_ROOT / "config/runs").rglob("*.json"),
    ]
    for path in sorted(layers):
        analysis = load_mapping(path).get("analysis") or {}
        assert "detectors" not in analysis, path


@pytest.mark.parametrize(("experiment", "run"), all_runs())
def test_every_run_names_a_declared_network(experiment: str, run: str) -> None:
    """`AnalysisConfig.network` is optional at the model level; here it is not.

    The model must allow its absence so a saved config -- which carries the
    resolved detectors and no [networks] table -- re-validates. That every
    *committed* run names one is this repo's rule, so it is this repo's test.
    """
    merged = assemble_run(experiment, run, root=REPO_ROOT)
    name = (merged.get("analysis") or {}).get("network")

    assert name, f"{experiment}/{run} names no network"
    assert name in merged["networks"], f"{experiment}/{run} names unknown {name!r}"


# --------------------------------------------------------------------------- #
# Every run assembles into a valid config
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("experiment", "run"), all_runs())
def test_every_run_merge_is_a_valid_run_config(experiment: str, run: str) -> None:
    config = build_run_config(assemble_run(experiment, run))

    amplitude_parameter = config.analysis.amplitude_parameter
    assert set(config.priors) == set(config.fiducials)
    assert set(config.analysis.sampled_params) <= set(config.fiducials)
    if config.analysis.likelihood == "amplitude_marginalized":
        assert amplitude_parameter is not None
    assert config.analysis.detectors


@pytest.mark.parametrize(("experiment", "run"), all_runs())
def test_every_run_names_declared_catalogs(experiment: str, run: str) -> None:
    config = build_run_config(assemble_run(experiment, run))

    check_catalog_references(config, label=f"{experiment}/{run}")


# --------------------------------------------------------------------------- #
# PolarizationPowerCatalog selection
# --------------------------------------------------------------------------- #
#: The eight catalogs the 26 runs share between them. Two pairs of specs
#: collapsed into one file when catalogs stopped being composed in memory:
#: astrophysical-parameters reuses the eps=0.1 guard catalog, and
#: waveform-approximant/IMRPhenom reuses the injection catalog.


@pytest.mark.parametrize(
    ("catalog", "message"),
    [
        ({"injection": "a"}, r"analysis\.catalog\.proposal must be a catalog name"),
        (None, r"\[analysis\.catalog\] table"),
        (
            {"injection": "a", "proposal": 3},
            r"analysis\.catalog\.proposal must be a catalog name",
        ),
    ],
)
def test_resolve_catalog_names_rejects_malformed_catalogs(
    tmp_path: Path, catalog: dict[str, object] | None, message: str
) -> None:
    experiment = tmp_path / "config/runs/demo"
    experiment.mkdir(parents=True)
    analysis: dict[str, object] = {"minimum_frequency": 2.0}
    if catalog is not None:
        analysis["catalog"] = catalog
    write_root_layers(tmp_path, analysis=analysis)
    (experiment / EXPERIMENT_BASE).write_text("{}", encoding="utf-8")
    (experiment / "only.json").write_text("{}", encoding="utf-8")

    with pytest.raises(TypeError, match=message):
        resolve_catalog_names("demo", "only", root=tmp_path)


# --------------------------------------------------------------------------- #
# PolarizationPowerCatalog reference validation
# --------------------------------------------------------------------------- #
def test_a_run_naming_an_unknown_catalog_is_rejected() -> None:
    raw = assemble_run("cosmological-parameters", "ET-triangular")
    raw["analysis"]["catalog"]["proposal"] = "does-not-exist"
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
# PolarizationPowerCatalog configs
# --------------------------------------------------------------------------- #
def test_the_shared_blocks_are_declared_once() -> None:
    """Every catalog inherits [waveform], [population] and [fiducials].

    Editing any of the three must therefore invalidate all eight catalogs,
    which is only true because they are declared as inputs of every one.
    ``fiducials.json`` is a run layer as well, so the hyperparameters a catalog
    is drawn at and the ones a run initializes at cannot drift.
    """
    for name in discover_catalog_names():
        layers = catalog_config_paths(name)
        assert [path.name for path in layers[:-1]] == [
            "waveform.json",
            "population.json",
            "fiducials.json",
        ]
        assert layers[-1].stem == name
        assert layers[-1].suffix == ".json"
        own = load_mapping(layers[-1])
        # Only the TaylorF2 catalog overrides anything in the shared waveform
        # block, and only the guard catalogs touch the shared population --
        # naming a different population and adding the construction setting it
        # takes, never restating the shared window and grid. No def overrides
        # [fiducials]: every committed catalog is drawn at them.
        assert set(own.get("waveform", {})) <= {"approximant"}, name
        assert set(own.get("population", {})) <= {"model_name", "model_kwargs"}, name
        assert "fiducials" not in own, name


def test_run_mcmc_validates_the_blocks_the_workflow_folds() -> None:
    """The whole path: `jq` folds the layers, `run_mcmc` parses and validates.

    `run_mcmc` is the one entrypoint handed blocks rather than layer paths, and
    nothing else executes its CLI -- ty does not check `argparse.Namespace`
    attributes, so a renamed flag would surface only in a submitted job. The
    run chosen is the one that overrides a prior, so the shallow `[priors]`
    fold has to survive validation and not merely parse.
    """
    jq = shutil.which("jq")
    if jq is None:
        pytest.skip("jq is not installed")

    run_mcmc = _import_script("run_mcmc")
    layers = [str(path) for path in run_config_paths("modified-propagation", "Xi_0-H0")]
    argv: list[str] = []
    for block, program in BLOCK_FOLDS.items():
        folded = subprocess.run(
            [jq, "-s", program, *layers], capture_output=True, text=True, check=True
        )
        argv += [f"--{block}", folded.stdout]
    argv += [
        "--injection-catalog",
        "outputs/catalogs/md-imrphenom-s41-n32768.h5",
        "--proposal-catalog",
        "outputs/catalogs/md-imrphenom-s41-n32768.h5",
        "--label",
        "Xi_0-H0",
    ]

    args = run_mcmc.parse_args(argv)
    config = build_run_config(load_config_blocks(args))

    # The tight H0 prior the run overrides, which a deep fold would have
    # corrupted, reached the validated config as a Normal.
    assert type(config.priors["H0"]).__name__ == "Normal"
    assert config.analysis.sampled_params == ("xi_0",)
    assert config.analysis.catalog.injection == "md-imrphenom-s41-n32768"
    assert config.analysis.grid.n_grid == 256


def test_the_jq_block_folds_match_the_python_merge() -> None:
    """`run_mcmc` is handed blocks `jq` folded; this pins them against Python.

    The workflow folds one block per `--<block>` flag, `*` everywhere and `+`
    for `priors`, while the notebooks and the validation gate reach the same
    mapping through `merge_config_layers`. Two implementations of one fold, so
    the agreement is checked rather than argued -- over every run, because
    exactly one of them (`modified-propagation/Xi_0-H0`) overrides a prior, and
    that is the only run where `+` and `*` differ.
    """
    jq = shutil.which("jq")
    if jq is None:
        pytest.skip("jq is not installed")

    overrode_a_prior = False
    for experiment, runs in discover_runs().items():
        for run in runs:
            layers = [str(path) for path in run_config_paths(experiment, run)]
            expected = merge_config_layers(run_config_paths(experiment, run))
            folded = {}
            for block, program in BLOCK_FOLDS.items():
                result = subprocess.run(
                    [jq, "-s", program, *layers],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                folded[block] = json.loads(result.stdout)
            assert folded == expected, f"{experiment}/{run}"
            if json.loads(Path(layers[-1]).read_text(encoding="utf-8")).get("priors"):
                overrode_a_prior = True
    assert overrode_a_prior, "no run exercises the shallow [priors] fold"


def test_a_deep_fold_would_corrupt_the_one_prior_override() -> None:
    """Why `priors` folds with `+`, as a failure rather than a comment.

    `config/priors.json` gives H0 a Uniform; `modified-propagation/Xi_0-H0`
    replaces it with a Normal. Key-merging the two leaves the Uniform's `low`
    and `high` beside the Normal's `loc` and `scale`, which `materialize_prior`
    rejects -- loudly here, but only because the fold is shallow in production.
    """
    jq = shutil.which("jq")
    if jq is None:
        pytest.skip("jq is not installed")

    layers = [str(path) for path in run_config_paths("modified-propagation", "Xi_0-H0")]
    deep = subprocess.run(
        [jq, "-s", BLOCK_FOLDS["priors"].replace(". + $b", ". * $b"), *layers],
        capture_output=True,
        text=True,
        check=True,
    )

    assert set(json.loads(deep.stdout)["H0"]["kwargs"]) == {
        "low",
        "high",
        "loc",
        "scale",
    }
    assert set(
        merge_config_layers(run_config_paths("modified-propagation", "Xi_0-H0"))[
            "priors"
        ]["H0"]["kwargs"]
    ) == {"loc", "scale"}


def test_the_jq_merge_matches_the_python_merge() -> None:
    """The workflow merges catalog layers with `jq`; this pins the two agree.

    `rule waveform_catalog` does its own merge in the shell so it can hand the
    generator the three blocks rather than a list of paths, which leaves two
    implementations of one fold. jq's `*` is a recursive merge, and the catalog
    layers carry no [priors] block, so the shallow-merge rule
    `_merge_run_overlay` exists for never applies -- but that is an argument,
    not a check.
    """
    jq = shutil.which("jq")
    if jq is None:
        pytest.skip("jq is not installed")
    for name in discover_catalog_names():
        layers = [str(path) for path in catalog_config_paths(name)]
        merged = subprocess.run(
            [jq, "-s", "reduce .[] as $layer ({}; . * $layer)", *layers],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(merged.stdout) == merge_config_layers(
            catalog_config_paths(name)
        ), name
