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

from astrogwb.metadata import CatalogRequest
from astrogwb.paper.config.catalogs import (
    check_catalog_requests,
    resolve_run_catalogs,
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
    catalog_blocks,
    discover_runs,
    load_base,
    load_config_blocks,
    merge_config_layers,
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
    "time-delay",
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
def test_discovery_finds_seven_experiments_and_27_runs() -> None:
    runs = discover_runs()

    assert set(runs) == EXPERIMENTS
    assert sum(len(names) for names in runs.values()) == 27
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
    # config/catalogs went when catalogs became content-addressed: a run
    # declares what it draws, and the file is named by the request's key.
    assert not (PAPER_ROOT / "config/catalogs").exists()
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
        "waveform.json",
        "population.json",
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
    assert set(base) == {
        "analysis",
        "fiducials",
        "networks",
        "priors",
        "sampler",
        "waveform",
        "population",
    }


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
def test_every_run_asks_for_catalogs_that_can_be_drawn(
    experiment: str, run: str
) -> None:
    config = build_run_config(assemble_run(experiment, run))

    check_catalog_requests(config, label=f"{experiment}/{run}")


@pytest.mark.parametrize(("experiment", "run"), all_runs())
@pytest.mark.parametrize("role", ["injection", "proposal"])
def test_run_config_and_raw_merge_key_a_catalog_identically(
    experiment: str, run: str, role: str
) -> None:
    """The workflow keys the raw merge; `run_mcmc` keys the validated config.

    A disagreement would hand every run a file its own check then refuses.
    """
    config = build_run_config(assemble_run(experiment, run))
    workflow = resolve_run_catalogs().by_run[(experiment, run)][role]

    assert config.catalog_request(role).key() == workflow


# --------------------------------------------------------------------------- #
# Catalog resolution
# --------------------------------------------------------------------------- #
def test_catalog_blocks_inherit_the_run_blocks_under_a_role_override() -> None:
    """The eps=0.1 guard names its own population but keeps the shared window."""
    blocks = catalog_blocks("variable-proposal-guard", "eps1e-1", "proposal")
    shared = load_base()["population"]["model_kwargs"]

    assert blocks["population"]["model_name"] == "bns_md_uniform_mixture"
    assert blocks["population"]["model_kwargs"] == {
        **shared,
        "uniform_mixing_fraction": 0.1,
    }
    assert (blocks["seed"], blocks["num_samples"]) == (61, 16384)


def test_the_injection_is_drawn_at_the_fiducials_the_run_initializes_at() -> None:
    raw = assemble_run("time-delay", "delay-slope")
    blocks = catalog_blocks("time-delay", "delay-slope", "injection")

    assert blocks["fiducials"] == raw["fiducials"]
    assert blocks["fiducials"]["delay_slope"] == -1.0
    assert blocks["population"]["model_name"] == "bns_md_time_delayed_cosmological"


def test_the_catalog_size_series_differs_only_in_size() -> None:
    requests = [
        CatalogRequest.from_blocks(
            **catalog_blocks("variable-catalog-size", run, "proposal")
        )
        for run in ("n8192", "n16384", "n32768")
    ]

    assert [request.num_samples for request in requests] == [8192, 16384, 32768]
    assert len({request.key() for request in requests}) == 3
    assert (
        len(
            {
                request.model_copy(update={"num_samples": 1}).key()
                for request in requests
            }
        )
        == 1
    )


def test_runs_asking_for_the_same_draw_share_one_catalog() -> None:
    by_run = resolve_run_catalogs().by_run

    shared = by_run[("cosmological-parameters", "ET-triangular")]["injection"]
    assert by_run[("modified-propagation", "Xi_0")]["injection"] == shared
    assert by_run[("waveform-approximant", "IMRPhenom")]["proposal"] == shared
    assert by_run[("waveform-approximant", "TaylorF2")]["proposal"] != shared


@pytest.mark.parametrize(
    ("catalog", "message"),
    [
        ({"injection": {"seed": 1, "num_samples": 8}}, r"analysis\.catalog\.proposal"),
        (None, r"analysis\.catalog\.injection"),
        (
            {
                "injection": {"seed": 1, "num_samples": 8},
                "proposal": {"num_samples": 8},
            },
            r"analysis\.catalog\.proposal must declare seed",
        ),
    ],
    ids=["missing-role", "missing-table", "missing-seed"],
)
def test_catalog_blocks_reject_malformed_catalogs(
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
        for role in ("injection", "proposal"):
            catalog_blocks("demo", "only", role, root=tmp_path)


def test_a_guard_mixture_is_rejected_as_an_injection() -> None:
    raw = assemble_run("variable-proposal-guard", "eps1e-1")
    raw["analysis"]["catalog"]["injection"] = raw["analysis"]["catalog"]["proposal"]
    config = build_run_config(raw)

    with pytest.raises(ValueError, match="declares no merger rate"):
        check_catalog_requests(config, label="demo/only")


def test_an_unregistered_catalog_population_is_rejected() -> None:
    raw = assemble_run("cosmological-parameters", "ET-triangular")
    raw["analysis"]["catalog"]["proposal"]["population"] = {
        "model_name": "no_such_population"
    }
    config = build_run_config(raw)

    with pytest.raises(ValueError, match="unknown population 'no_such_population'"):
        check_catalog_requests(config, label="demo/only")


# --------------------------------------------------------------------------- #
# The assemble_config CLI
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The pre-flight gate that replaced `astrogwb-assemble-config --all`
# --------------------------------------------------------------------------- #
def test_the_validation_gate_covers_every_run() -> None:
    labels = validate_all_runs()

    assert len(labels) == 27
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
        "outputs/catalogs/injection.h5",
        "--proposal-catalog",
        "outputs/catalogs/proposal.h5",
        "--label",
        "Xi_0-H0",
    ]

    args = run_mcmc.parse_args(argv)
    config = build_run_config(load_config_blocks(args))

    # The tight H0 prior the run overrides, which a deep fold would have
    # corrupted, reached the validated config as a Normal.
    assert type(config.priors["H0"]).__name__ == "Normal"
    assert config.analysis.sampled_params == ("xi_0",)
    assert config.analysis.catalog.injection.seed == 41
    assert config.waveform["approximant"] == "IMRPhenomXAS_NRTidalv3"
    assert config.analysis.population.model_kwargs["n_grid"] == 256


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
