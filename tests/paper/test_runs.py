"""Discovery and the layered merge behind ``config/``.

Replaces ``test_experiments.py``: there is no inventory to validate any more,
so what is tested is the convention (which files exist and what they map to)
and the merge that turns them into a run config.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import knf
import pytest
from config_fixtures import write_defaults
from repo import REPO_ROOT

from astrogwb.metadata import CatalogMetadata
from astrogwb.paper.config.catalogs import (
    check_catalog_requests,
    resolve_run_catalogs,
    validate_all_runs,
)
from astrogwb.paper.config.mcmc import build_run_config
from astrogwb.paper.config.runs import (
    EXPERIMENT_BASE,
    RUNS_DIR,
    assemble_run,
    base_config_paths,
    catalog_blocks,
    discover_runs,
    load_base,
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


def test_every_experiment_has_the_required_base_overlay() -> None:
    for experiment in discover_runs():
        assert (PAPER_ROOT / RUNS_DIR / experiment / EXPERIMENT_BASE).is_file()


def test_run_config_paths_are_the_layers_in_merge_order() -> None:
    """The shared layer first, then the experiment, then the run.

    Asserted literally rather than against `base_config_paths()`: a
    self-consistent comparison against the helper would pass whatever list the
    helper happened to return, and what matters here is that a run is three
    layers and that its own file is last.
    """
    paths = run_config_paths("cosmological-parameters", "ET-triangular")

    assert paths[:1] == base_config_paths()
    assert [str(path) for path in paths] == [
        "config/defaults.toml",
        f"config/runs/cosmological-parameters/{EXPERIMENT_BASE}",
        "config/runs/cosmological-parameters/ET-triangular.toml",
    ]
    assert all(path.is_file() for path in paths)


def test_plotting_settings_are_not_a_run_layer() -> None:
    """`config/plotting.toml` is presentation and must not reach a RunConfig.

    It sits in the same directory as the shared layer, so a glob would sweep it
    in and `extra="forbid"` would then reject every run.
    """
    names = {
        path.name
        for path in run_config_paths("cosmological-parameters", "ET-triangular")
    }

    assert "plotting.toml" not in names


def test_merge_config_layers_rejects_an_empty_layer_list() -> None:
    with pytest.raises(ValueError, match="no config layers"):
        merge_config_layers([])


def test_merge_config_layers_rejects_an_all_json_layer_list(tmp_path: Path) -> None:
    """knf would merge JSON layers too; the run config is one format, TOML."""
    path = tmp_path / "defaults.json"
    path.write_text('{"analysis": {"network": "demo"}}', encoding="utf-8")

    with pytest.raises(ValueError, match="config layers are TOML"):
        merge_config_layers([path])


def test_assembling_an_unknown_run_names_the_missing_file() -> None:
    with pytest.raises(ValueError, match="unknown run cosmological-parameters/nope"):
        assemble_run("cosmological-parameters", "nope")


def test_an_experiment_without_a_base_overlay_is_rejected(tmp_path: Path) -> None:
    write_defaults(tmp_path)
    (tmp_path / "config/runs/demo").mkdir(parents=True)
    (tmp_path / "config/runs/demo/only.toml").write_text("")

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
    # The shared layer declares every block of a run config.
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
        REPO_ROOT / "config/defaults.toml",
        *(REPO_ROOT / "config/runs").rglob("*.toml"),
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
        CatalogMetadata.from_blocks(
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
    write_defaults(tmp_path, analysis=analysis)
    (experiment / EXPERIMENT_BASE).write_text("", encoding="utf-8")
    (experiment / "only.toml").write_text("", encoding="utf-8")

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


def test_run_mcmc_validates_the_layers_the_workflow_passes() -> None:
    """The whole path: the workflow's `--config` layers, parsed and validated.

    `run_mcmc` is the one entrypoint the workflow runs per chain, and nothing
    else executes its CLI -- ty does not check `argparse.Namespace`
    attributes, so a renamed flag would surface only in a submitted job. The
    run chosen is the one that overrides a prior, so the shallow `[priors]`
    merge has to survive validation and not merely parse.
    """
    run_mcmc = _import_script("run_mcmc")
    argv: list[str] = []
    for path in run_config_paths("modified-propagation", "Xi_0-H0"):
        argv += ["--config", str(path)]
    argv += [
        "--injection-catalog",
        "outputs/catalogs/injection.h5",
        "--proposal-catalog",
        "outputs/catalogs/proposal.h5",
        "--label",
        "Xi_0-H0",
    ]

    args = run_mcmc.parse_args(argv)
    config = build_run_config(run_mcmc.load_merged_config(args))

    # The tight H0 prior the run overrides, which a deep merge would have
    # corrupted, reached the validated config as a Normal.
    assert type(config.priors["H0"]).__name__ == "Normal"
    assert config.analysis.sampled_params == ("xi_0",)
    assert config.analysis.catalog.injection.seed == 41
    assert config.waveform["approximant"] == "IMRPhenomXAS_NRTidalv3"
    assert config.analysis.population.model_kwargs["n_grid"] == 256


def test_a_deep_fold_would_corrupt_the_one_prior_override() -> None:
    """Why `PRIOR_SHALLOW` exists, as a failure rather than a comment.

    `config/defaults.toml` gives H0 a Uniform; `modified-propagation/Xi_0-H0`
    replaces it with a Normal. Key-merging the two leaves the Uniform's `low`
    and `high` beside the Normal's `loc` and `scale`, which `materialize_prior`
    rejects -- loudly here, but only because the merge is shallow in production.
    """
    layers = run_config_paths("modified-propagation", "Xi_0-H0")
    deep = knf.load(list(layers))

    assert set(deep["priors"]["H0"]["kwargs"]) == {
        "low",
        "high",
        "loc",
        "scale",
    }
    shallow = merge_config_layers(layers)
    assert set(shallow["priors"]["H0"]["kwargs"]) == {"loc", "scale"}
