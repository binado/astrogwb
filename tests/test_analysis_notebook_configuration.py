import ast
from collections.abc import Callable
from pathlib import Path
import tomllib
from typing import cast

import pytest


ROOT = Path(__file__).parents[1]
AMPLITUDE_NOTEBOOK = ROOT / "notebooks" / "amplitude_toy_model.py"
SNR_NOTEBOOK = ROOT / "notebooks" / "snr_by_detector.py"
PAPER_CONFIG = ROOT / "configs" / "paper.toml"

ParseNetwork = Callable[[str], tuple[str, tuple[str, ...]]]
ResolveNetworks = Callable[
    [dict[str, tuple[str, ...]], list[str], list[str]],
    dict[str, tuple[str, ...]],
]


def _source(path: Path) -> str:
    return path.read_text()


def _literal_assignments(path: Path) -> dict[str, object]:
    assignments = {}
    for node in ast.parse(_source(path)).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id.startswith("DEFAULT_"):
            try:
                assignments[target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return assignments


def _network_helpers() -> dict[str, object]:
    tree = ast.parse(_source(SNR_NOTEBOOK))
    wanted = {"_parse_network_definition", "_resolve_networks"}
    functions: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            functions.append(node)
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), SNR_NOTEBOOK, "exec"),
        namespace,
    )
    return namespace


def test_amplitude_defaults_match_paper_config() -> None:
    with PAPER_CONFIG.open("rb") as handle:
        paper = tomllib.load(handle)
    defaults = _literal_assignments(AMPLITUDE_NOTEBOOK)

    assert defaults["DEFAULT_CHAINS_DIR"] == paper["paths"]["chains_dir"]
    assert defaults["DEFAULT_OBSERVATION_TIME"] == paper["analysis"]["observation_time"]
    assert defaults["DEFAULT_F_MIN"] == paper["analysis"]["f_min"]
    assert defaults["DEFAULT_F_MAX"] == paper["analysis"]["f_max"]


def test_snr_defaults_match_paper_config() -> None:
    with PAPER_CONFIG.open("rb") as handle:
        paper = tomllib.load(handle)
    defaults = _literal_assignments(SNR_NOTEBOOK)
    analysis = paper["analysis"]
    cosmology = analysis["cosmology"]
    fiducials = analysis["fiducials"]

    expected_scalars = {
        "DEFAULT_OBSERVATION_TIME": analysis["observation_time"],
        "DEFAULT_F_MIN": analysis["f_min"],
        "DEFAULT_F_MAX": analysis["f_max"],
        "DEFAULT_Z_MIN": cosmology["z_min"],
        "DEFAULT_Z_MAX": cosmology["z_max"],
        "DEFAULT_N_GRID": cosmology["n_grid"],
        "DEFAULT_H0": fiducials["H0"],
        "DEFAULT_OMEGA_M": fiducials["Omega_m"],
        "DEFAULT_XI_0": fiducials["xi_0"],
        "DEFAULT_XI_N": fiducials["xi_n"],
        "DEFAULT_GAMMA": fiducials["gamma"],
        "DEFAULT_KAPPA": fiducials["kappa"],
        "DEFAULT_Z_PEAK": fiducials["z_peak"],
        "DEFAULT_LOCAL_MERGER_RATE": fiducials["local_merger_rate"],
    }
    assert {name: defaults[name] for name in expected_scalars} == expected_scalars

    expected_networks = {
        name: tuple(detectors) for name, detectors in paper["detector_networks"].items()
    }
    assert defaults["DEFAULT_DETECTOR_NETWORKS"] == expected_networks
    assert defaults["DEFAULT_NETWORKS"] == list(expected_networks)


@pytest.mark.parametrize(
    "definition",
    ["missing-equals", "=E1,E2", "empty=", "empty-first=,E2", "empty-last=E1,"],
)
def test_network_definition_rejects_malformed_values(definition: str) -> None:
    parse_network = cast(ParseNetwork, _network_helpers()["_parse_network_definition"])

    with pytest.raises(ValueError, match="invalid network definition"):
        parse_network(definition)


def test_network_definitions_overlay_defaults_and_select_names() -> None:
    resolve_networks = cast(ResolveNetworks, _network_helpers()["_resolve_networks"])

    resolved = resolve_networks(
        {"existing": ("E1", "E2")},
        ["existing=S1,R1", "new=C1,E3"],
        ["new", "existing"],
    )

    assert resolved == {"new": ("C1", "E3"), "existing": ("S1", "R1")}


def test_network_definitions_reject_duplicate_cli_names() -> None:
    resolve_networks = cast(ResolveNetworks, _network_helpers()["_resolve_networks"])

    with pytest.raises(ValueError, match="duplicate --network definition"):
        resolve_networks({}, ["same=E1,E2", "same=S1,R1"], [])


def test_network_selection_rejects_unknown_names() -> None:
    resolve_networks = cast(ResolveNetworks, _network_helpers()["_resolve_networks"])

    with pytest.raises(ValueError, match="unknown selected network"):
        resolve_networks({"known": ("E1", "E2")}, [], ["missing"])


@pytest.mark.parametrize("notebook", [AMPLITUDE_NOTEBOOK, SNR_NOTEBOOK])
def test_analysis_notebooks_have_no_config_loader_or_argument(notebook: Path) -> None:
    source = _source(notebook)
    tree = ast.parse(source)

    assert "--config" not in source
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "astrogwb.config.loading"
        for node in ast.walk(tree)
    )
