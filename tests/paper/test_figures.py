"""Figure presentation constants and the detector lists they resolve against.

Presentation -- which runs a figure shows, in what order, under which label --
is hard-coded in the scripts and in :mod:`astrogwb.paper.plotting`. These tests
pin the contract that survived the move out of TOML: every hard-coded run name
is a real run of its experiment, and resolved networks carry that experiment's
detectors in declaration order -- the order the workflow also expands its chain
paths and its ``--network-run`` flags from.

Detectors are read from each run's own committed config layers rather than from
an assembled artifact, so these tests need no build step and no tmp tree: they
run against the checkout as committed. A run names a network and its layers
carry the table that resolves the name, so the detectors a figure reports an
SNR for are still the ones its chain was sampled with.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from config_fixtures import write_root_layers
from repo import REPO_ROOT

from astrogwb.paper.config import networks
from astrogwb.paper.config.runs import assemble_run, discover_runs, resolve_networks
from astrogwb.paper.plotting import DETECTOR_NETWORK_RUNS, DETECTOR_NETWORKS

PAPER_ROOT = REPO_ROOT

# The experiments whose runs the network legend is resolved against. The
# fiducial-spectrum figure borrows cosmological-parameters' detector lists.
NETWORK_EXPERIMENTS = ("cosmological-parameters", "modified-propagation")


def network_references(experiment: str) -> list[tuple[str, str]]:
    """The ``--network-run`` list a figure rule passes, in legend order."""
    return [(experiment, run) for run in DETECTOR_NETWORK_RUNS]


def test_the_figure_config_directory_is_gone() -> None:
    # Presentation moved into the scripts; nothing should reintroduce a
    # parallel TOML copy of it for the scripts or the workflow to reload.
    assert not (PAPER_ROOT / "inputs/figures").exists()


def test_no_assembled_config_tree_is_rebuilt() -> None:
    # Every entrypoint merges its own layers now. A reappearing
    # `outputs/configs/` would mean something started writing the intermediate
    # artifact again, and figures could then read a stale copy.
    assert not (PAPER_ROOT / "outputs/configs").exists()


def test_every_hard_coded_network_is_a_declared_run() -> None:
    runs = discover_runs()
    for name in NETWORK_EXPERIMENTS:
        assert set(DETECTOR_NETWORK_RUNS) <= set(runs[name]), (
            f"{name} lacks a compared network"
        )


def test_network_labels_are_unique_and_non_empty() -> None:
    labels = [label for _, label in DETECTOR_NETWORKS]

    assert all(labels)
    assert len(set(labels)) == len(labels)
    assert len(set(DETECTOR_NETWORK_RUNS)) == len(DETECTOR_NETWORK_RUNS)


def test_committed_latex_labels_survive_the_move_out_of_toml() -> None:
    labels = dict(DETECTOR_NETWORKS)

    assert labels["ET-triangular"] == r"ET-$\Delta$"
    assert labels["ET-2L-misaligned-CE-Hanford"] == r"ET-2L $+$ CE"


def test_resolve_networks_preserves_order_and_attaches_detectors() -> None:
    networks = resolve_networks(
        network_references("cosmological-parameters"), DETECTOR_NETWORKS
    )

    assert [network.name for network in networks] == list(DETECTOR_NETWORK_RUNS)
    for network in networks:
        # Through the run's own merge, exactly as `resolve_networks` does it:
        # the run names a network, and the [networks] table its layers carry
        # resolves that name. Not a direct `networks()` lookup, which would
        # assume the run name and the network name always agree.
        merged = assemble_run("cosmological-parameters", network.name)
        expected = merged["networks"][merged["analysis"]["network"]]
        assert network.detectors == tuple(expected)
    assert networks[0].detectors == ("E1", "E2", "E3")
    assert networks[-1].detectors == ("S2", "R2", "C1")


def test_both_network_experiments_resolve_to_identical_networks() -> None:
    resolved = [
        resolve_networks(network_references(name), DETECTOR_NETWORKS)
        for name in NETWORK_EXPERIMENTS
    ]

    assert resolved[0] == resolved[1]


def test_resolve_networks_rejects_empty_duplicate_and_unknown_runs() -> None:
    with pytest.raises(ValueError, match="no detector networks"):
        resolve_networks([], [])
    with pytest.raises(ValueError, match="duplicate"):
        resolve_networks(
            [("cosmological-parameters", "ET-triangular")] * 2,
            [("ET-triangular", "a"), ("ET-triangular", "b")],
        )
    with pytest.raises(ValueError, match="unknown run"):
        resolve_networks([("cosmological-parameters", "nope")], [("nope", "label")])


def test_resolve_networks_rejects_a_mis_ordered_network_run_list() -> None:
    # Order drives chain order, legend order, and color assignment, and a
    # swapped pair renders a perfectly good figure with the wrong labels on the
    # wrong curves. Positional matching makes that checkable; this is the check.
    references = network_references("cosmological-parameters")
    swapped = [references[1], references[0], *references[2:]]

    with pytest.raises(ValueError, match="must match the figure legend order"):
        resolve_networks(swapped, DETECTOR_NETWORKS)


def test_resolve_networks_rejects_a_short_or_mixed_network_run_list() -> None:
    references = network_references("cosmological-parameters")

    with pytest.raises(ValueError, match="legend declares"):
        resolve_networks(references[:-1], DETECTOR_NETWORKS)

    mixed = [*references[:-1], ("modified-propagation", references[-1][1])]
    with pytest.raises(ValueError, match="same experiment"):
        resolve_networks(mixed, DETECTOR_NETWORKS)


def test_only_the_network_experiments_have_figure_rules() -> None:
    # The remaining experiments have no figure script, so the workflow offers
    # only their run_experiment_* targets.
    with_figures = set(NETWORK_EXPERIMENTS)
    chains_only = set(discover_runs()) - with_figures

    assert chains_only == {
        "astrophysical-parameters",
        "variable-catalog-size",
        "variable-proposal-guard",
        "waveform-approximant",
    }


def test_every_legend_network_is_declared_in_the_table() -> None:
    """The legend cannot name a network `config/networks.json` does not have.

    Membership lives in the table; only the order and the LaTeX label live in
    `DETECTOR_NETWORKS`. This is the seam between them.
    """
    declared = networks(REPO_ROOT)

    assert set(DETECTOR_NETWORK_RUNS) <= set(declared)


def test_the_network_a_run_names_matches_its_legend_name() -> None:
    """Every network run is named after the network it uses.

    `resolve_networks` deliberately does not rely on this -- it merges each run
    and reads that run's own `analysis.network` -- but the property is worth
    pinning: it is what would make a future direct-lookup simplification safe,
    and its quiet loss is exactly the bug the indirection guards against.
    """
    for experiment in NETWORK_EXPERIMENTS:
        for run in DETECTOR_NETWORK_RUNS:
            merged = assemble_run(experiment, run, root=REPO_ROOT)

            assert merged["analysis"]["network"] == run, f"{experiment}/{run}"


def test_resolve_networks_rejects_an_undeclared_network(tmp_path: Path) -> None:
    """A run naming a network the table lacks fails, naming both."""
    write_root_layers(tmp_path, networks={"known": ["S1", "R1"]})
    experiment = tmp_path / "config/runs/demo"
    experiment.mkdir(parents=True)
    (experiment / "_base.json").write_text("{}", encoding="utf-8")
    (experiment / "only.json").write_text(
        '{"analysis": {"network": "absent"}}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"names network 'absent'"):
        resolve_networks([("demo", "only")], (("only", "label"),), root=tmp_path)


# --------------------------------------------------------------------------- #
# Where a hand-run figure writes
# --------------------------------------------------------------------------- #
#: The three figure scripts. Their `--output-*` flags are what a hand run writes
#: to, while the workflow hands each one explicit paths -- so the defaults are
#: the only place a script could grow a second `outputs/figures/...` literal.
FIGURE_SCRIPTS = (
    "scripts/importance_weights_grid.py",
    "scripts/mcmc_cosmological_parameters.py",
    "scripts/mcmc_modified_propagation.py",
)


def _output_flags(relative: str) -> dict[str, dict[str, ast.expr]]:
    """Every ``--output-*`` ``add_argument`` in a script, as flag -> keywords.

    Parsed rather than imported: the scripts reach JAX and matplotlib, and this
    is a property of the source, not of what importing it builds.
    """
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    flags: dict[str, dict[str, ast.expr]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        if isinstance(flag, str) and flag.startswith("--output-"):
            flags[flag] = {
                keyword.arg: keyword.value
                for keyword in node.keywords
                if keyword.arg is not None
            }
    return flags


def _rooted_name(expr: ast.expr) -> str | None:
    """The leftmost name of a `A / "b" / "c"` path chain."""
    while isinstance(expr, ast.BinOp):
        expr = expr.left
    return expr.id if isinstance(expr, ast.Name) else None


@pytest.mark.parametrize("relative", FIGURE_SCRIPTS)
def test_every_output_flag_defaults_under_figures_dir(relative: str) -> None:
    """A hand-run script writes under the same root the workflow declares.

    The default has to exist and the flag has to be optional, or the default is
    unreachable; and it has to be `FIGURES_DIR / ...` rather than a typed-out
    `outputs/figures/...`, which is how the two would drift apart.
    """
    flags = _output_flags(relative)

    assert flags, f"{relative} declares no --output-* flags"
    for flag, keywords in flags.items():
        assert "required" not in keywords, f"{relative} {flag} is still required"
        default = keywords.get("default")
        assert default is not None, f"{relative} {flag} has no default"
        assert _rooted_name(default) == "FIGURES_DIR", f"{relative} {flag}"
