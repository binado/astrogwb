"""Figure presentation constants and the detector lists they resolve against.

Presentation -- which runs a figure shows, in what order, under which label --
is hard-coded in the scripts and in :mod:`astrogwb.paper.plotting`. These tests
pin the contract that survived the move out of TOML: every hard-coded run name
is a real run of its experiment, and resolved networks carry that experiment's
detectors in declaration order -- the order the workflow also expands its chain
paths and its ``--network-run`` flags from.

Detectors are read from each run's own committed config layers rather than from
an assembled artifact, so these tests need no build step and no tmp tree: they
run against the checkout as committed.
"""

from __future__ import annotations

import pytest

from astrogwb.paper.config.runs import assemble_run, discover_runs, resolve_networks
from astrogwb.paper.paths import paper_project_root
from astrogwb.paper.plotting import DETECTOR_NETWORK_RUNS, DETECTOR_NETWORKS

PAPER_ROOT = paper_project_root()

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
        expected = assemble_run("cosmological-parameters", network.name)["analysis"][
            "detectors"
        ]
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


def test_fiducials_and_analysis_grid_come_from_a_merged_run() -> None:
    from astrogwb.paper.config.mcmc import build_run_config

    config = build_run_config(
        assemble_run("cosmological-parameters", "ET-2L-aligned-CE-Hanford")
    )
    fiducials = config.fiducials
    grid = config.analysis_grid

    # `importance_relative_ess` is a plotting truth line, not a fiducial: adding
    # it here would inject a spurious constant into the sampled model.
    assert set(fiducials) == {
        "H0",
        "Omega_m",
        "xi_0",
        "xi_n",
        "gamma",
        "kappa",
        "z_peak",
        "local_merger_rate",
    }
    assert fiducials["H0"] == 67.66
    assert fiducials["local_merger_rate"] == 770.0
    assert (grid.observation_time, grid.f_min, grid.f_max) == (1.0, 2.0, 4096.0)
    assert (grid.minimum_redshift, grid.maximum_redshift, grid.n_grid) == (
        0.3,
        20.0,
        256,
    )


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
