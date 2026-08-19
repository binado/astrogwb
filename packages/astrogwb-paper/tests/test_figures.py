"""Figure presentation constants and the detector lists they resolve against.

Presentation -- which runs a figure shows, in what order, under which label --
is hard-coded in the scripts and in :mod:`astrogwb_paper.plotting`. These tests
pin the contract that survived the move out of TOML: every hard-coded run name
is a real run of its experiment, and resolved networks carry that experiment's
detectors in declaration order -- the order the workflow also expands its chain
paths from.
"""

from __future__ import annotations

import pytest
from astrogwb_paper.config.experiments import experiment, load_experiments, overlay_for
from astrogwb_paper.config.figures import (
    load_analysis_grid,
    load_fiducials,
    resolve_networks,
)
from astrogwb_paper.paths import paper_project_root
from astrogwb_paper.plotting import DETECTOR_NETWORK_RUNS, DETECTOR_NETWORKS

PAPER_ROOT = paper_project_root()
BASE_CONFIG = PAPER_ROOT / "inputs/config.yaml"

# The experiments whose runs the network legend is resolved against. The
# fiducial-spectrum figure borrows cosmological-parameters' detector lists.
NETWORK_EXPERIMENTS = ("cosmological-parameters", "modified-propagation")


def test_the_figure_config_directory_is_gone() -> None:
    # Presentation moved into the scripts; nothing should reintroduce a
    # parallel TOML copy of it for the scripts or the workflow to reload.
    assert not (PAPER_ROOT / "inputs/figures").exists()


def test_every_hard_coded_network_is_a_declared_run() -> None:
    for name in NETWORK_EXPERIMENTS:
        runs = set(experiment(name).runs)

        assert set(DETECTOR_NETWORK_RUNS) <= runs, f"{name} lacks a compared network"


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
    spec = experiment("cosmological-parameters")
    networks = resolve_networks("cosmological-parameters", DETECTOR_NETWORKS)

    assert [network.name for network in networks] == list(DETECTOR_NETWORK_RUNS)
    for network in networks:
        expected = overlay_for(spec, network.name)["analysis"]["detectors"]
        assert network.detectors == tuple(expected)
    assert networks[0].detectors == ("E1", "E2", "E3")
    assert networks[-1].detectors == ("S2", "R2", "C1")


def test_both_network_experiments_resolve_to_identical_networks() -> None:
    resolved = [
        resolve_networks(name, DETECTOR_NETWORKS) for name in NETWORK_EXPERIMENTS
    ]

    assert resolved[0] == resolved[1]


def test_resolve_networks_rejects_empty_duplicate_and_unknown_runs() -> None:
    with pytest.raises(ValueError, match="no detector networks"):
        resolve_networks("cosmological-parameters", [])
    with pytest.raises(ValueError, match="duplicate"):
        resolve_networks(
            "cosmological-parameters",
            [("ET-triangular", "a"), ("ET-triangular", "b")],
        )
    with pytest.raises(ValueError, match="unknown run"):
        resolve_networks("cosmological-parameters", [("nope", "label")])


def test_base_fiducials_and_analysis_grid_are_read_from_the_base_config() -> None:
    fiducials = load_fiducials(BASE_CONFIG)
    grid = load_analysis_grid(BASE_CONFIG)

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
    assert (grid.observation_time, grid.f_min, grid.f_max) == (1.0, 2.0, 4096.0)
    assert (grid.z_min, grid.z_max, grid.n_grid) == (0.0, 20.0, 256)


def test_experiments_without_figures_are_chains_only() -> None:
    with_figures = set(NETWORK_EXPERIMENTS)
    chains_only = set(load_experiments()) - with_figures

    assert chains_only == {
        "astrophysical-parameters",
        "variable-injection-size",
    }
