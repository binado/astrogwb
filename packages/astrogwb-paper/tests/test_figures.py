"""Figure presentation constants and the detector lists they resolve against.

Presentation -- which runs a figure shows, in what order, under which label --
is hard-coded in the scripts and in :mod:`astrogwb_paper.plotting`. These tests
pin the contract that survived the move out of TOML: every hard-coded run name
is a real run of its experiment, and resolved networks carry that experiment's
detectors in declaration order -- the order the workflow also expands its chain
paths from.

Detectors, fiducials, and the analysis grid are read from the *assembled*
configs under ``outputs/configs/``, so these tests assemble into a tmp tree and
point the figure loader at it. That is the improvement the rework bought: a
figure reports what was sampled, not what an inventory said.
"""

from __future__ import annotations

import astrogwb_paper.config.figures as figures_module
import pytest
from astrogwb_paper.cli.assemble_config import main as assemble_configs
from astrogwb_paper.config.figures import (
    REFERENCE_RUN,
    load_analysis_grid,
    load_fiducials,
    reference_config_path,
    resolve_networks,
)
from astrogwb_paper.config.runs import assemble_run, config_path, discover_runs
from astrogwb_paper.paths import paper_project_root
from astrogwb_paper.plotting import DETECTOR_NETWORK_RUNS, DETECTOR_NETWORKS

PAPER_ROOT = paper_project_root()

# The experiments whose runs the network legend is resolved against. The
# fiducial-spectrum figure borrows cosmological-parameters' detector lists.
NETWORK_EXPERIMENTS = ("cosmological-parameters", "modified-propagation")


@pytest.fixture(autouse=True)
def assembled_configs(tmp_path_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Assemble every config into a tmp tree and read figures from there.

    The committed checkout may have no `outputs/configs/` at all (it is a build
    artifact), so the figure loader is pointed at a freshly assembled one rather
    than at whatever happens to be lying around.
    """
    root = tmp_path_factory.mktemp("paper-root")
    assemble_configs(["--all", "--output-dir", str(root / "outputs/configs")])
    monkeypatch.setattr(figures_module, "paper_project_root", lambda: root)


def test_the_figure_config_directory_is_gone() -> None:
    # Presentation moved into the scripts; nothing should reintroduce a
    # parallel TOML copy of it for the scripts or the workflow to reload.
    assert not (PAPER_ROOT / "inputs/figures").exists()


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
    networks = resolve_networks("cosmological-parameters", DETECTOR_NETWORKS)

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
    with pytest.raises(FileNotFoundError, match="assembled config not found"):
        resolve_networks("cosmological-parameters", [("nope", "label")])


def test_the_reference_run_is_a_real_run() -> None:
    experiment, run = REFERENCE_RUN
    assert run in discover_runs()[experiment]
    assert reference_config_path() == config_path(experiment, run)


def test_fiducials_and_analysis_grid_come_from_an_assembled_run() -> None:
    fiducials = load_fiducials()
    grid = load_analysis_grid()

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
