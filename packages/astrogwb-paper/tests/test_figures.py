"""Figure presentation configs under ``inputs/figures/``.

These pin the contract the figure scripts rely on: every run name a figure
declares is a real run of its experiment, every detector network resolves to a
registry label, and the resolved networks carry the experiment's detectors in
declaration order (which is also chain order and legend order).
"""

from __future__ import annotations

import pytest
from astrogwb_paper.config.experiments import experiment, load_experiments, overlay_for
from astrogwb_paper.config.figures import (
    NETWORK_LABELS_PATH,
    figure_networks,
    load_analysis_grid,
    load_fiducials,
    load_figure_config,
    load_network_labels,
    resolve_networks,
)
from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()
FIGURES_DIR = PAPER_ROOT / "inputs/figures"
BASE_CONFIG = PAPER_ROOT / "inputs/mcmc.base.toml"

# Committed inventory. ``inputs/figures/`` is never globbed: the registry is
# not a figure config, and ``fiducial-spectrum.toml`` borrows another
# experiment's networks, so filename and ``experiment`` key do not correspond
# one-to-one.
FIGURE_CONFIGS = (
    "H0-all-detectors",
    "H0-merger-rate",
    "H0-omega-m",
    "modified-propagation-all-detectors",
    "fiducial-spectrum",
)

# figure config -> keys holding a single run name, keys holding ordered
# detector-network run names.
RUN_KEYS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "H0-all-detectors": ((), ("posteriors",)),
    "H0-merger-rate": (("fixed_run", "sampled_run"), ()),
    "H0-omega-m": (("run",), ()),
    "modified-propagation-all-detectors": (
        ("xi0_run", "xi0_n_run", "h0_run"),
        ("detector_posteriors",),
    ),
}

DETECTOR_NETWORKS = (
    "ET-triangular",
    "ET-triangular-CE-Hanford",
    "ET-2L-aligned",
    "ET-2L-aligned-CE-Hanford",
    "ET-2L-misaligned",
    "ET-2L-misaligned-CE-Hanford",
)


def figure(name: str) -> dict:
    return load_figure_config(FIGURES_DIR / f"{name}.toml")


def test_figure_inputs_are_exactly_the_committed_files() -> None:
    discovered = {path.stem for path in FIGURES_DIR.glob("*.toml")}

    assert discovered == {*FIGURE_CONFIGS, NETWORK_LABELS_PATH.stem}
    assert list(FIGURES_DIR.glob("*/*.toml")) == []


def test_every_figure_run_name_is_a_declared_run() -> None:
    for name, (single_keys, network_keys) in RUN_KEYS.items():
        config = figure(name)
        runs = set(experiment(config["experiment"]).runs)
        declared = {config[key] for key in single_keys}
        for key in network_keys:
            declared.update(config[key])

        assert declared <= runs, f"{name} names runs outside its experiment"


def test_fiducial_spectrum_borrows_h0_all_detectors_networks() -> None:
    spectrum = figure("fiducial-spectrum")

    assert spectrum["experiment"] == "H0-all-detectors"
    assert tuple(spectrum["networks"]) == DETECTOR_NETWORKS


def test_every_referenced_network_resolves_to_a_registry_label() -> None:
    labels = load_network_labels()
    referenced = {
        *figure("H0-all-detectors")["posteriors"],
        *figure("modified-propagation-all-detectors")["detector_posteriors"],
        *figure("fiducial-spectrum")["networks"],
    }

    assert referenced <= set(labels)
    # No dead entries: every committed label is used by at least one figure.
    assert set(labels) == referenced


def test_resolve_networks_preserves_declaration_order_and_detectors() -> None:
    config = figure("H0-all-detectors")
    spec = experiment("H0-all-detectors")
    networks = figure_networks(config, "posteriors")

    assert [network.name for network in networks] == list(config["posteriors"])
    for network in networks:
        expected = overlay_for(spec, network.name)["analysis"]["detectors"]
        assert network.detectors == tuple(expected)
    assert networks[0].detectors == ("E1", "E2", "E3")
    assert networks[-1].detectors == ("S2", "R2", "C1")


def test_the_three_network_figures_resolve_to_identical_networks() -> None:
    h0 = figure_networks(figure("H0-all-detectors"), "posteriors")
    propagation = figure_networks(
        figure("modified-propagation-all-detectors"), "detector_posteriors"
    )
    spectrum = figure_networks(figure("fiducial-spectrum"), "networks")

    assert h0 == propagation == spectrum


def test_resolve_networks_rejects_empty_duplicate_and_unlabelled_arrays() -> None:
    labels = load_network_labels()

    with pytest.raises(ValueError, match="no detector networks"):
        resolve_networks("H0-all-detectors", [], labels)
    with pytest.raises(ValueError, match="duplicate"):
        resolve_networks("H0-all-detectors", ["ET-triangular", "ET-triangular"], labels)
    with pytest.raises(ValueError, match="no entry for"):
        resolve_networks("H0-all-detectors", ["ET-triangular"], {})
    with pytest.raises(ValueError, match="unknown run"):
        resolve_networks("H0-all-detectors", ["nope"], {"nope": "label"})


def test_committed_latex_labels_survive_the_move_out_of_experiments() -> None:
    labels = load_network_labels()

    assert labels["ET-triangular"] == r"ET-$\Delta$"
    assert labels["ET-2L-misaligned-CE-Hanford"] == r"ET-2L $+$ CE"
    assert figure("H0-merger-rate")["labels"] == [
        r"$H_0$ (fixed $\mathcal{R}_0$)",
        r"$H_0 + \mathcal{R}_0$ (narrow prior)",
    ]
    assert figure("H0-omega-m")["label"] == r"$H_0 + \Omega_m$"
    propagation = figure("modified-propagation-all-detectors")
    assert propagation["marginal_labels"] == [
        r"$\Xi_0$",
        r"$\Xi_0 + n$",
        r"$\Xi_0 + H_0$",
    ]
    assert propagation["h0_labels"] == [r"$\Xi_0 + H_0$"]


def test_base_fiducials_and_analysis_grid_are_read_from_the_base_config() -> None:
    fiducials = load_fiducials(BASE_CONFIG)
    grid = load_analysis_grid(BASE_CONFIG)

    # `importance_relative_ess` is a plotting truth line, not a fiducial: adding
    # it here would change every run's config_sha256 and inject a spurious
    # constant into the sampled model.
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


def test_figure_configs_cover_every_experiment_that_declares_figures() -> None:
    with_figures = {figure(name)["experiment"] for name in RUN_KEYS}
    chains_only = set(load_experiments()) - with_figures

    assert chains_only == {
        "astrophysical-parameters",
        "star-formation-peak",
        "variable-injection-size",
    }
