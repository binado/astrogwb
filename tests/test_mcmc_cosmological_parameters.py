from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr


matplotlib.use("Agg")
REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks/paper/mcmc_cosmological_parameters.py"


def _load_helpers() -> ModuleType:
    """Load notebook helpers without executing the analysis cells."""
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    marker = "\nargs = _parse_args()\n"
    cutoff = source.find(marker)
    if cutoff < 0:
        raise RuntimeError(
            f"could not find helper cutoff marker {marker!r} in {NOTEBOOK_PATH}"
        )
    module = ModuleType("mcmc_cosmological_parameters")
    module.__file__ = str(NOTEBOOK_PATH)
    if str(NOTEBOOK_PATH.parent) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK_PATH.parent))
    exec(compile(source[:cutoff], str(NOTEBOOK_PATH), "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def cosmology_notebook() -> ModuleType:
    return _load_helpers()


def _inference_tree(
    *,
    include_h0: bool = True,
    include_local_merger_rate: bool = False,
    include_omega_m: bool = False,
    seed: int = 0,
) -> xr.DataTree:
    rng = np.random.default_rng(seed)
    variables: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    if include_h0:
        variables["H0"] = (("chain", "draw"), rng.normal(67.66, 1.0, (2, 200)))
    if include_local_merger_rate:
        variables["local_merger_rate"] = (
            ("chain", "draw"),
            rng.normal(161.0, 3.0, (2, 200)),
        )
    if include_omega_m:
        variables["Omega_m"] = (("chain", "draw"), rng.normal(0.31, 0.02, (2, 200)))
    posterior = xr.Dataset(
        variables,
        coords={"chain": np.arange(2), "draw": np.arange(200)},
    )
    return xr.DataTree.from_dict({"posterior": posterior})


def test_validate_inference_data_rejects_mismatched_labels(
    cosmology_notebook: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="1 inference trees but 0 labels"):
        cosmology_notebook.validate_inference_data([_inference_tree()], [])


def test_validate_inference_data_requires_h0(
    cosmology_notebook: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="missing posterior variable.*H0"):
        cosmology_notebook.validate_inference_data(
            [_inference_tree(include_h0=False)], ["missing"]
        )


def test_select_corner_inference_data_finds_narrow_and_broad_chains(
    cosmology_notebook: ModuleType,
) -> None:
    inference_data = [
        _inference_tree(seed=1),
        _inference_tree(include_local_merger_rate=True, seed=2),
        _inference_tree(include_local_merger_rate=True, seed=3),
    ]
    selected, labels, indices = cosmology_notebook.select_corner_inference_data(
        inference_data, ["fixed", "narrow", "broad"]
    )

    assert selected == inference_data[1:]
    assert labels == ["narrow", "broad"]
    assert indices == [1, 2]


def test_select_corner_inference_data_requires_exactly_two_joint_chains(
    cosmology_notebook: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="exactly two.*found 1"):
        cosmology_notebook.select_corner_inference_data(
            [_inference_tree(), _inference_tree(include_local_merger_rate=True)],
            ["fixed", "narrow"],
        )


def test_detector_network_styles_pair_et_with_ce(
    cosmology_notebook: ModuleType,
) -> None:
    networks = {
        "ET-triangular": ("E1", "E2", "E3"),
        "ET-triangular-CE-Hanford": ("E1", "E2", "E3", "C1"),
        "ET-2L-aligned": ("S1", "R1"),
        "ET-2L-aligned-CE-Hanford": ("S1", "R1", "C1"),
    }

    colors, linestyles = cosmology_notebook.detector_network_styles(networks)

    assert colors[0] == colors[1]
    assert colors[2] == colors[3]
    assert colors[0] != colors[2]
    assert linestyles == ["-", "--", "-", "--"]


def test_posterior_and_corner_plot_helpers_accept_loaded_data(
    cosmology_notebook: ModuleType,
) -> None:
    narrow = _inference_tree(include_local_merger_rate=True, seed=4)
    broad = _inference_tree(include_local_merger_rate=True, seed=5)
    omega_m = _inference_tree(include_omega_m=True, seed=6)
    merger_rate_fiducials = {"H0": 67.66, "local_merger_rate": 161.0}
    omega_m_fiducials = {"H0": 67.66, "Omega_m": 0.3096}

    posterior_figure = cosmology_notebook.plot_h0_posteriors(
        [narrow, broad], ["narrow", "broad"]
    )
    narrow_corner_figure = cosmology_notebook.plot_corner(
        [narrow],
        ["narrow"],
        cosmology_notebook.MERGER_RATE_VAR_NAMES,
        fiducials=merger_rate_fiducials,
    )
    broad_corner_figure = cosmology_notebook.plot_corner(
        [broad],
        ["broad"],
        cosmology_notebook.MERGER_RATE_VAR_NAMES,
        fiducials=merger_rate_fiducials,
    )
    omega_m_corner_figure = cosmology_notebook.plot_corner(
        [omega_m],
        [r"$H_0 + \Omega_m$"],
        cosmology_notebook.OMEGA_M_VAR_NAMES,
        fiducials=omega_m_fiducials,
    )

    assert len(posterior_figure.axes[0].lines) == 2
    assert len(narrow_corner_figure.axes) == 4
    assert len(broad_corner_figure.axes) == 4
    assert len(omega_m_corner_figure.axes) == 4
    plt.close(posterior_figure)
    plt.close(narrow_corner_figure)
    plt.close(broad_corner_figure)
    plt.close(omega_m_corner_figure)


def test_build_snr_h0_constraint_table(
    cosmology_notebook: ModuleType,
) -> None:
    networks = {"network-a": ("E1", "E2", "E3")}
    snr_table = pd.DataFrame(
        [{"network": "network-a", "detectors": "E1,E2,E3", "snr": 10.0}]
    )

    result = cosmology_notebook.build_snr_h0_constraint_table(
        networks,
        [_inference_tree(seed=6)],
        ["ET"],
        snr_table,
        h0_fiducial=67.66,
    )

    assert result.loc[0, "sigma_h0_snr"] == pytest.approx(6.766)
    assert result.loc[0, "rel_sigma_h0_snr"] == pytest.approx(0.1)
    assert result.loc[0, "sigma_h0_hdi"] > 0
