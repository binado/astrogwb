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
NOTEBOOK_PATH = REPO_ROOT / "notebooks/paper/mcmc_modified_propagation.py"


def _load_helpers() -> ModuleType:
    """Load notebook helpers without executing the analysis cells."""
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    marker = "\nargs = _parse_args()\n"
    cutoff = source.find(marker)
    if cutoff < 0:
        raise RuntimeError(
            f"could not find helper cutoff marker {marker!r} in {NOTEBOOK_PATH}"
        )
    module = ModuleType("mcmc_modified_propagation")
    module.__file__ = str(NOTEBOOK_PATH)
    if str(NOTEBOOK_PATH.parent) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK_PATH.parent))
    exec(compile(source[:cutoff], str(NOTEBOOK_PATH), "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def modified_propagation_notebook() -> ModuleType:
    return _load_helpers()


def _inference_tree(
    *,
    xi_0: float = 1.0,
    xi_n: float = 1.91,
    include_importance_relative_ess: bool = False,
    seed: int = 0,
) -> xr.DataTree:
    rng = np.random.default_rng(seed)
    variables: dict[str, tuple[tuple[str, str], np.ndarray]] = {
        "xi_0": (("chain", "draw"), rng.normal(xi_0, 0.05, (2, 200))),
        "xi_n": (("chain", "draw"), rng.normal(xi_n, 0.3, (2, 200))),
    }
    if include_importance_relative_ess:
        variables["importance_relative_ess"] = (
            ("chain", "draw"),
            rng.uniform(0.2, 0.9, (2, 200)),
        )
    posterior = xr.Dataset(
        variables,
        coords={"chain": np.arange(2), "draw": np.arange(200)},
    )
    return xr.DataTree.from_dict({"posterior": posterior})


def test_detector_network_styles_pair_et_with_ce(
    modified_propagation_notebook: ModuleType,
) -> None:
    networks = {
        "ET-triangular": ("E1", "E2", "E3"),
        "ET-triangular-CE-Hanford": ("E1", "E2", "E3", "C1"),
        "ET-2L-aligned": ("S1", "R1"),
        "ET-2L-aligned-CE-Hanford": ("S1", "R1", "C1"),
    }

    colors, linestyles = modified_propagation_notebook.detector_network_styles(networks)

    assert colors[0] == colors[1]
    assert colors[2] == colors[3]
    assert colors[0] != colors[2]
    assert linestyles == ["-", "--", "-", "--"]


def test_resolve_networks_supports_cli_overrides(
    modified_propagation_notebook: ModuleType,
) -> None:
    defaults = {"ET-triangular": ("E1", "E2", "E3")}

    resolved = modified_propagation_notebook._resolve_networks(
        defaults, ["ET-extra=X1,X2"], ["ET-triangular", "ET-extra"]
    )

    assert resolved == {
        "ET-triangular": ("E1", "E2", "E3"),
        "ET-extra": ("X1", "X2"),
    }


def test_resolve_networks_rejects_duplicate_definitions(
    modified_propagation_notebook: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="duplicate --network definition"):
        modified_propagation_notebook._resolve_networks(
            {}, ["ET-extra=X1", "ET-extra=X2"], []
        )


def test_resolve_networks_rejects_unknown_selection(
    modified_propagation_notebook: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="unknown selected network"):
        modified_propagation_notebook._resolve_networks(
            {"ET-triangular": ("E1", "E2", "E3")}, [], ["missing-network"]
        )


def test_build_snr_xi0_n_constraint_table(
    modified_propagation_notebook: ModuleType,
) -> None:
    networks = {"network-a": ("E1", "E2", "E3")}
    snr_table = pd.DataFrame(
        [{"network": "network-a", "detectors": "E1,E2,E3", "snr": 10.0}]
    )

    result = modified_propagation_notebook.build_snr_xi0_n_constraint_table(
        networks,
        [_inference_tree(seed=6)],
        ["ET"],
        snr_table,
    )

    assert list(result.columns) == [
        "label",
        "snr",
        "sigma_xi0_hdi",
        "sigma_n_hdi",
        "rel_sigma_xi0_snr",
    ]
    assert result.loc[0, "rel_sigma_xi0_snr"] == pytest.approx(0.1)
    assert result.loc[0, "sigma_xi0_hdi"] > 0
    assert result.loc[0, "sigma_n_hdi"] > 0


def test_build_snr_xi0_n_constraint_table_requires_matching_snr_network(
    modified_propagation_notebook: ModuleType,
) -> None:
    networks = {"network-a": ("E1", "E2", "E3")}
    snr_table = pd.DataFrame([{"network": "other", "detectors": "E1", "snr": 10.0}])

    with pytest.raises(KeyError, match="missing configured network"):
        modified_propagation_notebook.build_snr_xi0_n_constraint_table(
            networks,
            [_inference_tree(seed=7)],
            ["ET"],
            snr_table,
        )


def test_marginal_posterior_plot_helper_accepts_loaded_data(
    modified_propagation_notebook: ModuleType,
) -> None:
    chain_a = _inference_tree(seed=4)
    chain_b = _inference_tree(seed=5)

    figure = modified_propagation_notebook.plot_marginal_posteriors(
        [chain_a, chain_b],
        ["a", "b"],
        var_name="xi_0",
        fiducial=1.0,
    )

    assert len(figure.axes[0].lines) == 3
    assert figure.axes[0].lines[-1].get_xdata()[0] == pytest.approx(1.0)
    plt.close(figure)


def test_xi_n_and_ess_corner_plot_helpers_accept_loaded_data(
    modified_propagation_notebook: ModuleType,
) -> None:
    xi_n = _inference_tree(seed=8)
    xi_n_ess = _inference_tree(include_importance_relative_ess=True, seed=9)
    xi_n_fiducials = {"xi_0": 1.0, "xi_n": 1.91}
    xi_n_ess_fiducials = {
        "xi_0": 1.0,
        "xi_n": 1.91,
        "importance_relative_ess": 1.0,
    }

    xi_n_corner_figure = modified_propagation_notebook.plot_corner(
        [xi_n],
        [r"$\Xi_0 + n$"],
        modified_propagation_notebook.XI_N_VAR_NAMES,
        fiducials=xi_n_fiducials,
    )
    xi_n_ess_corner_figure = modified_propagation_notebook.plot_corner(
        [xi_n_ess],
        [r"$\Xi_0 + n$"],
        modified_propagation_notebook.XI_N_ESS_VAR_NAMES,
        fiducials=xi_n_ess_fiducials,
    )

    assert len(xi_n_corner_figure.axes) == 4
    assert len(xi_n_ess_corner_figure.axes) == 9
    plt.close(xi_n_corner_figure)
    plt.close(xi_n_ess_corner_figure)


def test_xi0_n_constraint_table_latex_contains_snr_column(
    modified_propagation_notebook: ModuleType,
) -> None:
    table = pd.DataFrame(
        [
            {
                "label": "ET",
                "snr": 10.0,
                "sigma_xi0_hdi": 0.1,
                "sigma_n_hdi": 0.4,
                "rel_sigma_xi0_snr": 0.1,
            }
        ]
    )

    latex = modified_propagation_notebook.xi0_n_constraint_table_latex(table)

    assert "SNR" in latex
    assert "Detector Network" in latex
    assert "ET" in latex
    assert r"\sigma_{\Xi_0}^{\rm HDI}/\Xi_0" not in latex
