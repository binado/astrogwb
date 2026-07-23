from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import matplotlib
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
    *, xi_0: float = 1.0, xi_n: float = 1.91, seed: int = 0
) -> xr.DataTree:
    rng = np.random.default_rng(seed)
    posterior = xr.Dataset(
        {
            "xi_0": (("chain", "draw"), rng.normal(xi_0, 0.05, (2, 200))),
            "xi_n": (("chain", "draw"), rng.normal(xi_n, 0.3, (2, 200))),
        },
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
        xi_0_fiducial=1.0,
    )

    assert list(result.columns) == [
        "network",
        "label",
        "detectors",
        "n_detectors",
        "snr",
        "xi0_hdi_lower",
        "xi0_hdi_upper",
        "sigma_xi0_hdi",
        "sigma_xi0_snr",
        "rel_sigma_xi0_hdi",
        "rel_sigma_xi0_snr",
        "n_hdi_lower",
        "n_hdi_upper",
        "sigma_n_hdi",
    ]
    assert result.loc[0, "sigma_xi0_snr"] == pytest.approx(0.1)
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
            xi_0_fiducial=1.0,
        )


def test_xi0_n_constraint_table_latex_contains_snr_column(
    modified_propagation_notebook: ModuleType,
) -> None:
    table = pd.DataFrame(
        [
            {
                "network": "network-a",
                "label": "ET",
                "detectors": "E1,E2,E3",
                "n_detectors": 3,
                "snr": 10.0,
                "xi0_hdi_lower": 0.9,
                "xi0_hdi_upper": 1.1,
                "sigma_xi0_hdi": 0.1,
                "sigma_xi0_snr": 0.1,
                "rel_sigma_xi0_hdi": 0.1,
                "rel_sigma_xi0_snr": 0.1,
                "n_hdi_lower": 1.5,
                "n_hdi_upper": 2.3,
                "sigma_n_hdi": 0.4,
            }
        ]
    )

    latex = modified_propagation_notebook.xi0_n_constraint_table_latex(table)

    assert "SNR" in latex
    assert "network-a" in latex
