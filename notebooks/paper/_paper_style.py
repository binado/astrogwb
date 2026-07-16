"""Shared paper-figure styling for the notebooks in this directory.

Presentation-only helpers: colorblind-safe palettes, the neutral truth-line
style, and a loader for ``paper.mplstyle``. This module is independent of the
``astrogwb`` package: it imports nothing from it, and nothing in the package
imports this.

Convention:
- category accents are the default color for single-posterior figures;
- the network palette colors the detector-network comparison figures;
- ``combo_colors`` orders the per-parameter-combination marginal overlay;
- truth / fiducial markers are always neutral dashed (``TRUTH``), everywhere.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

_STYLE_PATH = Path(__file__).parent / "paper.mplstyle"

CATEGORY: dict[str, str] = {
    "cosmology": "#0072B2",
    "modified_propagation": "#D55E00",
    "astrophysical": "#009E73",
}

NETWORK: dict[str, str] = {
    "ET": "#E69F00",
    "CE": "#56B4E9",
    "LISA": "#CC79A7",
    "ET+CE": "#F0E442",
}

TRUTH: dict[str, object] = {"color": "0.15", "linestyle": "--", "linewidth": 1.0}

CORNER_LEVELS: tuple[float, ...] = (0.6827, 0.9545)


def use_paper_style() -> None:
    """Apply ``paper.mplstyle`` to the current matplotlib session."""
    plt.style.use(str(_STYLE_PATH))


def get_corner_kwargs(**overrides: object) -> dict[str, object]:
    """Shared ``corner.corner()`` defaults.

    Corner does not read most rcParams, so the axis-label and title font sizes
    are pulled from the active rcParams here. Pass keyword overrides to replace
    any default for a single call.
    """
    kwargs: dict[str, object] = {
        "plot_datapoints": True,
        "plot_density": True,
        "fill_contours": True,
        "levels": CORNER_LEVELS,
        "truth_color": str(TRUTH["color"]),
        "max_n_ticks": 4,
        "label_kwargs": {"fontsize": plt.rcParams["axes.labelsize"]},
        "title_kwargs": {"fontsize": plt.rcParams["axes.titlesize"]},
    }
    return kwargs | overrides


def combo_colors(n: int) -> list[str]:
    """Ordered qualitative colors for the per-parameter-combination overlay."""
    return [to_hex(c) for c in plt.cm.viridis(np.linspace(0.1, 0.9, n))]
