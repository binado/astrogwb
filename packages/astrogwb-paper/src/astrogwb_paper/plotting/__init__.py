"""Shared paper-figure styling for ``astrogwb-paper``.

Presentation-only helpers: colorblind-safe palettes, the neutral truth-line
style (solid), and a loader for ``paper.mplstyle``. This module is independent
of the ``astrogwb`` package: it imports nothing from it.

Convention:
- category accents are the default color for single-posterior figures;
- the network palette colors the detector-network comparison figures;
- ``SPECTRUM`` colors the fiducial dual-axis spectrum figure
  (Tol muted indigo + ColorBrewer Greens);
- ``combo_colors`` orders the per-parameter-combination marginal overlay;
- ``detector_network_styles`` pairs each ET network with its ET+CE companion;
- truth / fiducial markers are always neutral solid (``TRUTH``), everywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

from astrogwb_paper.config.figures import Network

_STYLE_PATH = Path(__file__).parent / "paper.mplstyle"

CATEGORY: dict[str, str] = {
    "cosmology": "#0072B2",
    "modified_propagation": "#D55E00",
    "astrophysical": "#009E73",
}

# Dual-axis accents for the fiducial spectrum figure: Paul Tol muted indigo
# and a darker ColorBrewer Greens step for S_h.
SPECTRUM: dict[str, str] = {
    "omega_gw": "#332288",
    "sh": "#006D2C",
}

NETWORK: dict[str, str] = {
    "ET": "#E69F00",
    "CE": "#56B4E9",
    "LISA": "#CC79A7",
    "ET+CE": "#F0E442",
}

# linewidth matches matplotlib's default lines.linewidth (and corner's truth
# bars, which only accept truth_color and inherit the rcParam).
TRUTH: dict[str, object] = {"color": "0.15", "linestyle": "-", "linewidth": 1.5}

CORNER_LEVELS: tuple[float, ...] = (0.6827, 0.9545)

DETECTOR_COMPARISON_LEGEND: dict[str, object] = {
    "ncol": 3,
    "loc": "lower center",
    "bbox_to_anchor": (0.5, 1.02),
    "frameon": False,
    "borderaxespad": 0,
    "handlelength": 2.5,
}

MERGER_RATE_LEGEND: dict[str, object] = {
    "loc": "best",
    "frameon": False,
}


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
        "smooth1d": 1.0,
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


def _base_network_name(name: str) -> str:
    """Map an ET+CE network name onto its ET-only counterpart."""
    suffix = "-CE-Hanford"
    return name.removesuffix(suffix)


def detector_network_styles(
    networks: Sequence[Network],
) -> tuple[list[str], list[str]]:
    """Shared color per ET / ET+CE pair; dashed linestyle for CE companions.

    Both lists are positional: they index the ``networks`` sequence, whose
    order comes from the figure config's run array.
    """
    bases: list[str] = []
    for network in networks:
        base = _base_network_name(network.name)
        if base not in bases:
            bases.append(base)
    palette = combo_colors(len(bases))
    color_by_base = dict(zip(bases, palette, strict=True))
    colors = [color_by_base[_base_network_name(n.name)] for n in networks]
    linestyles = ["--" if n.name.endswith("-CE-Hanford") else "-" for n in networks]
    return colors, linestyles
