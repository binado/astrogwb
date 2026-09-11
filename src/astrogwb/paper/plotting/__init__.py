"""Shared styling for the astrogwb paper application.

Presentation-only helpers: colorblind-safe palettes, the neutral truth-line
style (solid), a loader for ``paper.mplstyle``, the LaTeX parameter labels
and savefig settings that ``config/plotting.json`` carries, and
:func:`save_figures`, the single writer that applies them. This module is
independent of the ``astrogwb`` package and of the config layer: it imports
nothing from either, at module scope or inside a function body -- it reads that
one JSON file with stdlib ``json``, lazily, so the ``Snakefile`` can import
``DETECTOR_NETWORK_RUNS`` while building the DAG without paying for any of it.

Convention:
- category accents are the default color for single-posterior figures;
- the network palette colors the detector-network comparison figures;
- ``SPECTRUM`` / ``SPECTRUM_LINESTYLES`` style the fiducial dual-axis
  spectrum figure (black curves; dotted $S_h$, solid $\\Omega_{\\mathrm{GW}}$;
  dashed $\\sigma$ in the Okabe-Ito blue);
- ``combo_colors`` orders the per-parameter-combination marginal overlay;
- ``detector_network_styles`` pairs each ET network with its ET+CE companion;
- truth / fiducial markers are always neutral solid (``TRUTH``), everywhere.

``DETECTOR_NETWORKS`` lives here for the same reason: it is the ordered legend
of the three network-comparison figures, and order is presentation.

The split with ``config/``: *order and structure* are Python, *values* are
data. So the ordered network legend is ``DETECTOR_NETWORKS`` here, while the
detector list behind each name is ``config/networks.json``; and the LaTeX label
for a *parameter* is ``config/plotting.json``, reached through
:func:`parameter_label`, while the label for a *network* stays in
``DETECTOR_NETWORKS`` because nothing can read it without also needing the
order it sits in.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

_STYLE_PATH = Path(__file__).parent / "paper.mplstyle"

#: Relative to the working directory -- the repository root for the workflow
#: and every script -- matching `astrogwb.paper.config.runs`. Read lazily, never
#: at import: the `Snakefile` imports this module for `DETECTOR_NETWORK_RUNS`
#: while building the DAG, and that import must stay free of file I/O.
_SETTINGS_PATH = Path("config/plotting.json")


@cache
def _figure_settings(root: Path | None = None) -> Mapping[str, Any]:
    """Parse ``config/plotting.json`` once.

    Read with stdlib ``json`` rather than ``astrogwb.paper.utils.load_mapping``
    so this module keeps importing nothing from the config layer, at module
    scope or inside a function body.

    Proxied because ``@cache`` hands every caller the same object.
    """
    return MappingProxyType(
        json.loads(((root or Path()) / _SETTINGS_PATH).read_text(encoding="utf-8"))
    )


def figure_dpi(root: Path | None = None) -> int:
    """Raster resolution for saved figures. Applied by :func:`use_paper_style`."""
    return int(_figure_settings(root)["figure_dpi"])


def figure_format(root: Path | None = None) -> str:
    """Default savefig container format. Applied by :func:`use_paper_style`.

    Near-inert in practice: ``savefig.format`` only decides anything when
    ``savefig`` is handed a path with no extension, and every figure script
    receives an explicit ``.pdf`` path from a ``Snakefile`` ``output:``
    declaration. It records intent; changing it does not re-target the
    workflow.
    """
    return str(_figure_settings(root)["figure_format"])


def parameter_labels(root: Path | None = None) -> dict[str, str]:
    """LaTeX display labels by parameter name, from ``config/plotting.json``.

    Covers every fiducial plus ``importance_relative_ess``, which is a derived
    diagnostic rather than a parameter -- so this is a superset of
    :func:`astrogwb.paper.config.fiducials`, not a match.

    Parameter labels are data; *network* labels are not. ``DETECTOR_NETWORKS``
    stays in Python because its order is load-bearing presentation.
    """
    return dict(_figure_settings(root)["labels"])


def parameter_label(name: str, root: Path | None = None) -> str:
    """The LaTeX label for ``name``, or ``name`` itself if none is declared."""
    return parameter_labels(root).get(name, name)


@dataclass(frozen=True)
class Network:
    """One detector network: its run name, LaTeX label, and detector list.

    The name and label are presentation, owned here alongside
    ``DETECTOR_NETWORKS``. The detector list is scientific and is filled in
    from the run's own config by
    :func:`astrogwb.paper.config.runs.resolve_networks`.
    """

    name: str
    label: str
    detectors: tuple[str, ...]


CATEGORY: dict[str, str] = {
    "cosmology": "#0072B2",
    "modified_propagation": "#D55E00",
}

# Dual-axis style for the fiducial spectrum figure: $S_h$ and
# $\Omega_{\mathrm{GW}}$ are black (dotted / solid); $\sigma$ is
# dashed Okabe-Ito blue so it is readable on the shared $S_h$ axis.
SPECTRUM: dict[str, str] = {
    "omega_gw": "k",
    "sh": "k",
    "sigma": "#0072B2",
}
SPECTRUM_LINESTYLES: dict[str, str] = {
    "omega_gw": "-",
    "sh": ":",
    "sigma": "--",
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

# The detector networks compared in every network figure, as (run name, LaTeX
# label) in legend order. Declaration order is load-bearing: it drives chain
# order on argv, legend order, and the color/linestyle assignment made by
# `detector_network_styles`.
#
# Only the label is owned here. Each run's *detector list* is read back out of
# that run's own config layers by
# `astrogwb.paper.config.runs.resolve_networks`, so the detectors a figure
# reports an SNR for are always the ones its chain was sampled with.
DETECTOR_NETWORKS: tuple[tuple[str, str], ...] = (
    ("ET-triangular", r"ET-$\Delta$"),
    ("ET-triangular-CE-Hanford", r"ET-$\Delta$ $+$ CE"),
    ("ET-2L-aligned", "ET-2L-par"),
    ("ET-2L-aligned-CE-Hanford", r"ET-2L-par $+$ CE"),
    ("ET-2L-misaligned", "ET-2L"),
    ("ET-2L-misaligned-CE-Hanford", r"ET-2L $+$ CE"),
)

DETECTOR_NETWORK_RUNS: tuple[str, ...] = tuple(name for name, _ in DETECTOR_NETWORKS)


def use_paper_style(root: Path | None = None) -> None:
    """Apply ``paper.mplstyle`` and configured savefig settings."""
    plt.style.use(str(_STYLE_PATH))
    plt.rcParams["savefig.dpi"] = figure_dpi(root)
    plt.rcParams["savefig.format"] = figure_format(root)


def save_figures(
    figures: Mapping[Path, Figure],
    *,
    figure_dpi: int | None = None,
    figure_format: str | None = None,
    root: Path | None = None,
) -> list[Path]:
    """Write each figure to its path, creating the parent directory.

    ``figures`` maps the output path to the figure to write, so the caller keeps
    owning *where* -- its ``--output-*`` flags -- while this owns *how*: the dpi
    and format come from ``config/plotting.json``, and the written paths are
    returned in iteration order so a caller can report them without restating
    them.

    ``figure_format`` names the extension a path that has none is given. A path
    that already names one keeps it, which is why the setting is near-inert
    here: every caller's ``--output-*`` flag ends in ``.pdf``.
    """
    settings = _figure_settings(root)
    dpi = int(settings["figure_dpi"]) if figure_dpi is None else figure_dpi
    fallback_format = (
        str(settings["figure_format"]) if figure_format is None else figure_format
    )

    saved: list[Path] = []
    for output, figure in figures.items():
        path = Path(output)
        if not path.suffix:
            path = path.with_suffix(f".{fallback_format}")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, format=path.suffix.lstrip("."))
        print("saved figure:", path)
        saved.append(path)


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


def plot_corner_for_posterior_grid(
    grids: Sequence[ArrayLike],
    log_density: ArrayLike,
    *,
    labels: Sequence[str] | None = None,
    truths: Sequence[float | None] | None = None,
    smooth: float | None = None,
    **corner_kwargs: object,
) -> plt.Figure:
    """Corner plot of a posterior evaluated on a grid, via ``corner.corner()``.

    The posterior density is recovered by exponentiating ``log_density``
    (unnormalized; stabilized by subtracting its maximum), and every grid
    point is passed to ``corner.corner()`` as a pseudo-sample weighted by its
    density value. ``bins`` are set to coincide with the grid cells and
    ``range`` with the cell-edge extents, so corner's histograms *are* the
    grid and its credible-region levels and quantiles carry the same meaning
    as for chains -- the density is not discretized a second time.

    Each grid must be strictly increasing and *uniform*: uniform cell areas
    are what make the density value proportional to the enclosed mass. For
    non-uniform grids, use the density-native renderer in
    ``notebooks/logposterior_grid.py`` instead.

    ``plot_datapoints`` and ``plot_density`` default to ``False`` here: the
    pseudo-samples cover the full parameter rectangle uniformly, so corner's
    datapoint scatter would be a featureless gray fog. These and every other
    default can be overridden through ``corner_kwargs``.

    Parameters
    ----------
    grids
        Bin-center coordinates, one 1D array per dimension (1 or 2).
    log_density
        Unnormalized log-posterior with shape ``tuple(len(g) for g in
        grids)``; ``log_density[i, j]`` corresponds to ``(grids[0][i],
        grids[1][j])``. ``-inf`` entries are allowed and become zero-weight
        cells (e.g. points outside a bounded prior); NaN and ``+inf`` are
        rejected.
    labels
        Axis labels, one per dimension.
    truths
        Marker positions, one per dimension (``None`` entries omit the
        marker). The axis limits are expanded to include them -- after
        corner has rendered, so the histogram range stays aligned with the
        grid cells and the density is unaffected.
    smooth
        Gaussian smoothing width in grid cells for the 2D panel, forwarded
        to ``corner.corner()``.
    corner_kwargs
        Additional keyword arguments forwarded to ``corner.corner()``; they
        take precedence over this function's defaults.

    Returns
    -------
    matplotlib.figure.Figure
        The figure produced by ``corner.corner()``.
    """
    import corner

    grids = tuple(np.asarray(grid, dtype=np.float64) for grid in grids)
    ndim = len(grids)
    if ndim not in (1, 2):
        raise ValueError(f"expected 1 or 2 grid dimensions, got {ndim}")
    for grid in grids:
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError("each grid must be a 1D array with at least 2 points")
        spacings = np.diff(grid)
        if np.any(spacings <= 0):
            raise ValueError("each grid must be strictly increasing")
        if not np.allclose(spacings, spacings[0]):
            raise ValueError(
                "each grid must be uniform so that cell areas are equal and "
                "density values are proportional to enclosed mass; for "
                "non-uniform grids use notebooks/logposterior_grid.py"
            )

    log_density = np.asarray(log_density, dtype=np.float64)
    expected_shape = tuple(grid.size for grid in grids)
    if log_density.shape != expected_shape:
        raise ValueError(
            f"log_density shape {log_density.shape} does not match "
            f"the grids {expected_shape}"
        )
    if np.any(np.isnan(log_density)) or np.any(log_density == np.inf):
        raise ValueError(
            "log_density must not contain NaN or +inf; use -inf for zero density"
        )
    if not np.any(np.isfinite(log_density)):
        raise ValueError(
            "log_density must contain at least one finite entry; maximum "
            "subtraction is undefined for an all -inf grid"
        )

    # Cell-edge extents. The histogram range is kept exactly at these bounds:
    # widening it to include out-of-grid truths would rebin the grid points
    # into non-cell-aligned bins and distort the density and its contours.
    extents: list[list[float]] = []
    for grid in grids:
        half = 0.5 * (grid[1] - grid[0])
        extents.append([float(grid[0] - half), float(grid[-1] + half)])
    if truths is not None:
        truths = tuple(truths)
        if len(truths) != ndim:
            raise ValueError(f"expected {ndim} truths, got {len(truths)}")

    weights = np.exp(log_density - log_density.max()).ravel()
    # corner requires 2D (nsamples, ndim) input; the pinned revision's 1D
    # ndarray path reshapes as data[None, :, :] and crashes, so present the
    # 1D grid as (npoints, 1).
    data = (
        grids[0][:, None]
        if ndim == 1
        else np.column_stack(
            [coords.ravel() for coords in np.meshgrid(*grids, indexing="ij")]
        )
    )

    kwargs: dict[str, object] = {
        **get_corner_kwargs(),
        "bins": [grid.size for grid in grids],
        "range": [tuple(extent) for extent in extents],
        "weights": weights,
        "plot_datapoints": False,
        "plot_density": False,
        "smooth": smooth,
        # corner's overplot_lines crashes for 1D figures in the pinned
        # revision (axes[k1, k1] on a non-subscriptable Axes), so in 1D the
        # truth line is drawn manually after rendering, below.
        "truths": list(truths) if truths is not None and ndim == 2 else None,
        "labels": list(labels) if labels is not None else None,
    }
    kwargs |= corner_kwargs
    fig = corner.corner(data, **kwargs)  # type: ignore[arg-type]
    if fig is None:  # pragma: no cover - corner always creates a figure here
        raise RuntimeError("corner did not create a figure")

    # Expand the visible axes to include out-of-grid truths after rendering:
    # corner's truth artists (truly spanning axvline/axhline plus square
    # markers) are drawn regardless of the limits and simply reappear once
    # the limits cover them.
    if truths is not None:
        axes = np.asarray(fig.axes).reshape(ndim, ndim)
        for index, truth in enumerate(truths):
            if truth is None:
                continue
            lo = min(extents[index][0], float(truth))
            hi = max(extents[index][1], float(truth))
            for row in range(index, ndim):
                axes[row, index].set_xlim(lo, hi)
            for col in range(index):
                axes[index, col].set_ylim(lo, hi)
        if ndim == 1 and truths[0] is not None:
            fig.axes[0].axvline(truths[0], color=str(kwargs["truth_color"]))
    return fig


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
