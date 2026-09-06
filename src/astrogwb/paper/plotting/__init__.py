"""Shared styling for the astrogwb paper application.

Presentation-only helpers: colorblind-safe palettes, the neutral truth-line
style (solid), and a loader for ``paper.mplstyle``. This module is independent
of the ``astrogwb`` package and of the config layer: it imports nothing from
either.

Convention:
- category accents are the default color for single-posterior figures;
- the network palette colors the detector-network comparison figures;
- ``SPECTRUM`` / ``SPECTRUM_LINESTYLES`` style the fiducial dual-axis
  spectrum figure (black curves; dotted $S_h$, solid $\\Omega_{\\mathrm{GW}}$);
- ``combo_colors`` orders the per-parameter-combination marginal overlay;
- ``detector_network_styles`` pairs each ET network with its ET+CE companion;
- truth / fiducial markers are always neutral solid (``TRUTH``), everywhere.

``DETECTOR_NETWORKS`` lives here for the same reason: it is the ordered legend
of the three network-comparison figures, and order is presentation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

_STYLE_PATH = Path(__file__).parent / "paper.mplstyle"


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

# Dual-axis style for the fiducial spectrum figure: both curves and axes
# are black, with dotted $S_h$ against solid $\Omega_{\mathrm{GW}}$.
SPECTRUM: dict[str, str] = {
    "omega_gw": "k",
    "sh": "k",
}
SPECTRUM_LINESTYLES: dict[str, str] = {
    "omega_gw": "-",
    "sh": ":",
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


def plot_corner_for_posterior_grid(
    grids: Sequence[np.ndarray],
    log_density: np.ndarray,
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
        grids[1][j])``.
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
    if not np.all(np.isfinite(log_density)):
        raise ValueError("log_density must be finite everywhere")

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
