from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

from astrogwb.paper import plotting
from astrogwb.paper.plotting import Network


def test_paper_mplstyle_sits_next_to_the_module() -> None:
    style_path = Path(plotting.__file__).parent / "paper.mplstyle"
    assert style_path.is_file()


def test_use_paper_style_loads_stylesheet() -> None:
    plotting.use_paper_style()
    assert matplotlib.pyplot.rcParams["savefig.format"] == "pdf"


def test_spectrum_style_is_black_with_dotted_sh() -> None:
    assert plotting.SPECTRUM["omega_gw"] == "k"
    assert plotting.SPECTRUM["sh"] == "k"
    assert plotting.SPECTRUM_LINESTYLES["omega_gw"] == "-"
    assert plotting.SPECTRUM_LINESTYLES["sh"] == ":"


def test_get_corner_kwargs_and_combo_colors() -> None:
    kwargs = plotting.get_corner_kwargs(max_n_ticks=3)
    assert kwargs["levels"] == plotting.CORNER_LEVELS
    assert kwargs["truth_color"] == str(plotting.TRUTH["color"])
    assert kwargs["max_n_ticks"] == 3

    colors = plotting.combo_colors(4)
    assert len(colors) == 4
    assert all(color.startswith("#") for color in colors)


def _gaussian_grids_and_log_density(
    nx: int = 64, ny: int = 48
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A separable 2D Gaussian log-posterior on uniform bin-center grids."""
    x = np.linspace(-3.0, 3.0, nx)
    y = np.linspace(-4.0, 4.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return x, y, -0.5 * (xx**2 + (yy / 2.0) ** 2)


def test_plot_corner_for_posterior_grid_2d() -> None:
    x, y, log_density = _gaussian_grids_and_log_density()
    fig = plotting.plot_corner_for_posterior_grid(
        (x, y), log_density, labels=["x", "y"], truths=[0.1, -0.2]
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    # 2x2 corner layout; the joint panel (1, 0) carries the filled contours
    # and the diagonal panels the 1D marginals.
    assert len(fig.axes) == 4
    joint = np.asarray(fig.axes).reshape(2, 2)[1, 0]
    assert joint.collections
    for index in range(2):
        assert np.asarray(fig.axes).reshape(2, 2)[index, index].lines
    matplotlib.pyplot.close(fig)


def test_plot_corner_for_posterior_grid_expands_axes_with_truths() -> None:
    x, y, log_density = _gaussian_grids_and_log_density()
    grid_extent = (x[0] - (x[1] - x[0]) / 2, x[-1] + (x[1] - x[0]) / 2)

    fig = plotting.plot_corner_for_posterior_grid(
        (x, y), log_density, truths=[4.0, 0.0]
    )
    axes = np.asarray(fig.axes).reshape(2, 2)
    # The grid spans [-3, 3]; after rendering, both the joint panel and the
    # x marginal must show the out-of-grid truth.
    assert axes[1, 0].get_xlim()[1] == pytest.approx(4.0)
    assert axes[0, 0].get_xlim()[1] == pytest.approx(4.0)
    matplotlib.pyplot.close(fig)

    # Without truths, the limits are exactly the grid's cell-edge extent:
    # the histogram range must never exceed it (otherwise corner's bins are
    # wider than the grid cells and the density/contours are distorted).
    fig = plotting.plot_corner_for_posterior_grid((x, y), log_density)
    axes = np.asarray(fig.axes).reshape(2, 2)
    assert axes[1, 0].get_xlim() == pytest.approx(grid_extent)
    matplotlib.pyplot.close(fig)


def test_plot_corner_for_posterior_grid_1d() -> None:
    x = np.linspace(-3.0, 3.0, 64)
    fig = plotting.plot_corner_for_posterior_grid((x,), -0.5 * x**2, labels=["x"])
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1
    assert fig.axes[0].lines
    matplotlib.pyplot.close(fig)


def test_plot_corner_for_posterior_grid_1d_truth() -> None:
    x = np.linspace(-3.0, 3.0, 64)
    # An out-of-grid truth must not crash (the pinned corner revision's
    # overplot_lines breaks on 1D figures) and must stay visible.
    fig = plotting.plot_corner_for_posterior_grid((x,), -0.5 * x**2, truths=[4.0])
    assert fig.axes[0].get_xlim()[1] == pytest.approx(4.0)
    matplotlib.pyplot.close(fig)


def test_plot_corner_for_posterior_grid_validation() -> None:
    x, y, log_density = _gaussian_grids_and_log_density()

    nonuniform = x.copy()
    nonuniform[1] += 0.01
    with pytest.raises(ValueError, match="uniform"):
        plotting.plot_corner_for_posterior_grid((nonuniform, y), log_density)

    with pytest.raises(ValueError, match="strictly increasing"):
        plotting.plot_corner_for_posterior_grid((x[::-1], y), log_density)

    with pytest.raises(ValueError, match="does not match"):
        plotting.plot_corner_for_posterior_grid((x, y), np.zeros((x.size, x.size)))

    with pytest.raises(ValueError, match="1 or 2 grid dimensions"):
        plotting.plot_corner_for_posterior_grid(
            (x, y, x), np.zeros((x.size, y.size, x.size))
        )

    nan_density = log_density.copy()
    nan_density[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        plotting.plot_corner_for_posterior_grid((x, y), nan_density)

    with pytest.raises(ValueError, match="truths"):
        plotting.plot_corner_for_posterior_grid((x, y), log_density, truths=[0.0])


def test_detector_network_styles_pairs_et_and_et_plus_ce() -> None:
    # `detector_network_styles` reads only `Network.name`, so the detectors are
    # irrelevant here; hard-code them instead of resolving gitignored configs.
    networks = [Network(name, label, ()) for name, label in plotting.DETECTOR_NETWORKS]
    colors, linestyles = plotting.detector_network_styles(networks)

    assert len(colors) == len(linestyles) == len(networks)
    # Each ET-only network shares its color with its CE companion, and the
    # companion is the dashed one.
    for base, companion in zip(colors[::2], colors[1::2], strict=True):
        assert base == companion
    assert linestyles == ["-", "--"] * 3
    assert len(set(colors)) == 3
