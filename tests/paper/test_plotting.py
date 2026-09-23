from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from repo import REPO_ROOT

matplotlib = pytest.importorskip("matplotlib")

from astrogwb.paper import plotting
from astrogwb.paper.config import fiducials
from astrogwb.paper.plotting import Network


def test_paper_mplstyle_sits_next_to_the_module() -> None:
    style_path = Path(plotting.__file__).parent / "paper.mplstyle"
    assert style_path.is_file()


def test_use_paper_style_loads_stylesheet_and_applies_savefig_settings() -> None:
    """`root=` explicitly: the settings file resolves against the checkout root.

    Passing it makes this test independent of where pytest was invoked from
    (and of the `root_dir()` fallback), and turns the assertions into a check
    that `use_paper_style` actually applies what the JSON says rather than that
    a literal survived in the stylesheet.
    """
    plotting.use_paper_style(REPO_ROOT)

    assert matplotlib.pyplot.rcParams["savefig.format"] == plotting.figure_format(
        REPO_ROOT
    )
    assert matplotlib.pyplot.rcParams["savefig.dpi"] == plotting.figure_dpi(REPO_ROOT)
    assert matplotlib.pyplot.rcParams["savefig.bbox"] == "tight"


def test_the_stylesheet_declares_no_savefig_dpi_or_format() -> None:
    """One source for these two, not a literal here and another in each script.

    This is the only thing stopping them being re-added to the stylesheet,
    where they would silently win or lose against `use_paper_style` depending
    on call order.
    """
    text = (Path(plotting.__file__).parent / "paper.mplstyle").read_text()
    directives = [
        line.split(":", 1)[0].strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "savefig.dpi" not in directives
    assert "savefig.format" not in directives
    assert "savefig.bbox" in directives


def test_committed_latex_labels_survive_the_move_out_of_python() -> None:
    """The guard on JSON backslash escaping.

    Every backslash in `config/plotting.json` is doubled. A *missed* doubling
    is loud -- `\\,` and `\\m` are invalid JSON escapes, so the file will not
    parse. Over-doubling is the quiet one: it parses, then emits a literal
    `\\,` into the TeX stream, and with `text.usetex: True` that surfaces as a
    LaTeX compile failure deep inside a figure job. These are the exact strings
    the three figure scripts each carried as `r""` literals before the move.
    """
    assert (
        plotting.parameter_label("H0", REPO_ROOT)
        == r"$H_0\,[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$"
    )
    assert (
        plotting.parameter_label("local_merger_rate", REPO_ROOT)
        == r"$\mathcal{R}_0\,[\mathrm{Gpc^{-3}\,yr^{-1}}]$"
    )
    assert plotting.parameter_label("Omega_m", REPO_ROOT) == r"$\Omega_m$"
    assert plotting.parameter_label("xi_0", REPO_ROOT) == r"$\Xi_0$"
    assert plotting.parameter_label("xi_n", REPO_ROOT) == r"$n$"
    assert (
        plotting.parameter_label("importance_relative_ess", REPO_ROOT)
        == r"$N_{\mathrm{eff}} / N_{\mathrm{inj}}$"
    )


def test_every_fiducial_parameter_has_a_label() -> None:
    """Subset, not equality: the labels also cover a derived diagnostic.

    `importance_relative_ess` is reported beside the parameters and needs a
    label, but it is not a fiducial and must never acquire one.
    """
    labels = plotting.parameter_labels(REPO_ROOT)

    assert set(fiducials(REPO_ROOT)) <= set(labels)
    assert "importance_relative_ess" in labels
    assert "importance_relative_ess" not in fiducials(REPO_ROOT)


def test_an_unlabelled_name_falls_back_to_itself() -> None:
    assert (
        plotting.parameter_label("no_such_parameter", REPO_ROOT) == "no_such_parameter"
    )


def test_spectrum_style_is_black_with_dotted_sh() -> None:
    assert plotting.SPECTRUM["omega_gw"] == "k"
    assert plotting.SPECTRUM["sh"] == "k"
    assert plotting.SPECTRUM["sigma"] == "#0072B2"
    assert plotting.SPECTRUM_LINESTYLES["omega_gw"] == "-"
    assert plotting.SPECTRUM_LINESTYLES["sh"] == ":"
    assert plotting.SPECTRUM_LINESTYLES["sigma"] == "--"


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
    with pytest.raises(ValueError, match="NaN"):
        plotting.plot_corner_for_posterior_grid((x, y), nan_density)

    inf_density = log_density.copy()
    inf_density[0, 0] = np.inf
    with pytest.raises(ValueError, match=r"\+inf"):
        plotting.plot_corner_for_posterior_grid((x, y), inf_density)

    with pytest.raises(ValueError, match="at least one finite entry"):
        plotting.plot_corner_for_posterior_grid(
            (x, y), np.full_like(log_density, -np.inf)
        )


def test_plot_corner_for_posterior_grid_accepts_neg_inf_zero_density() -> None:
    x, y, log_density = _gaussian_grids_and_log_density()
    # Mimic a bounded prior: zero density outside |x| < 2, |y| < 1.
    xx, yy = np.meshgrid(x, y, indexing="ij")
    log_density = np.where(
        (np.abs(xx) < 2.0) & (np.abs(yy) < 1.0), log_density, -np.inf
    )
    fig = plotting.plot_corner_for_posterior_grid((x, y), log_density)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert np.asarray(fig.axes).reshape(2, 2)[1, 0].collections
    matplotlib.pyplot.close(fig)

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
