from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")

from astrogwb_paper import plotting


def test_paper_mplstyle_sits_next_to_the_module() -> None:
    style_path = Path(plotting.__file__).parent / "paper.mplstyle"
    assert style_path.is_file()


def test_use_paper_style_loads_stylesheet() -> None:
    plotting.use_paper_style()
    assert matplotlib.pyplot.rcParams["savefig.format"] == "pdf"


def test_get_corner_kwargs_and_combo_colors() -> None:
    kwargs = plotting.get_corner_kwargs(max_n_ticks=3)
    assert kwargs["levels"] == plotting.CORNER_LEVELS
    assert kwargs["truth_color"] == str(plotting.TRUTH["color"])
    assert kwargs["max_n_ticks"] == 3

    colors = plotting.combo_colors(4)
    assert len(colors) == 4
    assert all(color.startswith("#") for color in colors)


def test_detector_network_styles_pairs_et_and_et_plus_ce() -> None:
    from astrogwb_paper.config.figures import resolve_networks

    networks = resolve_networks("cosmological-parameters", plotting.DETECTOR_NETWORKS)
    colors, linestyles = plotting.detector_network_styles(networks)

    assert len(colors) == len(linestyles) == len(networks)
    # Each ET-only network shares its color with its CE companion, and the
    # companion is the dashed one.
    for base, companion in zip(colors[::2], colors[1::2], strict=True):
        assert base == companion
    assert linestyles == ["-", "--"] * 3
    assert len(set(colors)) == 3
