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
