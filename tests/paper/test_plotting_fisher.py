"""Degeneracy figures of a Fisher forecast."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from astrogwb.paper.plotting.fisher import (
    plot_fisher_eigenmodes,
    plot_template_composition,
    plot_whitened_derivatives,
)


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


@pytest.fixture
def frequencies() -> np.ndarray:
    return np.geomspace(2.0, 200.0, 16)


@pytest.fixture
def whitened(frequencies: np.ndarray) -> np.ndarray:
    return np.stack([frequencies**-1.0, -2.0 * frequencies**-0.5], axis=-1)


def test_whitened_derivative_curves_have_unit_norm(
    frequencies: np.ndarray, whitened: np.ndarray
) -> None:
    fig = plot_whitened_derivatives(frequencies, whitened, ("a", "b"))

    for line in fig.axes[0].get_lines()[:2]:
        assert float(np.sum(np.asarray(line.get_ydata()) ** 2)) == pytest.approx(1.0)


def test_information_kept_is_all_at_the_lowest_frequency_and_falls(
    frequencies: np.ndarray, whitened: np.ndarray
) -> None:
    fig = plot_whitened_derivatives(frequencies, whitened, ("a", "b"))

    kept = np.asarray(fig.axes[1].get_lines()[0].get_ydata())
    assert kept[0] == pytest.approx(1.0)
    assert np.all(np.diff(kept) <= 0.0)


def test_whitened_derivative_without_information_is_named_not_drawn(
    frequencies: np.ndarray, whitened: np.ndarray
) -> None:
    whitened[:, 1] = 0.0

    fig = plot_whitened_derivatives(frequencies, whitened, ("a", "b"))

    legend_box = fig.axes[0].get_legend()
    assert legend_box is not None
    legend = [text.get_text() for text in legend_box.get_texts()]
    assert legend == ["a", "b (no information)"]
    assert len(fig.axes[1].get_lines()) == 1


def test_whitened_derivatives_reject_a_shape_mismatch(
    frequencies: np.ndarray, whitened: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="whitened has shape"):
        plot_whitened_derivatives(frequencies, whitened, ("a",))


def test_eigenmode_unconstrained_mode_reaches_the_top_marked_infinite() -> None:
    fig = plot_fisher_eigenmodes(np.array([0.1, np.inf]), np.eye(2), ("a", "b"))

    width_ax = fig.axes[0]
    bar = width_ax.patches[1]
    assert isinstance(bar, Rectangle)
    assert bar.get_y() + bar.get_height() == pytest.approx(width_ax.get_ylim()[1])
    assert r"$\infty$" in [text.get_text() for text in width_ax.texts]


def test_eigenmode_prior_markers_skip_modes_without_a_prior() -> None:
    fig = plot_fisher_eigenmodes(
        np.array([0.1, 2.0]),
        np.eye(2),
        ("a", "b"),
        prior_sigmas=np.array([np.inf, 1.0]),
    )

    offsets = np.asarray(fig.axes[0].collections[0].get_offsets(), dtype=float)
    np.testing.assert_allclose(offsets, [[2.0, 1.0]])


def test_eigenmode_loadings_are_annotated_per_cell() -> None:
    directions = np.array([[0.6, -0.8], [0.8, 0.6]])

    fig = plot_fisher_eigenmodes(np.array([0.1, 1.0]), directions, ("a", "b"))

    texts = [text.get_text() for text in fig.axes[1].texts]
    assert texts == ["$+0.60$", "$-0.80$", "$+0.80$", "$+0.60$"]


def test_eigenmodes_reject_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="expected sigmas"):
        plot_fisher_eigenmodes(np.ones(2), np.eye(3), ("a", "b"))


def test_template_composition_bars_fill_each_mode_to_one() -> None:
    fractions = np.array([[0.9, 0.99], [0.1, 0.7]])

    fig = plot_template_composition(fractions, ("amplitude", "+1PN"), ("1", "2"))

    widths = np.zeros(2)
    for bar in fig.axes[0].patches:
        assert isinstance(bar, Rectangle)
        widths[round(bar.get_y() + bar.get_height() / 2)] += bar.get_width()
    np.testing.assert_allclose(widths, 1.0)


def test_template_composition_segments_are_what_each_column_adds() -> None:
    fig = plot_template_composition(np.array([[0.25, 0.75]]), ("a", "b"), ("1",))

    segments = [
        bar.get_width() for bar in fig.axes[0].patches if isinstance(bar, Rectangle)
    ]
    np.testing.assert_allclose(segments, [0.25, 0.5, 0.25])


def test_template_composition_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="fractions has shape"):
        plot_template_composition(np.ones((2, 2)), ("a",), ("1", "2"))
