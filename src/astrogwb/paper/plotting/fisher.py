r"""Figures for the degeneracies of a Fisher forecast.

Two views, each taking plain arrays so this module keeps the package's rule of
importing nothing from ``astrogwb``:

- :func:`plot_whitened_derivatives` draws the derivative of the spectrum in
  noise units, :math:`\partial_a S_i / \sigma_i`, one curve per parameter, each
  scaled to unit norm. Two curves of the same shape (up to sign) are a
  degeneracy: the parameters bend the spectrum the same way. A second panel
  shows the fraction of each parameter's information a low-frequency cutoff
  at :math:`f` keeps.
- :func:`plot_fisher_eigenmodes` draws the principal axes of the Fisher
  matrix: the width of each constrained combination, beside the prior's width
  along it, and the loading of every parameter on it.
- :func:`plot_template_composition` names each mode's spectral template: the
  share of it that the spectrum itself, then each added frequency correction,
  explains, and what is left over.

The arrays come from :func:`astrogwb.sampling.whitened_jacobian`,
:func:`astrogwb.sampling.fisher_eigenmodes`, :func:`astrogwb.sampling.fisher_svd`
and :func:`astrogwb.sampling.cumulative_template_fractions`. A mode's template,
``fisher_svd(...).templates``, is itself a set of whitened curves, so
:func:`plot_whitened_derivatives` draws it too.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

__all__ = [
    "plot_fisher_eigenmodes",
    "plot_template_composition",
    "plot_whitened_derivatives",
]

#: Cycled with the colors: degenerate parameters draw the same curve, and a
#: second linestyle keeps the one underneath visible.
_LINESTYLES = ("-", "--", ":", "-.")


def plot_whitened_derivatives(
    frequencies: ArrayLike,
    whitened: ArrayLike,
    labels: Sequence[str],
    *,
    colors: Sequence[str] | None = None,
    cutoffs: Sequence[float] = (),
) -> Figure:
    r"""Unit-norm derivative shapes and the information a cutoff keeps.

    Parameters
    ----------
    frequencies:
        Bin frequencies in Hz, shape ``(F,)``. Pass only the bins the
        likelihood uses.
    whitened:
        :math:`\partial_a S_i / \sigma_i` on those bins, shape ``(F, P)``.
    labels:
        One display label per column.
    colors:
        One color per column; the matplotlib cycle when omitted.
    cutoffs:
        Low-frequency cutoffs in Hz, marked on both panels.

    Returns
    -------
    Figure
        Top: :math:`w_a / \|w_a\|`, signed. Bottom:
        :math:`\sum_{f_i \ge f} w_{ia}^2 / \sum_i w_{ia}^2`, the fraction of
        parameter :math:`a`'s Fisher diagonal a cutoff at :math:`f` keeps. A
        column with no information is left out and named in the legend.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    whitened = np.asarray(whitened, dtype=float)
    if whitened.shape != (frequencies.size, len(labels)):
        raise ValueError(
            f"whitened has shape {whitened.shape}, expected "
            f"{(frequencies.size, len(labels))}"
        )
    if colors is not None and len(colors) != len(labels):
        raise ValueError(f"{len(colors)} colors for {len(labels)} parameters")
    order = np.argsort(frequencies)
    frequencies, whitened = frequencies[order], whitened[order]

    fig, (shape_ax, kept_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(8.5, 6.0), height_ratios=(3, 2)
    )
    squared = whitened**2
    totals = squared.sum(axis=0)
    for column, label in enumerate(labels):
        style = {
            "color": None if colors is None else colors[column],
            "linestyle": _LINESTYLES[column % len(_LINESTYLES)],
        }
        if not totals[column] > 0.0:
            shape_ax.plot([], [], label=f"{label} (no information)", **style)
            continue
        shape_ax.plot(
            frequencies,
            whitened[:, column] / np.sqrt(totals[column]),
            label=label,
            **style,
        )
        # Reverse cumulative sum: what survives a cutoff at each frequency.
        kept = np.cumsum(squared[::-1, column])[::-1] / totals[column]
        kept_ax.plot(frequencies, kept, **style)
    for cutoff in cutoffs:
        for ax in (shape_ax, kept_ax):
            ax.axvline(cutoff, color="0.6", linestyle=":", linewidth=1.0)
    shape_ax.axhline(0.0, color="0.6", linewidth=0.8)
    shape_ax.set_xscale("log")
    shape_ax.set_ylabel(r"$w_a / \|w_a\|$")
    shape_ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    kept_ax.set_ylim(0.0, 1.05)
    kept_ax.set_xlabel(r"$f\,[\mathrm{Hz}]$")
    kept_ax.set_ylabel("information kept")
    fig.tight_layout()
    return fig


def plot_fisher_eigenmodes(
    sigmas: ArrayLike,
    directions: ArrayLike,
    labels: Sequence[str],
    *,
    prior_sigmas: ArrayLike | None = None,
    sigma_label: str = r"$\sigma$ along mode",
) -> Figure:
    r"""Width and parameter loadings of each Fisher eigenmode.

    Parameters
    ----------
    sigmas:
        Likelihood width of each mode, shape ``(P,)``, best-constrained first;
        ``inf`` for an unconstrained mode.
    directions:
        Unit eigenvectors as columns, shape ``(P, P)``; row ``a`` is
        parameter ``labels[a]``.
    labels:
        One display label per parameter.
    prior_sigmas:
        Prior width along each mode, shape ``(P,)``, drawn as a marker over
        its bar; ``inf`` (no prior along that mode) draws nothing. A bar
        reaching above its marker is a prior-dominated mode.
    sigma_label:
        Y-axis label of the width panel, naming the units the widths are in.

    Returns
    -------
    Figure
        Left: the widths on a log axis, an unconstrained mode drawn to the top
        edge and marked :math:`\infty`. Right: the loadings, one column per
        mode, on a diverging scale from -1 to 1.
    """
    sigmas = np.asarray(sigmas, dtype=float)
    directions = np.asarray(directions, dtype=float)
    n = len(labels)
    if sigmas.shape != (n,) or directions.shape != (n, n):
        raise ValueError(
            f"expected sigmas {(n,)} and directions {(n, n)}, got "
            f"{sigmas.shape} and {directions.shape}"
        )
    priors = (
        np.full(n, np.inf)
        if prior_sigmas is None
        else np.asarray(prior_sigmas, dtype=float)
    )
    if priors.shape != (n,):
        raise ValueError(f"expected prior_sigmas {(n,)}, got {priors.shape}")

    modes = np.arange(1, n + 1)
    finite = np.concatenate([sigmas[np.isfinite(sigmas)], priors[np.isfinite(priors)]])
    if finite.size == 0:
        finite = np.ones(1)
    bottom, top = finite.min() / 10.0, finite.max() * 10.0

    fig, (width_ax, loading_ax) = plt.subplots(
        1,
        2,
        figsize=(4.0 + 1.3 * n, 1.2 + 0.9 * max(n, 3)),
        width_ratios=(1, 1),
    )
    heights = np.where(np.isfinite(sigmas), sigmas, top)
    bars = width_ax.bar(modes, heights - bottom, bottom=bottom, color="0.55")
    for bar, sigma in zip(bars, sigmas, strict=True):
        if not np.isfinite(sigma):
            bar.set_hatch("//")
            bar.set_facecolor("0.85")
            width_ax.annotate(
                r"$\infty$",
                (bar.get_x() + bar.get_width() / 2, top),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "none"},
            )
    with_prior = np.isfinite(priors)
    if np.any(with_prior):
        width_ax.scatter(
            modes[with_prior],
            priors[with_prior],
            marker="_",
            s=900,
            linewidths=2.0,
            color="0.1",
            zorder=3,
            label="prior",
        )
        width_ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)
    width_ax.set_yscale("log")
    width_ax.set_ylim(bottom, top)
    width_ax.set_xticks(modes)
    width_ax.minorticks_off()
    width_ax.set_xlabel("mode")
    width_ax.set_ylabel(sigma_label)

    image = loading_ax.imshow(
        directions, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto"
    )
    loading_ax.grid(False)
    for (row, column), value in np.ndenumerate(directions):
        loading_ax.text(
            column,
            row,
            f"${value:+.2f}$",
            ha="center",
            va="center",
            color="white" if abs(value) > 0.6 else "0.1",
        )
    loading_ax.set_xticks(np.arange(n), [str(mode) for mode in modes])
    loading_ax.set_yticks(np.arange(n), list(labels))
    loading_ax.minorticks_off()
    loading_ax.tick_params(top=False, right=False)
    loading_ax.set_xlabel("mode")
    fig.colorbar(image, ax=loading_ax, fraction=0.046, pad=0.04, label="loading")
    fig.tight_layout()
    return fig


def plot_template_composition(
    fractions: ArrayLike,
    basis_labels: Sequence[str],
    mode_labels: Sequence[str],
    *,
    colors: Sequence[str] | None = None,
) -> Figure:
    r"""Stacked share of each mode's template explained by a nested basis.

    Parameters
    ----------
    fractions:
        Cumulative fractions, shape ``(K, B)``, from
        :func:`astrogwb.sampling.cumulative_template_fractions`: entry
        ``[k, j]`` is the share of mode ``k`` spanned by basis columns
        ``0 .. j``.
    basis_labels:
        One label per basis column, naming what that column adds.
    mode_labels:
        One label per mode.
    colors:
        One color per basis column; the matplotlib cycle when omitted.

    Returns
    -------
    Figure
        One horizontal bar per mode, split into what each basis column adds
        and a hatched remainder the whole basis leaves unexplained.
    """
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (len(mode_labels), len(basis_labels)):
        raise ValueError(
            f"fractions has shape {fractions.shape}, expected "
            f"{(len(mode_labels), len(basis_labels))}"
        )
    if colors is not None and len(colors) != len(basis_labels):
        raise ValueError(f"{len(colors)} colors for {len(basis_labels)} basis columns")
    fractions = np.clip(np.nan_to_num(fractions), 0.0, 1.0)
    added = np.diff(fractions, axis=1, prepend=0.0)
    rows = np.arange(len(mode_labels))
    fig, ax = plt.subplots(figsize=(7.5, 1.0 + 0.55 * len(mode_labels)))
    left = np.zeros(len(mode_labels))
    for column, label in enumerate(basis_labels):
        ax.barh(
            rows,
            added[:, column],
            left=left,
            color=None if colors is None else colors[column],
            label=label,
        )
        left += added[:, column]
    ax.barh(
        rows,
        1.0 - left,
        left=left,
        color="white",
        edgecolor="0.5",
        hatch="//",
        label="unexplained",
    )
    ax.set_yticks(rows, list(mode_labels))
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.minorticks_off()
    ax.tick_params(top=False, right=False)
    ax.set_xlabel("share of the template's squared norm")
    ax.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=min(len(basis_labels) + 1, 4)
    )
    fig.tight_layout()
    return fig
