"""Presentation of a single network's spectrum-realization SNRs."""

from __future__ import annotations

import arviz_plots as azp
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def plot_snr_distribution(
    snrs: ArrayLike, *, network: str, distribution_label: str
) -> Figure:
    """Plot the SNR density under the caller's active paper style.

    Constant ensembles (including a single draw) use an ECDF because a KDE
    requires nonzero scatter. These are simulation draws, not posterior chains.
    """
    values = np.asarray(snrs, dtype=np.float64)
    kind = "ecdf" if np.ptp(values) == 0 else "kde"
    collection = azp.plot_dist(
        xr.DataTree.from_dict({"simulations": xr.Dataset({"snr": ("draw", values)})}),
        group="simulations",
        sample_dims=["draw"],
        kind=kind,
        backend="matplotlib",
        visuals={
            "credible_interval": False,
            "point_estimate": False,
            "point_estimate_text": False,
            "title": False,
            "remove_axis": False,
        },
        figure_kwargs={
            "figsize": (6.4, 4.8),
            "layout": "constrained",
            # gwpy registers replacement default axes that ArviZ cannot identify.
            "subplot_kws": {"axes_class": Axes},
        },
    )
    figure = collection.viz["figure"].item()
    axis = figure.axes[0]
    axis.set_xlabel("SNR")
    axis.set_ylabel("Cumulative probability" if kind == "ecdf" else "Density")
    axis.set_ylim(bottom=0)
    axis.set_title(f"{network}\n{distribution_label}")
    return figure
