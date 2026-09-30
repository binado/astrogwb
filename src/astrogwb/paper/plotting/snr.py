"""Presentation of a single network's spectrum-realization SNRs."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def plot_snr_histogram(
    snrs: ArrayLike, *, network: str, distribution_label: str
) -> Figure:
    """Create a raw-SNR count histogram under the caller's active paper style."""
    figure, axis = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    axis.hist(snrs, bins="auto", edgecolor="white")
    axis.set_xlabel("SNR")
    axis.set_ylabel("Realizations")
    axis.set_title(f"{network}\n{distribution_label}")
    return figure
