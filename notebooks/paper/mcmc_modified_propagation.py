# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: astrogwb (3.12.9)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Modified-propagation constraints
#
# This notebook assembles the modified-GW-propagation figures used by the paper
# workflow. The propagation effect is parameterized by the phenomenological
# GW-to-EM luminosity-distance ratio $(\Xi_0, n)$ of Belgacem et al. It compares
# four inference runs on the same detector network:
#
# 1. $\Xi_0$ only (population and cosmology fixed at their fiducial values),
# 2. $\Xi_0 + n$ (both propagation parameters sampled),
# 3. $\Xi_0 + \mathcal{R}_0$ with a *narrow* prior on the local merger rate, and
# 4. $\Xi_0 + \mathcal{R}_0$ with a *broad* prior on the local merger rate.
#
# It produces three figures:
#
# - a corner plot of the $\Xi_0$--$n$ posterior (chain 2),
# - an overlay of the marginal $\Xi_0$ posterior across all four chains, and
# - an overlaid $\Xi_0$--$\mathcal{R}_0$ corner plot comparing the narrow and
#   broad merger-rate priors (chains 3 and 4).
#
# Chain paths and labels are separate inputs. This keeps chain loading outside the
# plotting helpers and makes it possible to select different inference runs without
# changing the plotting code.

# %%
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import arviz_stats as azs
import corner
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from arviz_base.labels import MapLabeller
from matplotlib.axes import Axes as MplAxes
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection

from astrogwb.utils import repo_root

# gwpy (via gwmock-signal) replaces matplotlib's rectilinear axes. ArviZ can then
# mis-detect the backend, so restore the standard matplotlib projection.
register_projection(MplAxes)

# %config InlineBackend.figure_format = "retina"

# %% [markdown]
# ## Defaults
#
# Direct notebook runs use the defaults below. Edit them in Jupyter or override
# with flags when running headless; the paper workflow passes the ordered chain
# paths as declared inputs.

# %%
_CHAIN_DIR = Path("chains/bns-n16384-df1/modified-propagation")
_NETWORK = "ET-2L-aligned-CE-Hanford"

DEFAULT_XI0_CHAIN = _CHAIN_DIR / f"{_NETWORK}__Xi0__baseline.nc"
DEFAULT_XI0_N_CHAIN = _CHAIN_DIR / f"{_NETWORK}__Xi0-n__baseline.nc"
DEFAULT_PRIOR_NARROW_CHAIN = (
    _CHAIN_DIR / f"{_NETWORK}__Xi0-merger-rate-gauss__baseline.nc"
)
DEFAULT_PRIOR_BROAD_CHAIN = _CHAIN_DIR / f"{_NETWORK}__Xi0-merger-rate__baseline.nc"

# Fiducial (injected) values marked as truths on the corner plots.
DEFAULT_XI_0 = 1.0
DEFAULT_XI_N = 1.91
DEFAULT_LOCAL_MERGER_RATE = 161.0

XI_0_LABEL = r"$\Xi_0$"
XI_N_LABEL = r"$n$"
LOCAL_MERGER_RATE_LABEL = r"$\mathcal{R}_0\,[\mathrm{Gpc^{-3}\,yr^{-1}}]$"

VAR_LABELS = {
    "xi_0": XI_0_LABEL,
    "xi_n": XI_N_LABEL,
    "local_merger_rate": LOCAL_MERGER_RATE_LABEL,
}

# Variable groups for each corner plot.
XI_N_VAR_NAMES = ("xi_0", "xi_n")
MERGER_RATE_VAR_NAMES = ("xi_0", "local_merger_rate")
CORNER_LEVELS = (0.6827, 0.9545)

# Labels for the four-chain marginal overlay, in chain order.
DEFAULT_MARGINAL_LABELS = [
    r"$\Xi_0$",
    r"$\Xi_0 + n$",
    r"$\Xi_0 + \mathcal{R}_0$ (narrow prior)",
    r"$\Xi_0 + \mathcal{R}_0$ (broad prior)",
]
# Labels for the narrow/broad merger-rate corner comparison, in chain order.
DEFAULT_PRIOR_LABELS = [
    r"narrow $\mathcal{R}_0$ prior",
    r"broad $\mathcal{R}_0$ prior",
]

PUBLICATION_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.labelsize": "medium",
    "axes.unicode_minus": False,
    "axes.titlesize": "medium",
    "figure.labelsize": "medium",
    "figure.titlesize": "medium",
    "legend.fontsize": "small",
    "legend.title_fontsize": "small",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "xtick.direction": "in",
    "xtick.minor.visible": True,
    "xtick.top": True,
    "ytick.direction": "in",
    "ytick.minor.visible": True,
    "ytick.right": True,
}


# %% [markdown]
# ## Input and validation helpers


# %%
def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_inference_data(path: Path) -> xr.DataTree:
    """Load one ArviZ-compatible inference tree from NetCDF/HDF5."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".nc", ".h5", ".hdf5"}:
        return xr.open_datatree(path, engine="h5netcdf")
    return xr.open_datatree(path)


def validate_inference_data(
    inference_data: Sequence[xr.DataTree],
    labels: Sequence[str],
    *,
    required_vars: Sequence[str] = ("xi_0",),
    group: str = "posterior",
    expected_count: int | None = None,
) -> None:
    """Validate aligned labels and required posterior variables."""
    if len(inference_data) != len(labels):
        raise ValueError(
            f"received {len(inference_data)} inference trees but {len(labels)} labels"
        )
    if expected_count is not None and len(inference_data) != expected_count:
        raise ValueError(
            f"expected {expected_count} inference trees, received {len(inference_data)}"
        )
    if not inference_data:
        raise ValueError("at least one inference tree is required")

    for index, tree in enumerate(inference_data):
        try:
            posterior = tree[group]
        except KeyError as error:
            raise ValueError(
                f"inference tree {index} ({labels[index]!r}) has no {group!r} group"
            ) from error
        missing = [name for name in required_vars if name not in posterior.data_vars]
        if missing:
            raise ValueError(
                f"inference tree {index} ({labels[index]!r}) is missing "
                f"{group} variable(s): {', '.join(missing)}"
            )


def _validate_styles(
    count: int,
    colors: Sequence[str] | None,
    linestyles: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    resolved_colors = (
        list(colors) if colors is not None else [f"C{i}" for i in range(count)]
    )
    resolved_linestyles = list(linestyles) if linestyles is not None else ["-"] * count
    if len(resolved_colors) != count or len(resolved_linestyles) != count:
        raise ValueError("color and linestyle counts must match the inference data")
    return resolved_colors, resolved_linestyles


# %% [markdown]
# ## Posterior plotting helpers


# %%
def plot_marginal_posteriors(
    inference_data: Sequence[xr.DataTree],
    labels: Sequence[str],
    *,
    var_name: str = "xi_0",
    colors: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
    group: str = "posterior",
    ax_kwargs: Mapping[str, Any] | None = None,
    legend_kwargs: Mapping[str, Any] | None = None,
) -> plt.Figure:
    """Overlay the marginalized 1D posterior density of `var_name` across chains."""
    validate_inference_data(
        inference_data, labels, required_vars=(var_name,), group=group
    )
    resolved_colors, resolved_linestyles = _validate_styles(
        len(inference_data), colors, linestyles
    )

    fig, ax = plt.subplots()
    for tree, label, color, linestyle in zip(
        inference_data,
        labels,
        resolved_colors,
        resolved_linestyles,
        strict=True,
    ):
        kde = azs.kde(tree, group=group, var_names=var_name)[var_name]
        x = kde.sel(plot_axis="x").to_numpy()
        probability = kde.sel(plot_axis="y").to_numpy()
        ax.plot(x, probability, label=label, color=color, linestyle=linestyle)

    resolved_ax_kwargs = {
        "xlabel": VAR_LABELS.get(var_name, var_name),
        "ylabel": "Posterior density",
        **dict(ax_kwargs or {}),
    }
    ax.set(**resolved_ax_kwargs)
    ax.legend(**dict(legend_kwargs or {}))
    fig.tight_layout()
    return fig


def _pooled_corner_range(
    inference_data: Sequence[xr.DataTree],
    var_names: Sequence[str],
    *,
    group: str,
    padding_fraction: float = 0.02,
) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for name in var_names:
        values = np.concatenate(
            [np.asarray(tree[group][name]).reshape(-1) for tree in inference_data]
        )
        lower = float(np.nanmin(values))
        upper = float(np.nanmax(values))
        width = upper - lower
        padding = padding_fraction * width if width > 0 else max(abs(lower), 1.0) * 0.02
        ranges.append((lower - padding, upper + padding))
    return ranges


def plot_corner(
    inference_data: Sequence[xr.DataTree],
    labels: Sequence[str],
    var_names: Sequence[str],
    *,
    colors: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
    group: str = "posterior",
    fiducials: Mapping[str, float] | None = None,
    legend_kwargs: Mapping[str, Any] | None = None,
) -> plt.Figure:
    """Overlay one or more corner posteriors over the same `var_names`."""
    var_names = tuple(var_names)
    validate_inference_data(
        inference_data, labels, required_vars=var_names, group=group
    )
    resolved_colors, resolved_linestyles = _validate_styles(
        len(inference_data), colors, linestyles
    )
    labeller = MapLabeller(
        var_name_map={name: VAR_LABELS.get(name, name) for name in var_names}
    )
    truths = None
    if fiducials is not None:
        truths = {name: fiducials[name] for name in var_names}

    fig: plt.Figure | None = None
    plot_range = _pooled_corner_range(inference_data, var_names, group=group)
    for index, (tree, color, linestyle) in enumerate(
        zip(inference_data, resolved_colors, resolved_linestyles, strict=True)
    ):
        fig = corner.corner(
            tree,
            group=group,
            var_names=list(var_names),
            labeller=labeller,
            range=plot_range,
            color=color,
            fig=fig,
            plot_datapoints=False,
            plot_density=False,
            fill_contours=False,
            levels=list(CORNER_LEVELS),
            hist_kwargs={
                "density": True,
                "linestyle": linestyle,
                "linewidth": 1.5,
            },
            contour_kwargs={"linestyles": linestyle, "linewidths": 1.5},
            truths=truths if index == 0 else None,
            truth_color="C3",
            max_n_ticks=4,
        )

    if fig is None:  # pragma: no cover - guarded by validation
        raise RuntimeError("corner did not create a figure")
    if len(inference_data) > 1:
        handles = [
            Line2D([], [], color=color, linestyle=linestyle, label=label)
            for label, color, linestyle in zip(
                labels, resolved_colors, resolved_linestyles, strict=True
            )
        ]
        resolved_legend_kwargs = {
            "loc": "upper right",
            "frameon": False,
            **dict(legend_kwargs or {}),
        }
        fig.legend(handles=handles, **resolved_legend_kwargs)
    fig.tight_layout()
    return fig


# %% [markdown]
# ## Command-line configuration and execution


# %%
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi0-chain", type=Path, default=DEFAULT_XI0_CHAIN)
    parser.add_argument("--xi0-n-chain", type=Path, default=DEFAULT_XI0_N_CHAIN)
    parser.add_argument(
        "--prior-narrow-chain", type=Path, default=DEFAULT_PRIOR_NARROW_CHAIN
    )
    parser.add_argument(
        "--prior-broad-chain", type=Path, default=DEFAULT_PRIOR_BROAD_CHAIN
    )
    parser.add_argument("--marginal-labels", nargs=4, default=DEFAULT_MARGINAL_LABELS)
    parser.add_argument("--prior-labels", nargs=2, default=DEFAULT_PRIOR_LABELS)
    parser.add_argument(
        "--output-xi-n-corner-pdf",
        type=Path,
        default=Path("figures/mcmc_modified_propagation_Xi0_n_corner.pdf"),
    )
    parser.add_argument(
        "--output-xi0-marginal-pdf",
        type=Path,
        default=Path("figures/mcmc_modified_propagation_Xi0_marginal.pdf"),
    )
    parser.add_argument(
        "--output-merger-rate-corner-pdf",
        type=Path,
        default=Path("figures/mcmc_modified_propagation_Xi0_merger_rate_corner.pdf"),
    )
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument("--group", default="posterior")
    parser.add_argument("--xi-0", type=float, default=DEFAULT_XI_0)
    parser.add_argument("--xi-n", type=float, default=DEFAULT_XI_N)
    parser.add_argument(
        "--local-merger-rate", type=float, default=DEFAULT_LOCAL_MERGER_RATE
    )
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    root = repo_root()

    # Chain order: [xi_0 only, xi_0 + n, narrow R0 prior, broad R0 prior].
    chain_paths = [
        _resolve_path(args.xi0_chain, root),
        _resolve_path(args.xi0_n_chain, root),
        _resolve_path(args.prior_narrow_chain, root),
        _resolve_path(args.prior_broad_chain, root),
    ]

    inference_data: list[xr.DataTree] = []
    try:
        inference_data = [load_inference_data(path) for path in chain_paths]
        validate_inference_data(
            inference_data,
            args.marginal_labels,
            group=args.group,
            expected_count=4,
        )

        xi_n_data = [inference_data[1]]
        xi_n_labels = [args.marginal_labels[1]]
        prior_data = [inference_data[2], inference_data[3]]

        fiducials = {
            "xi_0": args.xi_0,
            "xi_n": args.xi_n,
            "local_merger_rate": args.local_merger_rate,
        }

        plt.rcParams.update(**PUBLICATION_RC)

        # (i) Xi_0--n corner from the two-parameter propagation chain.
        xi_n_corner_figure = plot_corner(
            xi_n_data,
            xi_n_labels,
            XI_N_VAR_NAMES,
            group=args.group,
            fiducials=fiducials,
        )

        # (ii) Marginal Xi_0 posterior overlaid across all four chains.
        xi0_marginal_figure = plot_marginal_posteriors(
            inference_data,
            args.marginal_labels,
            var_name="xi_0",
            group=args.group,
        )

        # (iii) Xi_0--R0 corner comparing the narrow and broad merger-rate priors.
        merger_rate_corner_figure = plot_corner(
            prior_data,
            args.prior_labels,
            MERGER_RATE_VAR_NAMES,
            group=args.group,
            fiducials=fiducials,
        )

        outputs = {
            _resolve_path(args.output_xi_n_corner_pdf, root): xi_n_corner_figure,
            _resolve_path(args.output_xi0_marginal_pdf, root): xi0_marginal_figure,
            _resolve_path(
                args.output_merger_rate_corner_pdf, root
            ): merger_rate_corner_figure,
        }
        for output_path, figure in outputs.items():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=args.figure_dpi, bbox_inches="tight")
            print("saved figure:", output_path)
    finally:
        for tree in inference_data:
            tree.close()


# %%
if __name__ == "__main__":
    main()
