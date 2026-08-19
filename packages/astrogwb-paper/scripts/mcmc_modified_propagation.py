r"""Modified-GW-propagation figures and constraint tables.

Compares three inference runs on the same detector network: $\Xi_0$ only,
$\Xi_0 + n$, and $\Xi_0 + H_0$ with a Gaussian prior on the Hubble constant.
Produces corners, a marginal overlay, and a matched-filter SNR /
$\Xi_0$--$n$ HDI table across detector networks.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import arviz_stats as azs
import corner
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from arviz_base.labels import MapLabeller
from astrogwb_paper.config.figures import (
    Network,
    load_analysis_grid,
    load_fiducials,
    resolve_networks,
)
from astrogwb_paper.paths import paper_project_root, resolve_paper_path
from astrogwb_paper.plotting import (
    CATEGORY,
    CORNER_LEVELS,
    DETECTOR_NETWORKS,
    TRUTH,
    combo_colors,
    get_corner_kwargs,
    use_paper_style,
)
from astrogwb_paper.snr import compute_network_snrs
from matplotlib.axes import Axes as MplAxes
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection

# gwpy (via gwmock-signal) replaces matplotlib's rectilinear axes. ArviZ can then
# mis-detect the backend, so restore the standard matplotlib projection.
register_projection(MplAxes)
jax.config.update("jax_enable_x64", True)

XI_0_LABEL = r"$\Xi_0$"
XI_N_LABEL = r"$n$"
H0_LABEL = r"$H_0\,[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$"
IMPORTANCE_RELATIVE_ESS_LABEL = r"$N_{\mathrm{eff}} / N_{\mathrm{inj}}$"

VAR_LABELS = {
    "xi_0": XI_0_LABEL,
    "xi_n": XI_N_LABEL,
    "H0": H0_LABEL,
    "importance_relative_ess": IMPORTANCE_RELATIVE_ESS_LABEL,
}

# Variable groups for each corner plot.
XI_N_VAR_NAMES = ("xi_0", "xi_n")
XI_N_ESS_VAR_NAMES = ("xi_0", "xi_n", "importance_relative_ess")
H0_VAR_NAMES = ("xi_0", "H0")

# Labels for this figure, in legend order. The marginal overlay compares three
# parameter combinations of one network, so its labels name the combinations and
# follow the order the rule passes --xi0-chain, --xi0-n-chain, --h0-chain. The
# by-detector table borrows the shared network legend (`DETECTOR_NETWORKS`),
# which the workflow also expands its chain paths from.
PROPAGATION_EXPERIMENT = "modified-propagation"
MARGINAL_LABELS = (r"$\Xi_0$", r"$\Xi_0 + n$", r"$\Xi_0 + H_0$")
H0_LABELS = (r"$\Xi_0 + H_0$",)


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


def plot_marginal_posteriors(
    inference_data: Sequence[xr.DataTree],
    labels: Sequence[str],
    *,
    var_name: str = "xi_0",
    colors: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
    group: str = "posterior",
    fiducial: float | None = None,
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

    if fiducial is not None:
        ax.axvline(fiducial, **TRUTH)

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


def _expand_corner_range_with_truths(
    plot_range: Sequence[tuple[float, float]],
    var_names: Sequence[str],
    truths: Mapping[str, float] | None,
    *,
    padding_fraction: float = 0.02,
) -> list[tuple[float, float]]:
    """Ensure corner axis limits include configured truth markers."""
    if truths is None:
        return list(plot_range)
    expanded: list[tuple[float, float]] = []
    for (lower, upper), name in zip(plot_range, var_names, strict=True):
        if name not in truths:
            expanded.append((lower, upper))
            continue
        lower = min(lower, float(truths[name]))
        upper = max(upper, float(truths[name]))
        width = upper - lower
        padding = padding_fraction * width if width > 0 else 0.0
        expanded.append((lower - padding, upper + padding))
    return expanded


def _disable_relative_ess_axis_offsets(
    fig: plt.Figure, var_names: Sequence[str]
) -> None:
    """Avoid matplotlib offset ticks that look like relative ESS > 1."""
    if "importance_relative_ess" not in var_names:
        return
    n_vars = len(var_names)
    ess_index = list(var_names).index("importance_relative_ess")
    axes = np.asarray(fig.axes).reshape(n_vars, n_vars)

    def _disable_offset(axis: object) -> None:
        formatter = axis.get_major_formatter()  # type: ignore[attr-defined]
        if hasattr(formatter, "set_useOffset"):
            formatter.set_useOffset(False)

    for index in range(n_vars):
        _disable_offset(axes[ess_index, index].yaxis)
        _disable_offset(axes[index, ess_index].xaxis)


def plot_corner(
    inference_data: Sequence[xr.DataTree],
    labels: Sequence[str],
    var_names: Sequence[str],
    *,
    colors: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
    group: str = "posterior",
    fiducials: Mapping[str, float] | None = None,
    truth_color: str = str(TRUTH["color"]),
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
    plot_range = _expand_corner_range_with_truths(
        _pooled_corner_range(inference_data, var_names, group=group),
        var_names,
        truths,
    )
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
            hist_kwargs={
                "linestyle": linestyle,
                "linewidth": 1.5,
            },
            contour_kwargs={"linestyles": linestyle, "linewidths": 1.5},
            truths=truths if index == 0 else None,
            **get_corner_kwargs(truth_color=truth_color),
        )

    if fig is None:  # pragma: no cover - guarded by validation
        raise RuntimeError("corner did not create a figure")
    _disable_relative_ess_axis_offsets(fig, var_names)
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
    return fig


def xi0_hdi_table(
    inference_data: Sequence[xr.DataTree],
    labels: Sequence[str],
    *,
    var_name: str = "xi_0",
    prob: float = CORNER_LEVELS[0],
    group: str = "posterior",
) -> pd.DataFrame:
    """1-sigma HDI table for `var_name`: left/right bounds and sigma."""
    validate_inference_data(
        inference_data, labels, required_vars=(var_name,), group=group
    )
    rows = []
    for tree, label in zip(inference_data, labels, strict=True):
        hdi = azs.hdi(tree, prob=prob, group=group, var_names=var_name)[var_name]
        left = float(hdi.sel(ci_bound="lower"))
        right = float(hdi.sel(ci_bound="upper"))
        rows.append(
            {
                "chain": label,
                "left": left,
                "right": right,
                "sigma": (right - left) / 2.0,
            }
        )
    return pd.DataFrame(rows).set_index("chain")


def _hdi(
    tree: xr.DataTree,
    var_name: str,
    *,
    group: str,
    probability: float,
) -> tuple[float, float]:
    interval = azs.hdi(tree, group=group, var_names=var_name, prob=probability)[
        var_name
    ]
    return (
        float(interval.sel(ci_bound="lower")),
        float(interval.sel(ci_bound="upper")),
    )


def _posterior_median(
    tree: xr.DataTree,
    var_name: str,
    *,
    group: str,
) -> float:
    return float(np.nanmedian(np.asarray(tree[group][var_name]).reshape(-1)))


def build_snr_xi0_n_constraint_table(
    networks: Sequence[Network],
    inference_data: Sequence[xr.DataTree],
    snr_table: pd.DataFrame,
    *,
    group: str = "posterior",
    probability: float = CORNER_LEVELS[0],
) -> pd.DataFrame:
    """Combine SNR scaling and sampled $\\Xi_0$/$n$ HDI constraints by network.

    `n` has no simple SNR-scaling analog (unlike `xi_0`, which enters as an
    overall amplitude rescaling), so only its HDI-based half-width is reported.
    """
    validate_inference_data(
        inference_data,
        [network.label for network in networks],
        required_vars=XI_N_VAR_NAMES,
        group=group,
        expected_count=len(networks),
    )
    snr_by_network = snr_table.set_index("network")
    missing_networks = [
        network.name for network in networks if network.name not in snr_by_network.index
    ]
    if missing_networks:
        raise KeyError(
            "SNR table is missing configured network(s): " + ", ".join(missing_networks)
        )

    rows: list[dict[str, Any]] = []
    for network, tree in zip(networks, inference_data, strict=True):
        snr = float(snr_by_network.loc[network.name, "snr"])
        xi0_lower, xi0_upper = _hdi(tree, "xi_0", group=group, probability=probability)
        n_lower, n_upper = _hdi(tree, "xi_n", group=group, probability=probability)
        sigma_xi0_hdi = (xi0_upper - xi0_lower) / 2
        sigma_n_hdi = (n_upper - n_lower) / 2
        rows.append(
            {
                "label": network.label,
                "snr": snr,
                "sigma_xi0_hdi": sigma_xi0_hdi,
                "sigma_n_hdi": sigma_n_hdi,
                "rel_sigma_xi0_snr": 1.0 / snr,
            }
        )
    return pd.DataFrame(rows)


def xi0_n_constraint_table_latex(table: pd.DataFrame) -> str:
    """Format the $\\Xi_0$/$n$ constraint table as a publication LaTeX tabular."""
    latex_table = table.rename(
        columns={
            "label": "Detector Network",
            "snr": "SNR",
            "sigma_xi0_hdi": r"$\sigma_{\Xi_0}^{\rm HDI}$",
            "sigma_n_hdi": r"$\sigma_n^{\rm HDI}$",
            "rel_sigma_xi0_snr": r"$1/{\rm SNR}$",
        }
    )
    return latex_table.to_latex(
        index=False,
        escape=False,
        float_format="%.3g",
        caption=(
            "Matched-filter SNR and $1\\sigma$ credible intervals for $\\Xi_0$ and $n$ "
            "for each detector network."
        ),
        label="tab:mcmc_modified_propagation_xi0_n_by_detector",
    )


def write_xi0_n_constraint_table(
    table: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
) -> str:
    """Write the machine-readable and publication-formatted constraint tables."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    latex = xi0_n_constraint_table_latex(table)
    tex_path.write_text(latex, encoding="utf-8")
    return latex


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--xi0-chain", type=Path, required=True)
    parser.add_argument("--xi0-n-chain", type=Path, required=True)
    parser.add_argument("--h0-chain", type=Path, required=True)
    parser.add_argument("--detector-xi0-n-chains", type=Path, nargs="+", required=True)
    parser.add_argument("--output-xi-n-corner-pdf", type=Path, required=True)
    parser.add_argument("--output-xi-n-ess-corner-pdf", type=Path, required=True)
    parser.add_argument("--output-xi0-marginal-pdf", type=Path, required=True)
    parser.add_argument("--output-h0-corner-pdf", type=Path, required=True)
    parser.add_argument("--output-xi0-n-csv", type=Path, required=True)
    parser.add_argument("--output-xi0-n-tex", type=Path, required=True)
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument("--group", default="posterior")
    # Not a fiducial: N_eff/N_inj is a plotting truth line at its definitional
    # maximum. Adding it to [fiducials] would inject a spurious constant into
    # the sampled model.
    parser.add_argument("--importance-relative-ess", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    root = paper_project_root()
    grid = load_analysis_grid()
    fiducials = {
        **load_fiducials(),
        "importance_relative_ess": args.importance_relative_ess,
    }
    networks = resolve_networks(PROPAGATION_EXPERIMENT, DETECTOR_NETWORKS)
    marginal_labels = list(MARGINAL_LABELS)
    h0_labels = list(H0_LABELS)
    detector_labels = [network.label for network in networks]
    if len(args.detector_xi0_n_chains) != len(networks):
        raise SystemExit(
            f"--detector-xi0-n-chains has {len(args.detector_xi0_n_chains)} paths "
            f"but {PROPAGATION_EXPERIMENT} declares {len(networks)} networks"
        )

    chain_paths = [
        resolve_paper_path(args.xi0_chain, root),
        resolve_paper_path(args.xi0_n_chain, root),
        resolve_paper_path(args.h0_chain, root),
    ]
    inference_data = [load_inference_data(path) for path in chain_paths]
    validate_inference_data(
        inference_data,
        marginal_labels,
        group=args.group,
        expected_count=3,
    )
    xi_n_data = [inference_data[1]]
    xi_n_labels = [marginal_labels[1]]
    h0_data = [inference_data[2]]
    detector_xi0_n_data = [
        load_inference_data(resolve_paper_path(path, root))
        for path in args.detector_xi0_n_chains
    ]
    validate_inference_data(
        detector_xi0_n_data,
        detector_labels,
        required_vars=XI_N_VAR_NAMES,
        group=args.group,
        expected_count=len(networks),
    )
    use_paper_style()

    xi_n_corner_figure = plot_corner(
        xi_n_data,
        xi_n_labels,
        XI_N_VAR_NAMES,
        group=args.group,
        fiducials=fiducials,
        colors=[CATEGORY["modified_propagation"]],
    )
    xi_n_ess_corner_figure = plot_corner(
        xi_n_data,
        xi_n_labels,
        XI_N_ESS_VAR_NAMES,
        group=args.group,
        fiducials=fiducials,
        colors=[CATEGORY["modified_propagation"]],
    )
    xi0_marginal_figure = plot_marginal_posteriors(
        inference_data,
        marginal_labels,
        var_name="xi_0",
        group=args.group,
        colors=combo_colors(len(inference_data)),
        fiducial=fiducials["xi_0"],
    )
    h0_corner_figure = plot_corner(
        h0_data,
        h0_labels,
        H0_VAR_NAMES,
        group=args.group,
        fiducials=fiducials,
        colors=[CATEGORY["modified_propagation"]],
    )
    snr_table = compute_network_snrs(
        resolve_paper_path(args.catalog, root),
        networks,
        fiducials,
        grid=grid,
        jnp=jnp,
    )
    xi0_n_constraint_table = build_snr_xi0_n_constraint_table(
        networks,
        detector_xi0_n_data,
        snr_table,
        group=args.group,
    )
    print(xi0_n_constraint_table_latex(xi0_n_constraint_table))

    outputs = {
        resolve_paper_path(args.output_xi_n_corner_pdf, root): xi_n_corner_figure,
        resolve_paper_path(
            args.output_xi_n_ess_corner_pdf, root
        ): xi_n_ess_corner_figure,
        resolve_paper_path(args.output_xi0_marginal_pdf, root): xi0_marginal_figure,
        resolve_paper_path(args.output_h0_corner_pdf, root): h0_corner_figure,
    }
    for output_path, figure in outputs.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=args.figure_dpi, bbox_inches="tight")
        print("saved figure:", output_path)

    csv_path = resolve_paper_path(args.output_xi0_n_csv, root)
    tex_path = resolve_paper_path(args.output_xi0_n_tex, root)
    write_xi0_n_constraint_table(xi0_n_constraint_table, csv_path, tex_path)
    print("saved constraint table:", csv_path)
    print("saved LaTeX table:", tex_path)

    for tree in [*inference_data, *detector_xi0_n_data]:
        tree.close()


if __name__ == "__main__":
    main()
