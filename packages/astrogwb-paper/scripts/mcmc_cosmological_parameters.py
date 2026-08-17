r"""Cosmological-parameter constraints by detector network.

Evaluates the fiducial SGWB once, computes matched-filter SNR for each
detector network, and compares those estimates with sampled $H_0$ posteriors.
Also compares fixed and narrow priors on the local merger rate
$\mathcal{R}_0$, the joint $H_0$--$\mathcal{R}_0$ corner, and an
$H_0$--$\Omega_m$ corner.
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
from _paper_style import (
    CATEGORY,
    CORNER_LEVELS,
    DETECTOR_COMPARISON_LEGEND,
    MERGER_RATE_LEGEND,
    TRUTH,
    combo_colors,
    get_corner_kwargs,
    use_paper_style,
)
from arviz_base.labels import MapLabeller
from astrogwb.detector import effective_psd, load_sensitivity_map
from astrogwb.frequency import (
    frequency_mask as make_frequency_mask,
)
from astrogwb.frequency import (
    frequency_spacing as compute_frequency_spacing,
)
from astrogwb.gwb import spectral_density, spectral_snr
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
    make_merger_rate_and_log_weights_fn,
)
from astrogwb.utils import years_to_seconds
from astrogwb.waveform import polarization_power as compute_polarization_power
from astrogwb_paper.paths import paper_project_root
from matplotlib.axes import Axes as MplAxes
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection
from pluscross import load_catalog

# gwpy (via gwmock-signal) replaces matplotlib's rectilinear axes. ArviZ can then
# mis-detect the backend, so restore the standard matplotlib projection.
register_projection(MplAxes)
jax.config.update("jax_enable_x64", True)

H0_LABEL = r"$H_0\,[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$"
LOCAL_MERGER_RATE_LABEL = r"$\mathcal{R}_0\,[\mathrm{Gpc^{-3}\,yr^{-1}}]$"
OMEGA_M_LABEL = r"$\Omega_m$"
IMPORTANCE_RELATIVE_ESS_LABEL = r"$N_{\mathrm{eff}} / N_{\mathrm{inj}}$"
VAR_LABELS = {
    "H0": H0_LABEL,
    "local_merger_rate": LOCAL_MERGER_RATE_LABEL,
    "Omega_m": OMEGA_M_LABEL,
    "importance_relative_ess": IMPORTANCE_RELATIVE_ESS_LABEL,
}
MERGER_RATE_VAR_NAMES = ("H0", "local_merger_rate")
OMEGA_M_VAR_NAMES = ("H0", "Omega_m")
OMEGA_M_ESS_VAR_NAMES = ("H0", "Omega_m", "importance_relative_ess")
# Kept for select_corner_inference_data / older call sites.
CORNER_VAR_NAMES = MERGER_RATE_VAR_NAMES


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _parse_network_definition(value: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in value:
        raise ValueError(
            f"invalid network definition {value!r}; expected NAME=DET1,DET2,..."
        )
    name, detector_list = value.split("=", 1)
    name = name.strip()
    detectors = tuple(detector.strip() for detector in detector_list.split(","))
    if not name or not detectors or any(not detector for detector in detectors):
        raise ValueError(
            f"invalid network definition {value!r}; expected NAME=DET1,DET2,..."
        )
    return name, detectors


def parse_networks(definitions: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Parse repeatable ``--network NAME=DET1,DET2`` flags in declaration order."""
    if not definitions:
        raise ValueError("at least one --network NAME=DET1,DET2,... is required")
    networks: dict[str, tuple[str, ...]] = {}
    for definition in definitions:
        name, detectors = _parse_network_definition(definition)
        if name in networks:
            raise ValueError(f"duplicate --network definition for {name!r}")
        networks[name] = detectors
    return networks


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
    required_vars: Sequence[str] = ("H0",),
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


def select_corner_inference_data(
    inference_data: Sequence[xr.DataTree],
    labels: Sequence[str],
    *,
    group: str = "posterior",
) -> tuple[list[xr.DataTree], list[str], list[int]]:
    """Select the H0--local-merger-rate trees from the prior comparison."""
    validate_inference_data(inference_data, labels, group=group)
    indices = [
        index
        for index, tree in enumerate(inference_data)
        if "local_merger_rate" in tree[group].data_vars
    ]
    if not indices:
        raise ValueError(
            "expected at least one prior-comparison chain containing local_merger_rate"
        )
    selected_data = [inference_data[index] for index in indices]
    selected_labels = [labels[index] for index in indices]
    validate_inference_data(
        selected_data,
        selected_labels,
        required_vars=CORNER_VAR_NAMES,
        group=group,
        expected_count=len(selected_data),
    )
    return selected_data, selected_labels, indices


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


def _base_network_name(name: str) -> str:
    """Map an ET+CE network name onto its ET-only counterpart."""
    suffix = "-CE-Hanford"
    return name.removesuffix(suffix)


def detector_network_styles(
    networks: Mapping[str, tuple[str, ...]],
) -> tuple[list[str], list[str]]:
    """Shared color per ET / ET+CE pair; dashed linestyle for CE companions."""
    names = list(networks)
    bases: list[str] = []
    for name in names:
        base = _base_network_name(name)
        if base not in bases:
            bases.append(base)
    palette = combo_colors(len(bases))
    color_by_base = dict(zip(bases, palette, strict=True))
    colors = [color_by_base[_base_network_name(name)] for name in names]
    linestyles = ["--" if name.endswith("-CE-Hanford") else "-" for name in names]
    return colors, linestyles


def plot_h0_posteriors(
    inference_data: Sequence[xr.DataTree],
    labels: Sequence[str],
    *,
    colors: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
    group: str = "posterior",
    fiducial: float | None = None,
    ax_kwargs: Mapping[str, Any] | None = None,
    legend_kwargs: Mapping[str, Any] | None = None,
) -> plt.Figure:
    """Overlay marginalized H0 posterior densities from loaded inference trees."""
    validate_inference_data(inference_data, labels, group=group)
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
        kde = azs.kde(tree, group=group, var_names="H0")["H0"]
        x = kde.sel(plot_axis="x").to_numpy()
        probability = kde.sel(plot_axis="y").to_numpy()
        ax.plot(
            x,
            probability,
            label=label,
            color=color,
            linestyle=linestyle,
        )

    if fiducial is not None:
        ax.axvline(fiducial, **TRUTH)

    resolved_ax_kwargs = {
        "xlabel": H0_LABEL,
        "ylabel": "Posterior density",
        **dict(ax_kwargs or {}),
    }
    ax.set(**resolved_ax_kwargs)
    handles = [
        Line2D([], [], color=color, linestyle=linestyle, label=label)
        for label, color, linestyle in zip(
            labels, resolved_colors, resolved_linestyles, strict=True
        )
    ]
    resolved_legend_kwargs = {
        "handlelength": 2.5,
        **dict(legend_kwargs or {}),
    }
    ax.legend(handles=handles, **resolved_legend_kwargs)
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
    legend_kwargs: Mapping[str, Any] | None = None,
) -> plt.Figure:
    """Overlay one or more corner posteriors over the same `var_names`."""
    var_names = tuple(var_names)
    validate_inference_data(
        inference_data,
        labels,
        required_vars=var_names,
        group=group,
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
            **get_corner_kwargs(),
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


def plot_h0_merger_rate_corner(
    inference_data: Sequence[xr.DataTree],
    labels: Sequence[str],
    *,
    colors: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
    group: str = "posterior",
    fiducials: Mapping[str, float] | None = None,
    legend_kwargs: Mapping[str, Any] | None = None,
) -> plt.Figure:
    """Overlay one or more H0--local-merger-rate corner posteriors."""
    return plot_corner(
        inference_data,
        labels,
        MERGER_RATE_VAR_NAMES,
        colors=colors,
        linestyles=linestyles,
        group=group,
        fiducials=fiducials,
        legend_kwargs=legend_kwargs,
    )


def compute_network_snrs(
    catalog_path: Path,
    networks: Mapping[str, tuple[str, ...]],
    fiducials: Mapping[str, float],
    *,
    observation_time: float,
    f_min: float,
    f_max: float,
    z_min: float,
    z_max: float,
    n_grid: int,
) -> pd.DataFrame:
    """Compute the fiducial matched-filter SNR for each detector network."""
    catalog = load_catalog(catalog_path)
    frequencies = jnp.asarray(catalog.frequencies)
    polarization_power = jnp.asarray(compute_polarization_power(catalog))
    samples = {
        name: jnp.asarray(values) for name, values in catalog.source_parameters.items()
    }
    del catalog

    missing = [
        name for name in ("redshift", "luminosity_distance") if name not in samples
    ]
    if missing:
        raise ValueError(
            "catalog samples are missing required parameter(s): " + ", ".join(missing)
        )

    z_grid = jnp.linspace(z_min, z_max, n_grid)
    _, _, proposal_logprob = compute_merger_rate_distance_and_logprob(
        fiducials, samples, redshift_grid=z_grid
    )
    merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
        fiducials=fiducials,
        redshift_grid=z_grid,
        proposal_logprob=proposal_logprob,
    )
    total_rate, log_weights = merger_rate_and_log_weights_fn(fiducials, samples)
    observed_spectral_density = spectral_density(
        polarization_power,
        jnp.exp(log_weights),
        total_rate,
        average_mode="analytic_inclination",
    )

    mask = make_frequency_mask(frequencies, fmin=f_min, fmax=f_max)
    frequency_spacing = compute_frequency_spacing(frequencies)
    observation_seconds = years_to_seconds(observation_time)

    rows: list[dict[str, Any]] = []
    for network, detectors in networks.items():
        sensitivities = load_sensitivity_map(detectors)
        effective_noise = jnp.asarray(
            effective_psd(frequencies, list(detectors), sensitivities)
        )
        snr = float(
            spectral_snr(
                observed_spectral_density[mask],
                effective_noise[mask],
                observation_seconds,
                frequency_spacing,
            )
        )
        rows.append(
            {
                "network": network,
                "detectors": ",".join(detectors),
                "n_detectors": len(detectors),
                "snr": snr,
            }
        )
    return pd.DataFrame(rows)


def _hdi(
    tree: xr.DataTree,
    var_name: str = "H0",
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


def format_median_hdi(
    median: float,
    lower: float,
    upper: float,
    *,
    precision: str = ".3g",
) -> str:
    """Format a posterior as journal-style $x_{-l}^{+u}$ from median and HDI."""
    return (
        f"${median:{precision}}"
        f"_{{-{(median - lower):{precision}}}}"
        f"^{{+{(upper - median):{precision}}}}$"
    )


def build_h0_r0_uncertainty_table(
    inference_data: Sequence[xr.DataTree],
    labels: Sequence[str],
    *,
    group: str = "posterior",
    probability: float = CORNER_LEVELS[0],
) -> pd.DataFrame:
    """Compare $H_0$ and $\\mathcal{R}_0$ constraints across prior-comparison runs.

    Rows are analyses (fixed $\\mathcal{R}_0$, then joint $H_0+\\mathcal{R}_0$
    runs). Cells are median with 68% HDI as $x_{-l}^{+u}$; missing parameters
    are shown as an em dash.
    """
    validate_inference_data(inference_data, labels, group=group)
    rows: list[dict[str, str]] = []
    for tree, label in zip(inference_data, labels, strict=True):
        h0_median = _posterior_median(tree, "H0", group=group)
        h0_lower, h0_upper = _hdi(tree, "H0", group=group, probability=probability)
        row = {
            "analysis": label,
            "H0": format_median_hdi(h0_median, h0_lower, h0_upper),
            "local_merger_rate": "—",
        }
        if "local_merger_rate" in tree[group].data_vars:
            rate_median = _posterior_median(tree, "local_merger_rate", group=group)
            rate_lower, rate_upper = _hdi(
                tree, "local_merger_rate", group=group, probability=probability
            )
            row["local_merger_rate"] = format_median_hdi(
                rate_median, rate_lower, rate_upper
            )
        rows.append(row)
    return pd.DataFrame(rows)


def h0_r0_uncertainty_table_latex(table: pd.DataFrame) -> str:
    """Format the H0 / R0 uncertainty comparison as publication LaTeX."""
    latex_table = table.rename(
        columns={
            "analysis": "Analysis",
            "H0": H0_LABEL,
            "local_merger_rate": LOCAL_MERGER_RATE_LABEL,
        }
    )
    return latex_table.to_latex(
        index=False,
        escape=False,
        caption=(
            "Median and 68.27\\% HDI constraints on $H_0$ and "
            r"$\mathcal{R}_0$ for fixed versus joint local-merger-rate analyses. "
            "Entries are $x_{-l}^{+u}$."
        ),
        label="tab:mcmc_cosmological_parameters_h0_r0",
    )


def build_snr_h0_constraint_table(
    networks: Mapping[str, tuple[str, ...]],
    inference_data: Sequence[xr.DataTree],
    labels: Sequence[str],
    snr_table: pd.DataFrame,
    *,
    h0_fiducial: float,
    group: str = "posterior",
    probability: float = CORNER_LEVELS[0],
) -> pd.DataFrame:
    """Combine SNR scaling and sampled H0 HDI constraints by network."""
    validate_inference_data(
        inference_data,
        labels,
        group=group,
        expected_count=len(networks),
    )
    snr_by_network = snr_table.set_index("network")
    missing_networks = [name for name in networks if name not in snr_by_network.index]
    if missing_networks:
        raise KeyError(
            "SNR table is missing configured network(s): " + ", ".join(missing_networks)
        )

    rows: list[dict[str, Any]] = []
    for network, tree, label in zip(networks, inference_data, labels, strict=True):
        snr = float(snr_by_network.loc[network, "snr"])
        lower, upper = _hdi(tree, "H0", group=group, probability=probability)
        sigma_hdi = (upper - lower) / 2
        rows.append(
            {
                "label": label,
                "snr": snr,
                "sigma_h0_hdi": sigma_hdi,
                "rel_sigma_h0_hdi": sigma_hdi / h0_fiducial,
                "rel_sigma_h0_snr": 1.0 / snr,
            }
        )
    return pd.DataFrame(rows)


def constraint_table_latex(table: pd.DataFrame) -> str:
    """Format the constraint table as a publication LaTeX tabular."""
    latex_table = table.rename(
        columns={
            "label": "Detector Network",
            "snr": "SNR",
            "sigma_h0_hdi": r"$\sigma_{H_0}^{\rm HDI}$",
            "rel_sigma_h0_hdi": r"$\sigma_{H_0}^{\rm HDI}/H_0$",
            "rel_sigma_h0_snr": r"$1/{\rm SNR}$",
        }
    )
    return latex_table.to_latex(
        index=False,
        escape=False,
        float_format="%.3g",
        caption=(
            "Matched-filter SNR and $1\\sigma$ credible intervals for $H_0$ "
            "for each detector network."
        ),
        label="tab:mcmc_cosmological_parameters_h0_by_detector",
    )


def write_constraint_table(
    table: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
) -> str:
    """Write the machine-readable and publication-formatted constraint tables."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    latex = constraint_table_latex(table)
    tex_path.write_text(latex, encoding="utf-8")
    return latex


def _add_fiducial_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--h0", type=float, required=True)
    parser.add_argument("--omega-m", type=float, required=True)
    parser.add_argument("--xi-0", type=float, required=True)
    parser.add_argument("--xi-n", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--z-peak", type=float, required=True)
    parser.add_argument("--local-merger-rate", type=float, required=True)
    parser.add_argument("--importance-relative-ess", type=float, default=1.0)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        choices=("detectors", "merger-rate", "omega-m"),
        required=True,
    )
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--detector-chains", type=Path, nargs="+")
    parser.add_argument("--detector-labels", nargs="+")
    parser.add_argument("--prior-chains", type=Path, nargs="+")
    parser.add_argument("--prior-labels", nargs="+")
    parser.add_argument("--omega-m-chain", type=Path)
    parser.add_argument("--omega-m-label")
    parser.add_argument("--output-detector-pdf", type=Path)
    parser.add_argument("--output-prior-pdf", type=Path)
    parser.add_argument("--output-narrow-corner-pdf", type=Path)
    parser.add_argument("--output-omega-m-corner-pdf", type=Path)
    parser.add_argument("--output-omega-m-ess-corner-pdf", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-tex", type=Path)
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument("--group", default="posterior")
    parser.add_argument("--observation-time", type=float)
    parser.add_argument("--f-min", type=float)
    parser.add_argument("--f-max", type=float)
    parser.add_argument("--z-min", type=float)
    parser.add_argument("--z-max", type=float)
    parser.add_argument("--n-grid", type=int)
    parser.add_argument(
        "--network",
        action="append",
        default=None,
        metavar="NAME=DET1,DET2,...",
        help="Detector-network definition (repeatable, declaration order).",
    )
    _add_fiducial_arguments(parser)
    return parser.parse_args(argv)


def _fiducials(args: argparse.Namespace) -> dict[str, float]:
    return {
        "H0": args.h0,
        "Omega_m": args.omega_m,
        "xi_0": args.xi_0,
        "xi_n": args.xi_n,
        "gamma": args.gamma,
        "kappa": args.kappa,
        "z_peak": args.z_peak,
        "local_merger_rate": args.local_merger_rate,
        "importance_relative_ess": args.importance_relative_ess,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    root = paper_project_root()
    fiducials = _fiducials(args)
    use_paper_style()
    outputs: dict[Path, plt.Figure] = {}
    opened_data: list[xr.DataTree] = []

    if args.section == "detectors":
        if args.detector_chains is None or args.detector_labels is None:
            raise SystemExit("--detector-chains and --detector-labels are required")
        if args.catalog is None or args.output_detector_pdf is None:
            raise SystemExit("--catalog and --output-detector-pdf are required")
        if args.output_csv is None or args.output_tex is None:
            raise SystemExit("--output-csv and --output-tex are required")
        missing = [
            name
            for name in (
                "observation_time",
                "f_min",
                "f_max",
                "z_min",
                "z_max",
                "n_grid",
            )
            if getattr(args, name) is None
        ]
        if missing:
            raise SystemExit("missing required analysis flags: " + ", ".join(missing))
        try:
            networks = parse_networks(args.network or [])
        except ValueError as error:
            raise SystemExit(str(error)) from error
        detector_labels = list(args.detector_labels)
        if len(args.detector_chains) != len(detector_labels):
            raise SystemExit("--detector-chains length must match --detector-labels")
        if len(args.detector_chains) != len(networks):
            raise SystemExit("--detector-chains length must match --network count")
        detector_data = [
            load_inference_data(_resolve_path(path, root))
            for path in args.detector_chains
        ]
        opened_data.extend(detector_data)
        validate_inference_data(
            detector_data,
            detector_labels,
            group=args.group,
            expected_count=len(networks),
        )
        detector_colors, detector_linestyles = detector_network_styles(networks)
        outputs[_resolve_path(args.output_detector_pdf, root)] = plot_h0_posteriors(
            detector_data,
            detector_labels,
            colors=detector_colors,
            linestyles=detector_linestyles,
            group=args.group,
            fiducial=fiducials["H0"],
            legend_kwargs=DETECTOR_COMPARISON_LEGEND,
        )
        snr_table = compute_network_snrs(
            _resolve_path(args.catalog, root),
            networks,
            fiducials,
            observation_time=args.observation_time,
            f_min=args.f_min,
            f_max=args.f_max,
            z_min=args.z_min,
            z_max=args.z_max,
            n_grid=args.n_grid,
        )
        table = build_snr_h0_constraint_table(
            networks,
            detector_data,
            detector_labels,
            snr_table,
            h0_fiducial=fiducials["H0"],
            group=args.group,
        )
        write_constraint_table(
            table,
            _resolve_path(args.output_csv, root),
            _resolve_path(args.output_tex, root),
        )

    elif args.section == "merger-rate":
        if args.prior_chains is None or args.prior_labels is None:
            raise SystemExit("--prior-chains and --prior-labels are required")
        if args.output_prior_pdf is None or args.output_narrow_corner_pdf is None:
            raise SystemExit(
                "--output-prior-pdf and --output-narrow-corner-pdf are required"
            )
        if args.output_csv is None or args.output_tex is None:
            raise SystemExit("--output-csv and --output-tex are required")
        prior_labels = list(args.prior_labels)
        if len(args.prior_chains) != len(prior_labels) or len(args.prior_chains) != 2:
            raise SystemExit(
                "the merger-rate comparison requires two chains and labels"
            )
        prior_data = [
            load_inference_data(_resolve_path(path, root)) for path in args.prior_chains
        ]
        opened_data.extend(prior_data)
        validate_inference_data(
            prior_data,
            prior_labels,
            group=args.group,
            expected_count=2,
        )
        corner_data, corner_labels, _ = select_corner_inference_data(
            prior_data, prior_labels, group=args.group
        )
        colors = combo_colors(len(prior_data))
        outputs[_resolve_path(args.output_prior_pdf, root)] = plot_h0_posteriors(
            prior_data,
            prior_labels,
            colors=colors,
            linestyles=["-"] * len(prior_data),
            group=args.group,
            fiducial=fiducials["H0"],
            legend_kwargs=MERGER_RATE_LEGEND,
        )
        outputs[_resolve_path(args.output_narrow_corner_pdf, root)] = plot_corner(
            [corner_data[0]],
            [corner_labels[0]],
            MERGER_RATE_VAR_NAMES,
            colors=[colors[1]],
            linestyles=["-"],
            group=args.group,
            fiducials=fiducials,
        )
        table = build_h0_r0_uncertainty_table(
            prior_data, prior_labels, group=args.group
        )
        csv_path = _resolve_path(args.output_csv, root)
        tex_path = _resolve_path(args.output_tex, root)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path)
        tex_path.write_text(h0_r0_uncertainty_table_latex(table), encoding="utf-8")

    else:
        if args.omega_m_chain is None or args.omega_m_label is None:
            raise SystemExit("--omega-m-chain and --omega-m-label are required")
        if args.output_omega_m_corner_pdf is None:
            raise SystemExit("--output-omega-m-corner-pdf is required")
        if args.output_omega_m_ess_corner_pdf is None:
            raise SystemExit("--output-omega-m-ess-corner-pdf is required")
        omega_m_data = [load_inference_data(_resolve_path(args.omega_m_chain, root))]
        opened_data.extend(omega_m_data)
        omega_m_labels = [args.omega_m_label]
        validate_inference_data(
            omega_m_data,
            omega_m_labels,
            required_vars=OMEGA_M_VAR_NAMES,
            group=args.group,
            expected_count=1,
        )
        outputs[_resolve_path(args.output_omega_m_corner_pdf, root)] = plot_corner(
            omega_m_data,
            omega_m_labels,
            OMEGA_M_VAR_NAMES,
            colors=[CATEGORY["cosmology"]],
            group=args.group,
            fiducials=fiducials,
        )
        outputs[_resolve_path(args.output_omega_m_ess_corner_pdf, root)] = plot_corner(
            omega_m_data,
            omega_m_labels,
            OMEGA_M_ESS_VAR_NAMES,
            colors=[CATEGORY["cosmology"]],
            group=args.group,
            fiducials=fiducials,
        )

    for output_path, figure in outputs.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=args.figure_dpi, bbox_inches="tight")
        print("saved figure:", output_path)

    for tree in opened_data:
        tree.close()


if __name__ == "__main__":
    main()
