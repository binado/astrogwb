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
# # Cosmological-parameter constraints by detector network
#
# This notebook assembles the cosmology figures and table used by the paper
# workflow. It evaluates the fiducial SGWB once, computes the matched-filter SNR
# for each detector network, and compares those estimates with sampled $H_0$
# posteriors. It also compares fixed and narrow priors on the local merger
# rate $\mathcal{R}_0$, shows the joint $H_0$--$\mathcal{R}_0$ corner for the
# narrow merger-rate prior, and an $H_0$--$\Omega_m$ corner.
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
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from _paper_style import (
    CATEGORY,
    CORNER_LEVELS,
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
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.paths import paper_project_root
from matplotlib.axes import Axes as MplAxes
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection
from pluscross import load_catalog

# gwpy (via gwmock-signal) replaces matplotlib's rectilinear axes. ArviZ can then
# mis-detect the backend, so restore the standard matplotlib projection.
register_projection(MplAxes)
jax.config.update("jax_enable_x64", True)

# %config InlineBackend.figure_format = "retina"

# %% [markdown]
# ## Defaults
#
# Direct notebook runs use the defaults below. The paper workflow passes the same
# values explicitly from the selected experiment and shared MCMC base config.

# %%
DEFAULT_CATALOG_PATH = Path("outputs/catalogs/bns-n16384-df1.h5")
DEFAULT_CONFIG_PATH = Path("experiments/H0-all-detectors/figure.toml")
DEFAULT_BASE_CONFIG_PATH = Path("inputs/mcmc.base.toml")
DEFAULT_OBSERVATION_TIME = 1.0
DEFAULT_F_MIN = 2.0
DEFAULT_F_MAX = 4096.0
DEFAULT_Z_MIN = 0.0
DEFAULT_Z_MAX = 20.0
DEFAULT_N_GRID = 256
DEFAULT_H0 = 67.66
DEFAULT_OMEGA_M = 0.3096
DEFAULT_XI_0 = 1.0
DEFAULT_XI_N = 1.91
DEFAULT_GAMMA = 1.42
DEFAULT_KAPPA = 4.62
DEFAULT_Z_PEAK = 1.84
DEFAULT_LOCAL_MERGER_RATE = 161.0
DEFAULT_IMPORTANCE_RELATIVE_ESS = 1.0

DEFAULT_DETECTOR_NETWORKS = {
    "ET-triangular": ("E1", "E2", "E3"),
    "ET-triangular-CE-Hanford": ("E1", "E2", "E3", "C1"),
    "ET-2L-aligned": ("S1", "R1"),
    "ET-2L-aligned-CE-Hanford": ("S1", "R1", "C1"),
    "ET-2L-misaligned": ("S2", "R2"),
    "ET-2L-misaligned-CE-Hanford": ("S2", "R2", "C1"),
}
DEFAULT_NETWORKS = list(DEFAULT_DETECTOR_NETWORKS)

DEFAULT_DETECTOR_CHAINS = [
    Path(
        "outputs/chains/H0-all-detectors/ET-triangular.nc"
    ),
    Path(
        "outputs/chains/H0-all-detectors/ET-triangular-CE-Hanford.nc"
    ),
    Path(
        "outputs/chains/H0-all-detectors/ET-2L-aligned.nc"
    ),
    Path(
        "outputs/chains/H0-all-detectors/ET-2L-aligned-CE-Hanford.nc"
    ),
    Path(
        "outputs/chains/H0-all-detectors/ET-2L-misaligned.nc"
    ),
    Path(
        "outputs/chains/H0-all-detectors/ET-2L-misaligned-CE-Hanford.nc"
    ),
]
DEFAULT_DETECTOR_LABELS = [
    r"ET-$\Delta$",
    r"ET-$\Delta +$ CE",
    "ET-2L-par",
    r"ET-2L-par $+$ CE",
    "ET-2L",
    r"ET-2L $+$ CE",
]

DEFAULT_PRIOR_CHAINS = [
    Path(
        "outputs/chains/H0-merger-rate/fixed.nc"
    ),
    Path(
        "outputs/chains/H0-merger-rate/sampled.nc"
    ),
]
DEFAULT_PRIOR_LABELS = [
    r"$H_0$ (fixed $\mathcal{R}_0$)",
    r"$H_0 + \mathcal{R}_0$ (narrow prior)",
]

DEFAULT_OMEGA_M_CHAIN = Path(
    "outputs/chains/H0-omega-m/H0-Omega_m.nc"
)
DEFAULT_OMEGA_M_LABEL = r"$H_0 + \Omega_m$"

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


# %% [markdown]
# ## Input and validation helpers


# %%
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


def _resolve_networks(
    defaults: Mapping[str, tuple[str, ...]],
    definitions: Sequence[str],
    selected: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    networks = dict(defaults)
    cli_names: set[str] = set()
    for definition in definitions:
        name, detectors = _parse_network_definition(definition)
        if name in cli_names:
            raise ValueError(f"duplicate --network definition for {name!r}")
        cli_names.add(name)
        networks[name] = detectors

    names = list(selected) or list(networks)
    unknown = [name for name in names if name not in networks]
    if unknown:
        raise ValueError(f"unknown selected network(s): {', '.join(unknown)}")
    return {name: networks[name] for name in names}


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


# %% [markdown]
# ## Posterior plotting helpers


# %%
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


# %% [markdown]
# ## Fiducial SNR and constraint table


# %%
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


# %% [markdown]
# ## Command-line configuration
#
# Every setting below can be overridden with a CLI flag when running this
# notebook headless; otherwise the defaults above are used.


# %%
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        choices=("detectors", "merger-rate", "omega-m"),
        default="detectors",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG_PATH)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument(
        "--detector-chains", type=Path, nargs="+", default=DEFAULT_DETECTOR_CHAINS
    )
    parser.add_argument("--detector-labels", nargs="+", default=DEFAULT_DETECTOR_LABELS)
    parser.add_argument(
        "--prior-chains", type=Path, nargs="+", default=DEFAULT_PRIOR_CHAINS
    )
    parser.add_argument("--prior-labels", nargs="+", default=DEFAULT_PRIOR_LABELS)
    parser.add_argument("--omega-m-chain", type=Path, default=DEFAULT_OMEGA_M_CHAIN)
    parser.add_argument("--omega-m-label", default=DEFAULT_OMEGA_M_LABEL)
    parser.add_argument(
        "--output-detector-pdf",
        type=Path,
        default=Path("outputs/figures/H0-all-detectors/H0-by-detector.pdf"),
    )
    parser.add_argument(
        "--output-prior-pdf",
        type=Path,
        default=Path("outputs/figures/H0-merger-rate/H0-merger-rate-priors.pdf"),
    )
    parser.add_argument(
        "--output-narrow-corner-pdf",
        type=Path,
        default=Path(
            "outputs/figures/H0-merger-rate/H0-merger-rate-corner.pdf"
        ),
    )
    parser.add_argument(
        "--output-omega-m-corner-pdf",
        type=Path,
        default=Path("outputs/figures/H0-omega-m/H0-Omega_m-corner.pdf"),
    )
    parser.add_argument(
        "--output-omega-m-ess-corner-pdf",
        type=Path,
        default=Path("outputs/figures/H0-omega-m/H0-Omega_m-ess-corner.pdf"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/figures/H0-all-detectors/H0-by-detector.csv"),
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("outputs/figures/H0-all-detectors/H0-by-detector.tex"),
    )
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument("--group", default="posterior")
    parser.add_argument(
        "--observation-time", type=float, default=DEFAULT_OBSERVATION_TIME
    )
    parser.add_argument("--f-min", type=float, default=DEFAULT_F_MIN)
    parser.add_argument("--f-max", type=float, default=DEFAULT_F_MAX)
    parser.add_argument("--z-min", type=float, default=DEFAULT_Z_MIN)
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX)
    parser.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    parser.add_argument("--h0", type=float, default=DEFAULT_H0)
    parser.add_argument("--omega-m", type=float, default=DEFAULT_OMEGA_M)
    parser.add_argument("--xi-0", type=float, default=DEFAULT_XI_0)
    parser.add_argument("--xi-n", type=float, default=DEFAULT_XI_N)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    parser.add_argument("--z-peak", type=float, default=DEFAULT_Z_PEAK)
    parser.add_argument(
        "--local-merger-rate", type=float, default=DEFAULT_LOCAL_MERGER_RATE
    )
    parser.add_argument(
        "--importance-relative-ess",
        type=float,
        default=DEFAULT_IMPORTANCE_RELATIVE_ESS,
    )
    parser.add_argument(
        "--network",
        action="append",
        default=[],
        metavar="NAME=DET1,DET2,...",
        help="Add or override a detector-network definition (repeatable).",
    )
    parser.add_argument(
        "--networks",
        nargs="*",
        default=DEFAULT_NETWORKS,
        help="Defined network names to include in detector-chain order.",
    )
    args, _ = parser.parse_known_args(argv)
    try:
        args.resolved_networks = _resolve_networks(
            DEFAULT_DETECTOR_NETWORKS, args.network, args.networks
        )
    except ValueError as error:
        parser.error(str(error))
    return args


args = _parse_args()
root = paper_project_root()

# %% [markdown]
# ## Load the chains
#
# Read only the chains owned by the selected experiment-facing section.

# %%
figure_config = load_mapping(_resolve_path(args.config, root))
base_config = load_mapping(_resolve_path(args.base_config, root))
networks = args.resolved_networks
fiducials = {
    **base_config["fiducials"],
    "importance_relative_ess": args.importance_relative_ess,
}
use_paper_style()
outputs = {}
opened_data = []

if args.section == "detectors":
    if len(args.detector_chains) != len(args.detector_labels):
        raise ValueError(
            "--detector-chains and --detector-labels must have equal length"
        )
    if len(args.detector_chains) != len(networks):
        raise ValueError(
            "detector chain count must match --networks: "
            f"received {len(args.detector_chains)} chains for {len(networks)} networks"
        )
    detector_data = [
        load_inference_data(_resolve_path(path, root))
        for path in args.detector_chains
    ]
    opened_data.extend(detector_data)
    validate_inference_data(
        detector_data,
        args.detector_labels,
        group=args.group,
        expected_count=len(networks),
    )
    detector_colors, detector_linestyles = detector_network_styles(networks)
    outputs[_resolve_path(args.output_detector_pdf, root)] = plot_h0_posteriors(
        detector_data,
        args.detector_labels,
        colors=detector_colors,
        linestyles=detector_linestyles,
        group=args.group,
        fiducial=fiducials["H0"],
        ax_kwargs=figure_config.get("axis"),
        legend_kwargs=figure_config.get("legend"),
    )
    snr_table = compute_network_snrs(
        _resolve_path(args.catalog, root),
        networks,
        fiducials,
        observation_time=base_config["observation_time"],
        f_min=base_config["analysis"]["f_min"],
        f_max=base_config["analysis"]["f_max"],
        z_min=base_config["cosmology"]["z_min"],
        z_max=base_config["cosmology"]["z_max"],
        n_grid=base_config["cosmology"]["n_grid"],
    )
    table = build_snr_h0_constraint_table(
        networks,
        detector_data,
        args.detector_labels,
        snr_table,
        h0_fiducial=fiducials["H0"],
        group=args.group,
    )
    csv_path = _resolve_path(args.output_csv, root)
    tex_path = _resolve_path(args.output_tex, root)
    write_constraint_table(table, csv_path, tex_path)

elif args.section == "merger-rate":
    if len(args.prior_chains) != len(args.prior_labels) or len(args.prior_chains) != 2:
        raise ValueError("the merger-rate comparison requires two chains and labels")
    prior_data = [
        load_inference_data(_resolve_path(path, root)) for path in args.prior_chains
    ]
    opened_data.extend(prior_data)
    validate_inference_data(
        prior_data,
        args.prior_labels,
        group=args.group,
        expected_count=2,
    )
    corner_data, corner_labels, _ = select_corner_inference_data(
        prior_data, args.prior_labels, group=args.group
    )
    colors = combo_colors(len(prior_data))
    outputs[_resolve_path(args.output_prior_pdf, root)] = plot_h0_posteriors(
        prior_data,
        args.prior_labels,
        colors=colors,
        linestyles=["-"] * len(prior_data),
        group=args.group,
        fiducial=fiducials["H0"],
        ax_kwargs=figure_config.get("axis"),
        legend_kwargs=figure_config.get("legend"),
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
        prior_data, args.prior_labels, group=args.group
    )
    csv_path = _resolve_path(args.output_csv, root)
    tex_path = _resolve_path(args.output_tex, root)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path)
    tex_path.write_text(h0_r0_uncertainty_table_latex(table), encoding="utf-8")

else:
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
