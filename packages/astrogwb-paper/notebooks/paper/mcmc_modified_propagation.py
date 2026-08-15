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
# three inference runs on the same detector network:
#
# 1. $\Xi_0$ only (population and cosmology fixed at their fiducial values),
# 2. $\Xi_0 + n$ (both propagation parameters sampled), and
# 3. $\Xi_0 + H_0$ with a Gaussian prior on the Hubble constant.
#
# It produces three figures and two tables:
#
# - a corner plot of the $\Xi_0$--$n$ posterior (chain 2),
# - an overlay of the marginal $\Xi_0$ posterior across all three chains,
# - a $\Xi_0$--$H_0$ corner plot (chain 3),
# - a $\Xi_0$ 1$\sigma$ HDI table across the three chains, and
# - a matched-filter-SNR / $\Xi_0$-$n$ HDI constraint table across detector
#   networks, using the joint $\Xi_0 + n$ chain for each network.
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
from astrogwb_paper.catalog import apply_gw_distance_at_fiducial
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
# Direct notebook runs use the defaults below. Edit them in Jupyter or override
# with flags when running headless; the paper workflow passes the ordered chain
# paths as declared inputs.

# %%
_CHAIN_DIR = Path("outputs/chains/modified-propagation-all-detectors")

DEFAULT_XI0_CHAIN = _CHAIN_DIR / "Xi_0.nc"
DEFAULT_XI0_N_CHAIN = _CHAIN_DIR / "ET-2L-aligned-CE-Hanford.nc"
DEFAULT_H0_CHAIN = _CHAIN_DIR / "Xi_0-H0.nc"
DEFAULT_BASE_CONFIG_PATH = Path("inputs/mcmc.base.toml")

# Fiducial (injected) values marked as truths on the corner plots.
DEFAULT_XI_0 = 1.0
DEFAULT_XI_N = 1.91
DEFAULT_H0 = 67.66
DEFAULT_IMPORTANCE_RELATIVE_ESS = 1.0

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

# Labels for the three-chain marginal overlay, in chain order.
DEFAULT_MARGINAL_LABELS = [
    r"$\Xi_0$",
    r"$\Xi_0 + n$",
    r"$\Xi_0 + H_0$",
]
# Label for the single-chain Xi_0-H0 corner plot.
DEFAULT_H0_LABELS = [
    r"$\Xi_0 + H_0$",
]

# Fiducial population/cosmology hyperparameters needed to evaluate the SGWB and
# its matched-filter SNR for the by-detector constraint table below.
DEFAULT_OMEGA_M = 0.3096
DEFAULT_GAMMA = 1.42
DEFAULT_KAPPA = 4.62
DEFAULT_Z_PEAK = 1.84
DEFAULT_LOCAL_MERGER_RATE = 161.0

DEFAULT_CATALOG_PATH = Path("outputs/catalogs/bns-n16384-df1.h5")
DEFAULT_OBSERVATION_TIME = 1.0
DEFAULT_F_MIN = 2.0
DEFAULT_F_MAX = 4096.0
DEFAULT_Z_MIN = 0.0
DEFAULT_Z_MAX = 20.0
DEFAULT_N_GRID = 256

DEFAULT_DETECTOR_NETWORKS = {
    "ET-triangular": ("E1", "E2", "E3"),
    "ET-triangular-CE-Hanford": ("E1", "E2", "E3", "C1"),
    "ET-2L-aligned": ("S1", "R1"),
    "ET-2L-aligned-CE-Hanford": ("S1", "R1", "C1"),
    "ET-2L-misaligned": ("S2", "R2"),
    "ET-2L-misaligned-CE-Hanford": ("S2", "R2", "C1"),
}
DEFAULT_NETWORKS = list(DEFAULT_DETECTOR_NETWORKS)
DEFAULT_DETECTOR_LABELS = [
    r"ET-$\Delta$",
    r"ET-$\Delta +$ CE",
    "ET-2L-par",
    r"ET-2L-par $+$ CE",
    "ET-2L",
    r"ET-2L $+$ CE",
]
DEFAULT_DETECTOR_XI0_N_CHAINS = [
    _CHAIN_DIR / f"{name}.nc" for name in DEFAULT_NETWORKS
]

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


# %% [markdown]
# ## Fiducial SNR and $\Xi_0$/$n$ constraint table by detector


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
    # Rescale to live-GW distances at the fiducial modified-propagation
    # parameters before reducing to polarization power.
    catalog = apply_gw_distance_at_fiducial(
        catalog,
        xi_0=float(fiducials["xi_0"]),
        xi_n=float(fiducials["xi_n"]),
    )
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
    networks: Mapping[str, tuple[str, ...]],
    inference_data: Sequence[xr.DataTree],
    labels: Sequence[str],
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
        labels,
        required_vars=XI_N_VAR_NAMES,
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
        xi0_lower, xi0_upper = _hdi(tree, "xi_0", group=group, probability=probability)
        n_lower, n_upper = _hdi(tree, "xi_n", group=group, probability=probability)
        sigma_xi0_hdi = (xi0_upper - xi0_lower) / 2
        sigma_n_hdi = (n_upper - n_lower) / 2
        rows.append(
            {
                "label": label,
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


# %% [markdown]
# ## Command-line configuration
#
# Every setting below can be overridden with a CLI flag when running this
# notebook headless; otherwise the defaults above are used.


# %%
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG_PATH)
    parser.add_argument("--xi0-chain", type=Path, default=DEFAULT_XI0_CHAIN)
    parser.add_argument("--xi0-n-chain", type=Path, default=DEFAULT_XI0_N_CHAIN)
    parser.add_argument("--h0-chain", type=Path, default=DEFAULT_H0_CHAIN)
    parser.add_argument("--marginal-labels", nargs=3, default=DEFAULT_MARGINAL_LABELS)
    parser.add_argument("--h0-labels", nargs=1, default=DEFAULT_H0_LABELS)
    parser.add_argument(
        "--output-xi-n-corner-pdf",
        type=Path,
        default=Path("figures/mcmc_modified_propagation_Xi0_n_corner.pdf"),
    )
    parser.add_argument(
        "--output-xi-n-ess-corner-pdf",
        type=Path,
        default=Path("figures/mcmc_modified_propagation_Xi0_n_ess_corner.pdf"),
    )
    parser.add_argument(
        "--output-xi0-marginal-pdf",
        type=Path,
        default=Path("figures/mcmc_modified_propagation_Xi0_marginal.pdf"),
    )
    parser.add_argument(
        "--output-h0-corner-pdf",
        type=Path,
        default=Path("figures/mcmc_modified_propagation_Xi0_H0_corner.pdf"),
    )
    parser.add_argument(
        "--detector-xi0-n-chains",
        type=Path,
        nargs="+",
        default=DEFAULT_DETECTOR_XI0_N_CHAINS,
    )
    parser.add_argument("--detector-labels", nargs="+", default=DEFAULT_DETECTOR_LABELS)
    parser.add_argument(
        "--output-xi0-n-csv",
        type=Path,
        default=Path("figures/mcmc_modified_propagation_Xi0_n_by_detector.csv"),
    )
    parser.add_argument(
        "--output-xi0-n-tex",
        type=Path,
        default=Path("figures/mcmc_modified_propagation_Xi0_n_by_detector.tex"),
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
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
    parser.add_argument(
        "--observation-time", type=float, default=DEFAULT_OBSERVATION_TIME
    )
    parser.add_argument("--f-min", type=float, default=DEFAULT_F_MIN)
    parser.add_argument("--f-max", type=float, default=DEFAULT_F_MAX)
    parser.add_argument("--z-min", type=float, default=DEFAULT_Z_MIN)
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX)
    parser.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    parser.add_argument("--omega-m", type=float, default=DEFAULT_OMEGA_M)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    parser.add_argument("--z-peak", type=float, default=DEFAULT_Z_PEAK)
    parser.add_argument(
        "--local-merger-rate", type=float, default=DEFAULT_LOCAL_MERGER_RATE
    )
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument("--group", default="posterior")
    parser.add_argument("--xi-0", type=float, default=DEFAULT_XI_0)
    parser.add_argument("--xi-n", type=float, default=DEFAULT_XI_N)
    parser.add_argument("--h0", type=float, default=DEFAULT_H0)
    parser.add_argument(
        "--importance-relative-ess",
        type=float,
        default=DEFAULT_IMPORTANCE_RELATIVE_ESS,
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
base_config = load_mapping(_resolve_path(args.base_config, root))

# %% [markdown]
# ## Load the chains
#
# Read the three inference runs from disk and check that each one carries the
# posterior variables the figures below need.

# %%
# Chain order: [xi_0 only, xi_0 + n, xi_0 + H0].
chain_paths = [
    _resolve_path(args.xi0_chain, root),
    _resolve_path(args.xi0_n_chain, root),
    _resolve_path(args.h0_chain, root),
]

inference_data: list[xr.DataTree] = [load_inference_data(path) for path in chain_paths]
validate_inference_data(
    inference_data,
    args.marginal_labels,
    group=args.group,
    expected_count=3,
)

xi_n_data = [inference_data[1]]
xi_n_labels = [args.marginal_labels[1]]
h0_data = [inference_data[2]]

fiducials = {
    **base_config["fiducials"],
    "importance_relative_ess": args.importance_relative_ess,
}

networks = args.resolved_networks
if len(args.detector_xi0_n_chains) != len(args.detector_labels):
    raise ValueError(
        "--detector-xi0-n-chains and --detector-labels must have equal length"
    )
if len(args.detector_xi0_n_chains) != len(networks):
    raise ValueError(
        "detector chain count must match --networks: "
        f"received {len(args.detector_xi0_n_chains)} chains for {len(networks)} networks"
    )
detector_xi0_n_paths = [
    _resolve_path(path, root) for path in args.detector_xi0_n_chains
]
detector_xi0_n_data = [load_inference_data(path) for path in detector_xi0_n_paths]
validate_inference_data(
    detector_xi0_n_data,
    args.detector_labels,
    required_vars=XI_N_VAR_NAMES,
    group=args.group,
    expected_count=len(networks),
)

use_paper_style()

# %% [markdown]
# ## Figure (i): $\Xi_0$--$n$ corner
#
# Both propagation parameters are sampled together in this chain, so this
# corner plot shows how well they can be told apart from each other.

# %%
xi_n_corner_figure = plot_corner(
    xi_n_data,
    xi_n_labels,
    XI_N_VAR_NAMES,
    group=args.group,
    fiducials=fiducials,
    colors=[CATEGORY["modified_propagation"]],
)

# %% [markdown]
# ## Figure (i-b): $\Xi_0$--$n$--relative-ESS corner
#
# Mirror of the $\Xi_0$--$n$ corner that also shows the importance-sampling
# relative effective sample size $N_{\mathrm{eff}} / N_{\mathrm{inj}}$.

# %%
xi_n_ess_corner_figure = plot_corner(
    xi_n_data,
    xi_n_labels,
    XI_N_ESS_VAR_NAMES,
    group=args.group,
    fiducials=fiducials,
    colors=[CATEGORY["modified_propagation"]],
)

# %% [markdown]
# ## Figure (ii): $\Xi_0$ marginal posterior overlay
#
# Compares how tightly $\Xi_0$ is constrained across all three inference
# setups, from the simplest (fixed cosmology and population) to the most
# flexible.

# %%
xi0_marginal_figure = plot_marginal_posteriors(
    inference_data,
    args.marginal_labels,
    var_name="xi_0",
    group=args.group,
    colors=combo_colors(len(inference_data)),
    fiducial=fiducials["xi_0"],
)

# %% [markdown]
# ## Table: $\Xi_0$ 1$\sigma$ HDI per chain
#
# Reports the 68.27% highest-density interval of $\Xi_0$ for each chain, with
# the left/right bounds and the half-width $\sigma = (\mathrm{right} -
# \mathrm{left}) / 2$.

# %%
xi0_hdi = xi0_hdi_table(inference_data, args.marginal_labels, group=args.group)

# %% [markdown]
# ## Figure (iii): $\Xi_0$--$H_0$ corner
#
# Shows the joint constraint on $\Xi_0$ and the Hubble constant $H_0$ when
# $H_0$ is sampled under a Gaussian prior.

# %%
h0_corner_figure = plot_corner(
    h0_data,
    args.h0_labels,
    H0_VAR_NAMES,
    group=args.group,
    fiducials=fiducials,
    colors=[CATEGORY["modified_propagation"]],
)

# %% [markdown]
# ## Fiducial SNR and $\Xi_0$/$n$ constraint table by detector
#
# Evaluate the fiducial SGWB once, compute the matched-filter SNR for each
# detector network, and combine those estimates with the sampled $\Xi_0$/$n$
# HDI widths from the joint $\Xi_0 + n$ chain for that network.

# %%
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
xi0_n_constraint_table = build_snr_xi0_n_constraint_table(
    networks,
    detector_xi0_n_data,
    args.detector_labels,
    snr_table,
    group=args.group,
)

# %% [markdown]
# ## LaTeX $\Xi_0$/$n$ constraint table
#
# Publication-formatted version of the table above.

# %%
print(xi0_n_constraint_table_latex(xi0_n_constraint_table))

# %% [markdown]
# ## Save figures and table
#
# Write the three figures above, and the machine-readable / LaTeX constraint
# tables, to the configured output paths.

# %%
outputs = {
    _resolve_path(args.output_xi_n_corner_pdf, root): xi_n_corner_figure,
    _resolve_path(args.output_xi_n_ess_corner_pdf, root): xi_n_ess_corner_figure,
    _resolve_path(args.output_xi0_marginal_pdf, root): xi0_marginal_figure,
    _resolve_path(args.output_h0_corner_pdf, root): h0_corner_figure,
}
for output_path, figure in outputs.items():
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.figure_dpi, bbox_inches="tight")
    print("saved figure:", output_path)

xi0_n_csv_path = _resolve_path(args.output_xi0_n_csv, root)
xi0_n_tex_path = _resolve_path(args.output_xi0_n_tex, root)
write_xi0_n_constraint_table(xi0_n_constraint_table, xi0_n_csv_path, xi0_n_tex_path)
print("saved constraint table:", xi0_n_csv_path)
print("saved LaTeX table:", xi0_n_tex_path)
