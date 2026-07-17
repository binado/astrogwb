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
# posteriors. It also compares fixed, narrow, and broad priors on the local merger
# rate $\mathcal{R}_0$, including separate joint $H_0$--$\mathcal{R}_0$ corners
# for the narrow and broad merger-rate priors.
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
from arviz_base.labels import MapLabeller
from matplotlib.axes import Axes as MplAxes
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection
from pluscross import load_catalog

from _paper_style import (
    CORNER_LEVELS,
    combo_colors,
    get_corner_kwargs,
    use_paper_style,
)
from astrogwb.config.loading import load_mapping
from astrogwb.detector import effective_psd, load_sensitivity_map
from astrogwb.gwb import frequency_mask as make_frequency_mask
from astrogwb.gwb import spectral_density, spectral_snr
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_proposal_logpdf,
    make_merger_rate_and_log_weights_fn,
)
from astrogwb.utils import repo_root, years_to_seconds
from astrogwb.waveform import polarization_power as compute_polarization_power

# gwpy (via gwmock-signal) replaces matplotlib's rectilinear axes. ArviZ can then
# mis-detect the backend, so restore the standard matplotlib projection.
register_projection(MplAxes)
jax.config.update("jax_enable_x64", True)

# %config InlineBackend.figure_format = "retina"

# %% [markdown]
# ## Defaults
#
# Direct notebook runs use the defaults below. The paper workflow passes the same
# values explicitly from `configs/paper.toml` and `configs/workflow.yaml`.

# %%
DEFAULT_CATALOG_PATH = Path("out/catalogs/bns-n16384-df1.h5")
DEFAULT_CONFIG_PATH = Path("configs/paper.toml")
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
        "chains/bns-n16384-df1/cosmology-all-detectors/ET-triangular__H0__baseline.nc"
    ),
    Path(
        "chains/bns-n16384-df1/cosmology-all-detectors/"
        "ET-triangular-CE-Hanford__H0__baseline.nc"
    ),
    Path(
        "chains/bns-n16384-df1/cosmology-all-detectors/ET-2L-aligned__H0__baseline.nc"
    ),
    Path(
        "chains/bns-n16384-df1/cosmology-all-detectors/"
        "ET-2L-aligned-CE-Hanford__H0__baseline.nc"
    ),
    Path(
        "chains/bns-n16384-df1/cosmology-all-detectors/"
        "ET-2L-misaligned__H0__baseline.nc"
    ),
    Path(
        "chains/bns-n16384-df1/cosmology-all-detectors/"
        "ET-2L-misaligned-CE-Hanford__H0__baseline.nc"
    ),
]
DEFAULT_DETECTOR_LABELS = [
    r"ET-$\Delta$",
    r"ET-$\Delta +$ CE",
    "ET-2L",
    r"ET-2L $+$ CE",
    r"ET-2L-$\alpha$",
    r"ET-2L-$\alpha +$ CE",
]

DEFAULT_PRIOR_CHAINS = [
    Path(
        "chains/bns-n16384-df1/cosmology-all-detectors/"
        "ET-2L-aligned-CE-Hanford__H0__baseline.nc"
    ),
    Path(
        "chains/bns-n16384-df1/cosmology-all-detectors/"
        "ET-2L-aligned-CE-Hanford__H0-merger-rate-gauss__baseline.nc"
    ),
    Path(
        "chains/bns-n16384-df1/cosmology-all-detectors/"
        "ET-2L-aligned-CE-Hanford__H0-merger-rate__baseline.nc"
    ),
]
DEFAULT_PRIOR_LABELS = [
    r"$H_0$ (fixed $\mathcal{R}_0$)",
    r"$H_0 + \mathcal{R}_0$ (narrow prior)",
    r"$H_0 + \mathcal{R}_0$ (broad prior)",
]

H0_LABEL = r"$H_0\,[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$"
LOCAL_MERGER_RATE_LABEL = r"$\mathcal{R}_0\,[\mathrm{Gpc^{-3}\,yr^{-1}}]$"
CORNER_VAR_NAMES = ("H0", "local_merger_rate")


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
    """Select the two H0--local-merger-rate trees from the prior comparison."""
    validate_inference_data(inference_data, labels, group=group)
    indices = [
        index
        for index, tree in enumerate(inference_data)
        if "local_merger_rate" in tree[group].data_vars
    ]
    if len(indices) != 2:
        raise ValueError(
            "expected exactly two prior-comparison chains containing "
            f"local_merger_rate, found {len(indices)}"
        )
    selected_data = [inference_data[index] for index in indices]
    selected_labels = [labels[index] for index in indices]
    validate_inference_data(
        selected_data,
        selected_labels,
        required_vars=CORNER_VAR_NAMES,
        group=group,
        expected_count=2,
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

    resolved_ax_kwargs = {
        "xlabel": H0_LABEL,
        "ylabel": "Posterior density",
        **dict(ax_kwargs or {}),
    }
    ax.set(**resolved_ax_kwargs)
    ax.legend(**dict(legend_kwargs or {}))
    fig.tight_layout()
    return fig


def _pooled_corner_range(
    inference_data: Sequence[xr.DataTree],
    *,
    group: str,
    padding_fraction: float = 0.02,
) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for name in CORNER_VAR_NAMES:
        values = np.concatenate(
            [np.asarray(tree[group][name]).reshape(-1) for tree in inference_data]
        )
        lower = float(np.nanmin(values))
        upper = float(np.nanmax(values))
        width = upper - lower
        padding = padding_fraction * width if width > 0 else max(abs(lower), 1.0) * 0.02
        ranges.append((lower - padding, upper + padding))
    return ranges


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
    validate_inference_data(
        inference_data,
        labels,
        required_vars=CORNER_VAR_NAMES,
        group=group,
    )
    resolved_colors, resolved_linestyles = _validate_styles(
        len(inference_data), colors, linestyles
    )
    labeller = MapLabeller(
        var_name_map={
            "H0": H0_LABEL,
            "local_merger_rate": LOCAL_MERGER_RATE_LABEL,
        }
    )
    truths = None
    if fiducials is not None:
        truths = {name: fiducials[name] for name in CORNER_VAR_NAMES}

    fig: plt.Figure | None = None
    plot_range = _pooled_corner_range(inference_data, group=group)
    for index, (tree, color, linestyle) in enumerate(
        zip(inference_data, resolved_colors, resolved_linestyles, strict=True)
    ):
        fig = corner.corner(
            tree,
            group=group,
            var_names=list(CORNER_VAR_NAMES),
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
            **get_corner_kwargs(
                plot_datapoints=False,
                plot_density=False,
                fill_contours=False,
            ),
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
    proposal_log_pdf = compute_proposal_logpdf(
        samples["redshift"], z_grid=z_grid, fiducials=fiducials
    )
    merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
        z_grid=z_grid,
        proposal_log_pdf=proposal_log_pdf,
        fiducial_xi_0=fiducials["xi_0"],
        fiducial_xi_n=fiducials["xi_n"],
    )
    total_rate, log_weights = merger_rate_and_log_weights_fn(fiducials, samples)
    observed_spectral_density = spectral_density(
        polarization_power,
        jnp.exp(log_weights),
        total_rate,
        average_mode="analytic_inclination",
    )

    mask = make_frequency_mask(frequencies, fmin=f_min, fmax=f_max)
    frequency_spacing = jnp.mean(jnp.diff(frequencies))
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
    *,
    group: str,
    probability: float,
) -> tuple[float, float]:
    interval = azs.hdi(tree, group=group, var_names="H0", prob=probability)["H0"]
    return (
        float(interval.sel(ci_bound="lower")),
        float(interval.sel(ci_bound="upper")),
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
    for (network, detectors), tree, label in zip(
        networks.items(), inference_data, labels, strict=True
    ):
        snr = float(snr_by_network.loc[network, "snr"])
        lower, upper = _hdi(tree, group=group, probability=probability)
        sigma_hdi = (upper - lower) / 2
        sigma_snr = h0_fiducial / snr
        rows.append(
            {
                "network": network,
                "label": label,
                "detectors": ",".join(detectors),
                "n_detectors": len(detectors),
                "snr": snr,
                "h0_hdi_lower": lower,
                "h0_hdi_upper": upper,
                "sigma_h0_hdi": sigma_hdi,
                "sigma_h0_snr": sigma_snr,
                "rel_sigma_h0_hdi": sigma_hdi / h0_fiducial,
                "rel_sigma_h0_snr": 1.0 / snr,
            }
        )
    return pd.DataFrame(rows)


def constraint_table_latex(table: pd.DataFrame) -> str:
    """Format the constraint table as a publication LaTeX tabular."""
    latex_table = table.rename(
        columns={
            "network": "Network",
            "label": "Label",
            "detectors": "Detectors",
            "n_detectors": r"$N_{\rm det}$",
            "snr": "SNR",
            "h0_hdi_lower": r"$H_{0,\rm low}$",
            "h0_hdi_upper": r"$H_{0,\rm high}$",
            "sigma_h0_hdi": r"$\sigma_{H_0}^{\rm HDI}$",
            "sigma_h0_snr": r"$\sigma_{H_0}^{\rm SNR}$",
            "rel_sigma_h0_hdi": r"$\sigma_{H_0}^{\rm HDI}/H_0$",
            "rel_sigma_h0_snr": r"$1/{\rm SNR}$",
        }
    )
    return latex_table.to_latex(
        index=False,
        escape=False,
        float_format="%.3g",
        caption=(
            "Matched-filter SNR and $H_0$ constraints by detector network. "
            "Posterior constraints are half-widths of the 68.27\\% HDI."
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument(
        "--detector-chains", type=Path, nargs="+", default=DEFAULT_DETECTOR_CHAINS
    )
    parser.add_argument("--detector-labels", nargs="+", default=DEFAULT_DETECTOR_LABELS)
    parser.add_argument(
        "--prior-chains", type=Path, nargs="+", default=DEFAULT_PRIOR_CHAINS
    )
    parser.add_argument("--prior-labels", nargs="+", default=DEFAULT_PRIOR_LABELS)
    parser.add_argument(
        "--output-detector-pdf",
        type=Path,
        default=Path("figures/mcmc_cosmological_parameters_H0_by_detector.pdf"),
    )
    parser.add_argument(
        "--output-prior-pdf",
        type=Path,
        default=Path("figures/mcmc_cosmological_parameters_H0_merger_rate_priors.pdf"),
    )
    parser.add_argument(
        "--output-narrow-corner-pdf",
        type=Path,
        default=Path(
            "figures/mcmc_cosmological_parameters_H0_merger_rate_narrow_corner.pdf"
        ),
    )
    parser.add_argument(
        "--output-broad-corner-pdf",
        type=Path,
        default=Path(
            "figures/mcmc_cosmological_parameters_H0_merger_rate_broad_corner.pdf"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("figures/mcmc_cosmological_parameters_H0_by_detector.csv"),
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("figures/mcmc_cosmological_parameters_H0_by_detector.tex"),
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
root = repo_root()

# %% [markdown]
# ## Load the chains
#
# Read the detector-network and prior-comparison inference runs from disk and
# check that each one carries the posterior variables the figures below need.

# %%
config = load_mapping(_resolve_path(args.config, root))
figure_config = config["figures"]["mcmc_cosmological_parameters"]
networks = args.resolved_networks

if len(args.detector_chains) != len(args.detector_labels):
    raise ValueError("--detector-chains and --detector-labels must have equal length")
if len(args.detector_chains) != len(networks):
    raise ValueError(
        "detector chain count must match --networks: "
        f"received {len(args.detector_chains)} chains for {len(networks)} networks"
    )
if len(args.prior_chains) != len(args.prior_labels):
    raise ValueError("--prior-chains and --prior-labels must have equal length")
if len(args.prior_chains) != 3:
    raise ValueError("the prior comparison requires exactly three chains")

detector_paths = [_resolve_path(path, root) for path in args.detector_chains]
prior_paths = [_resolve_path(path, root) for path in args.prior_chains]
detector_data = [load_inference_data(path) for path in detector_paths]
prior_data = [load_inference_data(path) for path in prior_paths]

validate_inference_data(
    detector_data,
    args.detector_labels,
    group=args.group,
    expected_count=len(networks),
)
validate_inference_data(
    prior_data,
    args.prior_labels,
    group=args.group,
    expected_count=3,
)
corner_data, corner_labels, corner_indices = select_corner_inference_data(
    prior_data, args.prior_labels, group=args.group
)
narrow_data = [corner_data[0]]
narrow_labels = [corner_labels[0]]
broad_data = [corner_data[1]]
broad_labels = [corner_labels[1]]

detector_colors = combo_colors(len(networks))
detector_linestyles = ["-"] * len(networks)
prior_colors = combo_colors(len(prior_data))
prior_linestyles = ["-"] * len(prior_data)
corner_colors = [prior_colors[index] for index in corner_indices]
corner_linestyles = [prior_linestyles[index] for index in corner_indices]

fiducials = {
    "H0": args.h0,
    "Omega_m": args.omega_m,
    "xi_0": args.xi_0,
    "xi_n": args.xi_n,
    "gamma": args.gamma,
    "kappa": args.kappa,
    "z_peak": args.z_peak,
    "local_merger_rate": args.local_merger_rate,
}

use_paper_style()

# %% [markdown]
# ## Figure (i): $H_0$ by detector network
#
# Marginalized $H_0$ posteriors for each configured detector network under the
# baseline cosmology analysis.

# %%
detector_figure = plot_h0_posteriors(
    detector_data,
    args.detector_labels,
    colors=detector_colors,
    linestyles=detector_linestyles,
    group=args.group,
    ax_kwargs=figure_config.get("detector_ax_kwargs"),
    legend_kwargs=figure_config.get("detector_legend_kwargs"),
)

# %% [markdown]
# ## Figure (ii): $H_0$ prior comparison
#
# Compares fixed, narrow, and broad priors on the local merger rate
# $\mathcal{R}_0$ for a single network.

# %%
prior_figure = plot_h0_posteriors(
    prior_data,
    args.prior_labels,
    colors=prior_colors,
    linestyles=prior_linestyles,
    group=args.group,
    ax_kwargs=figure_config.get("prior_ax_kwargs"),
    legend_kwargs=figure_config.get("prior_legend_kwargs"),
)

# %% [markdown]
# ## Figure (iii): Narrow-prior $H_0$--$\mathcal{R}_0$ corner
#
# Joint constraint when the local merger rate carries a narrow Gaussian prior.

# %%
narrow_corner_figure = plot_h0_merger_rate_corner(
    narrow_data,
    narrow_labels,
    colors=[corner_colors[0]],
    linestyles=[corner_linestyles[0]],
    group=args.group,
    fiducials=fiducials,
)

# %% [markdown]
# ## Figure (iv): Broad-prior $H_0$--$\mathcal{R}_0$ corner
#
# Joint constraint when the local merger rate carries a broad prior. Plotted
# separately from the narrow-prior corner because the posterior masses differ
# enough that a shared axis range is unhelpful.

# %%
broad_corner_figure = plot_h0_merger_rate_corner(
    broad_data,
    broad_labels,
    colors=[corner_colors[1]],
    linestyles=[corner_linestyles[1]],
    group=args.group,
    fiducials=fiducials,
)

# %% [markdown]
# ## Fiducial SNR and $H_0$ constraint table
#
# Evaluate the fiducial SGWB once, compute the matched-filter SNR for each
# detector network, and combine those estimates with sampled $H_0$ HDI widths.

# %%
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
constraint_table = build_snr_h0_constraint_table(
    networks,
    detector_data,
    args.detector_labels,
    snr_table,
    h0_fiducial=args.h0,
    group=args.group,
)
constraint_table

# %% [markdown]
# ## LaTeX constraint table
#
# Publication-formatted version of the table above.

# %%
print(constraint_table_latex(constraint_table))

# %% [markdown]
# ## Save figures and table
#
# Write the figures and machine-readable / LaTeX constraint tables to the
# configured output paths.

# %%
outputs = {
    _resolve_path(args.output_detector_pdf, root): detector_figure,
    _resolve_path(args.output_prior_pdf, root): prior_figure,
    _resolve_path(args.output_narrow_corner_pdf, root): narrow_corner_figure,
    _resolve_path(args.output_broad_corner_pdf, root): broad_corner_figure,
}
for output_path, figure in outputs.items():
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.figure_dpi, bbox_inches="tight")
    print("saved figure:", output_path)

csv_path = _resolve_path(args.output_csv, root)
tex_path = _resolve_path(args.output_tex, root)
write_constraint_table(constraint_table, csv_path, tex_path)
print("saved constraint table:", csv_path)
print("saved LaTeX table:", tex_path)

for tree in [*detector_data, *prior_data]:
    tree.close()
