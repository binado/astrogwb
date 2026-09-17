# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: astrogwb (3.12.9)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Expansion-history parameters from grid-evaluated posteriors
#
# This notebook reproduces the grid-evaluated $H_0$ and $(H_0, \Omega_m)$
# figures. Each detector network gets its own $H_0$ grid, sized from its
# matched-filter SNR through the Fisher prediction
# $\sigma_{H_0} \approx H_0 / \mathrm{SNR}$.
#
# The notebook resolves the checkout root with `astrogwb.paper.paths.root_dir`,
# so it runs from the repository root or from this directory; the catalog is
# read from outputs/catalogs/md-imrphenom-s41-n32768.h5 below it.

# %% [markdown]
# ## Imports and JAX configuration

# %%
import json
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection
from numpyro.distributions import Distribution, Normal, Uniform

from astrogwb.metadata import PopulationMetadata, WaveformMetadata
from astrogwb.paper.catalogs import load_run_catalog
from astrogwb.paper.config import fiducials, networks, priors
from astrogwb.paper.config.mcmc import AnalysisGrid
from astrogwb.paper.inference import InferenceInputs, prepare_inference_inputs
from astrogwb.paper.paths import root_dir
from astrogwb.paper.plotting import (
    CORNER_LEVELS,
    DETECTOR_COMPARISON_LEGEND,
    DETECTOR_NETWORKS,
    TRUTH,
    Network,
    detector_network_styles,
    parameter_labels,
    plot_corner_for_posterior_grid,
    use_paper_style,
)
from astrogwb.paper.snr import compute_network_snrs
from astrogwb.populations import build_population
from astrogwb.sampling import LogDensityFn, gwb_spectral_density_model

register_projection(MplAxes)
use_paper_style()
jax.config.update("jax_enable_x64", True)
# %config InlineBackend.figure_format = 'retina'

# %% [markdown]
# ## Notebook configuration

# %%
#: Jupyter's kernel cwd is this notebook's own directory, so anchor the
#: notebook's own file paths on the checkout root -- the same root the
#: `astrogwb.paper` accessors default to when they are given no `root=`.
ROOT_DIR = root_dir()
DEBUG: bool = False
INJECTION_CATALOG_PATH = ROOT_DIR / "outputs/catalogs/md-imrphenom-s41-n32768.h5"
PROPOSAL_CATALOG_PATH = INJECTION_CATALOG_PATH
ANALYSIS_GRID = AnalysisGrid(
    observation_time=1.0,
    minimum_frequency=2.0,
    maximum_frequency=2048.0,
    minimum_redshift=0.3,
    maximum_redshift=20.0,
    n_grid=256,
)
FIDUCIALS = fiducials()
PRIORS: dict[str, Distribution] = priors()
NETWORK_CONFIG = networks()
PARAMETER_LABELS = parameter_labels()
DEFAULT_NETWORK = "ET-2L-aligned-CE-Hanford"
COVERAGE_SIGMAS: float = 5.0
NPOINTS_1D: int = 256
NPOINTS_2D: int = 96
CHUNK_SIZE: int = 64
if DEBUG:
    NPOINTS_1D, NPOINTS_2D = 33, 17
SAVE_OUTPUTS: bool = True
GRID_DIR = ROOT_DIR / "grids"
FIGURE_DIR = ROOT_DIR / "figures"

# %% [markdown]
# ## Networks, catalogs, and SNRs

# %%
NETWORKS: tuple[Network, ...] = tuple(
    Network(name, label, NETWORK_CONFIG[name]) for name, label in DETECTOR_NETWORKS
)
injection_catalog = load_run_catalog(INJECTION_CATALOG_PATH, label="injection")
proposal_catalog = load_run_catalog(PROPOSAL_CATALOG_PATH, label="proposal")
PROPOSAL_POPULATION_METADATA: PopulationMetadata = proposal_catalog.population
PROPOSAL_WAVEFORM_METADATA: WaveformMetadata = proposal_catalog.waveform_metadata
TARGET_MODEL = build_population(
    "bns_md_modified_propagation",
    minimum_redshift=ANALYSIS_GRID.minimum_redshift,
    maximum_redshift=ANALYSIS_GRID.maximum_redshift,
    n_grid=ANALYSIS_GRID.n_grid,
)
print(
    "proposal:",
    PROPOSAL_POPULATION_METADATA.model_name,
    "seed=",
    PROPOSAL_POPULATION_METADATA.seed,
    "n_samples=",
    proposal_catalog.num_samples,
)
print(
    "proposal waveform:",
    PROPOSAL_WAVEFORM_METADATA.approximant,
    "frequency_resolution=",
    PROPOSAL_WAVEFORM_METADATA.frequency_resolution,
)
SNR_TABLE = compute_network_snrs(
    INJECTION_CATALOG_PATH, NETWORKS, FIDUCIALS, grid=ANALYSIS_GRID
)
SNR_TABLE = SNR_TABLE.assign(
    sigma_h0_fisher=FIDUCIALS["H0"] / SNR_TABLE["snr"],
    rel_sigma_h0_fisher=1.0 / SNR_TABLE["snr"],
)
SNR_TABLE

# %% [markdown]
# ## Grid helpers and windows


# %%
def fisher_window(
    center: float, sigma: float, *, sigmas: float, support: tuple[float, float]
) -> tuple[float, float]:
    low = max(center - sigmas * sigma, support[0])
    high = min(center + sigmas * sigma, support[1])
    return low, high


def prior_window(prior: Distribution, *, sigmas: float) -> tuple[float, float]:
    if isinstance(prior, Uniform):
        return float(prior.low), float(prior.high)
    if isinstance(prior, Normal):
        return (
            float(prior.loc - sigmas * prior.scale),
            float(prior.loc + sigmas * prior.scale),
        )
    raise NotImplementedError(f"prior_window unsupported for {type(prior).__name__}")


def uniform_grid(low: float, high: float, npoints: int) -> jax.Array:
    return jnp.linspace(low, high, npoints)


H0_SUPPORT = (float(PRIORS["H0"].low), float(PRIORS["H0"].high))
_snr_by_network = SNR_TABLE.set_index("network")["snr"]
H0_WINDOWS: dict[str, tuple[float, float]] = {
    network.name: fisher_window(
        FIDUCIALS["H0"],
        FIDUCIALS["H0"] / float(_snr_by_network[network.name]),
        sigmas=COVERAGE_SIGMAS,
        support=H0_SUPPORT,
    )
    for network in NETWORKS
}
H0_GRIDS: dict[str, jax.Array] = {
    name: uniform_grid(low, high, NPOINTS_1D)
    for name, (low, high) in H0_WINDOWS.items()
}
H0_GRIDS_2D: dict[str, jax.Array] = {
    name: uniform_grid(low, high, NPOINTS_2D)
    for name, (low, high) in H0_WINDOWS.items()
}
OMEGA_M_GRID = uniform_grid(
    *prior_window(PRIORS["Omega_m"], sigmas=COVERAGE_SIGMAS), NPOINTS_2D
)
H0_GRID_2D = H0_GRIDS_2D[DEFAULT_NETWORK]
pd.DataFrame(
    [
        {"network": name, "low": low, "high": high}
        for name, (low, high) in H0_WINDOWS.items()
    ]
)

# %% [markdown]
# ## Log-density evaluators and grid evaluations


# %%
def build_network_inputs(network: Network) -> InferenceInputs:
    return prepare_inference_inputs(
        injection_catalog,
        proposal_catalog,
        grid=ANALYSIS_GRID,
        detectors=network.detectors,
        target=TARGET_MODEL,
        density_sites=[],
    )


def evaluate_joint(
    log_density_fn: LogDensityFn,
    grids: dict[str, jax.Array],
    *,
    model_kwargs: dict[str, jax.Array],
) -> jax.Array:
    fixed = {name: FIDUCIALS[name] for name in PRIORS if name not in grids}
    return jax.block_until_ready(log_density_fn(grids, fixed=fixed, **model_kwargs))


NETWORK_INPUTS: dict[str, InferenceInputs] = {
    network.name: build_network_inputs(network) for network in NETWORKS
}
# Detector choice changes only the effective PSD and frequency mask carried by
# `model_kwargs`; the catalog-bound spectral-density function is shared across
# networks. Keeping this model and evaluator outside the loop lets JAX reuse
# each grid-signature compilation for every network.
SHARED_MODEL = partial(
    gwb_spectral_density_model,
    spectral_density_fn=next(iter(NETWORK_INPUTS.values())).spectral_density_fn,
    priors=PRIORS,
)
SHARED_LOG_DENSITY_FN = LogDensityFn(SHARED_MODEL, chunk_size=CHUNK_SIZE)
MODEL_KWARGS: dict[str, dict[str, jax.Array]] = {}
H0_LOGPOSTERIORS: dict[str, jax.Array] = {}
H0_OMEGA_M_LOGPOSTERIORS: dict[str, jax.Array] = {}
for network in NETWORKS:
    model_kwargs = NETWORK_INPUTS[network.name].model_kwargs()
    MODEL_KWARGS[network.name] = model_kwargs
    start = time.perf_counter()
    H0_LOGPOSTERIORS[network.name] = evaluate_joint(
        SHARED_LOG_DENSITY_FN,
        {"H0": H0_GRIDS[network.name]},
        model_kwargs=model_kwargs,
    )
    H0_OMEGA_M_LOGPOSTERIORS[network.name] = evaluate_joint(
        SHARED_LOG_DENSITY_FN,
        {"H0": H0_GRIDS_2D[network.name], "Omega_m": OMEGA_M_GRID},
        model_kwargs=model_kwargs,
    )
    print(
        f"{network.label}: {time.perf_counter() - start:.2f}s for "
        f"the 1D and {NPOINTS_2D}x{NPOINTS_2D} grids"
    )

# %% [markdown]
# ## Detector overlays and constraints


# %%
def safe_exponentiate(log_values: jax.Array) -> np.ndarray:
    values = np.asarray(log_values, dtype=np.float64)
    return np.exp(values - values.max())


def marginal_along(
    log_density: jax.Array, axis_grid: jax.Array, *, axis: int
) -> np.ndarray:
    return np.trapezoid(
        safe_exponentiate(log_density), np.asarray(axis_grid), axis=axis
    )


def plot_parameter_by_detector(
    grids: dict[str, jax.Array],
    densities: dict[str, np.ndarray],
    *,
    param: str,
) -> plt.Figure:
    colors, linestyles = detector_network_styles(NETWORKS)
    fig, ax = plt.subplots()
    for network, color, linestyle in zip(NETWORKS, colors, linestyles, strict=True):
        grid = np.asarray(grids[network.name])
        density = np.asarray(densities[network.name], dtype=np.float64)
        density = density / np.trapezoid(density, grid)
        ax.plot(grid, density, label=network.label, color=color, linestyle=linestyle)
    ax.axvline(FIDUCIALS[param], **TRUTH)
    ax.set(xlabel=PARAMETER_LABELS[param], ylabel="Posterior density")
    ax.legend(**DETECTOR_COMPARISON_LEGEND)
    fig.tight_layout()
    return fig


H0_OMEGA_M_H0_MARGINAL = {
    name: marginal_along(logpost, OMEGA_M_GRID, axis=1)
    for name, logpost in H0_OMEGA_M_LOGPOSTERIORS.items()
}
fig_h0_by_detector = plot_parameter_by_detector(
    H0_GRIDS,
    {name: safe_exponentiate(logpost) for name, logpost in H0_LOGPOSTERIORS.items()},
    param="H0",
)
fig_h0_omega_m_by_detector = plot_parameter_by_detector(
    H0_GRIDS_2D, H0_OMEGA_M_H0_MARGINAL, param="H0"
)

# %%
fig_h0_by_detector

# %%
fig_h0_omega_m_by_detector


def hpd_half_width(
    grid: np.ndarray, density: np.ndarray, *, probability: float
) -> float:
    weights = density / density.sum()
    order = np.argsort(density)[::-1]
    n_included = int(np.searchsorted(np.cumsum(weights[order]), probability)) + 1
    included_grid = grid[order[:n_included]]
    return float(included_grid.max() - included_grid.min()) / 2.0


def hpd_half_width_1d(
    grid: np.ndarray, log_density: np.ndarray, *, probability: float
) -> float:
    return hpd_half_width(grid, safe_exponentiate(log_density), probability=probability)


CONSTRAINT_TABLE = pd.DataFrame(
    [
        {
            "label": network.label,
            "snr": float(_snr_by_network[network.name]),
            "sigma_h0_grid": hpd_half_width_1d(
                np.asarray(H0_GRIDS[network.name]),
                H0_LOGPOSTERIORS[network.name],
                probability=CORNER_LEVELS[0],
            ),
            "sigma_h0_omega_m_grid": hpd_half_width(
                np.asarray(H0_GRIDS_2D[network.name]),
                H0_OMEGA_M_H0_MARGINAL[network.name],
                probability=CORNER_LEVELS[0],
            ),
        }
        for network in NETWORKS
    ]
)
CONSTRAINT_TABLE = CONSTRAINT_TABLE.assign(
    rel_sigma_h0_grid=CONSTRAINT_TABLE["sigma_h0_grid"] / FIDUCIALS["H0"],
    rel_sigma_h0_omega_m_grid=CONSTRAINT_TABLE["sigma_h0_omega_m_grid"]
    / FIDUCIALS["H0"],
    rel_sigma_h0_snr=1.0 / CONSTRAINT_TABLE["snr"],
)
CONSTRAINT_TABLE

# %% [markdown]
# ## $H_0-\Omega_m$ corner plots

# %%
H0_OMEGA_M_GRIDS = {"H0": H0_GRID_2D, "Omega_m": OMEGA_M_GRID}
H0_OMEGA_M_LOGPOST = H0_OMEGA_M_LOGPOSTERIORS[DEFAULT_NETWORK]
fig_h0_omega_m_corner = plot_corner_for_posterior_grid(
    tuple(H0_OMEGA_M_GRIDS.values()),
    H0_OMEGA_M_LOGPOST,
    labels=[PARAMETER_LABELS["H0"], PARAMETER_LABELS["Omega_m"]],
    truths=[FIDUCIALS["H0"], FIDUCIALS["Omega_m"]],
    smooth=1.0,
)
fig_h0_omega_m_corner

# %% [markdown]
# ## Saving the grids and figures

# %%
if SAVE_OUTPUTS:
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        GRID_DIR / "inference_grid_expansion_history.npz",
        **{f"h0_grid_{name}": np.asarray(grid) for name, grid in H0_GRIDS.items()},
        **{
            f"h0_logpost_{name}": np.asarray(logpost)
            for name, logpost in H0_LOGPOSTERIORS.items()
        },
        **{
            f"h0_grid_2d_{name}": np.asarray(grid) for name, grid in H0_GRIDS_2D.items()
        },
        **{
            f"h0_omega_m_logpost_{name}": np.asarray(logpost)
            for name, logpost in H0_OMEGA_M_LOGPOSTERIORS.items()
        },
        omega_m_grid=np.asarray(OMEGA_M_GRID),
        h0_omega_m_logpost=np.asarray(H0_OMEGA_M_LOGPOST),
    )
    (GRID_DIR / "inference_grid_expansion_history.json").write_text(
        json.dumps(
            {
                "fiducials": FIDUCIALS,
                "coverage_sigmas": COVERAGE_SIGMAS,
                "npoints_1d": NPOINTS_1D,
                "npoints_2d": NPOINTS_2D,
                "default_network": DEFAULT_NETWORK,
                "networks": [network.name for network in NETWORKS],
                "h0_windows": H0_WINDOWS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    CONSTRAINT_TABLE.to_csv(FIGURE_DIR / "H0-by-detector-grid.csv", index=False)
    fig_h0_by_detector.savefig(
        FIGURE_DIR / "H0-by-detector-grid.pdf", bbox_inches="tight"
    )
    fig_h0_omega_m_by_detector.savefig(
        FIGURE_DIR / "H0-Omega_m-by-detector-grid.pdf", bbox_inches="tight"
    )
    fig_h0_omega_m_corner.savefig(
        FIGURE_DIR / "H0-Omega_m-corner-grid.pdf", bbox_inches="tight"
    )
    print("saved expansion-history grids to", GRID_DIR, "and figures to", FIGURE_DIR)
