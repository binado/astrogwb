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
# # Population parameters from grid-evaluated posteriors
#
# This notebook evaluates the population-shape parameter $z_{\mathrm{peak}}$
# across detector networks, including the corresponding detector overlays
# and joint $(H_0, z_{\mathrm{peak}})$ corners.
#
# The notebook resolves the checkout root with `astrogwb.paper.paths.root_dir`,
# so it runs from the repository root or from this directory. The injection is
# the shared observed catalog `md-imrphenom-s41-n32768`; the proposal is the
# ε = 0.1 uniform-mixture catalog `md-uniform-imrphenom-s61-n16384-eps1e-1`,
# the same pairing `astrophysical-parameters` uses when sampling $z_{\mathrm{peak}}$.

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
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection
from numpyro.distributions import Distribution, Normal, Uniform

from astrogwb.metadata import PopulationMetadata, WaveformMetadata
from astrogwb.paper.catalogs import load_run_catalog
from astrogwb.paper.config import fiducials, networks, priors
from astrogwb.paper.config.runs import CATALOGS_ROOT
from astrogwb.paper.inference import prepare_inference_inputs
from astrogwb.paper.paths import root_dir
from astrogwb.paper.plotting import (
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

# %% [markdown]
# ## Notebook configuration

# %%
#: Jupyter's kernel cwd is this notebook's own directory, so anchor the
#: notebook's own file paths on the checkout root -- the same root the
#: `astrogwb.paper` accessors default to when they are given no `root=`.
ROOT_DIR = root_dir()
DEBUG: bool = False
INJECTION_CATALOG = "md-imrphenom-s41-n32768"
PROPOSAL_CATALOG = "md-uniform-imrphenom-s61-n16384-eps1e-1"
INJECTION_CATALOG_PATH = ROOT_DIR / CATALOGS_ROOT / f"{INJECTION_CATALOG}.h5"
PROPOSAL_CATALOG_PATH = ROOT_DIR / CATALOGS_ROOT / f"{PROPOSAL_CATALOG}.h5"
observation_time = 1.0
minimum_frequency = 2.0
maximum_frequency = 2048.0
minimum_redshift = 0.3
maximum_redshift = 20.0
n_grid = 256
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
    minimum_redshift=minimum_redshift,
    maximum_redshift=maximum_redshift,
    n_grid=n_grid,
)
print(
    "proposal:",
    PROPOSAL_POPULATION_METADATA.model_name,
    PROPOSAL_POPULATION_METADATA.model_kwargs,
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
    INJECTION_CATALOG_PATH,
    NETWORKS,
    FIDUCIALS,
    observation_time=observation_time,
    minimum_redshift=minimum_redshift,
    maximum_redshift=maximum_redshift,
    minimum_frequency=minimum_frequency,
    maximum_frequency=maximum_frequency,
)
_snr_by_network = SNR_TABLE.set_index("network")["snr"]

# %% [markdown]
# ## Grids and evaluators


# %%
def fisher_window(
    center: float, sigma: float, *, sigmas: float, support: tuple[float, float]
) -> tuple[float, float]:
    return (
        max(center - sigmas * sigma, support[0]),
        min(center + sigmas * sigma, support[1]),
    )


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


H0_WINDOWS: dict[str, tuple[float, float]] = {
    network.name: fisher_window(
        FIDUCIALS["H0"],
        FIDUCIALS["H0"] / float(_snr_by_network[network.name]),
        sigmas=COVERAGE_SIGMAS,
        support=(float(PRIORS["H0"].low), float(PRIORS["H0"].high)),
    )
    for network in NETWORKS
}
H0_GRIDS_2D = {
    name: uniform_grid(low, high, NPOINTS_2D)
    for name, (low, high) in H0_WINDOWS.items()
}
Z_PEAK_GRID = uniform_grid(
    *prior_window(PRIORS["z_peak"], sigmas=COVERAGE_SIGMAS), NPOINTS_2D
)

# %% [markdown]
# ## Grid evaluations


# %%
def build_log_density(network: Network) -> tuple[LogDensityFn, dict[str, jax.Array]]:
    inputs = prepare_inference_inputs(
        injection_catalog,
        proposal_catalog,
        observation_time=observation_time,
        minimum_redshift=minimum_redshift,
        maximum_redshift=maximum_redshift,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        detectors=network.detectors,
        target=TARGET_MODEL,
        density_sites=["redshift"],
    )
    model = partial(
        gwb_spectral_density_model,
        spectral_density_fn=inputs.spectral_density_fn,
        priors=PRIORS,
    )
    return LogDensityFn(model, chunk_size=CHUNK_SIZE), inputs.model_kwargs()


def evaluate_joint(
    log_density_fn: LogDensityFn,
    grids: dict[str, jax.Array],
    *,
    model_kwargs: dict[str, jax.Array],
) -> jax.Array:
    fixed = {name: FIDUCIALS[name] for name in PRIORS if name not in grids}
    return jax.block_until_ready(log_density_fn(grids, fixed=fixed, **model_kwargs))


LOG_DENSITY_FNS: dict[str, LogDensityFn] = {}
MODEL_KWARGS: dict[str, dict[str, jax.Array]] = {}
H0_Z_PEAK_LOGPOSTERIORS: dict[str, jax.Array] = {}
for network in NETWORKS:
    log_density_fn, model_kwargs = build_log_density(network)
    LOG_DENSITY_FNS[network.name] = log_density_fn
    MODEL_KWARGS[network.name] = model_kwargs
    start = time.perf_counter()
    H0_Z_PEAK_LOGPOSTERIORS[network.name] = evaluate_joint(
        log_density_fn,
        {"H0": H0_GRIDS_2D[network.name], "z_peak": Z_PEAK_GRID},
        model_kwargs=model_kwargs,
    )
    print(
        f"{network.label}: {time.perf_counter() - start:.2f}s for "
        f"{NPOINTS_2D}x{NPOINTS_2D} grid points"
    )

# %% [markdown]
# ## Marginals and detector overlays


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


H0_Z_PEAK_Z_PEAK_MARGINAL = {
    name: marginal_along(logpost, H0_GRIDS_2D[name], axis=0)
    for name, logpost in H0_Z_PEAK_LOGPOSTERIORS.items()
}
fig_z_peak_by_detector = plot_parameter_by_detector(
    {network.name: Z_PEAK_GRID for network in NETWORKS},
    H0_Z_PEAK_Z_PEAK_MARGINAL,
    param="z_peak",
)
fig_z_peak_by_detector

# %% [markdown]
# ## Joint corners


# %%
def plot_h0_z_peak_corner(network_name: str) -> plt.Figure:
    grids = {"H0": H0_GRIDS_2D[network_name], "z_peak": Z_PEAK_GRID}
    return plot_corner_for_posterior_grid(
        tuple(grids.values()),
        H0_Z_PEAK_LOGPOSTERIORS[network_name],
        labels=[PARAMETER_LABELS["H0"], PARAMETER_LABELS["z_peak"]],
        truths=[FIDUCIALS["H0"], FIDUCIALS["z_peak"]],
        smooth=1.0,
    )


fig_h0_z_peak_corner_et_2l_ce = plot_h0_z_peak_corner(DEFAULT_NETWORK)

# %%
fig_h0_z_peak_corner_et_2l_ce

# %%
fig_h0_z_peak_corner_et_triangular_ce = plot_h0_z_peak_corner(
    "ET-triangular-CE-Hanford"
)

# %%
fig_h0_z_peak_corner_et_triangular_ce

# %% [markdown]
# ## Saving the grids and figures

# %%
if SAVE_OUTPUTS:
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        GRID_DIR / "inference_grid_population.npz",
        **{
            f"h0_grid_2d_{name}": np.asarray(grid) for name, grid in H0_GRIDS_2D.items()
        },
        **{
            f"h0_z_peak_logpost_{name}": np.asarray(logpost)
            for name, logpost in H0_Z_PEAK_LOGPOSTERIORS.items()
        },
        z_peak_grid=np.asarray(Z_PEAK_GRID),
    )
    (GRID_DIR / "inference_grid_population.json").write_text(
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
    fig_z_peak_by_detector.savefig(
        FIGURE_DIR / "z_peak-by-detector-grid.pdf", bbox_inches="tight"
    )
    fig_h0_z_peak_corner_et_2l_ce.savefig(
        FIGURE_DIR / "H0-z_peak-corner-ET-2L-aligned-CE-Hanford-grid.pdf",
        bbox_inches="tight",
    )
    fig_h0_z_peak_corner_et_triangular_ce.savefig(
        FIGURE_DIR / "H0-z_peak-corner-ET-triangular-CE-Hanford-grid.pdf",
        bbox_inches="tight",
    )
    print("saved population grids to", GRID_DIR, "and figures to", FIGURE_DIR)
