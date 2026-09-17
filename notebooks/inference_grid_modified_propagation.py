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
# # Modified-propagation parameters from grid-evaluated posteriors
#
# This notebook evaluates the default detector network on a two-dimensional
# $(\Xi_0, n)$ grid and draws the corresponding corner plot. The $\Xi_0$
# window is Fisher-sized from the network SNR; $n$ spans its Uniform prior.
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
import numpy as np
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection
from numpyro.distributions import Distribution, Uniform

from astrogwb.metadata import PopulationMetadata, WaveformMetadata
from astrogwb.paper.catalogs import load_run_catalog
from astrogwb.paper.config import fiducials, networks, priors
from astrogwb.paper.config.mcmc import AnalysisGrid
from astrogwb.paper.inference import prepare_inference_inputs
from astrogwb.paper.paths import root_dir
from astrogwb.paper.plotting import (
    CATEGORY,
    DETECTOR_NETWORKS,
    Network,
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
NPOINTS_2D: int = 96
CHUNK_SIZE: int = 64
DEBUG: bool = False
if DEBUG:
    NPOINTS_2D = 17
SAVE_OUTPUTS: bool = True
GRID_DIR = ROOT_DIR / "grids"
FIGURE_DIR = ROOT_DIR / "figures"

# %% [markdown]
# ## Catalog and default network

# %%
network = next(
    Network(name, label, NETWORK_CONFIG[name])
    for name, label in DETECTOR_NETWORKS
    if name == DEFAULT_NETWORK
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
    INJECTION_CATALOG_PATH, (network,), FIDUCIALS, grid=ANALYSIS_GRID
)
snr = float(SNR_TABLE.iloc[0]["snr"])
snr

# %% [markdown]
# ## Grid and evaluator


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
    raise NotImplementedError(f"prior_window unsupported for {type(prior).__name__}")


def uniform_grid(low: float, high: float, npoints: int) -> jax.Array:
    return jnp.linspace(low, high, npoints)


XI0_WINDOW = fisher_window(
    FIDUCIALS["xi_0"],
    FIDUCIALS["xi_0"] / snr,
    sigmas=COVERAGE_SIGMAS,
    support=(float(PRIORS["xi_0"].low), float(PRIORS["xi_0"].high)),
)
XI0_GRID_2D = uniform_grid(*XI0_WINDOW, NPOINTS_2D)
XI_N_GRID = uniform_grid(
    *prior_window(PRIORS["xi_n"], sigmas=COVERAGE_SIGMAS), NPOINTS_2D
)

inputs = prepare_inference_inputs(
    injection_catalog,
    proposal_catalog,
    grid=ANALYSIS_GRID,
    detectors=network.detectors,
    target=TARGET_MODEL,
    density_sites=[],
)
model = partial(
    gwb_spectral_density_model,
    spectral_density_fn=inputs.spectral_density_fn,
    priors=PRIORS,
)
log_density_fn = LogDensityFn(model, chunk_size=CHUNK_SIZE)
model_kwargs = inputs.model_kwargs()


def evaluate_joint(
    grids: dict[str, jax.Array],
) -> jax.Array:
    fixed = {name: FIDUCIALS[name] for name in PRIORS if name not in grids}
    return jax.block_until_ready(log_density_fn(grids, fixed=fixed, **model_kwargs))


XI0_N_GRIDS = {"xi_0": XI0_GRID_2D, "xi_n": XI_N_GRID}
start = time.perf_counter()
XI0_N_LOGPOST = evaluate_joint(XI0_N_GRIDS)
print(f"{time.perf_counter() - start:.2f}s for {NPOINTS_2D}x{NPOINTS_2D} grid points")

# %% [markdown]
# ## Corner plot

# %%
fig_xi0_n_corner = plot_corner_for_posterior_grid(
    tuple(XI0_N_GRIDS.values()),
    XI0_N_LOGPOST,
    labels=[PARAMETER_LABELS["xi_0"], PARAMETER_LABELS["xi_n"]],
    truths=[FIDUCIALS["xi_0"], FIDUCIALS["xi_n"]],
    smooth=1.0,
    color=CATEGORY["modified_propagation"],
)
fig_xi0_n_corner

# %% [markdown]
# ## Saving the grid and figure

# %%
if SAVE_OUTPUTS:
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        GRID_DIR / "inference_grid_modified_propagation.npz",
        xi0_grid_2d=np.asarray(XI0_GRID_2D),
        xi_n_grid=np.asarray(XI_N_GRID),
        xi0_n_logpost=np.asarray(XI0_N_LOGPOST),
    )
    (GRID_DIR / "inference_grid_modified_propagation.json").write_text(
        json.dumps(
            {
                "fiducials": FIDUCIALS,
                "coverage_sigmas": COVERAGE_SIGMAS,
                "npoints_2d": NPOINTS_2D,
                "default_network": DEFAULT_NETWORK,
                "snr": snr,
                "xi0_window": XI0_WINDOW,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    fig_xi0_n_corner.savefig(FIGURE_DIR / "Xi0-n-corner-grid.pdf", bbox_inches="tight")
    print("saved modified-propagation grid to", GRID_DIR, "and figure to", FIGURE_DIR)
