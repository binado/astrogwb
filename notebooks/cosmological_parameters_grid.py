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
# # Cosmological parameters from grid-evaluated posteriors
#
# `scripts/mcmc_cosmological_parameters.py` produces the paper's cosmological-parameter
# figures from six NUTS chains plus two amplitude-marginalized runs. This notebook
# reproduces the same figures from **grid-evaluated log densities** instead: for each
# detector network we evaluate the model's constrained log density directly over a
# 1D $H_0$ grid (`astrogwb.sampling.LogDensityFn`), sized from that network's own
# matched-filter SNR via the Fisher prediction $\sigma_{H_0} \approx H_0 / \mathrm{SNR}$,
# with $\Omega_m$ pinned to the fiducial. An identical per-network scan replaces that
# pin with $\Omega_m$'s Gaussian prior, so each network also gets a 2D $(H_0,
# \Omega_m)$ grid whose $H_0$ marginal is the prior-marginalized counterpart of the
# 1D scan. A second per-network 2D scan grids $(H_0, z_{\mathrm{peak}})$ over the
# Uniform $z_{\mathrm{peak}}$ prior; its $z_{\mathrm{peak}}$ marginal is the
# by-detector overlay, and joint corners are drawn for ET-2L-aligned+CE and
# ET-triangular+CE. The default network additionally gets $(H_0, \mathcal{R}_0)$ and
# $(\Xi_0, n)$ joints -- the latter mirroring `scripts/mcmc_modified_propagation.py`'s
# $\Xi_0$--$n$ corner, the former an exact-quadrature cross-check against the
# script's amplitude-marginalized $H_0$-$\mathcal{R}_0$ run.
#
# **Inputs:** `outputs/catalogs/md-imrphenom-s41-n32768.h5`, used as *both* the
# injection and the proposal -- built by
# `snakemake --snakefile Snakefile --cores 1 catalogs`. Using two independently
# seeded catalogs (s41 injection / s42 proposal) leaves real Monte Carlo shot
# noise in the comparison: the spectral-density power sum is dominated by
# whichever handful of samples land nearest the `minimum_redshift` window edge
# (`1/d_L^2` weighting), so two independent draws disagree at the percent
# level even though both are unbiased and the importance weights are exact.
# Reusing the injection catalog as its own proposal -- the same trick the
# waveform-approximant/IMRPhenom systematics-baseline run uses -- removes that
# noise source so the grid posteriors sit on the fiducial line.
#
# **Outputs (when `SAVE_OUTPUTS`):** grid arrays and metadata under `grids/`, figures
# under `figures/`.
#
# This notebook needs the repository root as its working directory and does not read
# any intermediate artifact besides the two catalogs above.

# %% [markdown]
# ## Imports and JAX configuration

# %%
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection

from astrogwb.paper.catalogs import load_run_catalog
from astrogwb.paper.config.constants import (
    DEFAULT_NETWORK,
    FIDUCIALS,
    NETWORK_DETECTORS,
    NETWORK_EXPERIMENT,
    PARAMETER_LABELS,
)
from astrogwb.paper.config.mcmc import AnalysisGrid
from astrogwb.paper.inference import prepare_inference_inputs
from astrogwb.paper.plotting import (
    CATEGORY,
    CORNER_LEVELS,
    DETECTOR_COMPARISON_LEGEND,
    DETECTOR_NETWORKS,
    MERGER_RATE_LEGEND,
    TRUTH,
    Network,
    detector_network_styles,
    plot_corner_for_posterior_grid,
    use_paper_style,
)
from astrogwb.paper.snr import compute_network_snrs
from astrogwb.populations import build_population
from astrogwb.sampling import LogDensityFn, gwb_spectral_density_model

# gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes. Restore
# matplotlib axes so plotting behaves as expected after importing detector utilities.
register_projection(MplAxes)
# %config InlineBackend.figure_format = 'retina'
use_paper_style()

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Notebook configuration

# %%
DEBUG: bool = False  # shrinks NPOINTS_1D / NPOINTS_2D only; one code path in both modes

# Same catalog (same seed) for both roles: see the markdown cell above --
# an independently seeded proposal leaves catalog shot noise in the comparison
# that this notebook's tight, high-SNR grids are sensitive enough to show.
INJECTION_CATALOG_PATH = Path("outputs/catalogs/md-imrphenom-s41-n32768.h5")
PROPOSAL_CATALOG_PATH = INJECTION_CATALOG_PATH

# Frequency band and redshift grid, mirroring config/analysis/base/model.toml.
ANALYSIS_GRID = AnalysisGrid(
    observation_time=1.0,
    f_min=2.0,
    f_max=2048.0,
    minimum_redshift=0.3,
    maximum_redshift=20.0,
    n_grid=256,
)

# Inlined to match config/analysis/base/parameters.toml. Every fiducial carries a
# prior: gwb_spectral_density_model samples every key, and LogDensityFn's `fixed=`
# pins the ones a given sweep is not gridding.
PRIORS: dict[str, dist.Distribution] = {
    "H0": dist.Uniform(20.0, 140.0),
    "Omega_m": dist.Normal(0.3096, 0.006),
    "xi_0": dist.Uniform(0.5, 5.0),
    "xi_n": dist.Uniform(0.3, 3.0),
    "gamma": dist.Uniform(-10.0, 10.0),
    "kappa": dist.Uniform(-10.0, 10.0),
    "z_peak": dist.Uniform(0.0, 2.5),
    "local_merger_rate": dist.Normal(770.0, 7.7),
    "minimum_mass": dist.Uniform(0.5, 1.5),
    "mass_width": dist.Uniform(0.5, 3.0),
}

COVERAGE_SIGMAS: float = 5.0  # grid half-width, in predicted sigma
NPOINTS_1D: int = 256
NPOINTS_2D: int = 96  # per axis; the 2D grids cost NPOINTS_2D**2 evaluations
CHUNK_SIZE: int = 64  # LogDensityFn batch_size; bounds peak memory

if DEBUG:
    NPOINTS_1D, NPOINTS_2D = 33, 17

SAVE_OUTPUTS: bool = True
GRID_DIR = Path("grids")
FIGURE_DIR = Path("figures")

# %% [markdown]
# ## Detector networks and fiducials
#
# `NETWORK_DETECTORS` and `plotting.DETECTOR_NETWORKS` are keyed by the same run
# names, so they compose directly into `Network` objects without a lookup table.

# %%
NETWORKS: tuple[Network, ...] = tuple(
    Network(name, label, NETWORK_DETECTORS[name]) for name, label in DETECTOR_NETWORKS
)

print(f"Detector networks ({NETWORK_EXPERIMENT}):")
pd.DataFrame(
    [
        {"name": n.name, "label": n.label, "detectors": ", ".join(n.detectors)}
        for n in NETWORKS
    ]
)

# %% [markdown]
# ## Loading the catalogs
#
# `prepare_inference_inputs` restricts both catalogs to the analysis window
# (samples and recorded density together), builds the fiducial observation from
# the injection catalog's own population, and prepares the estimator against the
# proposal catalog's own recorded density. Nothing here restates the proposal:
# the file carries it.
#
# `FIDUCIALS` still come from `astrogwb.paper.config.constants` -- they are the
# analysis truth the grids are centred on, not a second copy of the catalog
# record. `tests/paper/test_config_constants.py` guards them against the run
# TOMLs.

# %%
injection_catalog = load_run_catalog(INJECTION_CATALOG_PATH, label="injection")
proposal_catalog = load_run_catalog(PROPOSAL_CATALOG_PATH, label="proposal")

# Bound to the analysis grid once: it is static pytree metadata on the
# estimator. At xi_0 = 1 this reduces to the catalog's cosmological law.
TARGET_MODEL = build_population(
    "bns_md_modified_propagation",
    settings={
        "z_min": ANALYSIS_GRID.minimum_redshift,
        "z_max": ANALYSIS_GRID.maximum_redshift,
        "n_grid": ANALYSIS_GRID.n_grid,
    },
)
print(
    "proposal:",
    proposal_catalog.population_model_name,
    "n_samples=",
    proposal_catalog.num_samples,
)

# %% [markdown]
# ## Matched-filter SNR per network

# %%
SNR_TABLE = compute_network_snrs(
    INJECTION_CATALOG_PATH, NETWORKS, FIDUCIALS, grid=ANALYSIS_GRID
)
SNR_TABLE

# %% [markdown]
# ## The Fisher prediction for $\sigma(H_0)$
#
# A matched-filter analysis with a single overall amplitude has Fisher information
# $\mathcal{I}_{H_0} \approx (\mathrm{SNR}/H_0)^2$, so $\sigma_{H_0} \approx H_0 /
# \mathrm{SNR}$. This is what sizes the per-network $H_0$ grid below; it is not itself
# the constraint the notebook reports.

# %%
SNR_TABLE = SNR_TABLE.assign(
    sigma_h0_fisher=FIDUCIALS["H0"] / SNR_TABLE["snr"],
    rel_sigma_h0_fisher=1.0 / SNR_TABLE["snr"],
)
SNR_TABLE

# %% [markdown]
# ## Sizing the parameter grids
#
# `H0` gets a **per-network** window centred on the fiducial with half-width
# `COVERAGE_SIGMAS * H0 / snr`, clipped to the `H0` prior support so no grid points are
# spent on `-inf` cells. `xi_0` is amplitude-like in the same way
# ($\sigma_{\Xi_0} \approx \Xi_0 / \mathrm{SNR}$), so its 2D window uses that Fisher
# prediction on the default network. `n` has no SNR-scaling analog -- the same
# observation `scripts/mcmc_modified_propagation.py` makes in its constraint table --
# so its window is the Uniform prior support. `Omega_m` and `local_merger_rate` are
# prior-dominated at these priors (their scale is much narrower than what any network's
# SNR could resolve), so their windows come from the prior itself.


# %%
def fisher_window(
    center: float, sigma: float, *, sigmas: float, support: tuple[float, float]
) -> tuple[float, float]:
    """A Fisher-predicted window, centred on `center`, clipped to `support`."""
    low = max(center - sigmas * sigma, support[0])
    high = min(center + sigmas * sigma, support[1])
    return low, high


def prior_window(prior: dist.Distribution, *, sigmas: float) -> tuple[float, float]:
    """Grid window implied by a prior: full range for Uniform, loc +/- sigmas*scale
    for Normal."""
    if isinstance(prior, dist.Uniform):
        return float(prior.low), float(prior.high)
    if isinstance(prior, dist.Normal):
        return (
            float(prior.loc - sigmas * prior.scale),
            float(prior.loc + sigmas * prior.scale),
        )
    raise NotImplementedError(f"prior_window unsupported for {type(prior).__name__}")


def uniform_grid(low: float, high: float, npoints: int) -> jax.Array:
    """A uniform grid; `plot_corner_for_posterior_grid` requires evenly spaced points."""
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

OMEGA_M_GRID = uniform_grid(
    *prior_window(PRIORS["Omega_m"], sigmas=COVERAGE_SIGMAS), NPOINTS_2D
)
Z_PEAK_GRID = uniform_grid(
    *prior_window(PRIORS["z_peak"], sigmas=COVERAGE_SIGMAS), NPOINTS_2D
)
LOCAL_MERGER_RATE_GRID = uniform_grid(
    *prior_window(PRIORS["local_merger_rate"], sigmas=COVERAGE_SIGMAS), NPOINTS_2D
)
XI_N_GRID = uniform_grid(
    *prior_window(PRIORS["xi_n"], sigmas=COVERAGE_SIGMAS), NPOINTS_2D
)
# Per-network H0 windows at the coarser 2D resolution. The 1D scan uses
# NPOINTS_1D; every (H0, Omega_m) and (H0, z_peak) joint uses these, including
# the default network's remaining 2D sweeps.
H0_GRIDS_2D: dict[str, jax.Array] = {
    name: uniform_grid(low, high, NPOINTS_2D)
    for name, (low, high) in H0_WINDOWS.items()
}
H0_GRID_2D = H0_GRIDS_2D[DEFAULT_NETWORK]
XI0_SUPPORT = (float(PRIORS["xi_0"].low), float(PRIORS["xi_0"].high))
XI0_WINDOW = fisher_window(
    FIDUCIALS["xi_0"],
    FIDUCIALS["xi_0"] / float(_snr_by_network[DEFAULT_NETWORK]),
    sigmas=COVERAGE_SIGMAS,
    support=XI0_SUPPORT,
)
XI0_GRID_2D = uniform_grid(*XI0_WINDOW, NPOINTS_2D)

pd.DataFrame(
    [
        {"network": name, "low": low, "high": high}
        for name, (low, high) in H0_WINDOWS.items()
    ]
)

# %% [markdown]
# ## Building one network's log-density evaluator
#
# **One evaluator per network, by necessity.** `prepare_inference_inputs` drops bins
# where the network's effective PSD is infinite, so the masked array lengths differ
# between networks and the estimator holds band-restricted power. That is baked into
# the model closure, not traced, so each network pays one compilation.
#
# **No `handlers.condition` / `handlers.block`.** The model carries every prior;
# `LogDensityFn(...)(grids, fixed=...)` pins the rest. `fixed` is traced, so only a
# change to its *key set* recompiles -- which is exactly what lets one network's
# evaluator serve the 1D `H0` sweep, that network's `(H0, Omega_m)` and
# `(H0, z_peak)` joints, and (for the default network) the remaining 2D sweeps.
# The joint-posterior section reuses the evaluators built here rather than
# rebuilding them.
#
# `prepare_inference_inputs` re-runs `prepare_observation` (the fiducial spectrum
# contraction) once per network. That is redundant work -- the observation does not
# depend on the detectors -- but it is the honest reuse of the production path, and
# it is a one-off cost per network, not per grid point.


# %%
def build_log_density(network: Network) -> tuple[LogDensityFn, dict[str, jax.Array]]:
    inputs = prepare_inference_inputs(
        injection_catalog,
        proposal_catalog,
        grid=ANALYSIS_GRID,
        detectors=network.detectors,
        target_model=TARGET_MODEL,
    )
    model = partial(
        gwb_spectral_density_model,
        spectral_density_fn=inputs.estimator,
        priors=PRIORS,
    )
    return LogDensityFn(model, chunk_size=CHUNK_SIZE), inputs.masked_model_kwargs()


def evaluate_joint(
    log_density_fn: LogDensityFn,
    grids: dict[str, jax.Array],
    *,
    model_kwargs: dict[str, jax.Array],
) -> jax.Array:
    """Evaluate the constrained log density on a Cartesian product of `grids`."""
    fixed = {name: FIDUCIALS[name] for name in PRIORS if name not in grids}
    return jax.block_until_ready(log_density_fn(grids, fixed=fixed, **model_kwargs))


# %% [markdown]
# ## Evaluating the H0 posterior for each network
#
# $\Omega_m$ (and every other non-$H_0$ site) is pinned to its fiducial.

# %%
LOG_DENSITY_FNS: dict[str, LogDensityFn] = {}
MODEL_KWARGS: dict[str, dict[str, jax.Array]] = {}
H0_LOGPOSTERIORS: dict[str, jax.Array] = {}

for network in NETWORKS:
    log_density_fn, model_kwargs = build_log_density(network)
    LOG_DENSITY_FNS[network.name] = log_density_fn
    MODEL_KWARGS[network.name] = model_kwargs

    start = time.perf_counter()
    logpost = evaluate_joint(
        log_density_fn, {"H0": H0_GRIDS[network.name]}, model_kwargs=model_kwargs
    )
    elapsed = time.perf_counter() - start
    H0_LOGPOSTERIORS[network.name] = logpost
    print(f"{network.label}: {elapsed:.2f}s for {NPOINTS_1D} grid points")

# %% [markdown]
# ## Evaluating H0--Omega_m for each network
#
# Identical to the 1D scan -- same networks, same per-network $H_0$ windows -- except
# $\Omega_m$ is no longer pinned to the fiducial. It is gridded over
# `prior_window(PRIORS["Omega_m"])`, the Gaussian $\mathrm{loc} \pm
# \mathtt{COVERAGE\_SIGMAS}\,\sigma$, so the model prior is part of the density
# rather than a delta at the fiducial. Each call reuses the evaluator compiled
# above; only the `grids` / `fixed` key set changes (one extra compilation per
# network).


# %%
H0_OMEGA_M_LOGPOSTERIORS: dict[str, jax.Array] = {}
for network in NETWORKS:
    start = time.perf_counter()
    logpost = evaluate_joint(
        LOG_DENSITY_FNS[network.name],
        {"H0": H0_GRIDS_2D[network.name], "Omega_m": OMEGA_M_GRID},
        model_kwargs=MODEL_KWARGS[network.name],
    )
    elapsed = time.perf_counter() - start
    H0_OMEGA_M_LOGPOSTERIORS[network.name] = logpost
    print(f"{network.label}: {elapsed:.2f}s for {NPOINTS_2D}x{NPOINTS_2D} grid points")

# %% [markdown]
# ## Evaluating H0--$z_{\mathrm{peak}}$ for each network
#
# Same networks and per-network $H_0$ windows as the $\Omega_m$ joint. $z_{\mathrm{peak}}$
# is gridded over its Uniform prior support -- unlike $\Omega_m$ it is not
# prior-dominated, so the $z_{\mathrm{peak}}$ marginal can move with detector
# network. Joint corners below are drawn for ET-2L-aligned+CE and ET-triangular+CE;
# the by-detector overlay uses every network's $z_{\mathrm{peak}}$ marginal.


# %%
H0_Z_PEAK_LOGPOSTERIORS: dict[str, jax.Array] = {}
for network in NETWORKS:
    start = time.perf_counter()
    logpost = evaluate_joint(
        LOG_DENSITY_FNS[network.name],
        {"H0": H0_GRIDS_2D[network.name], "z_peak": Z_PEAK_GRID},
        model_kwargs=MODEL_KWARGS[network.name],
    )
    elapsed = time.perf_counter() - start
    H0_Z_PEAK_LOGPOSTERIORS[network.name] = logpost
    print(f"{network.label}: {elapsed:.2f}s for {NPOINTS_2D}x{NPOINTS_2D} grid points")

# %% [markdown]
# ## H0 posterior by detector network
#
# ≙ `H0-by-detector.pdf`. $\Omega_m$ is pinned to the fiducial. Each network's grid
# is normalized to a proper density with `np.trapezoid` before plotting; the grids
# differ in window per network, so the raw `log_density` values are not directly
# comparable across curves.


# %%
def safe_exponentiate(log_values: jax.Array) -> np.ndarray:
    """Stably exponentiate a log-density array, off the JAX device.

    Subtracts the max before `exp` so the largest exponent is 0 and overflow
    cannot occur; the overall normalization is irrelevant for a density that
    is about to be renormalized by `np.trapezoid` anyway.
    """
    values = np.asarray(log_values, dtype=np.float64)
    return np.exp(values - values.max())


def marginal_along(
    log_density: jax.Array, axis_grid: jax.Array, *, axis: int
) -> np.ndarray:
    """Trapezoidal marginal of a joint log-density along `axis`."""
    return np.trapezoid(
        safe_exponentiate(log_density), np.asarray(axis_grid), axis=axis
    )


def plot_parameter_by_detector(
    grids: dict[str, jax.Array],
    densities: dict[str, np.ndarray],
    *,
    param: str,
) -> plt.Figure:
    """Overlay per-network posterior densities with the detector-network styles."""
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


fig_h0_by_detector = plot_parameter_by_detector(
    H0_GRIDS,
    {name: safe_exponentiate(logpost) for name, logpost in H0_LOGPOSTERIORS.items()},
    param="H0",
)
fig_h0_by_detector

# %% [markdown]
# ## H0 posterior by detector network, Omega_m marginalized
#
# Same overlay as the 1D scan, from the $H_0$ marginal of each network's $(H_0,
# \Omega_m)$ grid (`np.trapezoid` over $\Omega_m$). $\Omega_m$ is prior-dominated at
# these priors, so the curves should closely track the fixed-$\Omega_m$ overlay
# above; a large disagreement would mean the Gaussian prior window is clipping the
# joint or the $H_0$--$\Omega_m$ degeneracy is not negligible.


# %%
H0_OMEGA_M_H0_MARGINAL: dict[str, np.ndarray] = {
    name: marginal_along(logpost, OMEGA_M_GRID, axis=1)
    for name, logpost in H0_OMEGA_M_LOGPOSTERIORS.items()
}
fig_h0_omega_m_by_detector = plot_parameter_by_detector(
    H0_GRIDS_2D, H0_OMEGA_M_H0_MARGINAL, param="H0"
)
fig_h0_omega_m_by_detector

# %% [markdown]
# ## $z_{\mathrm{peak}}$ posterior by detector network
#
# $z_{\mathrm{peak}}$ marginal of each network's $(H_0, z_{\mathrm{peak}})$ grid
# (`np.trapezoid` over $H_0$). Every network shares the same Uniform $z_{\mathrm{peak}}$
# grid, so the overlay is on a common x-axis.


# %%
H0_Z_PEAK_Z_PEAK_MARGINAL: dict[str, np.ndarray] = {
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
# ## SNR versus the measured H0 uncertainty
#
# ≙ `H0-by-detector.csv` / `.tex`. `sigma_h0_grid` is the half-width of the smallest
# highest-density interval enclosing `CORNER_LEVELS[0]` (68.27%) of the 1D
# (fixed-$\Omega_m$) grid's mass -- found by sorting the grid cells by density and
# thresholding at the enclosed-mass level, the same construction
# `plot_corner_for_posterior_grid` relies on for its credible-region contours.
# `sigma_h0_omega_m_grid` is the same HPD on the $H_0$ marginal of the 2D scan.
# `rel_sigma_h0_grid` should track `rel_sigma_h0_snr = 1/SNR` across the six
# networks; large disagreement means `COVERAGE_SIGMAS` is clipping the posterior.


# %%
def hpd_half_width(
    grid: np.ndarray, density: np.ndarray, *, probability: float
) -> float:
    """Half-width of the smallest region enclosing `probability` of the grid's mass."""
    grid = np.asarray(grid)
    density = np.asarray(density, dtype=np.float64)
    weights = density / density.sum()
    order = np.argsort(density)[::-1]
    cumulative = np.cumsum(weights[order])
    n_included = int(np.searchsorted(cumulative, probability)) + 1
    included_grid = grid[order[:n_included]]
    return float(included_grid.max() - included_grid.min()) / 2.0


def hpd_half_width_1d(
    grid: np.ndarray, log_density: np.ndarray, *, probability: float
) -> float:
    """Half-width of the HPD of a 1D log-density grid."""
    return hpd_half_width(grid, safe_exponentiate(log_density), probability=probability)


CONSTRAINT_TABLE = pd.DataFrame(
    [
        {
            "label": network.label,
            "snr": float(_snr_by_network[network.name]),
            "sigma_h0_grid": hpd_half_width_1d(
                H0_GRIDS[network.name],
                H0_LOGPOSTERIORS[network.name],
                probability=CORNER_LEVELS[0],
            ),
            "sigma_h0_omega_m_grid": hpd_half_width(
                H0_GRIDS_2D[network.name],
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
# ## Joint posteriors for the default network
#
# The $(H_0, \Omega_m)$ joint is the default network's slice of the per-network 2D
# scan above. The remaining two 2D sweeps reuse the same evaluator: only the
# `grids` / `fixed` key sets change between calls.

# %%
_default_log_density_fn = LOG_DENSITY_FNS[DEFAULT_NETWORK]
_default_model_kwargs = MODEL_KWARGS[DEFAULT_NETWORK]

H0_OMEGA_M_GRIDS = {"H0": H0_GRID_2D, "Omega_m": OMEGA_M_GRID}
H0_OMEGA_M_LOGPOST = H0_OMEGA_M_LOGPOSTERIORS[DEFAULT_NETWORK]

H0_MERGER_RATE_GRIDS = {"H0": H0_GRID_2D, "local_merger_rate": LOCAL_MERGER_RATE_GRID}
H0_MERGER_RATE_LOGPOST = evaluate_joint(
    _default_log_density_fn, H0_MERGER_RATE_GRIDS, model_kwargs=_default_model_kwargs
)

XI0_N_GRIDS = {"xi_0": XI0_GRID_2D, "xi_n": XI_N_GRID}
XI0_N_LOGPOST = evaluate_joint(
    _default_log_density_fn, XI0_N_GRIDS, model_kwargs=_default_model_kwargs
)

# %% [markdown]
# ## Corner plots
#
# ≙ `H0-Omega_m-corner.pdf`, `H0-merger-rate-corner.pdf`, `Xi0-n-corner.pdf`,
# `H0-z_peak-corner-ET-2L-aligned-CE-Hanford.pdf`,
# `H0-z_peak-corner-ET-triangular-CE-Hanford.pdf`. The grids `dict`s are handed to
# `tuple(...values())` rather than re-listed, so the dict-to-sequence handoff
# cannot silently transpose the axes.
#
# **No ESS panel.** `H0-Omega_m-ess-corner.pdf` and `Xi0-n-ess-corner.pdf` plot
# `importance_relative_ess`, a NumPyro `deterministic` site; `LogDensityFn` returns
# only the scalar log density, so it has no grid analogue -- there is no third panel
# here.

# %%
fig_h0_omega_m_corner = plot_corner_for_posterior_grid(
    tuple(H0_OMEGA_M_GRIDS.values()),
    H0_OMEGA_M_LOGPOST,
    labels=[PARAMETER_LABELS["H0"], PARAMETER_LABELS["Omega_m"]],
    truths=[FIDUCIALS["H0"], FIDUCIALS["Omega_m"]],
    smooth=1.0,
)
fig_h0_omega_m_corner

# %%
fig_h0_merger_rate_corner = plot_corner_for_posterior_grid(
    tuple(H0_MERGER_RATE_GRIDS.values()),
    H0_MERGER_RATE_LOGPOST,
    labels=[PARAMETER_LABELS["H0"], PARAMETER_LABELS["local_merger_rate"]],
    truths=[FIDUCIALS["H0"], FIDUCIALS["local_merger_rate"]],
    smooth=1.0,
)
fig_h0_merger_rate_corner

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


# %%
def plot_h0_z_peak_corner(network_name: str) -> plt.Figure:
    """$(H_0, z_{peak})$ corner for one network's 2D grid."""
    grids = {"H0": H0_GRIDS_2D[network_name], "z_peak": Z_PEAK_GRID}
    return plot_corner_for_posterior_grid(
        tuple(grids.values()),
        H0_Z_PEAK_LOGPOSTERIORS[network_name],
        labels=[PARAMETER_LABELS["H0"], PARAMETER_LABELS["z_peak"]],
        truths=[FIDUCIALS["H0"], FIDUCIALS["z_peak"]],
        smooth=1.0,
    )


fig_h0_z_peak_corner_et_2l_ce = plot_h0_z_peak_corner(DEFAULT_NETWORK)
fig_h0_z_peak_corner_et_2l_ce

# %%
fig_h0_z_peak_corner_et_triangular_ce = plot_h0_z_peak_corner(
    "ET-triangular-CE-Hanford"
)
fig_h0_z_peak_corner_et_triangular_ce

# %% [markdown]
# ## Fixed versus marginalized merger rate
#
# ≙ `H0-merger-rate-priors.pdf`. Overlays the default network's fixed-$\mathcal{R}_0$
# $H_0$ posterior (section 10) against the $H_0$ marginal of the $(H_0,
# \mathcal{R}_0)$ grid (section 13), obtained by `np.trapezoid` over the
# $\mathcal{R}_0$ axis. This is exact quadrature on the grid, where the chain-based
# figure instead compares a fixed-$\mathcal{R}_0$ run against an
# analytically-amplitude-marginalized one. The $H_0$ marginal should be visibly wider
# than the fixed-$\mathcal{R}_0$ posterior.

# %%
_fixed_density = safe_exponentiate(H0_LOGPOSTERIORS[DEFAULT_NETWORK])
_fixed_density /= np.trapezoid(_fixed_density, np.asarray(H0_GRIDS[DEFAULT_NETWORK]))

_h0_marginal = marginal_along(H0_MERGER_RATE_LOGPOST, LOCAL_MERGER_RATE_GRID, axis=1)
_h0_marginal /= np.trapezoid(_h0_marginal, np.asarray(H0_GRID_2D))

fig_h0_merger_rate_priors, ax = plt.subplots()
ax.plot(
    np.asarray(H0_GRIDS[DEFAULT_NETWORK]),
    _fixed_density,
    label=r"$H_0$ (fixed $\mathcal{R}_0$)",
)
ax.plot(
    np.asarray(H0_GRID_2D),
    _h0_marginal,
    label=r"$H_0$ ($\mathcal{R}_0$ marginalized, grid quadrature)",
)
ax.axvline(FIDUCIALS["H0"], **TRUTH)
ax.set(xlabel=PARAMETER_LABELS["H0"], ylabel="Posterior density")
ax.legend(**MERGER_RATE_LEGEND)
fig_h0_merger_rate_priors.tight_layout()
fig_h0_merger_rate_priors

# %% [markdown]
# ## Saving the grids and figures

# %%
if SAVE_OUTPUTS:
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        GRID_DIR / "cosmological_parameters_grid.npz",
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
        **{
            f"h0_z_peak_logpost_{name}": np.asarray(logpost)
            for name, logpost in H0_Z_PEAK_LOGPOSTERIORS.items()
        },
        h0_grid_2d=np.asarray(H0_GRID_2D),
        omega_m_grid=np.asarray(OMEGA_M_GRID),
        z_peak_grid=np.asarray(Z_PEAK_GRID),
        local_merger_rate_grid=np.asarray(LOCAL_MERGER_RATE_GRID),
        h0_omega_m_logpost=np.asarray(H0_OMEGA_M_LOGPOST),
        h0_merger_rate_logpost=np.asarray(H0_MERGER_RATE_LOGPOST),
        xi0_grid_2d=np.asarray(XI0_GRID_2D),
        xi_n_grid=np.asarray(XI_N_GRID),
        xi0_n_logpost=np.asarray(XI0_N_LOGPOST),
    )
    (GRID_DIR / "cosmological_parameters_grid.json").write_text(
        json.dumps(
            {
                "fiducials": FIDUCIALS,
                "coverage_sigmas": COVERAGE_SIGMAS,
                "npoints_1d": NPOINTS_1D,
                "npoints_2d": NPOINTS_2D,
                "default_network": DEFAULT_NETWORK,
                "networks": [network.name for network in NETWORKS],
                "h0_windows": H0_WINDOWS,
                "xi0_window": XI0_WINDOW,
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
    fig_z_peak_by_detector.savefig(
        FIGURE_DIR / "z_peak-by-detector-grid.pdf", bbox_inches="tight"
    )
    fig_h0_omega_m_corner.savefig(
        FIGURE_DIR / "H0-Omega_m-corner-grid.pdf", bbox_inches="tight"
    )
    fig_h0_merger_rate_corner.savefig(
        FIGURE_DIR / "H0-merger-rate-corner-grid.pdf", bbox_inches="tight"
    )
    fig_xi0_n_corner.savefig(FIGURE_DIR / "Xi0-n-corner-grid.pdf", bbox_inches="tight")
    fig_h0_z_peak_corner_et_2l_ce.savefig(
        FIGURE_DIR / "H0-z_peak-corner-ET-2L-aligned-CE-Hanford-grid.pdf",
        bbox_inches="tight",
    )
    fig_h0_z_peak_corner_et_triangular_ce.savefig(
        FIGURE_DIR / "H0-z_peak-corner-ET-triangular-CE-Hanford-grid.pdf",
        bbox_inches="tight",
    )
    fig_h0_merger_rate_priors.savefig(
        FIGURE_DIR / "H0-merger-rate-priors-grid.pdf", bbox_inches="tight"
    )
    print("saved grids to", GRID_DIR, "and figures to", FIGURE_DIR)
