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
# # Log-posterior grid for the astrophysical GWB
#
# This notebook shares the exact model setup of `mcmc.py` (importance-weighted SGWB
# inference over a fixed proposal catalog) but, instead of running NUTS, it compiles
# the NumPyro model and evaluates the **(unnormalized) log-posterior** on a grid over
# the prior ranges. We support one or two sampled parameters:
#
# - `compute_logposterior_1d` sweeps a single parameter and produces a line plot;
# - `compute_logposterior_2d` sweeps two parameters and produces a contour plot.
#
# We evaluate the model's log joint density directly at the physical parameter values
# via `numpyro.infer.util.log_density`, which gives the correct unnormalized
# log-posterior (`log prior + log likelihood`) and is robust at the prior bounds.
#
# To run the notebook end-to-end you must point `CATALOG_PATH` at an `.npz`
# polarization-power catalog (same schema as `mcmc.py`).

# %% [markdown]
# ## Imports and JAX configuration

# %%
import json
import multiprocessing
import time
from datetime import datetime
from functools import partial
from pathlib import Path

# Setting JAX to use all available CPU cores for parallelization
num_cpus = multiprocessing.cpu_count()
import numpyro

numpyro.set_host_device_count(num_cpus)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist

# gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes. Restore
# matplotlib axes so plotting behaves as expected after importing detector utilities.
from matplotlib.axes import Axes as MplAxes
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.projections import register_projection
from numpyro.infer.util import log_density
from scipy.ndimage import gaussian_filter

from astrogwb.detector import effective_psd, load_sensitivity_map
from astrogwb.gwb import (
    frequency_mask as make_frequency_mask,
)
from astrogwb.gwb import (
    omega_gw_from_spectral_density,
    spectral_density,
)
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_proposal_logpdf,
    make_merger_rate_and_log_weights_fn,
)
from astrogwb.sampling.numpyro_model import numpyro_model
from astrogwb.waveform import load_polarization_power_catalog

register_projection(MplAxes)

jax.config.update("jax_enable_x64", True)
# %config InlineBackend.figure_format = 'retina'


# %% [markdown]
# ## Pipeline configuration

# %%
DEBUG = False  # small smoke settings for first runs; set False for the production run

# --- Catalog input (placeholder — see schema markdown in mcmc.py) -----------
# No working polarization-power catalog exists yet; set this once one is produced.


def get_root_dir() -> Path:
    return Path.cwd().parent


ROOT_DIR = get_root_dir()
CATALOG_PATH = ROOT_DIR / "out/bns_polarization_power_catalog.npz"

# Detector settings
detnames = ("S1", "R1", "C1")  # resolve via bundled geometry.toml / sensitivity.toml
observation_time = 1.0  # [yr]; cancels in S_h, kept for the likelihood scale

# Grid settings
seed = 42
npoints = 256  # grid points per sampled parameter
chunk_size = 64  # grid points evaluated per vectorized batch (bounds peak memory)

if DEBUG:
    npoints = 21


# Redshift grid for the cosmology integrals (and MD normalization)
z_min = 0.0
z_max = 20.0
n_grid = 256  # grid points for cosmology integrals / MD normalization

# Frequency band for the analysis
f_min = 2
f_max = 4096

# Fiducial parameters
fiducials = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "xi_0": 1.0,
    "xi_n": 1.91,
    "gamma": 2.7,
    "kappa": 3.0,
    "z_peak": 2.0,
    "local_merger_rate": 161.0,
}

# --- Hyperprior bounds (order: cosmology, then population) -------------------
hyperprior_dists = {
    "H0": dist.Uniform(20.0, 140.0),
    "Omega_m": dist.Uniform(0.05, 0.95),
    "xi_0": dist.Uniform(0.5, 5.0),
    "xi_n": dist.Uniform(0.3, 3.0),
    "gamma": dist.Uniform(0.5, 10.0),
    "kappa": dist.Uniform(0.05, 10.0),
    "z_peak": dist.Uniform(0.05, 10.0),
}

# Optional custom per-parameter grid ranges as {name: (low, high)}. When a sampled
# parameter appears here, the grid is built over this range instead of the full
# prior (concentrating `npoints` for better resolution) and the plot axes are
# limited to it. Parameters without an entry fall back to their prior range and
# auto-scaled axes. Example: grid_ranges = {"H0": (55.0, 85.0)}.
grid_ranges: dict[str, tuple[float, float]] = {"xi_0": (0.9, 1.1), "xi_n": (0.3, 3.0)}

sampled_params = ("xi_0", "xi_n")

assert 1 <= len(sampled_params) <= 2, (
    "the log-posterior grid supports only 1 or 2 sampled parameters"
)

priors = {k: hyperprior_dists[k] for k in sampled_params}
constants = {k: v for k, v in fiducials.items() if k not in sampled_params}

# %% [markdown]
# ## Loading the waveform catalog
#
# See `mcmc.py` for the full catalog schema. The `.npz` file stores the FFT
# frequency grid, the per-source polarization power `(nfreq, nsamples)`, and the
# per-source parameter samples.

# %%
catalog = load_polarization_power_catalog(CATALOG_PATH)

frequencies = jnp.asarray(catalog.frequencies)
polarization_power = jnp.asarray(catalog.polarization_power)  # (nfreq, nsamples)
samples = {name: jnp.asarray(v) for name, v in catalog.samples.items()}

assert "redshift" in samples, "catalog samples must include 'redshift' for the weights"
assert "luminosity_distance" in samples, (
    "catalog samples must include 'luminosity_distance' for the weights"
)

n_freq, n_samples = polarization_power.shape
print(f"loaded catalog: n_frequency_bins={n_freq} n_proposal_samples={n_samples}")

# %% [markdown]
# ## Effective PSD and analysis band
#
# `load_sensitivity_map` resolves the str-named detectors (e.g. `S1`, `R1`, `C1`) via the bundled
# `geometry.toml` / `sensitivity.toml`. We then calculate the effective power spectral density
#
# $$
# S_{\mathrm{eff}} = \left(\sum_{a,b} \frac{\Gamma^2_{ab}(f)}{S_{n,a} S_{n,b}}  \right)^{-1/2}
# $$

# %%
sensitivities = load_sensitivity_map(detnames)
effective_psd_arr = jnp.asarray(
    effective_psd(frequencies, list(detnames), sensitivities)
)
mask = make_frequency_mask(frequencies, fmin=f_min, fmax=f_max)
print("band bins:", int(jnp.sum(mask)), "of", frequencies.shape[0])


# %%
def plot_effective_psd(
    frequencies: jax.Array,
    effective_psd: jax.Array,
    mask: jax.Array,
    *,
    color: str = "black",
):
    fig, ax = plt.subplots()
    ax.loglog(frequencies, effective_psd, color=color, ls="--")
    ax.loglog(frequencies[mask], effective_psd[mask], color=color)
    ax.set(
        xlabel=r"$f$ [Hz]", ylabel=r"$S_{\text{eff}}(f)$ [1/Hz]", title="Effective PSD"
    )
    return fig


plot_effective_psd(frequencies, effective_psd_arr, mask)

# %% [markdown]
# ## Modelling the astrophysical SGWB
#
# The importance-weighted spectral-density model is identical to `mcmc.py`. The
# proposal log-density `log p_proposal(z)` depends only on the fixed fiducial point,
# so we evaluate it once here and reuse it inside the weight callback.

# %%
z_samples = jnp.asarray(samples["redshift"])
z_grid = jnp.linspace(z_min, z_max, n_grid)

log_p_proposal = compute_proposal_logpdf(
    z_samples, z_grid=z_grid, fiducials=fiducials
)


# %%
merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
    z_grid=z_grid,
    proposal_log_pdf=log_p_proposal,
    fiducial_xi_0=fiducials["xi_0"],
    fiducial_xi_n=fiducials["xi_n"],
)


# %% [markdown]
# ## Visualizing $\Omega_{\mathrm{GW}}(f)$
#
# We build the fiducial observed spectral density used as the likelihood target,
# and plot $\Omega_{\mathrm{GW}}(f, \Lambda_0)$.


# %%
def plot_omegagw(
    spectral_density: jax.Array,
    frequencies: jax.Array,
    mask: jax.Array,
    *,
    color: str = "black",
    ymin: float = 1e-15,
):
    omega_gw = omega_gw_from_spectral_density(spectral_density, frequencies)
    pos = omega_gw > 0.0
    fig, ax = plt.subplots()
    ax.loglog(
        np.asarray(frequencies[mask & pos]),
        np.asarray(omega_gw[mask & pos]),
        color=color,
    )
    ax.set_xlabel(r"$f\ \mathrm{(Hz)}$")
    ax.set_ylabel(r"$\Omega_{\mathrm{GW}}(f)$")
    ax.set_ylim(ymin, None)
    return fig


ones_weights = jnp.ones((n_samples,))
rate0, _ = merger_rate_and_log_weights_fn(
    fiducials,
    samples,
)
observed_spectral_density = spectral_density(
    polarization_power, ones_weights, rate0, average_mode="analytic_inclination"
)
plot_omegagw(observed_spectral_density, frequencies, mask, color="black", ymin=1e-15)

# %% [markdown]
# ## Building the model
#
# We assemble the same `numpyro_model` used by the NUTS run. Rather than sampling
# it, we evaluate its log joint density on a grid below.

# %%
model = partial(
    numpyro_model,
    observation_time=observation_time,
    average_mode="analytic_inclination",
    merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
    priors=priors,
    constants=constants,
    frequency_mask=mask,
)

model_kwargs = {
    "frequencies": frequencies,
    "polarization_power": polarization_power,
    "samples": samples,
    "observed_spectral_density": observed_spectral_density,
    "effective_psd": effective_psd_arr,
}

# %% [markdown]
# ## Building the parameter grid
#
# For each sampled parameter we build a grid. `low`/`high` (e.g. from
# `grid_ranges`) override the corresponding bound and concentrate `npoints` there
# for better resolution; any bound left as `None` is derived from the prior:
#
# - `Uniform(low, high)` → `linspace(low, high, npoints)`;
# - `Normal(loc, scale)` → `linspace(loc - 5 scale, loc + 5 scale, npoints)`.


# %%
def make_param_grid(
    distribution: dist.Distribution,
    npoints: int,
    low: float | None = None,
    high: float | None = None,
) -> jax.Array:
    if low is None or high is None:
        if isinstance(distribution, dist.Uniform):
            prior_low, prior_high = distribution.low, distribution.high
        elif isinstance(distribution, dist.Normal):
            prior_low = distribution.loc - 5.0 * distribution.scale
            prior_high = distribution.loc + 5.0 * distribution.scale
        else:
            raise NotImplementedError(
                f"grid unsupported for distribution {type(distribution).__name__}"
            )
        low = prior_low if low is None else low
        high = prior_high if high is None else high
    return jnp.linspace(low, high, npoints)


# %% [markdown]
# ## Evaluating the log-posterior
#
# We evaluate the model's log joint density directly at the physical parameter
# values via `log_density`, which returns the unnormalized log-posterior
# `log p(\Lambda) + log L(\Lambda)`.
#
# Each evaluation builds catalog-sized importance weights and a full-frequency
# spectrum, so a single `vmap` over the whole grid would force XLA to materialize
# those arrays for every grid point at once (`npoints**2 x N`), which OOMs on
# realistic catalogs. We instead use `jax.lax.map` with `batch_size=chunk_size`:
# it vectorizes within each chunk and scans across chunks, bounding peak memory
# to `chunk_size x (N + F)`.
#
# We build the jitted evaluator **once** at module scope. `jax.jit` caches by
# `(function object, input shapes)`, so re-running the evaluation/plotting cells
# below reuses the compiled program; only changing `npoints` (which changes the
# input shape) triggers a recompile. The evaluator takes a `(M, D)` array of grid
# points, where `D = len(sampled_params)`, so the same function serves the 1D and
# 2D cases.


# %%
sampled_param_names = tuple(sampled_params)


def _log_posterior(param_values: dict[str, jax.Array]) -> jax.Array:
    log_joint, _ = log_density(model, (), model_kwargs, param_values)
    return log_joint


def _log_posterior_from_point(point: jax.Array) -> jax.Array:
    param_values = {name: point[i] for i, name in enumerate(sampled_param_names)}
    return _log_posterior(param_values)


eval_grid_points = jax.jit(
    lambda points: jax.lax.map(_log_posterior_from_point, points, batch_size=chunk_size)
)


def compute_logposterior_1d():
    (name,) = sampled_param_names
    low, high = grid_ranges.get(name, (None, None))
    grid = make_param_grid(priors[name], npoints, low, high)
    logpost = eval_grid_points(grid[:, None])
    return name, grid, logpost


def compute_logposterior_2d():
    name0, name1 = sampled_param_names
    low0, high0 = grid_ranges.get(name0, (None, None))
    low1, high1 = grid_ranges.get(name1, (None, None))
    grid0 = make_param_grid(priors[name0], npoints, low0, high0)
    grid1 = make_param_grid(priors[name1], npoints, low1, high1)
    mesh0, mesh1 = jnp.meshgrid(grid0, grid1, indexing="ij")
    points = jnp.stack([mesh0.ravel(), mesh1.ravel()], axis=-1)
    logpost = eval_grid_points(points).reshape(mesh0.shape)
    return (name0, grid0), (name1, grid1), logpost


# %% [markdown]
# ## Plotting the posterior
#
# We exponentiate the (unnormalized) log-posterior to recover the posterior up to
# a constant. To avoid overflow we use the standard log-sum-exp stabilization:
# subtract the maximum log-posterior before exponentiating. This is safe because
# the overall normalization is irrelevant for plotting the shape.
#
# For the 2D case we also compute the **marginal** posterior of each parameter by
# integrating the joint posterior over the other axis (trapezoidal rule on the
# grid). The marginals are then shown above and to the right of the joint panel in
# a corner-plot style layout.
#
# The joint panel reuses the styling of `corner.hist2d`: monochrome filled contours
# whose levels are **credible regions** (0.5/1/1.5/2-sigma, i.e. the density
# thresholds enclosing the corresponding fraction of the posterior mass) with
# increasing opacity toward the peak. We skip corner's `np.histogram2d` step because
# our grid already holds the posterior density; on a uniform grid the equal cell
# areas make the density value proportional to the enclosed mass, so corner's level
# computation carries over unchanged. Optional Gaussian smoothing uses
# `scipy.ndimage` (kept off the JAX device, since the device holds the large
# catalog/model and a round-trip here can stall plotting for tens of seconds).


# %%
def safe_exponentialize(logpost: np.ndarray) -> np.ndarray:
    """Convert a log-posterior array to an (unnormalized) posterior stably.

    Subtracts the max to keep the largest exponent at 0, so `exp` never overflows.
    """
    logpost = np.asarray(logpost, dtype=np.float64)
    return np.exp(logpost - np.max(logpost))


def compute_marginal_distributions(
    logpdf: jax.Array, x: jax.Array, y: jax.Array
) -> tuple[np.ndarray, np.ndarray]:
    """Marginalize a 2D log-posterior over each axis.

    `logpdf[i, j]` is the (unnormalized) log-posterior at `(x[i], y[j])`.
    Returns `(marginal_x, marginal_y)`, each a normalized 1D posterior density
    evaluated on its respective grid, obtained by integrating out the other
    variable via trapezoidal quadrature on the supplied grid.
    """
    posterior = safe_exponentialize(logpdf)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # Integrate out y (axis=1) -> function of x; integrate out x (axis=0) -> of y.
    marginal_x = np.trapezoid(posterior, x=y, axis=1)
    marginal_y = np.trapezoid(posterior, x=x, axis=0)
    # Normalize each to a proper density over its own grid.
    marginal_x = marginal_x / np.trapezoid(marginal_x, x=x)
    marginal_y = marginal_y / np.trapezoid(marginal_y, x=y)
    return marginal_x, marginal_y


def plot_posterior_1d(
    name: str,
    grid: jax.Array,
    logpost: jax.Array,
    *,
    color: str = "black",
    axlim: tuple[float, float] | None = None,
):
    posterior = safe_exponentialize(logpost)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(grid), posterior, color=color)
    ax.axvline(fiducials[name], color="tab:red", ls="--", label="fiducial")
    ax.set(xlabel=name, ylabel="posterior (unnormalized)")
    if axlim is not None:
        ax.set_xlim(axlim)
    ax.legend()
    return fig


def gaussian_filter_2d(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian smoothing of a 2D grid, `sigma` in grid cells.

    Deliberately uses `scipy.ndimage` rather than JAX: this runs during plotting,
    where the JAX devices hold the (large) catalog and compiled model. Routing the
    smoothing through JAX forces a host<->device round-trip that blocks on pending
    device work and, under memory pressure, can stall for tens of seconds. scipy
    keeps the plot path entirely on the CPU (and needs no XLA compilation).
    """
    return gaussian_filter(np.asarray(image, dtype=np.float64), sigma)


def hist2d_density(
    x: jax.Array,
    y: jax.Array,
    density: jax.Array,
    ax: MplAxes,
    *,
    color: str = "k",
    sigmas: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    smooth: float | None = None,
    fill_contours: bool = True,
    plot_contours: bool = True,
):
    """Render a precomputed posterior density with `corner.hist2d` styling.

    Adapted from `corner.hist2d`: we skip its `np.histogram2d` step because
    `density[i, j]` already estimates the (unnormalized) posterior at
    `(x[i], y[j])`. On a uniform grid the cell areas are equal, so the density
    value is proportional to the enclosed probability mass and corner's
    credible-region level computation applies unchanged. `x`/`y` are treated as
    the bin centers of the grid.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    H = np.asarray(density, dtype=np.float64)

    if smooth is not None:
        H = gaussian_filter_2d(H, smooth)

    # Sigma credible-region levels: fraction of total mass they enclose.
    levels = 1.0 - np.exp(-0.5 * np.asarray(sigmas, dtype=np.float64) ** 2)

    # Monochrome colormaps keyed off the axes background, exactly as corner does.
    base_color = ax.get_facecolor()
    base_cmap = LinearSegmentedColormap.from_list(
        "base_cmap", [base_color, base_color], N=2
    )
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for _ in levels] + [rgba_color]
    for i in range(len(levels)):
        contour_cmap[i][-1] *= float(i) / (len(levels) + 1)

    # Map each mass level to a density threshold V (highest density first).
    Hflat = H.flatten()
    Hflat = Hflat[np.argsort(Hflat)[::-1]]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except IndexError:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Pad by two cells so contours close cleanly at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate(
        [
            x[0] + np.array([-2, -1]) * np.diff(x[:2]),
            x,
            x[-1] + np.array([1, 2]) * np.diff(x[-2:]),
        ]
    )
    Y2 = np.concatenate(
        [
            y[0] + np.array([-2, -1]) * np.diff(y[:2]),
            y,
            y[-1] + np.array([1, 2]) * np.diff(y[-2:]),
        ]
    )

    if fill_contours:
        # White base fill hides the densest region before the alpha-graded fills.
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()], cmap=base_cmap, antialiased=False)
        ax.contourf(
            X2,
            Y2,
            H2.T,
            np.concatenate([[0], V, [H.max() * (1 + 1e-4)]]),
            colors=contour_cmap,
            antialiased=False,
        )
    if plot_contours:
        ax.contour(X2, Y2, H2.T, V, colors=color)


def plot_posterior_2d(
    axis0: tuple[str, jax.Array],
    axis1: tuple[str, jax.Array],
    logpost: jax.Array,
    *,
    marginal0: np.ndarray | None = None,
    marginal1: np.ndarray | None = None,
    color: str = "k",
    truth_color: str = "#4682b4",
    sigmas: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    smooth: float | None = 1.0,
    axlim0: tuple[float, float] | None = None,
    axlim1: tuple[float, float] | None = None,
):
    """Corner-style plot: joint 2D posterior with optional 1D marginals.

    The joint panel uses `hist2d_density` (corner styling: monochrome filled
    credible-region contours). If `marginal0`/`marginal1` are provided (posterior
    densities over `axis0`/`axis1` grids), they are drawn in panels above and to
    the right of the joint panel, mimicking a `corner`-style layout. `smooth` is a
    Gaussian sigma in grid cells (set to `None` to disable smoothing). `axlim0`/
    `axlim1` set the limits of the `axis0`/`axis1` axes; the shared marginal panels
    follow via `sharex`/`sharey`.
    """
    (name0, grid0), (name1, grid1) = axis0, axis1
    grid0 = np.asarray(grid0)
    grid1 = np.asarray(grid1)
    posterior = safe_exponentialize(logpost)

    show_marginals = marginal0 is not None and marginal1 is not None
    if show_marginals:
        fig = plt.figure()
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=(4, 1),
            height_ratios=(1, 4),
            wspace=0.05,
            hspace=0.05,
        )
        ax_top = fig.add_subplot(gs[0, 0])
        ax_joint = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_joint)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)
    else:
        fig, ax_joint = plt.subplots()
        ax_top = ax_right = None

    # Joint panel.
    hist2d_density(
        grid0, grid1, posterior, ax_joint, color=color, sigmas=sigmas, smooth=smooth
    )
    ax_joint.axvline(fiducials[name0], color=truth_color)
    ax_joint.axhline(fiducials[name1], color=truth_color)
    ax_joint.plot(fiducials[name0], fiducials[name1], marker="s", color=truth_color)
    ax_joint.set(xlabel=name0, ylabel=name1)
    if axlim0 is not None:
        ax_joint.set_xlim(axlim0)
    if axlim1 is not None:
        ax_joint.set_ylim(axlim1)

    if show_marginals:
        # Top marginal: p(name0) = integrate out name1.
        ax_top.plot(grid0, marginal0, color=color)
        ax_top.axvline(fiducials[name0], color=truth_color)
        ax_top.set_yticks([])

        # Right marginal: p(name1), rotated so its x-axis aligns with the joint y-axis.
        ax_right.plot(marginal1, grid1, color=color)
        ax_right.axhline(fiducials[name1], color=truth_color)
        ax_right.set_xticks([])

    return fig


# %% [markdown]
# ## Running the grid evaluation
#
# We dispatch to the 1D or 2D routine based on the number of sampled parameters.
#
# JAX dispatches the jitted evaluation asynchronously, so `compute_logposterior_*`
# returns almost instantly with an *unrealized* array; the real work only happens
# when the values are first read on the host. We call `jax.block_until_ready` here
# to force (and time) the computation in this cell, rather than having its cost
# surface later inside plotting. This cell is intentionally separate from the
# plotting cell so re-plotting never re-triggers the grid evaluation.

# %%
_t0 = time.perf_counter()
if len(sampled_params) == 1:
    name, grid, logpost = compute_logposterior_1d()
    logpost = jax.block_until_ready(logpost)
    print(
        f"evaluated log-posterior over {name}: {logpost.shape[0]} points "
        f"in {time.perf_counter() - _t0:.1f}s"
    )
else:
    axis0, axis1, logpost = compute_logposterior_2d()
    logpost = jax.block_until_ready(logpost)
    print(
        f"evaluated log-posterior over {axis0[0]} x {axis1[0]}: {logpost.shape} grid "
        f"in {time.perf_counter() - _t0:.1f}s"
    )

# %% [markdown]
# ## Plotting
#
# This cell only reads the already-materialized `logpost`, so it is fast and safe
# to re-run repeatedly (e.g. to tweak styling) without recomputing the grid.

# %%
if len(sampled_params) == 1:
    plot_posterior_1d(name, grid, logpost, axlim=grid_ranges.get(name))
else:
    marginal0, marginal1 = compute_marginal_distributions(logpost, axis0[1], axis1[1])
    plot_posterior_2d(
        axis0,
        axis1,
        logpost,
        marginal0=marginal0,
        marginal1=marginal1,
        axlim0=grid_ranges.get(axis0[0]),
        axlim1=grid_ranges.get(axis1[0]),
    )

# %% [markdown]
# ## Saving the grid
#
# We write the grid axes and log-posterior array to an `.npz` file, alongside a
# small JSON record of the run configuration.

# %%
out_dir = Path("grids")
out_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
params_suffix = "-".join(sorted(sampled_params))
det_suffix = ",".join(detnames)
base = f"logpost-grid-{params_suffix}-det={det_suffix}-seed{seed}-{timestamp}"

if len(sampled_params) == 1:
    np.savez(
        out_dir / f"{base}.npz",
        param_names=np.array([name]),
        grid=np.asarray(grid),
        logpost=np.asarray(logpost),
    )
else:
    np.savez(
        out_dir / f"{base}.npz",
        param_names=np.array([axis0[0], axis1[0]]),
        grid0=np.asarray(axis0[1]),
        grid1=np.asarray(axis1[1]),
        logpost=np.asarray(logpost),
    )

run_config = {
    "catalog_path": str(CATALOG_PATH),
    "detectors": list(detnames),
    "seed": seed,
    "observation_time": observation_time,
    "sampled_params": sorted(sampled_params),
    "fiducials": fiducials,
    "priors": {
        name: {
            "type": type(priors[name]).__name__,
            **{
                key: float(np.asarray(getattr(priors[name], key)))
                for key in ("low", "high", "loc", "scale")
                if hasattr(priors[name], key)
            },
        }
        for name in sorted(sampled_params)
    },
    "grid": {
        "npoints": npoints,
        "ranges": {
            name: list(grid_ranges[name])
            for name in sorted(sampled_params)
            if name in grid_ranges
        },
    },
}
(out_dir / f"{base}.json").write_text(json.dumps(run_config, indent=2))
print("saved:", base)
