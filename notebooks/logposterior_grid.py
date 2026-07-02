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
from datetime import datetime
from functools import partial
import json
import multiprocessing
from pathlib import Path

# Setting JAX to use all available CPU cores for parallelization
num_cpus = multiprocessing.cpu_count()
import numpyro

numpyro.set_host_device_count(num_cpus)

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.infer.util import log_density

import matplotlib.pyplot as plt

from astrogwb.sampling.numpyro_model import numpyro_model
from astrogwb.gwb import (
    spectral_density,
    omega_gw_from_spectral_density,
    frequency_mask as make_frequency_mask,
)
from astrogwb.detector import load_sensitivity_map, effective_psd
from astrogwb.waveform import load_polarization_power_catalog

# gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes. Restore
# matplotlib axes so plotting behaves as expected after importing detector utilities.
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection

register_projection(MplAxes)

# gwmock-pop: still needed for the proposal redshift PDF precompute.
from gwmock_pop.distributions.madau_dickinson import madau_dickinson_redshift_pdf

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
local_merger_rate = 161.0  # [Gpc^-3 yr^-1], matches COBA simulations
observation_time = 1.0  # [yr]; cancels in S_h, kept for the likelihood scale

# Grid settings
seed = 42
npoints = 101  # grid points per sampled parameter
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

sampled_params = set(("H0",))

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

log_p_proposal = jnp.log(
    madau_dickinson_redshift_pdf(
        z_samples,
        z_max=z_max,
        z_min=z_min,
        gamma=fiducials["gamma"],
        kappa=fiducials["kappa"],
        z_peak=fiducials["z_peak"],
        hubble_constant=fiducials["H0"],
        omega_m=fiducials["Omega_m"],
        n_grid=n_grid,
    )
)


# %%
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    make_merger_rate_and_log_weights_fn,
)

merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
    z_grid=jnp.linspace(z_min, z_max, n_grid),
    proposal_log_pdf=log_p_proposal,
    local_merger_rate=local_merger_rate,
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
# For each sampled parameter we build a grid from its prior:
#
# - `Uniform(low, high)` → `linspace(low, high, npoints)`;
# - `Normal(loc, scale)` → `linspace(loc - 5 scale, loc + 5 scale, npoints)`.


# %%
def make_param_grid(distribution: dist.Distribution, npoints: int) -> jax.Array:
    if isinstance(distribution, dist.Uniform):
        return jnp.linspace(distribution.low, distribution.high, npoints)
    if isinstance(distribution, dist.Normal):
        lo = distribution.loc - 5.0 * distribution.scale
        hi = distribution.loc + 5.0 * distribution.scale
        return jnp.linspace(lo, hi, npoints)
    raise NotImplementedError(
        f"grid unsupported for distribution {type(distribution).__name__}"
    )


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


# %%
def _log_posterior(param_values: dict[str, jax.Array]) -> jax.Array:
    log_joint, _ = log_density(model, (), model_kwargs, param_values)
    return log_joint


def compute_logposterior_1d():
    (name,) = sorted(sampled_params)
    grid = make_param_grid(priors[name], npoints)
    eval_grid = jax.jit(
        lambda values: jax.lax.map(
            lambda v: _log_posterior({name: v}), values, batch_size=chunk_size
        )
    )
    return name, grid, eval_grid(grid)


def compute_logposterior_2d():
    name0, name1 = sorted(sampled_params)
    grid0 = make_param_grid(priors[name0], npoints)
    grid1 = make_param_grid(priors[name1], npoints)
    mesh0, mesh1 = jnp.meshgrid(grid0, grid1, indexing="ij")
    points = jnp.stack([mesh0.ravel(), mesh1.ravel()], axis=-1)
    eval_grid = jax.jit(
        lambda pts: jax.lax.map(
            lambda p: _log_posterior({name0: p[0], name1: p[1]}),
            pts,
            batch_size=chunk_size,
        )
    )
    logpost = eval_grid(points).reshape(mesh0.shape)
    return (name0, grid0), (name1, grid1), logpost


# %% [markdown]
# ## Plotting the log-posterior


# %%
def plot_logposterior_1d(
    name: str, grid: jax.Array, logpost: jax.Array, *, color: str = "black"
):
    fig, ax = plt.subplots()
    ax.plot(np.asarray(grid), np.asarray(logpost), color=color)
    ax.axvline(fiducials[name], color="tab:red", ls="--", label="fiducial")
    ax.set(xlabel=name, ylabel="log posterior (unnormalized)")
    ax.legend()
    return fig


def plot_logposterior_2d(
    axis0: tuple[str, jax.Array],
    axis1: tuple[str, jax.Array],
    logpost: jax.Array,
    *,
    levels: int = 30,
):
    (name0, grid0), (name1, grid1) = axis0, axis1
    mesh0, mesh1 = np.meshgrid(np.asarray(grid0), np.asarray(grid1), indexing="ij")
    fig, ax = plt.subplots()
    cf = ax.contourf(mesh0, mesh1, np.asarray(logpost), levels=levels)
    ax.contour(
        mesh0, mesh1, np.asarray(logpost), levels=levels, colors="k", linewidths=0.3
    )
    ax.scatter(
        fiducials[name0],
        fiducials[name1],
        color="tab:red",
        marker="x",
        label="fiducial",
    )
    ax.set(xlabel=name0, ylabel=name1)
    ax.legend()
    fig.colorbar(cf, ax=ax, label="log posterior (unnormalized)")
    return fig


# %% [markdown]
# ## Running the grid evaluation
#
# We dispatch to the 1D or 2D routine based on the number of sampled parameters.

# %%
if len(sampled_params) == 1:
    name, grid, logpost = compute_logposterior_1d()
    print(f"evaluated log-posterior over {name}: {logpost.shape[0]} points")
    plot_logposterior_1d(name, grid, logpost)
else:
    axis0, axis1, logpost = compute_logposterior_2d()
    print(f"evaluated log-posterior over {axis0[0]} x {axis1[0]}: {logpost.shape} grid")
    plot_logposterior_2d(axis0, axis1, logpost)

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
    "local_merger_rate": local_merger_rate,
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
    "grid": {"npoints": npoints},
}
(out_dir / f"{base}.json").write_text(json.dumps(run_config, indent=2))
print("saved:", base)
