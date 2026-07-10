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
# # Cosmological parameter inference with the astrophysical GWB
#
# In this notebook we perform Bayesian inference on the cosmological and astrophysical
# parameters that drive the stochastic gravitational-wave background (SGWB) of
# stellar-mass compact binary coalescences (CBCs), such as binary neutron stars or
# black holes.
#
# The strategy is **importance sampling over a fixed proposal catalog**: a one-off catalog
# of CBC sources (drawn at a *fiducial* parameter point) provides the per-source
# polarization powers $|\tilde{h}_+ (f, \theta)|^2 + |\tilde{h}_\times (f, \theta)|^2$. During NUTS we never regenerate waveforms;
# instead we reweight the catalog with analytic, JAX-traceable importance weights so the
# likelihood depends on the sampled parameters $\Lambda$ through them and the
# total merger rate only.
#
# To run the notebook end-to-end you must point `CATALOG_PATH` at an `.npz`
# polarization-power catalog (schema documented below). The data-dependent cells are written
# to be correct by construction but will only execute once such a catalog exists.

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

import arviz_base as azb
import arviz_plots as azp
import arviz_stats as azs
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import matplotlib.pyplot as plt

from astrogwb.sampling.numpyro_model import numpyro_model
from astrogwb.gwb import (
    spectral_density,
    omega_gw_from_spectral_density,
    frequency_mask as make_frequency_mask,
)
from astrogwb.detector import load_sensitivity_map, effective_psd
from astrogwb.utils import repo_root
from astrogwb.waveform import polarization_power as compute_polarization_power
from pluscross import load_catalog

# gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes; ArviZ 1.2
# mis-detects gwpy axes and looks for arviz_plots.backend.gwpy. Restore matplotlib axes.
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection

register_projection(MplAxes)

jax.config.update("jax_enable_x64", True)
# %config InlineBackend.figure_format = 'retina'
azp.style.use("arviz-variat")


# %% [markdown]
# ## Pipeline configuration

# %%
DEBUG = False  # small smoke settings for first runs; set False for the production run

# --- Catalog input (placeholder — see schema markdown below) ----------------
# No working polarization-power catalog exists yet; set this once one is produced.


ROOT_DIR = repo_root()
CATALOG_PATH = ROOT_DIR / "out/bns_waveform_catalog.h5"

# Detector settings
detnames = ("S1", "R1", "C1")  # resolve via bundled geometry.toml / sensitivity.toml
observation_time = 1.0  # [yr]; cancels in S_h, kept for the likelihood scale

# MCMC settings
seed = 42
num_chains = num_cpus  # one chain per CPU core
num_warmup = 250
num_samples = 250
target_accept = 0.9

if DEBUG:
    num_warmup, num_samples, num_chains, target_accept = 100, 100, 1, 0.9

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
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
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
    "local_merger_rate": dist.Uniform(7.6, 250.0),
}

sampled_params = set(("H0",))

priors = {k: hyperprior_dists[k] for k in sampled_params}
constants = {k: v for k, v in fiducials.items() if k not in sampled_params}

# %% [markdown]
# ## Loading the waveform catalog
#
# Our method requires a waveform catalog computed for a fiducial population of CBCs. We offer a script to generate that in the [scripts directory](../scripts/generate_polarization_power_catalog.py).
#
# The catalog is an `.npz` file with this schema:
#
# - `frequencies` — shape `(nfreq,)`, the FFT frequency grid (Hz).
# - `polarization_power` — shape `(nfreq, nsamples)`, the raw per-source
#   $|\tilde{h}_+(f)|^2 + |\tilde{h}_\times(f)|^2$
# - `samples` — per-source parameters, stored as `sample__<name>` keys and restored into a dict.

# %%
catalog = load_catalog(CATALOG_PATH)

frequencies = jnp.asarray(catalog.frequencies)
polarization_power = jnp.asarray(compute_polarization_power(catalog))  # (nfreq, nsamples)
samples = {name: jnp.asarray(v) for name, v in catalog.source_parameters.items()}
del catalog

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
# The spectral density $S_h(f, \Lambda)$ due to the superposition of gravitational-wave sources from a CBC population is given by
#
# $$
# S_h(f, \Lambda) = \frac{1}{T} \left \langle \sum_{i=1}^{N(\Lambda)} |\tilde{h}_+(f, \theta_i)|^2 + |\tilde{h}_\times (f, \theta_i)|^2  \right \rangle_{\theta \sim p(\theta | \Lambda)}
# $$
#
# In our implementation, we calculate $S_h(f, \Lambda)$ with an importance sampling estimator over a fiducial proposal distribution $p(\theta | \Lambda_0)$:
#
# $$
# S_h(f, \Lambda) = \frac{N(\Lambda)}{T}\frac{1}{N_{\mathrm{inj}}} \sum_{i=1}^{N_{\mathrm{inj}}} \omega_i \left [ |\tilde{h}_+(f, \theta_i)|^2 + |\tilde{h}_\times (f, \theta_i)|^2 \right ],
# $$
#
# The waveforms are thus pre-computed for the fiducial population $\Lambda_0$, and the astrophysical + cosmological model specify two things:
#
# - The volume-integrated merger rate $N(\Lambda)/T$;
# - The importance weights $\omega_i$.
#
# ### Calculating the importance weights
#
# The importance weights are proportional to the ratio of probabilities at the particular sample points,
#
# $$
# \omega_i \propto \frac{p(\theta_i | \Lambda)}{p(\theta_i | \Lambda_0)}.
# $$
#
# While the single-event parameter samples are fixed, a different cosmology will change the amplitude of the waveforms due to the $\propto 1 / d_L$ dependence. Therefore, the importance weights must be rescaled by the inverse-squared distance ratio:
#
# $$
# \omega_i = \frac{p(\theta_i | \Lambda)}{p(\theta_i | \Lambda_0)} \frac{d_L(z, \Lambda_0)^2}{d_L(z, \Lambda)^2}
# $$
#
# ### Taking into account modified propagation
#
# When considering effects of deviation from GR on the propagation of gravitational-waves, we can generalize the above relation to
#
# $$
# \omega_i = \frac{p(\theta_i | \Lambda)}{p(\theta_i | \Lambda_0)} \frac{d_{GW}(z, \Lambda_0)^2}{d_{GW}(z, \Lambda)^2}
# $$
#
# which now also encodes the effect of modified propagation.
#
# ### Implementing the model
#
# To implement and use the importance sampling model in the pipeline, it suffices to implement a
#
# ```python
# def merger_rate_and_log_weights(parameters: jax.Array, samples: jax.Array):
#     pass
# ```
#
#
# which returns a tuple of (merger rate, log importance weights). While it would be conceptually simpler to pass separate functions for each quantity, encapsulating all the logic in a single function allows the caller to efficiently implement the cosmology integrals which are used in both calculations.
#
# The proposal log-density `log p_proposal(z)` depends only on the fixed fiducial point,
# so we evaluate it **once** here. The NUTS-time weight function then reuses this array
# and evaluates one shared redshift grid per proposed $\Lambda$ for target distances,
# target redshift density, and total merger-rate normalization.

# %%
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_proposal_logpdf,
    make_merger_rate_and_log_weights_fn,
)

z_samples = jnp.asarray(samples["redshift"])
z_grid = jnp.linspace(z_min, z_max, n_grid)

log_p_proposal = compute_proposal_logpdf(
    z_samples, z_grid=z_grid, fiducials=fiducials
)


# %% [markdown]
# The `merger_rate_and_log_weights` callback is packaged in
# `astrogwb.importance.models.bns_madau_dickinson_modified_propagation`
# (BNS + Madau-Dickinson rate + modified GW/EM propagation). We bind the
# canonical, tested factory to this notebook's proposal catalog here.

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
# In the cell below, we plot $\Omega_{GW}(f, \Lambda_0)$.


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
# ## Running the MCMC
#
# We run the NUTS sampler as implemented in the `numpyro` python package.

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

kernel = NUTS(
    model,
    target_accept_prob=target_accept,
    forward_mode_differentiation=True,
    dense_mass=True,
)
mcmc = MCMC(
    kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    progress_bar=True,
    jit_model_args=True,
    chain_method="vectorized",
)
rng_key = jax.random.PRNGKey(seed)
mcmc.run(
    rng_key,
    frequencies=frequencies,
    polarization_power=polarization_power,
    samples=samples,
    observed_spectral_density=observed_spectral_density,
    effective_psd=effective_psd_arr,
    extra_fields=("num_steps", "accept_prob", "diverging"),
)
mcmc.print_summary()

# %% [markdown]
# ## Saving the run
#
# We convert to an `xarray.DataTree` via `arviz_base` and write it to NetCDF,
# alongside a small JSON record of the run configuration.

# %%


out_dir = Path("chains")
out_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
params_suffix = "-".join(sampled_params)
det_suffix = ",".join(detnames)
base = f"chains-{params_suffix}-det={det_suffix}-seed{seed}-{timestamp}"

inference_data = azb.from_numpyro(mcmc)
inference_data.to_netcdf(out_dir / f"{base}.nc")

run_config = {
    "catalog_path": str(CATALOG_PATH),
    "detectors": list(detnames),
    "seed": seed,
    "observation_time": observation_time,
    "sampled_params": list(sampled_params),
    "fiducials": fiducials,
    "sampler": {
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "target_accept": target_accept,
    },
}
(out_dir / f"{base}.json").write_text(json.dumps(run_config, indent=2))
print("saved:", base)

# %% [markdown]
# ## Diagnostic plots

# %%
summary = azs.summary(inference_data, var_names=list(sampled_params))
summary

# %%
azp.plot_trace_dist(inference_data, var_names=list(sampled_params))

# %%
azp.plot_autocorr(inference_data, var_names=list(sampled_params))

# %%
if len(sampled_params) >= 2:
    azp.plot_pair(
        inference_data,
        var_names=list(sampled_params),
        marginal_kind="kde",
        marginal=True,
    )
else:
    azp.plot_dist(inference_data, var_names=list(sampled_params))
