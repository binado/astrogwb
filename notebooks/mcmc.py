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
# polarization powers $|A_+|^2 + |A_\times|^2$. During NUTS we never regenerate waveforms;
# instead we reweight the catalog with analytic, JAX-traceable importance weights so the
# likelihood depends on the sampled parameters $\Lambda$ through `exp(log_weights)` and the
# total merger rate only.
#
# To run the notebook end-to-end you must point `CATALOG_PATH` at an `.npz`
# polarization-power catalog (schema documented below). The data-dependent cells are written
# to be correct by construction but will only execute once such a catalog exists.

# %% [markdown]
# ## Imports and JAX configuration

# %%
from collections.abc import Mapping
from datetime import datetime
from functools import partial
import json
import multiprocessing
from pathlib import Path
from typing import Any

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

from astrogwb.sampling.numpyro_model import MergerRateAndLogWeightsFn, numpyro_model
from astrogwb.gwb import (
    spectral_density,
    omega_gw_from_spectral_density,
    frequency_mask as make_frequency_mask,
)
from astrogwb.detector import load_sensitivity_map, effective_psd
from astrogwb.utils import SECONDS_PER_YEAR
from astrogwb.waveform import load_polarization_power_catalog

# gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes; ArviZ 1.2
# mis-detects gwpy axes and looks for arviz_plots.backend.gwpy. Restore matplotlib axes.
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection

register_projection(MplAxes)

# gwmock-pop: JAX-traceable population + cosmology physics for the weights
from gwmock_pop.distributions.madau_dickinson import (
    madau_dickinson_redshift_pdf,
    madau_dickinson_rate,
)
from gwmock_pop.cosmology.flat_lambda_cdm import (
    SPEED_OF_LIGHT,
    build_distance_lookup,
    compute_normalized_hubble_parameter,
)

jax.config.update("jax_enable_x64", True)
# %config InlineBackend.figure_format = 'retina'
azp.style.use("arviz-variat")


# %% [markdown]
# ## Pipeline configuration

# %%
DEBUG = False  # small smoke settings for first runs; set False for the production run

# --- Catalog input (placeholder — see schema markdown below) ----------------
# No working polarization-power catalog exists yet; set this once one is produced.

def get_root_dir() -> Path:
    return Path.cwd().parent

ROOT_DIR = get_root_dir()
CATALOG_PATH = ROOT_DIR / "out/bns_polarization_power_catalog.npz"

# Detector settings
detnames = ("S1", "R1", "C1")  # resolve via bundled geometry.toml / sensitivity.toml
local_merger_rate = 161.0  # [Gpc^-3 yr^-1], matches COBA simulations
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
freq_mask = make_frequency_mask(frequencies, fmin=f_min, fmax=f_max)
print("band bins:", int(jnp.sum(freq_mask)), "of", frequencies.shape[0])


# %%
def plot_effective_psd(frequencies: jax.Array, effective_psd: jax.Array, mask: jax.Array, *, color: str = "black"):
    fig, ax = plt.subplots()
    ax.loglog(frequencies, effective_psd, color=color, ls="--")
    ax.loglog(frequencies[mask], effective_psd[mask], color=color)
    ax.set(xlabel=r"$f$ [Hz]", ylabel=r"$S_{\text{eff}}(f)$ [1/Hz]", title="Effective PSD")
    return fig

plot_effective_psd(frequencies, effective_psd_arr, freq_mask)

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


# %% [markdown]
# We implement the `merger_rate_and_log_weights` function in the cell below:

# %%
def log_gw_em_ratio(z, xi_0, xi_n):
    return jnp.log(xi_0 + (1.0 - xi_0) * jnp.exp(-xi_n * jnp.log1p(z)))


def flat_lcdm_grid(
    params: Mapping[str, Any],
    max_redshift: float,
    n_grid: int = 256
) -> tuple[jax.Array, jax.Array]:
    h0 = params["H0"]
    omega_m = params["Omega_m"]

    z, comoving_distance, luminosity_distance = build_distance_lookup(
        hubble_constant=h0,
        omega_m=omega_m,
        max_redshift=max_redshift,
        n_grid=n_grid,
    )
    e_z = compute_normalized_hubble_parameter(redshift=z, omega_m=omega_m)
    differential_comoving_volume = (
        4.0 * jnp.pi * comoving_distance**2 / (h0 * e_z) * SPEED_OF_LIGHT / 1000
    )
    return luminosity_distance, differential_comoving_volume


def make_merger_rate_and_log_weights_fn(
    *,
    z_grid: jax.Array,
    proposal_log_pdf: jax.Array,
    local_merger_rate: float,
    fiducial_xi_0: float,
    fiducial_xi_n: float,
) -> MergerRateAndLogWeightsFn:
    n_grid = z_grid.shape[0]
    max_redshift = float(z_grid[-1])
    
    def merger_rate_and_log_weights_fn(params, samples):
        z = samples["redshift"]
        d_l_fid = samples["luminosity_distance"]

        luminosity_distance_grid, dvc_dz_grid = flat_lcdm_grid(params, max_redshift, n_grid)
        d_l_theta = jnp.interp(
            z,
            z_grid,
            luminosity_distance_grid,
            left=luminosity_distance_grid[0],
            right=luminosity_distance_grid[-1],
        )

        rate_shape_grid = madau_dickinson_rate(
            z_grid, params["gamma"], params["kappa"], params["z_peak"]
        )
        unnormalized_pdf_grid = rate_shape_grid / (1.0 + z_grid) * dvc_dz_grid
        integral_Mpc3 = jnp.trapezoid(unnormalized_pdf_grid, z_grid)

        rate_shape_samples = madau_dickinson_rate(
            z, params["gamma"], params["kappa"], params["z_peak"]
        )
        dvc_dz_samples = jnp.interp(
            z,
            z_grid,
            dvc_dz_grid,
            left=dvc_dz_grid[0],
            right=dvc_dz_grid[-1],
        )
        target_pdf = rate_shape_samples / (1.0 + z) * dvc_dz_samples / integral_Mpc3

        log_fiducial_gw_em_ratio = log_gw_em_ratio(z, fiducial_xi_0, fiducial_xi_n)
        log_target_gw_em_ratio = log_gw_em_ratio(z, params["xi_0"], params["xi_n"])
        log_weights = (
            jnp.log(target_pdf)
            - proposal_log_pdf
            + 2.0 * jnp.log(d_l_fid)
            - 2.0 * jnp.log(d_l_theta)
            + 2.0 * log_fiducial_gw_em_ratio
            - 2.0 * log_target_gw_em_ratio
        )

        total_merger_rate = 1e-9 * local_merger_rate * integral_Mpc3 / SECONDS_PER_YEAR
        return total_merger_rate, log_weights

    return merger_rate_and_log_weights_fn


merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
    z_grid=jnp.linspace(z_min, z_max, n_grid),
    proposal_log_pdf=log_p_proposal,
    local_merger_rate=local_merger_rate,
    fiducial_xi_0=fiducials["xi_0"],
    fiducial_xi_n=fiducials["xi_n"],
)


# %% [markdown]
# ## Visualizing $\Omega_{\mathrm{GW}}$ at the fiducial point
#
# With weights $\equiv 1$ (the fiducial spectrum), `spectral_density` reduces to
# `0.4 · rate · mean_over_sources(polarization_power)`. We convert to $\Omega_{\mathrm{GW}}(f)$
# and plot the positive part on log-log axes (Julia `plot_fiducial_omega_gw`).

# %%
ones_weights = jnp.ones((n_samples,))
rate0, _ = merger_rate_and_log_weights_fn(
    fiducials,
    samples,
)
S_h0 = spectral_density(
    polarization_power, ones_weights, rate0, average_mode="analytic_inclination"
)

omega0 = omega_gw_from_spectral_density(S_h0, frequencies)
pos = omega0 > 0.0

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.loglog(np.asarray(frequencies[pos]), np.asarray(omega0[pos]))
ax.set_xlabel(r"$f\ \mathrm{(Hz)}$")
ax.set_ylabel(r"$\Omega_{\mathrm{GW}}(f)$")
ax.set_ylim(1e-15, None)
ax.set_title("Fiducial astrophysical GWB")
fig.tight_layout()

# %% [markdown]
# ## Observed (injected) data
#
# We inject the fiducial spectrum as the observed data, so the truth sits at the fiducial
# parameter point (the profile/recovery setup of the Julia notebook).

# %%
observed_spectral_density = S_h0

# %% [markdown]
# ## Running the MCMC
#
# We pass the large catalog arrays as dynamic JAX arguments to `mcmc.run`, not as closed-over
# constants. Like the Julia run (which used ForwardDiff), we use **forward-mode**
# differentiation — the weight chain goes through grid-based cosmology integrals that are
# forward-mode friendly. Production defaults mirror Julia (`num_warmup=3000,
# num_samples=3000, target_accept=0.9`); `DEBUG` uses a small smoke setting.

# %%
model = partial(
    numpyro_model,
    observation_time=observation_time,
    average_mode="analytic_inclination",
    merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
    priors=priors,
    constants=constants,
    frequency_mask=freq_mask,
)

kernel = NUTS(
    model,
    target_accept_prob=target_accept,
    forward_mode_differentiation=True,
    dense_mass=True
)
mcmc = MCMC(
    kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    progress_bar=True,
    jit_model_args=True,
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
    "local_merger_rate": local_merger_rate,
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
#
# `arviz_stats.summary`, then `arviz_plots` trace, autocorrelation, and marginal/pair
# plots. We also surface the model's `importance_relative_ess` and `total_merger_rate`
# deterministics — the relative effective sample size is the key health check that the
# proposal catalog still reweights well at the posterior.

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

# %%
# importance-sampling health: relative ESS should stay close to 1 across draws
post = inference_data["posterior"]
ress = post["importance_relative_ess"].values.ravel()
rate = post["total_merger_rate"].values.ravel()
print(f"relative_ess: mean={ress.mean():.3f} min={ress.min():.3f}")
print(f"total_merger_rate [/s]: mean={rate.mean():.4e}")
