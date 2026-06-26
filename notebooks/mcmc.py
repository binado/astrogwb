# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cosmological parameter inference with the astrophysical GWB
#
# In this notebook we perform Bayesian inference on the cosmological and astrophysical
# parameters that drive the stochastic gravitational-wave background (SGWB) of
# stellar-mass compact binary coalescences (CBCs), such as binary neutron stars or
# black holes. It is the NumPyro port of `ASGWB.jl/notebooks/mcmc.jl`.
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
#
# We enable 64-bit precision to match the Julia run (`Float64`). This must happen
# *before* any JAX array is created.

# %%
from collections.abc import Mapping
from functools import partial
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import matplotlib.pyplot as plt

# astrogwb: minimal "bring-your-own-physics" core
from astrogwb.sampling.numpyro_model import MergerRateAndLogWeightsFn, numpyro_model
from astrogwb.gwb import (
    spectral_density,
    omega_gw_from_spectral_density,
    frequency_mask as make_frequency_mask,
)
from astrogwb.detector import load_sensitivity_map, effective_psd
from astrogwb.waveform import load_polarization_power_catalog

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

print("jax x64:", jax.config.jax_enable_x64)

SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


# %% [markdown]
# ## Configuration
#
# Edit runtime settings here, mirroring the Julia `c3d4e5f6` cell: detector network,
# seed, local merger rate, observation time, redshift range, fiducial parameters,
# hyperprior bounds, and which parameters to actually sample.
#
# Naming note vs. Julia: the modified-propagation parameters $\Xi_0 \to$ `xi_0`,
# $\Xi_n \to$ `xi_n` (GR is `xi_0=1, xi_n=0`). Because gwmock-pop's Madau-Dickinson
# `kappa` is the *offset* exponent (denominator exponent `= gamma + kappa`, matching
# Julia's `γ + κ`), we set the fiducial `kappa = 3.0`.
#
# `sampled_params` is the `sample_only` analog: parameters listed here get a prior and are
# inferred; everything else in `fiducials` is held fixed as a `constant`.

# %%
DEBUG = True  # small smoke settings for first runs; set False for the production run

# --- Catalog input (placeholder — see schema markdown below) ----------------
# No working polarization-power catalog exists yet; set this once one is produced.
CATALOG_PATH = "catalog.npz"

# --- Detector network -------------------------------------------------------
detnames = ("S1", "R1", "C1")  # resolve via bundled geometry.toml / sensitivity.toml

# --- RNG / run settings -----------------------------------------------------
seed = 42
local_merger_rate = 161.0  # [Gpc^-3 yr^-1], matches COBA simulations
observation_time = 1.0  # [yr]; cancels in S_h, kept for the likelihood scale

# --- Redshift integration range ---------------------------------------------
z_min = 0.0
z_max = 10.0
n_grid = 4096  # grid points for cosmology integrals / MD normalization

# --- Analysis frequency band ------------------------------------------------
f_min = 10.0
f_max = 100.0

# --- Fiducial parameters (the catalog's proposal point) ---------------------
# Julia Ξ₀ -> xi_0, Ξₙ -> xi_n; flat ΛCDM here (no w0 — we sample H0 only).
fiducials = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "xi_0": 1.0,  # GR modified-propagation amplitude
    "xi_n": 0.0,  # GR modified-propagation slope
    "gamma": 2.7,
    "kappa": 3.0,  # gwmock-pop offset convention (Julia κ)
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

# --- Which parameters to sample (the `sample_only` analog) ------------------
sampled_params = ("H0",)

priors = {k: hyperprior_dists[k] for k in sampled_params}
constants = {k: v for k, v in fiducials.items() if k not in sampled_params}

print("sampling:", tuple(priors), "| fixed:", tuple(constants))

# %% [markdown]
# ## Load the proposal catalog
#
# The catalog is an `.npz` file written by
# `astrogwb.waveform.save_polarization_power_catalog`, with this schema:
#
# - `frequencies` — shape `(nfreq,)`, the FFT frequency grid (Hz).
# - `polarization_power` — shape `(nfreq, nsamples)`, the raw per-source
#   $|A_+(f)|^2 + |A_\times(f)|^2$ evaluated at the **fiducial** distances. **All distance
#   scaling lives here**: each column already carries $1/d_{L,\mathrm{fid}}^2$ at the
#   source's fiducial redshift, so the importance-weight math (which multiplies by
#   $d_{L,\mathrm{fid}}^2 / d_{L,\theta}^2$) is exact.
# - `samples` — per-source intrinsic parameters; **must include `redshift`** (consumed by
#   the importance weights) and `luminosity_distance` (the fiducial EM luminosity
#   distance). Stored as `sample__<name>` keys and restored into a dict.
#
# > **Deferred:** there is no working catalog file yet, so this cell (and everything
# > downstream of it) will only execute once `CATALOG_PATH` points at a real file. The cells
# > are nonetheless written to be correct by construction.

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
# `load_sensitivity_map` resolves the str-named detectors (`S1`, `R1`, `C1`) via the bundled
# `geometry.toml` / `sensitivity.toml`; `effective_psd` combines them (including the
# frequency-dependent overlap reduction function) into a single network PSD on our
# frequency grid. The `frequency_mask` restricts the likelihood to the analysis band.

# %%
sensitivities = load_sensitivity_map(detnames)
effective_psd_arr = jnp.asarray(
    effective_psd(frequencies, list(detnames), sensitivities)
)
freq_mask = make_frequency_mask(frequencies, fmin=f_min, fmax=f_max)
print("band bins:", int(jnp.sum(freq_mask)), "of", frequencies.shape[0])

# %% [markdown]
# ## Precompute fiducial constants for the weights
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
# ## Importance weights
#
# Per proposal sample $i$ with redshift $z_i$ (from `AstroSGWB/src/importance.jl`):
#
# $$\log w_i = \big[\log p_\mathrm{target}(z_i) - \log p_\mathrm{proposal}(z_i)\big]
#             + 2\log d_{L,\mathrm{fid}}(z_i) - 2\log d_{L,\theta}(z_i)
#             + 2\log \Xi_\mathrm{fid}(z_i) - 2\log \Xi_\theta(z_i)$$
#
# with the modified-propagation ratio
# $\Xi_\theta(z) = \Xi_0 + (1-\Xi_0)/(1+z)^{\Xi_n}$ (GR $\Rightarrow \Xi_0=1, \Xi_n=0
# \Rightarrow \Xi \equiv 1$). At $\theta = $ fiducial all correction terms vanish, so
# weights $\equiv 1$. `log_p_proposal` is precomputed above;
# `samples["luminosity_distance"]` supplies the fiducial EM luminosity distances.


# %%
def log_gw_em_ratio(z, xi_0, xi_n):
    return jnp.log(xi_0 + (1.0 - xi_0) * jnp.exp(-xi_n * jnp.log1p(z)))


def flat_lcdm_grid(
    params: Mapping[str, Any],
    z_grid: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    h0 = params["H0"]
    omega_m = params["Omega_m"]

    z, comoving_distance, luminosity_distance = build_distance_lookup(
        hubble_constant=h0,
        omega_m=omega_m,
        max_redshift=float(z_grid[-1]),
        n_grid=z_grid.shape[0],
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
    def merger_rate_and_log_weights_fn(params, samples):
        z = samples["redshift"]
        d_l_fid = samples["luminosity_distance"]

        luminosity_distance_grid, dvc_dz_grid = flat_lcdm_grid(params, z_grid)
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
# ## Total merger rate
#
# `total_merger_rate` (events/sec) integrates the **unnormalized** detector-frame weight
# $\psi(z)/(1+z)\,\mathrm{d}V_c/\mathrm{d}z$ — the same quantity
# `madau_dickinson_redshift_pdf` normalizes — times the local rate:
#
# $$\texttt{integral\_Mpc3} = \int_{z_\min}^{z_\max}
#    \frac{\psi(z)}{1+z}\,\frac{\mathrm{d}V_c}{\mathrm{d}z}\,\mathrm{d}z,\qquad
#    \texttt{rate} = 10^{-9}\,R_\mathrm{local}\,\frac{\texttt{integral\_Mpc3}}{T_\mathrm{yr}}.$$
#
# The $10^{-9}$ converts $R_\mathrm{local}$ from $\mathrm{Gpc}^{-3}$ to $\mathrm{Mpc}^{-3}$;
# dividing by seconds-per-year turns the per-year local rate into per-second.


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

if DEBUG:
    num_warmup, num_samples, num_chains, target_accept = 100, 100, 1, 0.9
else:
    num_warmup, num_samples, num_chains, target_accept = 3000, 3000, 1, 0.9

numpyro.set_host_device_count(num_chains)

kernel = NUTS(
    model,
    target_accept_prob=target_accept,
    forward_mode_differentiation=True,
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
# We convert to an ArviZ `InferenceData` (the JLD2 chain analog) and write it to NetCDF,
# alongside a small JSON record of the run configuration.

# %%
import json
from datetime import datetime
from pathlib import Path

import arviz as az

out_dir = Path("chains")
out_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
params_suffix = "-".join(sampled_params)
det_suffix = ",".join(detnames)
base = f"chains-{params_suffix}-det={det_suffix}-seed{seed}-{timestamp}"

idata = az.from_numpyro(mcmc)
idata.to_netcdf(out_dir / f"{base}.nc")

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
# `az.summary` (Julia `summarystats`), trace and autocorrelation plots, and a `corner`
# plot for $\ge 2$ sampled parameters (the `PairPlots` analog) or `az.plot_posterior` for a
# single parameter. We also surface the model's `importance_relative_ess` and
# `total_merger_rate` deterministics — the relative effective sample size is the key health
# check that the proposal catalog still reweights well at the posterior.

# %%
summary = az.summary(idata)
summary

# %%
az.plot_trace(idata, var_names=list(sampled_params))
plt.tight_layout()

# %%
az.plot_autocorr(idata, var_names=list(sampled_params))
plt.tight_layout()

# %%
if len(sampled_params) >= 2:
    import corner

    corner.corner(idata, var_names=list(sampled_params))
else:
    az.plot_posterior(idata, var_names=list(sampled_params))
plt.tight_layout()

# %%
# importance-sampling health: relative ESS should stay close to 1 across draws
post = idata.posterior
ress = post["importance_relative_ess"].values.ravel()
rate = post["total_merger_rate"].values.ravel()
print(f"relative_ess: mean={ress.mean():.3f} min={ress.min():.3f}")
print(f"total_merger_rate [/s]: mean={rate.mean():.4e}")
