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
# # Single-amplitude toy MCMC
#
# This notebook is a stripped-down sibling of `mcmc.py`: it keeps the same
# importance-weighted pipeline scaffolding (catalog load, effective PSD,
# Gaussian likelihood, NUTS, saving, diagnostics) but replaces the
# cosmology/population callback with a trivial one-parameter model. The SGWB
# spectral density $S_h(f, \Lambda)$ scales linearly with a single scalar
# `amplitude` ($A$) through the total merger rate; the importance weights are
# all unity (relative ESS = 1), so this exercises the same
# `merger_rate_and_log_weights_fn` interface without any real astrophysics.
#
# It is a smoke/sanity test: can NUTS recover a known injected amplitude from
# a fiducial catalog?
#
# To run the notebook end-to-end, select a catalog ID in
# [`configs/paper.toml`](../configs/paper.toml) (or pass `--config`); the
# registry resolves it to a pluscross `.h5` catalog of complex polarizations.
# Figure-local knobs
# (detectors, seed, sampler, outputs) are argparse defaults in the config
# cell — edit them in Jupyter, override with flags headless.

# %% [markdown]
# ## Imports and JAX configuration

# %%
import argparse
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
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from astrogwb.config.loading import load_mapping
from astrogwb.sampling.numpyro_model import numpyro_model
from astrogwb.gwb import (
    spectral_density,
    spectral_snr_squared,
    frequency_mask as make_frequency_mask,
)
from astrogwb.detector import load_sensitivity_map, effective_psd
from astrogwb.waveform import polarization_power as compute_polarization_power
from astrogwb.config.catalogs import catalog_path, load_catalog_recipes
from pluscross import load_catalog
from astrogwb.utils import repo_root, years_to_seconds

# gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes; ArviZ 1.2
# mis-detects gwpy axes and looks for arviz_plots.backend.gwpy. Restore matplotlib axes.
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection

register_projection(MplAxes)

plt.style.library["paper-figures"] = {
    # "figure.figsize": (colwidth, colwidth),
    "figure.dpi": 200,
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.labelsize": "medium",
    "axes.titlesize": "medium",
    "figure.labelsize": "medium",
    "figure.titlesize": "medium",
    # Make the legend/label fonts a little smaller
    "legend.fontsize": "small",
    "legend.title_fontsize": "small",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
}

jax.config.update("jax_enable_x64", True)
# %config InlineBackend.figure_format = 'retina'
azp.style.use("arviz-variat")


# %% [markdown]
# ## Pipeline configuration
#
# Shared paths and analysis band come from [`configs/paper.toml`](../configs/paper.toml)
# (`[paths]`, `[analysis]`). Figure-local knobs are argparse defaults below —
# edit them in Jupyter, override with flags headless (`--debug` for a short
# smoke run: 100 warmup / 100 samples / 1 chain). Promote happy values by
# updating the defaults (and toml for shared settings).


# %%
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/paper.toml"))
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("figures/amplitude_toy_fisher_overlay.pdf"),
    )
    parser.add_argument(
        "--detectors",
        nargs="*",
        default=["S1", "R1"],
        help="Detector site codes (resolved via bundled geometry/sensitivity).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Short smoke run (100 warmup / 100 samples / 1 chain).",
    )
    parser.add_argument("--merger-rate-norm", type=float, default=1e-3)
    parser.add_argument("--amplitude-fiducial", type=float, default=1.0)
    parser.add_argument("--prior-low", type=float, default=0.1)
    parser.add_argument("--prior-high", type=float, default=10.0)
    parser.add_argument("--num-warmup", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument(
        "--num-chains",
        default="auto",
        help='Chain count, or "auto" for one chain per CPU.',
    )
    parser.add_argument("--target-accept", type=float, default=0.9)
    args, _ = parser.parse_known_args()
    return args


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


ROOT_DIR = repo_root()
args = _parse_args()
config_path = _resolve_path(args.config, ROOT_DIR)
paper = load_mapping(config_path)

catalog_registry = _resolve_path(Path(paper["catalog"]["registry"]), ROOT_DIR)
catalog_id = paper["catalog"]["id"]
if catalog_id not in load_catalog_recipes(catalog_registry):
    raise ValueError(f"unknown catalog {catalog_id!r} in {catalog_registry}")
CATALOG_PATH = _resolve_path(catalog_path(catalog_id), ROOT_DIR)
output_path = _resolve_path(args.output_pdf, ROOT_DIR)

# Detector settings
detnames = tuple(args.detectors)  # resolve via bundled geometry.toml / sensitivity.toml
observation_time = paper["analysis"][
    "observation_time"
]  # [yr]; cancels in S_h, kept for the likelihood scale

# MCMC settings
seed = args.seed
num_chains_raw = args.num_chains
num_chains = num_cpus if num_chains_raw == "auto" else int(num_chains_raw)
num_warmup = args.num_warmup
num_samples = args.num_samples
target_accept = args.target_accept

if args.debug:
    num_warmup, num_samples, num_chains, target_accept = 100, 100, 1, 0.9

# Frequency band for the analysis
f_min = paper["analysis"]["f_min"]
f_max = paper["analysis"]["f_max"]

# The toy model: S_h is linear in `amplitude` through the merger rate; the
# importance weights are all unity, so `merger_rate_norm` sets the SNR scale.
merger_rate_norm = args.merger_rate_norm  # mergers/sec
amplitude_fiducial = args.amplitude_fiducial

fiducials = {"amplitude": amplitude_fiducial}
hyperprior_dists = {"amplitude": dist.Uniform(args.prior_low, args.prior_high)}

sampled_params = ("amplitude",)

priors = {k: hyperprior_dists[k] for k in sampled_params}
constants = {k: v for k, v in fiducials.items() if k not in sampled_params}

# %% [markdown]
# ## Loading the waveform catalog
#
# Our method requires a waveform catalog computed for a fiducial population of CBCs. We offer a script to generate that in the [scripts directory](../scripts/generate_waveform_catalog.py).
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
polarization_power = jnp.asarray(
    compute_polarization_power(catalog)
)  # (nfreq, nsamples)
samples = {name: jnp.asarray(v) for name, v in catalog.source_parameters.items()}
del catalog

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

# %% [markdown]
# ## The single-amplitude toy model
#
# The predicted spectral density is
#
# $$
# S_h(f, A) = \frac{N(A)}{T}\frac{1}{N_{\mathrm{inj}}} \sum_{i=1}^{N_{\mathrm{inj}}} \left [ |\tilde{h}_+(f, \theta_i)|^2 + |\tilde{h}_\times (f, \theta_i)|^2 \right ],
# $$
#
# i.e. it scales **linearly** in $A$ through the total merger rate
# $N(A)/T = A \times \texttt{merger\_rate\_norm}$. The importance weights are
# all unity (the catalog samples are drawn from the fiducial population and
# never reweighted), so the relative ESS stays at 1 throughout the run — this
# notebook is only exercising the amplitude scaling of the callback interface.


# %%
def merger_rate_and_log_weights_fn(params, samples):
    total_merger_rate = params["amplitude"] * merger_rate_norm
    log_weights = jnp.zeros(n_samples)
    return total_merger_rate, log_weights


weights_fid = jnp.ones((n_samples,))
rate_fid = amplitude_fiducial * merger_rate_norm
observed_spectral_density = spectral_density(
    polarization_power, weights_fid, rate_fid, average_mode="analytic_inclination"
)

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


out_dir = _resolve_path(Path(paper["paths"]["chains_dir"]), ROOT_DIR)
out_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
params_suffix = "-".join(sampled_params)
det_suffix = ",".join(detnames)
base = f"chains-amplitude-toy-{params_suffix}-det={det_suffix}-seed{seed}-{timestamp}"

inference_data = azb.from_numpyro(mcmc)
inference_data.to_netcdf(out_dir / f"{base}.nc")

run_config = {
    "config_path": str(config_path),
    "catalog_path": str(CATALOG_PATH),
    "detectors": list(detnames),
    "output_path": str(output_path),
    "seed": seed,
    "observation_time": observation_time,
    "sampled_params": list(sampled_params),
    "fiducials": fiducials,
    "merger_rate_norm": merger_rate_norm,
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

# %% [markdown]
# ## Posterior vs. the Fisher–Gaussian (Laplace) approximation
#
# For the linear model $S_h(f, A) = A \, s(f)$ with a per-bin Gaussian likelihood
# of scale `gaussian_bin_scale` $= S_{\mathrm{eff}} / \sqrt{2 T \Delta f}$, the
# Fisher information at the fiducial amplitude is
#
# $$
# I(A_{\mathrm{fid}}) = \sum_i \left(\frac{\partial S_h}{\partial A}\right)^2 \Big/ \sigma_i^2
# = \frac{\mathrm{SNR}^2}{A_{\mathrm{fid}}^2},
# $$
#
# since $\partial S_h/\partial A = s = S_h^{\mathrm{obs}}/A_{\mathrm{fid}}$ and
# $\mathrm{SNR}^2$ is exactly `spectral_snr_squared` of the fiducial (observed)
# spectral density. The Laplace/Cramér–Rao posterior width is therefore
# $\sigma_A \approx A_{\mathrm{fid}} / \mathrm{SNR}$. We overlay
# $\mathcal{N}(A_{\mathrm{fid}}, \sigma_A^2)$ on the sampled marginal as a
# sanity check.

# %%
df = float(jnp.mean(jnp.diff(frequencies)))
T_sec = float(years_to_seconds(observation_time))
snr_sq = spectral_snr_squared(
    observed_spectral_density[mask], effective_psd_arr[mask], T_sec, df
)
snr = float(jnp.sqrt(snr_sq))
sigma_fisher = float(amplitude_fiducial / snr)
print(f"SNR={snr:.3f} sigma_fisher={sigma_fisher:.3f}")

# %%
kde = azs.kde(inference_data, var_names=["amplitude"])["amplitude"]
x_kde = kde.sel(plot_axis="x").values
y_kde = kde.sel(plot_axis="y").values

mu = amplitude_fiducial
x = np.linspace(mu - 4 * sigma_fisher, mu + 4 * sigma_fisher, 400)
pdf = np.exp(-0.5 * ((x - mu) / sigma_fisher) ** 2) / (
    sigma_fisher * np.sqrt(2 * np.pi)
)

# Scoped reset: `azp.style.use` above mutated the global matplotlib rcParams,
# which would otherwise leak arviz's styling into this hand-built figure too.
with plt.style.context("paper-figures", after_reset=True):
    fig, ax = plt.subplots()
    ax.plot(x_kde, y_kde, label="MCMC posterior", color="black")
    ax.fill_between(x_kde, y_kde, alpha=0.2, color="black")
    ax.plot(
        x,
        pdf,
        linestyle="--",
        label=r"$\mathcal{N}(A_\mathrm{fid}, 1/\rho_0^2)$",
        color="black",
        lw=2.0,
    )
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Posterior density")
    ax.legend(loc="upper right")

output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, bbox_inches="tight")
print("saved figure:", output_path)
