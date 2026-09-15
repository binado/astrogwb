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
# The strategy is **importance sampling over a fixed guarded proposal catalog**:
# a mixture of fiducial and uniform-redshift sources provides the per-source
# polarization powers $|\tilde{h}_+ (f, \theta)|^2 + |\tilde{h}_\times (f, \theta)|^2$. During NUTS we never regenerate waveforms;
# instead we reweight the catalog with analytic, JAX-traceable importance weights so the
# likelihood depends on the sampled parameters $\Lambda$ through them and the
# total merger rate only.
#
# To run the notebook end-to-end, point `INJECTION_CATALOG_PATH` and
# `PROPOSAL_CATALOG_PATH` at the `astrogwb_catalog` HDF5 files generated
# by the catalog workflow (``outputs/catalogs/<name>.h5``).

# %% [markdown]
# ## Environment bootstrap (Colab vs. local)
#
# Detect whether this notebook is running on Google Colab. On Colab we probe
# for TPU hardware before installing the checkout. The checkout is cloned into
# `/content/astrogwb` and installed with the selected accelerator and notebook
# extras. Finally, mount Google Drive for the waveform catalog. Locally this
# cell is a no-op.

# %%
import os
import subprocess
import sys
from pathlib import Path

try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Stdlib-only probe — choose accelerator extras before resolving deps so a
# TPU VM never downloads the jax[cuda12] stack.
HAS_COLAB_TPU = IN_COLAB and (
    os.path.exists("/dev/accel0") or bool(os.environ.get("COLAB_TPU_ADDR"))
)

if IN_COLAB:
    _accelerator = "tpu" if HAS_COLAB_TPU else "cuda"
    _checkout = Path("/content/astrogwb")
    if not _checkout.is_dir():
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/binado/astrogwb.git",
                str(_checkout),
            ]
        )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "uv",
        ]
    )
    subprocess.check_call(
        [
            "uv",
            "pip",
            "install",
            "--system",
            "--editable",
            f"{_checkout}[{_accelerator},notebook]",
        ]
    )
    # Editable .pth files are processed on interpreter startup. Make the source
    # tree immediately visible to the already-running Colab kernel as well.
    sys.path.insert(0, str(_checkout / "src"))

    from google.colab import drive

    drive.mount("/content/drive")

# Force TPU only when Colab exposes its device; otherwise let JAX auto-detect
# CPU or CUDA.
platform = "tpu" if HAS_COLAB_TPU else "auto"

# %% [markdown]
# ## Runtime & sampler sizing
#
# `num_chains` must be fixed before we hand the device platform to
# `configure_runtime` below — it seeds `numpyro.set_host_device_count`, which
# has to run before JAX claims a device. Locally we run one chain per CPU
# core; Colab uses four chains on CPU, GPU, and TPU runtimes.

# %%
import multiprocessing

DEBUG = False  # small smoke settings for first runs; set False for the production run

seed = 42
# One chain per CPU core locally; a fixed, modest count on every Colab runtime.
num_chains = 4 if IN_COLAB else multiprocessing.cpu_count()
num_warmup = 250
num_samples = 250
target_accept = 0.9

if DEBUG:
    num_warmup, num_samples, num_chains, target_accept = 100, 100, 1, 0.9

# %% [markdown]
# ## Imports and JAX/device configuration
#
# `configure_runtime` (`astrogwb.paper/runtime.py`) is the single place allowed
# to set `XLA_FLAGS`/`JAX_PLATFORMS`/`numpyro.set_host_device_count` and import
# `jax`; it must run before any other cell imports `jax` or `numpyro`. It
# resolves `chain_method` from the visible device count: `"parallel"` when
# every chain has a device, `"vectorized"` when GPU/TPU devices are fewer than
# chains, and `"sequential"` when CPU devices are insufficient. The same
# helper is used by `scripts/run_mcmc.py` for the headless runner.

# %%
import json
from datetime import datetime
from functools import partial

from astrogwb.paper.runtime import configure_runtime

jax, chain_method = configure_runtime(num_chains=num_chains, platform=platform)

import arviz_base as azb
import arviz_plots as azp
import arviz_stats as azs
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes; ArviZ 1.2
# mis-detects gwpy axes and looks for arviz_plots.backend.gwpy. Restore matplotlib axes.
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

from astrogwb.gwb import (
    omega_gw_from_spectral_density,
)
from astrogwb.paper.catalogs import load_run_catalog
from astrogwb.paper.config import fiducials as committed_fiducials
from astrogwb.paper.config import networks as committed_networks
from astrogwb.paper.config import priors as committed_priors
from astrogwb.paper.config.mcmc import AnalysisGrid, build_run_config
from astrogwb.paper.config.runs import assemble_run
from astrogwb.paper.inference import prepare_inference_inputs
from astrogwb.populations import build_population
from astrogwb.sampling import gwb_spectral_density_model

register_projection(MplAxes)

# %config InlineBackend.figure_format = 'retina'
azp.style.use("arviz-variat")


# %% [markdown]
# ## Pipeline configuration

# %%
# --- Catalog input ---------------------------------------------------------

if IN_COLAB:
    INJECTION_CATALOG_PATH = Path(
        "/content/drive/MyDrive/asgwb/md-imrphenom-s41-n32768.h5"
    )
    PROPOSAL_CATALOG_PATH = Path(
        "/content/drive/MyDrive/asgwb/md-imrphenom-s42-n16384.h5"
    )
else:
    ROOT_DIR = Path()
    INJECTION_CATALOG_PATH = ROOT_DIR / "outputs/catalogs/md-imrphenom-s41-n32768.h5"
    PROPOSAL_CATALOG_PATH = ROOT_DIR / "outputs/catalogs/md-imrphenom-s42-n16384.h5"

# The run whose catalog composition this notebook reproduces. It used to be a
# default buried in `config.figures.load_injection_spec`; naming it here makes
# the notebook say which run it is standing in for.
REFERENCE_RUN = ("cosmological-parameters", "ET-2L-aligned-CE-Hanford")

# Detector settings. The network name resolves to its detector list through
# config/networks.json -- the same table the reference run resolves through --
# so this cell stays a knob (change the name) without keeping a second copy of
# the list. The detector names resolve further via the bundled geometry.toml /
# sensitivity.toml.
NETWORK = "ET-2L-aligned-CE-Hanford"
detnames = committed_networks()[NETWORK]
observation_time = 1.0  # [yr]; cancels in S_h, kept for the likelihood scale

# Redshift grid for the cosmology integrals (and MD normalization)
minimum_redshift = 0.3
maximum_redshift = 20.0
n_grid = 256  # grid points for cosmology integrals / MD normalization

# Frequency band for the analysis
f_min = 2
f_max = 4096

# Fiducial parameters and the prior on each, straight from config/fiducials.json
# and config/priors.json -- the same tables every committed run merges. These
# used to be hand-written here and had drifted: `local_merger_rate` carried a
# Uniform(7.6, 250) prior that excluded its own fiducial of 770, and `Omega_m`
# a broad uniform where the analysis assumes a Planck-tight normal. Reading
# them means this notebook samples what a run samples.
fiducials = committed_fiducials()
hyperprior_dists = committed_priors()

sampled_params = {"H0"}

priors = dict(hyperprior_dists)
fixed_params = {k: v for k, v in fiducials.items() if k not in sampled_params}

# %% [markdown]
# ## Loading the waveform catalogs
#
# The synthetic observation uses an independent fiducial injection catalog.
# The model uses a guarded proposal catalog carrying its analytic proposal
# redshift log-density.
#
# The catalog is an `astrogwb_catalog` HDF5 file containing:
#
# - `frequencies` — shape `(nfreq,)`, the FFT frequency grid (Hz).
# - `polarization_power` — shape `(nfreq, nsamples)`, the on-disk
#   $|\tilde{h}_+|^2 + |\tilde{h}_\times|^2$ power;
# - source parameters, exposed as `catalog.source_parameters`.

# %%
# The two catalogs come from a run's own config layers, merged here the same
# way the workflow merges them, so the notebook composes exactly what that run
# composes. `assemble_run` addresses a run by name -- the workflow passes the
# same layers on argv instead, but neither reads an intermediate artifact.
RUN_CONFIG = build_run_config(assemble_run(*REFERENCE_RUN))
injection_catalog = load_run_catalog(INJECTION_CATALOG_PATH, label="injection")
proposal_catalog = load_run_catalog(PROPOSAL_CATALOG_PATH, label="proposal")

# The frequency band and redshift grid stay the notebook's own knobs rather
# than the reference run's, so the settings cell above stays live -- they are
# deliberately explorable here, unlike the fiducials and priors, which have one
# home now and are read from it. `RUN_CONFIG` is still loaded for its catalogs.
analysis_grid = AnalysisGrid(
    observation_time=observation_time,
    f_min=f_min,
    f_max=f_max,
    minimum_redshift=minimum_redshift,
    maximum_redshift=maximum_redshift,
    n_grid=n_grid,
)
# The target population -- source model and merger rate together -- bound to
# the analysis grid once: a partial hashes by identity, so rebuilding one per
# step would retrace the whole model.
target_settings = {
    "z_min": minimum_redshift,
    "z_max": maximum_redshift,
    "n_grid": n_grid,
}
target = build_population("bns_md_modified_propagation", **target_settings)

# One call does every step the headless runner does: restrict both catalogs to
# the analysis window (samples *and* recorded density together), build the
# fiducial observation from the injection catalog's own population, build the
# effective PSD and the band mask, and prepare the importance arrays against the
# proposal catalog's own recorded density. Nothing here restates the proposal:
# the file carries it.
inputs = prepare_inference_inputs(
    injection_catalog,
    proposal_catalog,
    grid=analysis_grid,
    detectors=detnames,
    target=target,
)
observation = inputs.observation
proposal = inputs.proposal
spectral_density_fn = inputs.spectral_density_fn

frequencies = observation.frequencies
df = observation.df
mask = observation.frequency_mask
effective_psd_arr = inputs.effective_psd
samples = {name: jnp.asarray(v) for name, v in proposal.source_parameters.items()}
n_freq, n_samples = proposal.polarization_power.shape
print(f"loaded proposal: n_frequency_bins={n_freq} n_proposal_samples={n_samples}")
print("band bins:", int(jnp.sum(mask)), "of", frequencies.shape[0])

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
# `prepare_inference_inputs` already built both above: uncovered bins have an
# infinite effective PSD and would contribute a constant -inf to the
# log-density, so they are dropped along with the out-of-band ones.


# %%
def plot_effective_psd(
    frequencies: jax.Array,
    effective_psd: jax.Array,
    band_mask: jax.Array,
    *,
    color: str = "black",
):
    fig, ax = plt.subplots()
    ax.loglog(frequencies, effective_psd, color=color, ls="--")
    ax.loglog(frequencies[band_mask], effective_psd[band_mask], color=color)
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
# In our implementation, we calculate $S_h(f, \Lambda)$ with an importance
# sampling estimator over a guarded proposal distribution $q(\theta)$:
#
# $$
# S_h(f, \Lambda) = \frac{N(\Lambda)}{T}\frac{1}{N_{\mathrm{inj}}} \sum_{i=1}^{N_{\mathrm{inj}}} \omega_i \left [ |\tilde{h}_+(f, \theta_i)|^2 + |\tilde{h}_\times (f, \theta_i)|^2 \right ],
# $$
#
# The proposal waveforms are pre-computed once, while an independent fiducial
# catalog generates the synthetic observation. The astrophysical +
# cosmological model specifies two things:
#
# - The volume-integrated merger rate $N(\Lambda)/T$;
# - The importance weights $\omega_i$.
#
# ### Calculating the importance weights
#
# The importance weights are proportional to the ratio of probabilities at the particular sample points,
#
# $$
# \omega_i \propto \frac{p(\theta_i | \Lambda)}{q(\theta_i)}.
# $$
#
# While the single-event parameter samples are fixed, a different cosmology will change the amplitude of the waveforms due to the $\propto 1 / d_L$ dependence. Therefore, the importance weights must be rescaled by the inverse-squared distance ratio:
#
# $$
# \omega_i = \frac{p(\theta_i | \Lambda)}{q(\theta_i)} \frac{d_L(z, \Lambda_0)^2}{d_L(z, \Lambda)^2}
# $$
#
# ### Taking into account modified propagation
#
# When considering effects of deviation from GR on the propagation of gravitational-waves, we can generalize the above relation to
#
# $$
# \omega_i = \frac{p(\theta_i | \Lambda)}{q(\theta_i)} \frac{d_{GW}(z, \Lambda_0)^2}{d_{GW}(z, \Lambda)^2}
# $$
#
# which now also encodes the effect of modified propagation.
#
# ### Implementing the model
#
# The pipeline expresses this as a *source model*: one NumPyro declaration
# whose sample sites are the catalog's stored columns and whose deterministic
# sites include the luminosity distance governing waveform amplitude. One
# isolated execution supplies both the source density and that distance; the
# observer-frame total merger rate is a separate merger-rate function.
#
# `build_importance_spectrum` reads a fixed catalog once and binds it to a
# target source model. The denominator $q(\theta_i)$ is not configured
# anywhere: it is the proposal catalog's *own* recorded source model,
# evaluated at the parameters it was drawn at. The reference distance
# $d_{GW}(z, \Lambda_0)$ is the distance column the file already holds -- the
# one its stored power was generated at -- never a freshly interpolated
# cosmology table.

# %% [markdown]
# ## Visualizing $\Omega_{\mathrm{GW}}(f)$
#
# In the cell below, we plot $\Omega_{GW}(f, \Lambda_0)$.


# %%
def plot_omegagw(
    spectral_density: jax.Array,
    frequencies: jax.Array,
    band_mask: jax.Array,
    *,
    hubble_constant: float,
    color: str = "black",
    ymin: float = 1e-15,
):
    omega_gw = omega_gw_from_spectral_density(
        spectral_density, frequencies, hubble_constant=hubble_constant
    )
    band_frequencies = frequencies[band_mask]
    band_omega_gw = omega_gw[band_mask]
    pos = band_omega_gw > 0.0
    fig, ax = plt.subplots()
    ax.loglog(
        np.asarray(band_frequencies[pos]),
        np.asarray(band_omega_gw[pos]),
        color=color,
    )
    ax.set_xlabel(r"$f\ \mathrm{(Hz)}$")
    ax.set_ylabel(r"$\Omega_{\mathrm{GW}}(f)$")
    ax.set_ylim(ymin, None)
    return fig


rate0 = observation.total_merger_rate
observed_spectral_density = observation.spectral_density
plot_omegagw(
    observed_spectral_density,
    frequencies,
    mask,
    hubble_constant=fiducials["H0"],
    color="black",
    ymin=1e-15,
)

# The arrays the likelihood is evaluated against: the observed spectrum, the
# per-bin scale, and the analysis-band mask, all on the catalog's frequency
# grid. The bound spectrum holds the full-grid power; masking the source
# samples would silently truncate the population, so it never happens.
model_kwargs = inputs.model_kwargs()
observed_spectral_density = model_kwargs["observed_spectral_density"]

# %% [markdown]
# ## Running the MCMC
#
# We run the NUTS sampler as implemented in the `numpyro` python package. The
# band reaches the model as a boolean mask over the catalog grid rather than as
# a compressed array, so re-running on a sub-band (`inputs.model_kwargs(fmax=...)`)
# reuses this compiled sampler instead of recompiling it.

# %%
base_model = partial(
    gwb_spectral_density_model,
    spectral_density_fn=spectral_density_fn,
    priors=priors,
)
model = handlers.block(
    handlers.condition(base_model, data=fixed_params),
    hide=list(fixed_params),
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
    chain_method=chain_method,
)
rng_key = jax.random.PRNGKey(seed)
mcmc.run(
    rng_key,
    **model_kwargs,
    extra_fields=("num_steps", "accept_prob", "diverging"),
)
mcmc.print_summary()

# %% [markdown]
# ## Saving the run
#
# We convert to an `xarray.DataTree` via `arviz_base` and write it to NetCDF,
# alongside a small JSON record of the run configuration.

# %%


if IN_COLAB:
    out_dir = Path("chains")
else:
    out_dir = ROOT_DIR / "chains"
out_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
params_suffix = "-".join(sampled_params)
det_suffix = ",".join(detnames)
base = f"chains-{params_suffix}-det={det_suffix}-seed{seed}-{timestamp}"

inference_data = azb.from_numpyro(mcmc)
inference_data.to_netcdf(out_dir / f"{base}.nc")

run_config = {
    "injection_catalog_path": str(INJECTION_CATALOG_PATH),
    "proposal_catalog_path": str(PROPOSAL_CATALOG_PATH),
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
