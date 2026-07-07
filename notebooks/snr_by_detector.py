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
# # Matched-filter SNR by detector network
#
# The fiducial astrophysical SGWB (same proposal population as `notebooks/mcmc.py`)
# has a spectral density $S_h(f)$ that depends only on the source population and
# cosmology — the polarization-power catalog, the importance weights, and the total
# merger rate. It does **not** depend on the detector network. Only the effective PSD
# $S_{\mathrm{eff}}(f)$ varies from one network to another.
#
# This notebook exploits that factorization: it computes $S_h(f)$ **once** at the
# fiducial point, then loops over the candidate detector networks defined in
# `scripts/generate_mcmc_configs.py`, computing only `effective_psd` and the
# resulting matched-filter (cross-correlation) SNR for each, and tabulates the
# results.

# %% [markdown]
# ## Imports and JAX configuration

# %%
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import pandas as pd

from astrogwb.gwb import (
    spectral_density,
    frequency_mask as make_frequency_mask,
    spectral_snr,
)
from astrogwb.detector import load_sensitivity_map, effective_psd
from astrogwb.waveform import load_polarization_power_catalog
from astrogwb.utils import years_to_seconds

jax.config.update("jax_enable_x64", True)


# %% [markdown]
# ## Pipeline configuration
#
# Mirrors `notebooks/mcmc.py`'s fiducial configuration exactly (same catalog,
# cosmology grid, frequency band, and fiducial parameter point).


# %%
def get_root_dir() -> Path:
    return Path.cwd().parent


ROOT_DIR = get_root_dir()
CATALOG_PATH = ROOT_DIR / "out/bns_polarization_power_catalog.npz"

sys.path.insert(0, str(ROOT_DIR / "scripts"))
from generate_mcmc_configs import DETECTOR_NETWORKS

observation_time = 1.0  # [yr]

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

# %% [markdown]
# ## Loading the waveform catalog

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
# ## Fiducial spectral density (computed once)
#
# $S_h(f)$ depends only on the source population + cosmology, so we evaluate it a
# single time here at the fiducial point, using unit importance weights (the
# proposal *is* the fiducial population).

# %%
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_proposal_logpdf,
    make_merger_rate_and_log_weights_fn,
)

z_grid = jnp.linspace(z_min, z_max, n_grid)

log_p_proposal = compute_proposal_logpdf(
    samples["redshift"], z_grid=z_grid, fiducials=fiducials
)

merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
    z_grid=z_grid,
    proposal_log_pdf=log_p_proposal,
    fiducial_xi_0=fiducials["xi_0"],
    fiducial_xi_n=fiducials["xi_n"],
)

rate0, _ = merger_rate_and_log_weights_fn(fiducials, samples)
ones_weights = jnp.ones((n_samples,))
S_h = spectral_density(
    polarization_power, ones_weights, rate0, average_mode="analytic_inclination"
)

mask = make_frequency_mask(frequencies, fmin=f_min, fmax=f_max)
df = jnp.mean(jnp.diff(frequencies))
obs_sec = years_to_seconds(observation_time)
print("band bins:", int(jnp.sum(mask)), "of", frequencies.shape[0])

# %% [markdown]
# ## SNR sweep over detector networks
#
# Only `effective_psd` changes per network; `S_h` is reused unchanged. Applying
# `mask` enforces the $[f_{\min}, f_{\max}]$ analysis band (out-of-band bins already
# carry $S_{\mathrm{eff}} = \infty$ and drop out on their own).

# %%
rows = []
for label, dets in DETECTOR_NETWORKS.items():
    sensitivities = load_sensitivity_map(dets)
    eff = jnp.asarray(effective_psd(frequencies, list(dets), sensitivities))
    snr = float(spectral_snr(S_h[mask], eff[mask], obs_sec, df))
    rows.append(
        {
            "network": label,
            "detectors": ",".join(dets),
            "n_detectors": len(dets),
            "snr": snr,
        }
    )

# %% [markdown]
# ## Results table
#
# A pandas `DataFrame` is the natural fit for this small labeled table; call
# `.to_xarray()` on it if an `xarray.Dataset` is more convenient downstream.

# %%
df_snr = pd.DataFrame(rows).set_index("network").sort_values("snr", ascending=False)
print(df_snr.to_string(float_format=lambda x: f"{x:.2f}"))
