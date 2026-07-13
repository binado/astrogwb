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
# # Compare $\Omega_{\mathrm{GW}}$ across waveform catalogs
#
# Load several pluscross waveform catalogs, evaluate the fiducial spectral density
# for each (same importance-sampling pipeline as `mcmc.py`), convert to
# $\Omega_{\mathrm{GW}}(f)$, and overlay the spectra on one plot.

# %% [markdown]
# ## Imports and JAX configuration

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pluscross import load_catalog

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
from astrogwb.utils import repo_root
from astrogwb.waveform import polarization_power as compute_polarization_power

jax.config.update("jax_enable_x64", True)
# %config InlineBackend.figure_format = 'retina'

# %% [markdown]
# ## Pipeline configuration
#
# Edit `CATALOG_PATHS` and `LABELS` to choose which catalogs to compare.

# %%
ROOT_DIR = repo_root()
CATALOG_PATHS = [
    ROOT_DIR / "out/bns_waveforms_df=1Hz.h5",
    ROOT_DIR / "out/bns_waveforms_df=2Hz.h5",
]
LABELS = ["df = 1 Hz", "df = 2 Hz"]

assert len(CATALOG_PATHS) == len(LABELS), (
    "CATALOG_PATHS and LABELS must have equal length"
)

# Redshift grid for the cosmology integrals (and MD normalization)
z_min = 0.0
z_max = 20.0
n_grid = 256

# Frequency band for the plot
f_min = 2
f_max = 4096

# Fiducial parameters (H0 / Omega_m / xi / local_merger_rate from mcmc.example.toml;
# gamma / kappa / z_peak as specified for this comparison)
fiducials = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "xi_0": 1.0,
    "xi_n": 1.91,
    "gamma": 1.42,
    "kappa": 4.63,
    "z_peak": 1.84,
    "local_merger_rate": 161.0,
}

# %% [markdown]
# ## Fiducial $\Omega_{\mathrm{GW}}$ per catalog
#
# For each catalog: load with pluscross, reduce to polarization power, build the
# Madau–Dickinson + modified-propagation weight callback at the fiducials, then
# contract with unit weights (proposal = target at $\Lambda_0$).

# %%
z_grid = jnp.linspace(z_min, z_max, n_grid)
spectra: list[tuple[str, jax.Array, jax.Array, jax.Array]] = []

for path, label in zip(CATALOG_PATHS, LABELS, strict=True):
    catalog = load_catalog(path)
    frequencies = jnp.asarray(catalog.frequencies)
    polarization_power = jnp.asarray(compute_polarization_power(catalog))
    samples = {name: jnp.asarray(v) for name, v in catalog.source_parameters.items()}
    del catalog

    assert "redshift" in samples, f"{path}: catalog samples must include 'redshift'"
    assert "luminosity_distance" in samples, (
        f"{path}: catalog samples must include 'luminosity_distance'"
    )

    n_freq, n_samples = polarization_power.shape
    print(f"{label}: n_frequency_bins={n_freq} n_proposal_samples={n_samples}")

    z_samples = jnp.asarray(samples["redshift"])
    log_p_proposal = compute_proposal_logpdf(
        z_samples, z_grid=z_grid, fiducials=fiducials
    )
    merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
        z_grid=z_grid,
        proposal_log_pdf=log_p_proposal,
        fiducial_xi_0=fiducials["xi_0"],
        fiducial_xi_n=fiducials["xi_n"],
    )

    rate0, log_weights0 = merger_rate_and_log_weights_fn(fiducials, samples)
    weights0 = jnp.exp(log_weights0)
    sh = spectral_density(
        polarization_power,
        weights0,
        rate0,
        average_mode="analytic_inclination",
    )
    omega_gw = omega_gw_from_spectral_density(sh, frequencies)
    mask = make_frequency_mask(frequencies, fmin=f_min, fmax=f_max)
    spectra.append((label, frequencies, omega_gw, mask))

# %% [markdown]
# ## Overlay plot

# %%
ymin = 1e-15
fig, ax = plt.subplots()
for label, frequencies, omega_gw, mask in spectra:
    pos = omega_gw > 0.0
    ax.loglog(
        np.asarray(frequencies[mask & pos]),
        np.asarray(omega_gw[mask & pos]),
        label=label,
    )
ax.set_xlabel(r"$f\ \mathrm{(Hz)}$")
ax.set_ylabel(r"$\Omega_{\mathrm{GW}}(f)$")
ax.set_ylim(ymin, None)
ax.legend()
fig
