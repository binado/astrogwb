# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: astrogwb (3.12.9)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Matched-filter SNR by detector network
#
# A stochastic gravitational-wave background (SGWB) from the superposition of
# compact-binary coalescences (CBCs) can be searched for with a **cross-correlation**
# (matched-filter) analysis. For a given astrophysical model, the central question is:
# how detectable is that background with different future detector networks?
#
# The answer factorizes into a **signal** and a **noise** piece. The astrophysical
# strain spectral density $S_h(f)$ is set by the CBC population, cosmology, and
# propagation assumptions. It does **not** depend on where the detectors sit or how
# they are oriented. What changes from one network to another is the **network
# sensitivity** $S_{\mathrm{eff}}(f)$, which combines the one-sided noise PSDs of
# each instrument with the overlap reduction functions $\Gamma_{ab}(f)$ between
# baselines.
#
# For a diagonal Gaussian noise model, the matched-filter signal-to-noise ratio is
#
# $$
# \mathrm{SNR}^2 = 2 T \Delta f \sum_i \frac{S_{h,i}^2}{S_{\mathrm{eff},i}^2},
# $$
#
# where $T$ is the observation time and $\Delta f$ the frequency bin width. This
# notebook evaluates $S_h(f)$ **once** at a fiducial astrophysical point, then
# recomputes only $S_{\mathrm{eff}}(f)$ for each candidate Einstein Telescope (ET)
# and Cosmic Explorer (CE) network listed below. The population model is the same
# one used for cosmological inference in `notebooks/mcmc.py`.

# %%
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

from astrogwb.config.loading import load_mapping
from astrogwb.gwb import (
    spectral_density,
    frequency_mask as make_frequency_mask,
    spectral_snr,
    omega_gw_from_spectral_density,
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

jax.config.update("jax_enable_x64", True)
# %config InlineBackend.figure_format = 'retina'


# %% [markdown]
# ## Fiducial astrophysical model
#
# We hold the following fixed throughout the comparison:
#
# - **Population:** binary neutron star mergers with a Madau–Dickinson merger-rate
#   history, parameterized by $(\gamma, \kappa, z_{\mathrm{peak}})$ and normalized
#   by the local merger rate.
# - **Cosmology and propagation:** fiducial $H_0$, $\Omega_m$, and modified-propagation
#   parameters $(\xi_0, \xi_n)$.
# - **Observation:** $T$ and the analysis band $[f_{\min}, f_{\max}]$ come from
#   `[analysis]` in [`configs/paper.toml`](../configs/paper.toml) (defaults:
#   $T = 1\,\mathrm{yr}$, $f \in [2, 4096]\,\mathrm{Hz}$).
# - **Detector networks:** site labels live in `[detector_networks]`; which networks
#   to plot is set by the `--networks` argparse default below (edit in Jupyter, or
#   override headless). Promote happy values by updating those defaults.

# %%
_DEFAULT_NETWORKS = [
    "ET-triangular",
    "ET-triangular-CE-Hanford",
    "ET-2L-aligned",
    "ET-2L-aligned-CE-Hanford",
    "ET-2L-misaligned",
    "ET-2L-misaligned-CE-Hanford",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/paper.toml"))
    parser.add_argument(
        "--output-pdf", type=Path, default=Path("figures/snr_by_detector.pdf")
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path("figures/snr_by_detector.csv")
    )
    parser.add_argument(
        "--output-tex", type=Path, default=Path("figures/snr_by_detector.tex")
    )
    parser.add_argument(
        "--output-sigmas-csv",
        type=Path,
        default=Path("figures/snr_by_detector_sigmas.csv"),
    )
    parser.add_argument(
        "--output-sigmas-tex",
        type=Path,
        default=Path("figures/snr_by_detector_sigmas.tex"),
    )
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument(
        "--networks",
        nargs="*",
        default=_DEFAULT_NETWORKS,
        help="Network names from [detector_networks] to include (default: all six).",
    )
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
output_pdf = _resolve_path(args.output_pdf, ROOT_DIR)
output_csv = _resolve_path(args.output_csv, ROOT_DIR)
output_tex = _resolve_path(args.output_tex, ROOT_DIR)
output_sigmas_csv = _resolve_path(args.output_sigmas_csv, ROOT_DIR)
output_sigmas_tex = _resolve_path(args.output_sigmas_tex, ROOT_DIR)
figure_dpi = args.figure_dpi

detector_networks = paper["detector_networks"]
network_names = args.networks or list(detector_networks)
DETECTOR_NETWORKS = {name: tuple(detector_networks[name]) for name in network_names}

analysis = paper["analysis"]
observation_time = analysis["observation_time"]  # [yr]

# Redshift grid for the cosmology integrals (and MD normalization)
cosmology = analysis["cosmology"]
z_min = cosmology["z_min"]
z_max = cosmology["z_max"]
n_grid = cosmology["n_grid"]  # grid points for cosmology integrals / MD normalization

# Frequency band for the analysis
f_min = analysis["f_min"]
f_max = analysis["f_max"]

# Fiducial parameters
fiducials = analysis["fiducials"]
# %% [markdown]
# ## Proposal waveform ensemble
#
# Rather than drawing fresh waveforms for every calculation, we use a fixed Monte
# Carlo ensemble of CBC sources generated at a fiducial parameter point
# $\Lambda_0$. Each realization contributes polarization power
# $|\tilde{h}_+(f, \theta_i)|^2 + |\tilde{h}_\times(f, \theta_i)|^2$ as a function
# of frequency, together with redshift and luminosity distance so that population
# reweighting can be applied analytically. This is the same proposal catalog used
# in `notebooks/mcmc.py`.

# %%
catalog = load_catalog(CATALOG_PATH)

frequencies = jnp.asarray(catalog.frequencies)
polarization_power = jnp.asarray(
    compute_polarization_power(catalog)
)  # (nfreq, nsamples)
samples = {name: jnp.asarray(v) for name, v in catalog.source_parameters.items()}
del catalog

assert "redshift" in samples, "catalog samples must include 'redshift' for the weights"
assert "luminosity_distance" in samples, (
    "catalog samples must include 'luminosity_distance' for the weights"
)

n_freq, n_samples = polarization_power.shape
print(f"loaded catalog: n_frequency_bins={n_freq} n_proposal_samples={n_samples}")

# %% [markdown]
# ## Fiducial spectral density
#
# The strain spectral density of the astrophysical SGWB is the incoherent sum of
# many unresolved CBC signals. With importance sampling over the fixed proposal
# ensemble, it is estimated as
#
# $$
# S_h(f) = \frac{N(\Lambda_0)}{T}\,\frac{1}{N_{\mathrm{inj}}} \sum_{i=1}^{N_{\mathrm{inj}}}
# \omega_i(\Lambda_0)\,\bigl[|\tilde{h}_+(f, \theta_i)|^2 + |\tilde{h}_\times(f, \theta_i)|^2\bigr],
# $$
#
# where $N(\Lambda_0)/T$ is the volume-integrated merger rate at the fiducial point
# and $\omega_i(\Lambda_0)$ are the importance weights. Even at $\Lambda = \Lambda_0$
# these weights are not necessarily unity: the proposal redshift distribution need
# not coincide with the target Madau–Dickinson rate, and the weights also encode
# distance and modified-propagation corrections relative to the proposal draw.
#
# We evaluate $S_h(f)$ once at the fiducial parameters. Only frequency bins where
# the detectors are sensitive contribute appreciably to the SNR integral below.

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

rate0, log_weights = merger_rate_and_log_weights_fn(fiducials, samples)
weights = jnp.exp(log_weights)
observed_spectral_density = spectral_density(
    polarization_power, weights, rate0, average_mode="analytic_inclination"
)

mask = make_frequency_mask(frequencies, fmin=f_min, fmax=f_max)
df = jnp.mean(jnp.diff(frequencies))
obs_sec = years_to_seconds(observation_time)
print("band bins:", int(jnp.sum(mask)), "of", frequencies.shape[0])

# %% [markdown]
# ## Energy density spectrum $\Omega_{\mathrm{GW}}(f)$
#
# It is often convenient to express the background in terms of the fractional energy
# density per logarithmic frequency,
#
# $$
# \Omega_{\mathrm{GW}}(f) = \frac{1}{\rho_c}\,\frac{\mathrm{d}\rho_{\mathrm{GW}}}{\mathrm{d}\ln f},
# $$
#
# which is related to $S_h(f)$ by $\Omega_{\mathrm{GW}}(f) \propto f^3 S_h(f)$. The
# plot below shows the fiducial astrophysical spectrum that each network would
# attempt to detect — a useful sanity check before comparing SNRs.


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
        jnp.asarray(frequencies[mask & pos]),
        jnp.asarray(omega_gw[mask & pos]),
        color=color,
    )
    ax.set_xlabel(r"$f\ \mathrm{(Hz)}$")
    ax.set_ylabel(r"$\Omega_{\mathrm{GW}}(f)$")
    ax.set_ylim(ymin, None)
    return fig


plot_omegagw(observed_spectral_density, frequencies, mask, color="black", ymin=1e-12)
# %% [markdown]
# ## Network sensitivity and cross-correlation SNR
#
# For a network of detectors $a, b, \ldots$, the effective noise PSD that enters the
# cross-correlation search is
#
# $$
# S_{\mathrm{eff}}(f) = \left(\sum_{a,b} \frac{\Gamma_{ab}^2(f)}{S_{n,a}(f)\,S_{n,b}(f)}\right)^{-1/2},
# $$
#
# where $S_{n,a}(f)$ is the one-sided noise PSD of detector $a$ and $\Gamma_{ab}(f)$
# is the overlap reduction function for baselines $a$–$b$, encoding their separation
# and orientation relative to an isotropic, unpolarized background.
#
# More baselines and favourable geometry yield a lower $S_{\mathrm{eff}}$ and hence a
# higher SNR for the same $S_h(f)$. Among the networks below:
#
# - **ET triangular** uses three co-located ET sites, maximizing the number of
#   independent cross-correlations at a single location.
# - **ET two-L** variants place two ET arms in an L-shaped configuration; the
#   aligned and misaligned layouts differ in baseline orientations and hence in
#   $\Gamma_{ab}(f)$.
# - Adding **CE-Hanford** introduces a long Cosmic Explorer arm, extending sensitive
#   baselines and improving low-frequency sensitivity.
#
# $S_h(f)$ is held fixed across this sweep; only $S_{\mathrm{eff}}(f)$ changes.

# %%
AMPLITUDE_VALUES = {
    "H0": fiducials["H0"],
    "xi_0": fiducials["xi_0"],
    "merger_rate": fiducials["local_merger_rate"],
}

rows = []
for label, dets in DETECTOR_NETWORKS.items():
    sensitivities = load_sensitivity_map(dets)
    eff = jnp.asarray(effective_psd(frequencies, list(dets), sensitivities))
    snr = float(spectral_snr(observed_spectral_density[mask], eff[mask], obs_sec, df))
    row = {
        "network": label,
        "detectors": ",".join(dets),
        "n_detectors": len(dets),
        "snr": snr,
    }
    for name, value in AMPLITUDE_VALUES.items():
        row[f"sigma_{name}"] = value / snr
        row[f"rel_sigma_{name}"] = 1 / snr
    rows.append(row)

# %% [markdown]
# ## Comparing networks
#
# The table ranks the candidate networks by matched-filter SNR for the **same**
# astrophysical signal and observation time. A higher SNR means a stronger expected
# cross-correlation detection of the fiducial background; ratios of SNR values
# quantify how much one network improves over another at this fixed model point.

# %%
_snr_cols = ["network", "detectors", "n_detectors", "snr"]
_precision_cols = [
    col for name in AMPLITUDE_VALUES for col in (f"sigma_{name}", f"rel_sigma_{name}")
]
df_all = pd.DataFrame(rows).sort_values("snr", ascending=False).reset_index(drop=True)
df_snr = df_all[_snr_cols]
df_sigmas = df_all[_snr_cols + _precision_cols]

df_snr.style.format({"snr": "{:.2f}", "n_detectors": "{:.0f}"})

# %%
df_sigmas.style.format(
    {
        "snr": "{:.2f}",
        "n_detectors": "{:.0f}",
        **{col: "{:.3g}" for col in _precision_cols},
    }
)

# %%
output_csv.parent.mkdir(parents=True, exist_ok=True)
output_sigmas_csv.parent.mkdir(parents=True, exist_ok=True)
output_pdf.parent.mkdir(parents=True, exist_ok=True)

df_snr.to_csv(output_csv, index=False)
df_sigmas.to_csv(output_sigmas_csv, index=False)

latex_snr = df_snr.to_latex(
    index=False,
    float_format="%.3g",
    caption="Matched-filter SNR by detector network.",
    label="tab:snr_by_detector",
)
output_tex.write_text(latex_snr)

latex_sigmas = df_sigmas.to_latex(
    index=False,
    float_format="%.3g",
    caption="Matched-filter SNR and amplitude-parameter precision by network.",
    label="tab:snr_by_detector_sigmas",
)
output_sigmas_tex.write_text(latex_sigmas)
print(latex_snr)
print(latex_sigmas)

df_plot = df_snr.sort_values("snr", ascending=True)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.barh(df_plot["network"], df_plot["snr"], color="0.25")
ax.set_xlabel("Matched-filter SNR")
ax.set_ylabel("")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(output_pdf, dpi=figure_dpi, bbox_inches="tight")
print("saved table:", output_csv)
print("saved latex:", output_tex)
print("saved sigmas table:", output_sigmas_csv)
print("saved sigmas latex:", output_sigmas_tex)
print("saved figure:", output_pdf)
