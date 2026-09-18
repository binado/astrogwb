# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: astrogwb (3.12.9)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Waveform-approximant spectral draws
#
# Compare stochastic-background spectra made from the *same events* with four
# Ripple frequency-domain approximants. Reusing one PRNG key for each NumPyro
# `Predictive` call makes every call replay the same event count and source
# latent variables; only `WaveformMetadata.approximant` changes. The solid line
# is the median of the retained draws and the shaded region is the 10th--90th
# percentile interval.
#
# Higher-mode waveforms depend on inclination, so the source model is wrapped
# with `with_isotropic_inclination`: each event draws $\iota$ from the isotropic
# law ($\cos\iota$ uniform on $[-1, 1]$). Returning `inclination` also disables
# the analytic $2/5$ face-on-to-isotropic rescaling, which is only valid for
# quadrupole waveforms. The shared PRNG key then replays the same orientations
# for every approximant.
#
# The lower panel shows fractional residuals
# $(S_h^A-S_h^\mathrm{NRTidalv3})/S_h^\mathrm{NRTidalv3}$. Bins where the
# reference is exactly zero are undefined and are masked rather than divided.

# %% [markdown]
# ## Imports and JAX configuration

# %%
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpyro.infer import Predictive

from astrogwb.paper.config import fiducials, population_model, waveform_generator
from astrogwb.paper.config.runs import FIGURES_DIR
from astrogwb.paper.plotting import save_figures, use_paper_style
from astrogwb.populations import with_isotropic_inclination
from astrogwb.sampling import gwb_forward_model
from astrogwb.utils import years_to_seconds

# Configure precision before constructing a JAX array or querying a device.
jax.config.update("jax_enable_x64", True)

ROOT_DIR = Path() if Path("notebooks").is_dir() else Path("..")
use_paper_style(root=ROOT_DIR)
# %config InlineBackend.figure_format = 'retina'

# %% [markdown]
# ## Shared simulation configuration
#
# There is one configuration for the population, hyperparameters, observing
# duration, waveform grid, number of retained draws, batching, capacity-tail
# rule, and seed. The four generators below receive the same grid settings;
# their approximant is their only differing metadata field. Fiducials,
# population, and waveform settings come from the shared
# `astrogwb.paper.config` accessors, so this comparison cannot drift from the
# catalog configuration. Ripple calls its registered tidal model
# `IMRPhenomXAS_NRTidalv3` (lower-case `v`), while plot text uses the
# conventional `IMRPhenomXAS_NRTidalV3` spelling.


# %%
@dataclass(frozen=True)
class ComparisonConfig:
    """All stochastic and numerical choices shared by the comparison."""

    model_kwargs: dict[str, float | int]
    hyperparameters: dict[str, float]
    observation_time: float
    draw_count: int
    batch_size: int
    n_max_sigma: float
    seed: int


CONFIG = ComparisonConfig(
    model_kwargs={"minimum_redshift": 0.3, "maximum_redshift": 20.0, "n_grid": 256},
    hyperparameters=fiducials(root=ROOT_DIR),
    observation_time=1.0,
    draw_count=4,
    batch_size=128,
    n_max_sigma=5.0,
    seed=20250314,
)

APPROXIMANTS = (
    "TaylorF2",
    "IMRPhenomXAS",
    "IMRPhenomHM",
    "IMRPhenomXAS_NRTidalv3",
)
REFERENCE_APPROXIMANT = "IMRPhenomXAS_NRTidalv3"
DISPLAY_LABELS = {
    "TaylorF2": "TaylorF2",
    "IMRPhenomXAS": "IMRPhenomXAS",
    "IMRPhenomHM": "IMRPhenomHM",
    REFERENCE_APPROXIMANT: "IMRPhenomXAS_NRTidalV3 (reference)",
}
COLORS = {
    "TaylorF2": "#0072B2",
    "IMRPhenomXAS": "#E69F00",
    "IMRPhenomHM": "#009E73",
    REFERENCE_APPROXIMANT: "#D55E00",
}
OUTPUT_PATH = ROOT_DIR / FIGURES_DIR / "waveform_approximant_spectra.pdf"


generators = {
    approximant: waveform_generator(root=ROOT_DIR, approximant=approximant)
    for approximant in APPROXIMANTS
}

# %% [markdown]
# ## Draw matched spectra
#
# `Predictive` assigns keys deterministically by sample-site name. Calling the
# same model with the same fixed key therefore reproduces `n_events`, masses,
# redshifts, spins, tidal deformabilities, and inclinations exactly for every
# approximant. Splitting the key in the loop would instead produce unrelated
# catalogs and would confound waveform differences with Monte Carlo variation.
# Non-tidal approximants deliberately do not consume the shared tidal latent
# variables.

# %%
population = population_model(root=ROOT_DIR, **CONFIG.model_kwargs)
source_model = with_isotropic_inclination(population.source_model)
merger_rate_fn = population.merger_rate_fn
if merger_rate_fn is None:
    raise ValueError("configured population cannot simulate event counts")

rate = float(jnp.asarray(merger_rate_fn(CONFIG.hyperparameters)))
mean_count = rate * years_to_seconds(CONFIG.observation_time)
max_events = max(int(np.ceil(mean_count + CONFIG.n_max_sigma * np.sqrt(mean_count))), 1)
shared_key = jax.random.key(CONFIG.seed)

spectral_draws: dict[str, np.ndarray] = {}
event_counts: dict[str, np.ndarray] = {}
for approximant, generator in generators.items():
    predictive = Predictive(
        partial(
            gwb_forward_model,
            source_model=source_model,
            merger_rate_fn=merger_rate_fn,
            generator=generator,
            observation_time=CONFIG.observation_time,
            batch_size=CONFIG.batch_size,
            max_events=max_events,
        ),
        num_samples=CONFIG.draw_count,
        return_sites=("spectral_density", "n_events"),
    )
    result = predictive(shared_key, CONFIG.hyperparameters)
    spectral_draws[approximant] = np.asarray(result["spectral_density"])
    event_counts[approximant] = np.asarray(result["n_events"])

# The identical counts are a cheap explicit check that the stochastic traces
# stayed paired. The fixed-key construction also pairs every named source site.
reference_counts = event_counts[REFERENCE_APPROXIMANT]
for approximant, counts in event_counts.items():
    np.testing.assert_array_equal(
        counts, reference_counts, err_msg=f"unpaired event counts for {approximant}"
    )

# %% [markdown]
# ## Validate the common frequency grid
#
# A pointwise comparison is meaningful only if every generator returns exactly
# the same bins. Fail before plotting if shape or values differ.

# %%
frequencies = np.asarray(generators[REFERENCE_APPROXIMANT].frequencies)
for approximant, generator in generators.items():
    candidate = np.asarray(generator.frequencies)
    np.testing.assert_array_equal(
        candidate,
        frequencies,
        err_msg=f"frequency grid differs for {approximant}",
    )
    if spectral_draws[approximant].shape != (CONFIG.draw_count, frequencies.size):
        raise ValueError(
            f"unexpected spectral shape for {approximant}: "
            f"{spectral_draws[approximant].shape}"
        )

# %% [markdown]
# ## Median spectra and fractional residuals
#
# Both panels summarize all retained draws: lines are draw-wise medians and
# bands span the 10th--90th percentiles. Residuals are computed for each paired
# draw before taking percentiles, preserving event-level comparability. A
# reference bin equal to zero is represented by `NaN` and omitted safely.

# %%
fig, (spectrum_ax, residual_ax) = plt.subplots(
    2,
    1,
    figsize=(7.0, 6.5),
    sharex=True,
    gridspec_kw={"height_ratios": (2.1, 1.0), "hspace": 0.08},
)

reference_draws = spectral_draws[REFERENCE_APPROXIMANT]
for approximant in APPROXIMANTS:
    draws = spectral_draws[approximant]
    median, low, high = np.percentile(draws, (50.0, 10.0, 90.0), axis=0)
    color = COLORS[approximant]
    label = DISPLAY_LABELS[approximant]
    spectrum_ax.loglog(frequencies, median, color=color, label=label)
    spectrum_ax.fill_between(frequencies, low, high, color=color, alpha=0.18)

    residuals = np.full_like(draws, np.nan)
    np.divide(
        draws - reference_draws,
        reference_draws,
        out=residuals,
        where=reference_draws != 0.0,
    )
    residual_median = np.nanmedian(residuals, axis=0)
    residual_low, residual_high = np.nanpercentile(residuals, (10.0, 90.0), axis=0)
    residual_ax.semilogx(frequencies, residual_median, color=color, label=label)
    residual_ax.fill_between(
        frequencies, residual_low, residual_high, color=color, alpha=0.18
    )

spectrum_ax.set_ylabel(r"$S_h(f)\ [\mathrm{Hz}^{-1}]$")
spectrum_ax.legend(loc="best")
spectrum_ax.grid(alpha=0.25)
residual_ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
residual_ax.set_xlabel(r"Frequency $f\ [\mathrm{Hz}]$")
residual_ax.set_ylabel("Fractional\nresidual")
residual_ax.grid(alpha=0.25)
fig.align_ylabels()
save_figures({OUTPUT_PATH: fig}, root=ROOT_DIR)
