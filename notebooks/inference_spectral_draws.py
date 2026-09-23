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
# # H₀ posteriors from simulated spectral-density draws
#
# This notebook simulates independent spectral-density realizations from the
# configured population, treats the first realization as data, and evaluates
# the Gaussian H₀ posterior implied by every remaining realization. The
# likelihood is implemented with JAX and evaluated over both the H₀ grid and
# simulated draws with nested `vmap` calls inside `jit`.

# %% [markdown]
# ## Imports and plotting configuration

# %%
import os
from collections.abc import Mapping
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection
from numpyro.infer import Predictive

from astrogwb.detector import effective_psd, gaussian_bin_scale, load_sensitivity_map
from astrogwb.paper.config import (
    fiducials,
    networks,
    population_model,
    waveform_generator,
)
from astrogwb.paper.config.constants import DEFAULT_NETWORK
from astrogwb.paper.paths import root_dir
from astrogwb.paper.plotting import (
    DETECTOR_NETWORKS,
    TRUTH,
    Network,
    use_paper_style,
)
from astrogwb.sampling import gwb_forward_model, validate_source_model
from astrogwb.utils import years_to_seconds

register_projection(MplAxes)
use_paper_style()
jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Notebook configuration

# %%
ROOT_DIR = root_dir()
SMOKE: bool = os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") == "1"

NUM_DRAWS: int = 2 if SMOKE else 8
RNG_SEED: int = 0
OBSERVATION_TIME: float = 1.0
BATCH_SIZE: int = 128
N_MAX_SIGMA: float = 5.0

H0_LOW: float = 55.0
H0_HIGH: float = 85.0
N_H0: int = 33 if SMOKE else 256

NETWORK_NAME: str = DEFAULT_NETWORK
FIDUCIALS = fiducials(root=ROOT_DIR)
H0_FID: float = FIDUCIALS["H0"]
H0_GRID = jnp.linspace(H0_LOW, H0_HIGH, N_H0)

# %% [markdown]
# ## Configured population and detector network

# %%
NETWORK_CONFIG = networks(root=ROOT_DIR)
NETWORK_LABELS = dict(DETECTOR_NETWORKS)
network = Network(
    NETWORK_NAME,
    NETWORK_LABELS[NETWORK_NAME],
    NETWORK_CONFIG[NETWORK_NAME],
)

GENERATOR = waveform_generator(root=ROOT_DIR)
POPULATION = population_model(root=ROOT_DIR)

print(f"network: {network.label} ({' '.join(network.detectors)})")
print(f"waveform: {GENERATOR.metadata.approximant}")
print(f"frequency resolution: {GENERATOR.metadata.frequency_resolution:g} Hz")

# %% [markdown]
# ## Spectral-density simulator
#
# This follows `scripts/simulate_spectra.py`: the Poisson event count is drawn
# inside `gwb_forward_model`, while `max_events` remains a static capacity for
# the plated source draw and batched waveform reduction.


# %%
def padded_event_capacity(mean_count: float, n_max_sigma: float) -> int:
    """Return a static source-plate capacity deep in the Poisson tail."""
    if mean_count < 0.0:
        raise ValueError("Poisson mean must be non-negative")
    if n_max_sigma < 0.0:
        raise ValueError("n_max_sigma must be non-negative")
    return max(int(np.ceil(mean_count + n_max_sigma * np.sqrt(mean_count))), 1)


def simulate_spectra(
    params: Mapping[str, float],
    *,
    num_draws: int,
    seed: int,
    observation_time: float,
    batch_size: int,
    n_max_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``num_draws`` independent spectral-density realizations."""
    if num_draws <= 0:
        raise ValueError("num_draws must be positive")
    if observation_time <= 0.0:
        raise ValueError("observation_time must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if POPULATION.merger_rate_fn is None:
        raise ValueError("the configured population must declare a merger rate")

    validate_source_model(
        params,
        source_model=POPULATION.source_model,
        generator=GENERATOR,
        rng_key=jax.random.key(seed),
    )

    rate = float(jnp.asarray(POPULATION.merger_rate_fn(params)))
    mean_count = rate * years_to_seconds(observation_time)
    max_events = padded_event_capacity(mean_count, n_max_sigma)

    simulate = Predictive(
        partial(
            gwb_forward_model,
            source_model=POPULATION.source_model,
            merger_rate_fn=POPULATION.merger_rate_fn,
            generator=GENERATOR,
            observation_time=observation_time,
            batch_size=batch_size,
            max_events=max_events,
        ),
        num_samples=num_draws,
        return_sites=("spectral_density", "n_events"),
    )
    draws = simulate(jax.random.key(seed), params)
    frequencies = np.asarray(GENERATOR.frequencies, dtype=np.float64)
    spectra = np.asarray(draws["spectral_density"], dtype=np.float64)
    print(
        f"simulated {spectra.shape[0]} spectra with event counts "
        f"{np.asarray(draws['n_events'], dtype=np.int64)}"
    )
    return frequencies, spectra


# %% [markdown]
# ## Simulate data and model draws

# %%
frequencies, spectra = simulate_spectra(
    FIDUCIALS,
    num_draws=NUM_DRAWS + 1,
    seed=RNG_SEED,
    observation_time=OBSERVATION_TIME,
    batch_size=BATCH_SIZE,
    n_max_sigma=N_MAX_SIGMA,
)

data = jnp.asarray(spectra[0])
model_draws = jnp.asarray(spectra[1:])

# %% [markdown]
# ## Detector noise scale

# %%
sensitivities = load_sensitivity_map(network.detectors)
network_psd = jnp.asarray(effective_psd(frequencies, network.detectors, sensitivities))
scale = gaussian_bin_scale(
    network_psd,
    OBSERVATION_TIME,
    GENERATOR.metadata.frequency_resolution,
)
frequency_mask = jnp.isfinite(scale) & (scale > 0.0)
# Keep excluded bins numerically harmless while retaining the original grid.
safe_scale = jnp.where(frequency_mask, scale, 1.0)

print(f"usable frequency bins: {int(jnp.sum(frequency_mask))}/{frequencies.size}")

# %% [markdown]
# ## JAX likelihood
#
# The model spectrum is evaluated at the fiducial population parameters. H₀
# changes only its amplitude according to
# $S_h(H_0) = (H_0^{\rm fid}/H_0) S_h(H_0^{\rm fid})$.


# %%
def log_posterior(
    h0: jax.Array,
    model: jax.Array,
    data: jax.Array,
    scale: jax.Array,
    frequency_mask: jax.Array,
) -> jax.Array:
    """Return the Gaussian log posterior for one H₀ and one model draw."""
    mean = H0_FID / h0 * model
    log_prob = -0.5 * (((data - mean) / scale) ** 2 + jnp.log(2.0 * jnp.pi * scale**2))
    return jnp.sum(jnp.where(frequency_mask, log_prob, 0.0))


log_posterior_over_h0 = jax.vmap(
    log_posterior,
    in_axes=(0, None, None, None, None),
)
log_posterior_over_draws_and_h0 = jax.jit(
    jax.vmap(
        log_posterior_over_h0,
        in_axes=(None, 0, None, None, None),
    )
)

# %% [markdown]
# ## JIT-compiled posterior evaluation

# %%
compile_start = perf_counter()
log_posteriors = jax.block_until_ready(
    log_posterior_over_draws_and_h0(
        H0_GRID,
        model_draws,
        data,
        safe_scale,
        frequency_mask,
    )
)
compile_elapsed = perf_counter() - compile_start

evaluation_start = perf_counter()
_ = jax.block_until_ready(
    log_posterior_over_draws_and_h0(
        H0_GRID,
        model_draws,
        data,
        safe_scale,
        frequency_mask,
    )
)
evaluation_elapsed = perf_counter() - evaluation_start

print(f"compiled JAX/vmap evaluation: {compile_elapsed:.3f} s")
print(f"cached JAX/vmap evaluation: {evaluation_elapsed:.3f} s")
print(f"log-posterior shape: {log_posteriors.shape}")

# %% [markdown]
# ## H₀ posterior curves

# %%
h0_grid = np.asarray(H0_GRID)
log_posteriors_np = np.asarray(log_posteriors, dtype=np.float64)
posterior_densities = np.exp(
    log_posteriors_np - np.max(log_posteriors_np, axis=1, keepdims=True)
)
posterior_densities /= np.trapezoid(
    posterior_densities,
    h0_grid,
    axis=1,
)[:, None]

fig_posteriors, ax = plt.subplots(figsize=(7.0, 4.5))
for draw_index, posterior in enumerate(posterior_densities, start=1):
    ax.plot(h0_grid, posterior, lw=1.2, label=f"draw {draw_index}")
ax.axvline(H0_FID, **TRUTH)
ax.set_xlabel(r"$H_0\ [\mathrm{km\ s^{-1}\ Mpc^{-1}}]$")
ax.set_ylabel(r"$p(H_0\mid d,\,\mathrm{draw})$")
ax.set_title(f"Simulated spectral-draw posteriors: {network.label}")
ax.legend(fontsize=7, ncol=2)
fig_posteriors

# %% [markdown]
# ## Sample variance across model draws

# %%
model_draws_np = np.asarray(model_draws, dtype=np.float64)
sample_variance = np.var(model_draws_np, axis=0, ddof=1)
positive_variance = sample_variance > 0.0

fig_variance, ax = plt.subplots(figsize=(7.0, 4.5))
ax.loglog(
    frequencies[positive_variance],
    sample_variance[positive_variance],
    color="C0",
)
ax.set_xlabel(r"$f\ [\mathrm{Hz}]$")
ax.set_ylabel(r"Sample variance of $S_h(f)$")
ax.set_title(f"Spectral-density variance across draws: {network.label}")
fig_variance

# %% [markdown]
# ## Relative residual variance across draws
#
# For each draw the residual is measured relative to the ensemble mean at every
# frequency, $S_h / \overline{S_h} - 1$. Its sample variance is the raw variance
# above divided by $\overline{S_h}^2$, so it is dimensionless and comparable
# across the band. Bins where the ensemble mean vanishes (or the variance is
# non-positive) are dropped before the log-log plot.

# %%
mean_spectrum = np.mean(model_draws_np, axis=0)
relative_residual = model_draws_np / mean_spectrum - 1.0
residual_variance = np.var(relative_residual, axis=0, ddof=1)
finite_variance = np.isfinite(residual_variance) & (residual_variance > 0.0)

print(f"usable residual-variance bins: {int(finite_variance.sum())}/{frequencies.size}")

fig_residual_variance, ax = plt.subplots(figsize=(7.0, 4.5))
ax.loglog(
    frequencies[finite_variance],
    residual_variance[finite_variance],
    color="C1",
)
ax.set_xlabel(r"$f\ [\mathrm{Hz}]$")
ax.set_ylabel(r"Sample variance of $S_h / \overline{S_h} - 1$")
ax.set_title(f"Relative spectral-density residual variance: {network.label}")
fig_residual_variance
