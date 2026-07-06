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
# # $\Omega_{\mathrm{GW}}(f)$ parameter curves
#
# This notebook plots multiple stochastic gravitational-wave background spectra on a
# single axes. Each curve corresponds to one hyperparameter dict $\Lambda$; the
# importance-sampled spectral density $S_h(f, \Lambda)$ is computed with the same
# pipeline as [`mcmc.py`](mcmc.py) (fixed proposal catalog + analytic weights), then
# converted to $\Omega_{\mathrm{GW}}(f)$.
#
# To run end-to-end, point `CATALOG_PATH` at an `.npz` polarization-power catalog
# (same schema as `mcmc.py`).

# %% [markdown]
# ## Imports and JAX configuration

# %%
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from astrogwb.gwb import (
    frequency_mask as make_frequency_mask,
    omega_gw_from_spectral_density,
    spectral_density,
)
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    make_merger_rate_and_log_weights_fn,
)
from astrogwb.waveform import load_polarization_power_catalog

# gwmock-pop: still needed for the proposal redshift PDF precompute.
from gwmock_pop.distributions.madau_dickinson import madau_dickinson_redshift_pdf

jax.config.update("jax_enable_x64", True)
# %config InlineBackend.figure_format = 'retina'


# %% [markdown]
# ## Pipeline configuration
#
# Defaults match [`mcmc.py`](mcmc.py).

# %%
def get_root_dir() -> Path:
    return Path.cwd().parent


ROOT_DIR = get_root_dir()
CATALOG_PATH = ROOT_DIR / "out/bns_polarization_power_catalog.npz"

# Redshift grid for the cosmology integrals (and MD normalization)
z_min = 0.0
z_max = 20.0
n_grid = 256

# Frequency band for plotting
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

average_mode = "analytic_inclination"

# %% [markdown]
# ## Loading the waveform catalog

# %%
catalog = load_polarization_power_catalog(CATALOG_PATH)

frequencies = jnp.asarray(catalog.frequencies)
polarization_power = jnp.asarray(catalog.polarization_power)
samples = {name: jnp.asarray(v) for name, v in catalog.samples.items()}

assert "redshift" in samples, "catalog samples must include 'redshift' for the weights"
assert "luminosity_distance" in samples, (
    "catalog samples must include 'luminosity_distance' for the weights"
)

mask = make_frequency_mask(frequencies, fmin=f_min, fmax=f_max)
n_freq, n_samples = polarization_power.shape
print(f"loaded catalog: n_frequency_bins={n_freq} n_proposal_samples={n_samples}")
print("band bins:", int(jnp.sum(mask)), "of", frequencies.shape[0])

# %% [markdown]
# ## Importance weights callback
#
# The proposal log-density depends only on the fixed fiducial point, so we evaluate it
# once here and bind it into the canonical factory from
# `astrogwb.importance.models.bns_madau_dickinson_modified_propagation`.

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

merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
    z_grid=jnp.linspace(z_min, z_max, n_grid),
    proposal_log_pdf=log_p_proposal,
    fiducial_xi_0=fiducials["xi_0"],
    fiducial_xi_n=fiducials["xi_n"],
)

# %% [markdown]
# ## Curve specs and computation
#
# Each curve is described by a frozen :class:`ParameterCurveSpec` carrying the full
# hyperparameter dict, a legend label, and optional matplotlib `plot_kwargs`.

# %%
MPC_IN_METERS = 3.0856775814913673e22


def h0_km_s_mpc_to_si(h0_km_s_mpc: float) -> float:
    return h0_km_s_mpc * 1000.0 / MPC_IN_METERS


def with_params(base: Mapping[str, float], **overrides: float) -> dict[str, float]:
    return {**base, **overrides}


@dataclass(frozen=True)
class ParameterCurveSpec:
    params: Mapping[str, float]
    label: str
    plot_kwargs: Mapping[str, Any] = field(default_factory=dict)


def compute_omega_gw_for_params(params: Mapping[str, float]) -> jax.Array:
    total_merger_rate, log_weights = merger_rate_and_log_weights_fn(params, samples)
    weights = jnp.exp(log_weights)
    sh = spectral_density(
        polarization_power,
        weights,
        total_merger_rate,
        average_mode=average_mode,
    )
    return omega_gw_from_spectral_density(
        sh,
        frequencies,
        hubble_constant_si=h0_km_s_mpc_to_si(params["H0"]),
    )


# %% [markdown]
# ## Plotting

# %%
def plot_omega_gw_curves(
    specs: Sequence[ParameterCurveSpec],
    *,
    frequencies: jax.Array,
    mask: jax.Array,
    compute_omega_gw=compute_omega_gw_for_params,
    ax: plt.Axes | None = None,
    ymin: float = 1e-15,
) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        _, ax = plt.subplots()

    for spec in specs:
        omega_gw = compute_omega_gw(spec.params)
        pos = (omega_gw > 0.0) & mask
        ax.loglog(
            np.asarray(frequencies[pos]),
            np.asarray(omega_gw[pos]),
            label=spec.label,
            **spec.plot_kwargs,
        )

    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"$\Omega_{\mathrm{GW}}(f)$")
    ax.set_ylim(ymin, None)
    ax.legend()
    return ax.figure, ax


# %% [markdown]
# ## Example: $H_0$ sweep on one axes
#
# Five values around the fiducial using a relative offset grid (±20% of $|H_0|$),
# analogous to the legacy hyperparameter-sweep notebook but on a single plot.

# %%
def fid_offset_grid(fid: float, *, relative_step: float, n_points: int = 5) -> np.ndarray:
    """Grid centered on ``fid`` with spacing ``relative_step * |fid|``."""
    t = relative_step * abs(fid)
    offsets = np.linspace(-2.0, 2.0, n_points) * t
    return np.asarray(fid + offsets, dtype=np.float64)


RELATIVE_STEP = 0.2
h0_grid = fid_offset_grid(fiducials["H0"], relative_step=RELATIVE_STEP)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, h0_grid.size))

curve_specs = [
    ParameterCurveSpec(
        params=with_params(fiducials, H0=float(h0)),
        label=rf"$H_0 = {h0:.4g}$",
        plot_kwargs={
            "color": "black" if np.isclose(h0, fiducials["H0"]) else color,
            "lw": 2.5 if np.isclose(h0, fiducials["H0"]) else 1.5,
        },
    )
    for h0, color in zip(h0_grid, colors, strict=True)
]

fig, ax = plot_omega_gw_curves(
    curve_specs,
    frequencies=frequencies,
    mask=mask,
    ymin=1e-15,
)
ax.set_title(r"$\Omega_{\mathrm{GW}}(f)$ vs $H_0$ (other parameters at fiducial)")
plt.show()
