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
# # Importance-weight grids for two-parameter combinations
#
# For each hard-coded two-parameter combination, this notebook builds a
# prior-support grid via the inverse CDF of each parameter's prior, evaluates
# the BNS Madau–Dickinson modified-propagation
# `merger_rate_and_log_weights` callback on that 2D grid, and plots a heatmap
# of the relative effective sample size
# $N_{\mathrm{eff}}/N = (\sum_i w_i)^2 / (N\sum_i w_i^2)$ from the
# non-log importance weights $w_i=\mathrm{e}^{\log w_i}$ (same definition as
# `importance_relative_ess` in the NumPyro model).
#
# Combinations (for now hard-coded):
#
# 1. $H_0$ + $\Omega_m$ — uniform $H_0$ prior from `mcmc.sweeps.toml`;
#    $\Omega_m\sim\mathrm{Uniform}(0.05, 0.95)$;
# 2. $\Xi_0$ + $n$ — uniform priors from `mcmc.sweeps.toml`.
#
# Point `CATALOG_PATH` at a polarization-power catalog that provides
# `redshift` and `luminosity_distance` source parameters (same schema as the
# other paper notebooks).

# %% [markdown]
# ## Imports and JAX configuration

# %%
from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
from _paper_style import CATEGORY, TRUTH, use_paper_style
from matplotlib.axes import Axes as MplAxes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.projections import register_projection
from pluscross import load_catalog

from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
    make_merger_rate_and_log_weights_fn,
)
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.paths import paper_project_root
from astrogwb_paper.priors import build_prior

# gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes.
# Restore the standard projection for consistent notebook plotting.
register_projection(MplAxes)
jax.config.update("jax_enable_x64", True)

# %config InlineBackend.figure_format = "retina"

# %% [markdown]
# ## Top-level configuration
#
# `eps` insets each prior grid from the extreme quantiles via `Distribution.icdf`;
# `npoints` is the number of samples along each axis.

# %%
ROOT = paper_project_root()
SWEEP_CONFIG_PATH = ROOT / "configs/mcmc.sweeps.toml"
BASE_CONFIG_PATH = ROOT / "configs/mcmc.base.toml"
CATALOG_PATH = ROOT / "out/catalogs/bns-n16384-df1.h5"

eps = 1e-3
npoints = 64
chunk_size = 64

z_min = 0.0
z_max = 20.0
n_redshift_grid = 256

H0_LABEL = r"$H_0\,[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$"
OMEGA_M_LABEL = r"$\Omega_m$"
XI_0_LABEL = r"$\Xi_0$"
XI_N_LABEL = r"$n$"
RELATIVE_ESS_LABEL = r"$N_{\mathrm{eff}} / N_{\mathrm{inj}}$"

PARAM_LABELS = {
    "H0": H0_LABEL,
    "Omega_m": OMEGA_M_LABEL,
    "xi_0": XI_0_LABEL,
    "xi_n": XI_N_LABEL,
}

use_paper_style()

# %% [markdown]
# ## Priors and fiducials
#
# Read the uniform $H_0$, $\Xi_0$, and $n$ priors from the sweep library.
# $\Omega_m$ uses an explicit $\mathrm{Uniform}(0.05, 0.95)$ as requested
# (matching `[priors.Omega_m.uniform]` in the sweep file, rather than the
# narrow normal used by the $H_0$–$\Omega_m$ MCMC analysis).

# %%
sweep = load_mapping(SWEEP_CONFIG_PATH)
base = load_mapping(BASE_CONFIG_PATH)
fiducials: dict[str, float] = dict(base["fiducials"])

h0_prior = build_prior(sweep["priors"]["H0"]["uniform"])
omega_m_prior = dist.Uniform(0.05, 0.95)
xi_0_prior = build_prior(sweep["priors"]["xi_0"]["uniform"])
xi_n_prior = build_prior(sweep["priors"]["xi_n"]["uniform"])

COMBOS: tuple[
    tuple[tuple[str, dist.Distribution], tuple[str, dist.Distribution], str], ...
] = (
    (("H0", h0_prior), ("Omega_m", omega_m_prior), "cosmology"),
    (("xi_0", xi_0_prior), ("xi_n", xi_n_prior), "modified_propagation"),
)

# %% [markdown]
# ## Catalog and weight callback
#
# Only the catalog source parameters enter the weight callback; polarization
# power is not needed for this diagnostic.

# %%
catalog = load_catalog(CATALOG_PATH)
samples = {
    name: jnp.asarray(values) for name, values in catalog.source_parameters.items()
}
del catalog

missing = [name for name in ("redshift", "luminosity_distance") if name not in samples]
if missing:
    raise ValueError(
        "catalog samples are missing required parameter(s): " + ", ".join(missing)
    )

n_samples = int(np.asarray(samples["redshift"]).shape[0])
print(f"loaded catalog samples: n_proposal_samples={n_samples}")

# %%
z_grid = jnp.linspace(z_min, z_max, n_redshift_grid)
_, _, proposal_logprob = compute_merger_rate_distance_and_logprob(
    fiducials, samples, redshift_grid=z_grid
)
merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
    fiducials=fiducials,
    redshift_grid=z_grid,
    proposal_logprob=proposal_logprob,
)

# %% [markdown]
# ## Grid helpers
#
# For a prior $p$, the 1D grid is the quantile interval $[F^{-1}(\varepsilon),
# F^{-1}(1-\varepsilon)]$ sampled uniformly with `npoints` nodes. On the 2D
# mesh we evaluate the relative ESS at each node (chunked to bound peak
# memory).


# %%
def prior_grid(prior: dist.Distribution, *, eps: float, npoints: int) -> jax.Array:
    """Linspace between the ``eps`` and ``1 - eps`` prior quantiles."""
    low, high = prior.icdf(jnp.asarray([eps, 1.0 - eps]))
    return jnp.linspace(low, high, npoints)


def relative_ess(log_weights: jax.Array) -> jax.Array:
    """Relative ESS $(\\sum w)^2 / (N \\sum w^2)$ from log-importance weights."""
    weights = jnp.exp(log_weights)
    return jnp.sum(weights) ** 2 / (weights.shape[0] * jnp.sum(weights**2))


def _category_cmap(category: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"{category}_relative_ess",
        ["#ffffff", CATEGORY[category]],
    )


def evaluate_relative_ess_grid(
    axis0: tuple[str, jax.Array],
    axis1: tuple[str, jax.Array],
    *,
    constants: Mapping[str, float],
    chunk_size: int,
) -> jax.Array:
    """Return relative ESS on the Cartesian product of ``axis0/1``."""
    name0, grid0 = axis0
    name1, grid1 = axis1
    mesh0, mesh1 = jnp.meshgrid(grid0, grid1, indexing="ij")
    points = jnp.stack([mesh0.ravel(), mesh1.ravel()], axis=-1)

    def _relative_ess_at_point(point: jax.Array) -> jax.Array:
        params = {**constants, name0: point[0], name1: point[1]}
        _, log_weights = merger_rate_and_log_weights_fn(params, samples)
        return relative_ess(log_weights)

    ess = jax.lax.map(_relative_ess_at_point, points, batch_size=chunk_size).reshape(
        mesh0.shape
    )
    return jax.block_until_ready(ess)


def plot_relative_ess_heatmap(
    axis0: tuple[str, jax.Array],
    axis1: tuple[str, jax.Array],
    relative_ess_grid: jax.Array,
    *,
    category: str,
    fiducials: Mapping[str, float],
) -> Figure:
    """Heatmap of relative ESS over a two-parameter grid."""
    name0, grid0 = axis0
    name1, grid1 = axis1
    grid0_np = np.asarray(grid0, dtype=np.float64)
    grid1_np = np.asarray(grid1, dtype=np.float64)
    ess_np = np.asarray(relative_ess_grid, dtype=np.float64)

    # ``matshow`` uses image indexing (row, col) = (y, x); transpose so that
    # ``ess[i, j]`` at ``(grid0[i], grid1[j])`` lands on the correct axes.
    dx = 0.5 * (grid0_np[1] - grid0_np[0]) if grid0_np.size > 1 else 0.5
    dy = 0.5 * (grid1_np[1] - grid1_np[0]) if grid1_np.size > 1 else 0.5
    extent = (
        float(grid0_np[0] - dx),
        float(grid0_np[-1] + dx),
        float(grid1_np[0] - dy),
        float(grid1_np[-1] + dy),
    )

    fig, ax = plt.subplots()
    image = ax.matshow(
        ess_np.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=_category_cmap(category),
    )
    ax.xaxis.set_ticks_position("bottom")
    ax.axvline(fiducials[name0], **TRUTH)
    ax.axhline(fiducials[name1], **TRUTH)
    ax.plot(fiducials[name0], fiducials[name1], marker="s", **TRUTH)
    ax.set_xlabel(PARAM_LABELS.get(name0, name0))
    ax.set_ylabel(PARAM_LABELS.get(name1, name1))
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(RELATIVE_ESS_LABEL)
    return fig


def combo_constants(
    param_names: Sequence[str], fiducials: Mapping[str, float]
) -> dict[str, float]:
    """Fiducials with the scanned parameters removed."""
    return {key: value for key, value in fiducials.items() if key not in param_names}


# %% [markdown]
# ## Combination 1: $H_0$–$\Omega_m$ grid

# %%
(name_h0, prior_h0), (name_om, prior_om), category_cosmo = COMBOS[0]
grid_h0 = prior_grid(prior_h0, eps=eps, npoints=npoints)
grid_om = prior_grid(prior_om, eps=eps, npoints=npoints)
constants_h0_om = combo_constants((name_h0, name_om), fiducials)
print(
    f"{name_h0} grid: [{float(grid_h0[0]):.4g}, {float(grid_h0[-1]):.4g}] "
    f"({npoints} pts); "
    f"{name_om} grid: [{float(grid_om[0]):.4g}, {float(grid_om[-1]):.4g}] "
    f"({npoints} pts)"
)

# %%
relative_ess_h0_om = evaluate_relative_ess_grid(
    (name_h0, grid_h0),
    (name_om, grid_om),
    constants=constants_h0_om,
    chunk_size=chunk_size,
)
print(
    f"H0–Omega_m relative ESS: "
    f"min={float(relative_ess_h0_om.min()):.4g} "
    f"max={float(relative_ess_h0_om.max()):.4g}"
)

# %% [markdown]
# ## Combination 2: $\Xi_0$–$n$ grid

# %%
(name_xi0, prior_xi0), (name_xin, prior_xin), category_xi = COMBOS[1]
grid_xi0 = prior_grid(prior_xi0, eps=eps, npoints=npoints)
grid_xin = prior_grid(prior_xin, eps=eps, npoints=npoints)
constants_xi = combo_constants((name_xi0, name_xin), fiducials)
print(
    f"{name_xi0} grid: [{float(grid_xi0[0]):.4g}, {float(grid_xi0[-1]):.4g}] "
    f"({npoints} pts); "
    f"{name_xin} grid: [{float(grid_xin[0]):.4g}, {float(grid_xin[-1]):.4g}] "
    f"({npoints} pts)"
)

# %%
relative_ess_xi = evaluate_relative_ess_grid(
    (name_xi0, grid_xi0),
    (name_xin, grid_xin),
    constants=constants_xi,
    chunk_size=chunk_size,
)
print(
    f"Xi0–n relative ESS: "
    f"min={float(relative_ess_xi.min()):.4g} "
    f"max={float(relative_ess_xi.max()):.4g}"
)

# %% [markdown]
# ## Plot: $H_0$–$\Omega_m$ relative-ESS heatmap

# %%
plot_relative_ess_heatmap(
    (name_h0, grid_h0),
    (name_om, grid_om),
    relative_ess_h0_om,
    category=category_cosmo,
    fiducials=fiducials,
)

# %% [markdown]
# ## Plot: $\Xi_0$–$n$ relative-ESS heatmap

# %%
plot_relative_ess_heatmap(
    (name_xi0, grid_xi0),
    (name_xin, grid_xin),
    relative_ess_xi,
    category=category_xi,
    fiducials=fiducials,
)
