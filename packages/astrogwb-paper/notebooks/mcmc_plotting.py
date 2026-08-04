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
# ## MCMC plotting
#
# This notebook loads a chain file stored in netCDF format as an `arviz.InferenceData` (`xarray.DataTree`) object. We make several diagnostic plots for the MCMC with `arviz_plots`, and a corner plot with `corner.corner`.

# %%
from __future__ import annotations

from pathlib import Path

import corner
import xarray as xr
from arviz_base.labels import MapLabeller
import arviz_plots as azp

from astrogwb_paper.paths import paper_project_root

# %config InlineBackend.figure_format = "retina"
azp.style.use("arviz-variat")


# %%
def load_inference_datatree(path: Path) -> xr.DataTree:
    """Open InferenceData-on-disk as an xarray DataTree (matches `azb.from_numpyro` round-trip)."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return xr.open_datatree(path, engine="h5netcdf")
    return xr.open_datatree(path)


def infer_scalar_var_names(dt: xr.DataTree, *, group: str = "posterior") -> list[str]:
    """Variables with only `(chain, draw)` dimensions — same role as `settings.active_params` in the analysis notebook."""
    ds = dt[group]
    out: list[str] = []
    for name in ds.data_vars:
        dims = set(ds[name].dims)
        if dims == {"chain", "draw"}:
            out.append(name)
    return sorted(out)


# %%
# Copied from `notebooks/analysis_numpyro.ipynb` (fiducial hyperparameters / cosmology).
FIDUCIALS: dict[str, float] = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "chi0": 1.0,
    "chin": 1.91,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
}
VAR_LABELS = {
    "H0": r"$H_0$",
    "Omega_m": r"$\Omega_m$",
    "chi0": r"$\chi_0$",
    "chin": r"$\chi_n$",
    "gamma": r"$\gamma$",
    "kappa": r"$\kappa$",
    "z_peak": r"$z_\mathrm{peak}$",
}
labeller = MapLabeller(var_name_map=VAR_LABELS)
# FIDUCIALS["omega_m"] = FIDUCIALS["Omega_m"] * (FIDUCIALS["H0"] / 100.0) ** 2

# Default: sample file produced by `analysis_numpyro.ipynb` when `settings.outdir` / `settings.label` match.
INFERENCE_DATA_PATH = (
    paper_project_root()
    / "chains/mcmc-H0-Omega_m-det=S1,R1-seed42-20260630-013127.nc"
)

# Set to a non-empty list to override automatic detection (e.g. only cosmology parameters).
VAR_NAMES: list[str] | None = None
PLOT_EXTRA_FIELDS: bool = False

inference_data = load_inference_datatree(INFERENCE_DATA_PATH)
var_names = list(VAR_NAMES) if VAR_NAMES else infer_scalar_var_names(inference_data)
if not PLOT_EXTRA_FIELDS:
    var_names = [n for n in var_names if n in FIDUCIALS]
var_names

# %%
# inference_data
inference_data

# %% [markdown]
# ### Corner plot

# %%
corner.corner(
    inference_data,
    var_names=var_names,
    labeller=labeller,
    divergences=True,
    truths={k: FIDUCIALS[k] for k in var_names if k in FIDUCIALS},
    truth_color="C3",
    quantiles=[0.16, 0.5, 0.84],
)

# %% [markdown]
# ### MCMC diagnostics
#
# Plots: trace, autocorrelation, convergence, ESS, ESS evolution, rank.

# %%
azp.plot_trace_dist(inference_data, var_names=var_names, labeller=labeller)

# %%
azp.plot_autocorr(inference_data, var_names=var_names, max_lag=300, labeller=labeller)

# %%
azp.plot_convergence_dist(inference_data, var_names=var_names, ref_line=True, labeller=labeller)

# %%
azp.plot_ess(inference_data, var_names=var_names, extra_methods=True, labeller=labeller)

# %%
azp.plot_ess_evolution(inference_data, var_names=var_names, extra_methods=True, labeller=labeller)

# %%
azp.plot_rank(inference_data, var_names=var_names, labeller=labeller)
