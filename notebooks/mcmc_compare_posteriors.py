# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: asgwb (3.12.9)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Comparing 1D posteriors

# %%
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import arviz_stats as azs
import matplotlib.pyplot as plt
import numpy.typing as npt
import xarray as xr

# %config InlineBackend.figure_format = "retina"

# %% [markdown]
# Configuring matplotlib:

# %%
pub_rc = {
    # "figure.figsize": (colwidth, colwidth),
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.labelsize": "medium",
    "axes.unicode_minus": False,
    "axes.titlesize": "medium",
    "figure.labelsize": "medium",
    "figure.titlesize": "medium",
    # Make the legend/label fonts a little smaller
    "legend.fontsize": "small",
    "legend.title_fontsize": "small",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",

    # X-axis ticks
    'xtick.direction': 'in',       # Point ticks inward
    'xtick.minor.visible': True,   # Turn on minor ticks
    'xtick.top': True,             # Draw ticks on the top spine as well
    
    # Y-axis ticks
    'ytick.direction': 'in',       # Point ticks inward
    'ytick.minor.visible': True,   # Turn on minor ticks
    'ytick.right': True,           # Draw ticks on the right spine as well
}
plt.rcParams.update(**pub_rc)

# %% [markdown]
# ## Notebook configuration

# %%
BASE_DIR = Path.cwd().parent
OUT_DIR = BASE_DIR / "figures"

LABELS = [
    r"ET-$\Delta$",
    r"ET-$\Delta +$ CE",
    r"ET-2L",
    r"ET-2L $+$ CE",
    r"ET-2L-$\alpha$",
    r"ET-2L-$\alpha +$ CE",
]

PATHS = [
    "chains/mcmc-H0-det=E1,E2,E3-seed42-20260630-091326.nc",
    "chains/mcmc-H0-det=E1,E2,E3,C1-seed42-20260630-082136.nc",
    "chains/mcmc-H0-det=S1,R1-seed42-20260629-201613.nc",
    "chains/mcmc-H0-det=S1,R1,C1-seed42-20260629-105230.nc",
    "chains/mcmc-H0-det=S2,R2-seed42-20260630-053542.nc",
    "chains/mcmc-H0-det=S2,R2,C1-seed42-20260630-034738.nc"
]

COLORS = ["tab:blue", "tab:orange", "tab:green"]
STYLES = ["-", "--"]
color_styles = list(product(COLORS, STYLES))

VAR_NAME = "H0"
OUT_FILE = OUT_DIR / f"mcmc_compare_posteriors_{VAR_NAME}.pdf"
FIGURE_DPI = 300


# %%
@dataclass(frozen=True)
class PosteriorConfig:
    label: str
    path: str | Path
    color: str
    linestyle: str

    def load_posterior_samples(self, base_dir: Path) -> xr.DataTree:
        fullpath = base_dir / self.path
        return xr.open_datatree(fullpath, engine="h5netcdf")
    
    def get_x_and_prob_arrays(self, base_dir: Path, group: str, var_name: str) -> tuple[npt.NDArray, npt.NDArray]:
        dtree = self.load_posterior_samples(base_dir)
        kde = azs.kde(dtree, group=group, var_names=var_name)[var_name]
        x, prob = kde.sel(plot_axis="x").to_numpy(), kde.sel(plot_axis="y").to_numpy()
        return x, prob

configs = [
    PosteriorConfig(label=label, path=path, color=color, linestyle=linestyle)
    for label, path, (color, linestyle) in zip(LABELS, PATHS, color_styles)
]


# %% [markdown]
# Defining the plotting function:

# %%
def plot_posteriors(
        configs: list[PosteriorConfig],
        var_name: str,
        base_dir: Path, 
        *, 
        group: str = "posterior",
        ax_kwargs: dict[str, Any] | None = None,
        legend_kwargs: dict[str, Any] | None = None
    ):
    fig, ax = plt.subplots()
    for config in configs:
        x, prob = config.get_x_and_prob_arrays(base_dir, group, var_name)
        ax.plot(x, prob, label=config.label, color=config.color, linestyle=config.linestyle)

    ax_kwargs = ax_kwargs or {}
    legend_kwargs = legend_kwargs or {}
    ax.set(**ax_kwargs)
    ax.legend(**legend_kwargs)
    fig.tight_layout()
    return fig


# %% [markdown]
# Plotting:

# %%
ax_kwargs = {
    "xlabel": r"$H_0 \ [\mathrm{km \ s^{-1} \ Mpc^{-1}}]$", 
    "ylabel": "Posterior density"
}
legend_kwargs = {
    "ncol": 3,
    "loc": "lower center",           # anchor point on the legend box
    "bbox_to_anchor": (0.5, 1.02),   # above the axes (axes coords)
    "frameon": False,
    "borderaxespad": 0,
}
fig = plot_posteriors(configs, VAR_NAME, BASE_DIR, ax_kwargs=ax_kwargs, legend_kwargs=legend_kwargs)
fig.subplots_adjust(top=0.85)

if OUT_FILE is not None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=FIGURE_DPI, bbox_inches="tight")

# %%
