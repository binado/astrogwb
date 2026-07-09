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
# # Comparing 1D posteriors

# %%
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arviz_stats as azs
import matplotlib.pyplot as plt
import numpy.typing as npt
import pandas as pd
import xarray as xr

from astrogwb.config import load_mapping
from astrogwb.utils import repo_root

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
    "xtick.direction": "in",  # Point ticks inward
    "xtick.minor.visible": True,  # Turn on minor ticks
    "xtick.top": True,  # Draw ticks on the top spine as well
    # Y-axis ticks
    "ytick.direction": "in",  # Point ticks inward
    "ytick.minor.visible": True,  # Turn on minor ticks
    "ytick.right": True,  # Draw ticks on the right spine as well
}
plt.rcParams.update(**pub_rc)

# %% [markdown]
# ## Notebook configuration
#
# Nested plot data (posterior list, `ax_kwargs`, `legend_kwargs`) comes from
# `[figures.mcmc_compare_posteriors]` in
# [`configs/paper.toml`](../configs/paper.toml). Scalars (`var_name`, dpi,
# output path, group) are argparse defaults below — edit in Jupyter, override
# with flags headless. Promote happy values by updating those defaults / toml.


# %%
@dataclass(frozen=True)
class PosteriorConfig:
    label: str
    path: Path
    color: str
    linestyle: str

    def load_posterior_samples(self, base_dir: Path) -> xr.DataTree:
        fullpath = self.path if self.path.is_absolute() else base_dir / self.path
        return xr.open_datatree(fullpath, engine="h5netcdf")

    def get_x_and_prob_arrays(
        self, base_dir: Path, group: str, var_name: str
    ) -> tuple[npt.NDArray, npt.NDArray]:
        dtree = self.load_posterior_samples(base_dir)
        kde = azs.kde(dtree, group=group, var_names=var_name)[var_name]
        x, prob = kde.sel(plot_axis="x").to_numpy(), kde.sel(plot_axis="y").to_numpy()
        return x, prob

    def get_hdi(
        self,
        base_dir: Path,
        group: str,
        var_name: str,
        *,
        prob: float = 0.6827,
    ) -> tuple[float, float]:
        dtree = self.load_posterior_samples(base_dir)
        hdi = azs.hdi(dtree, group=group, var_names=var_name, prob=prob)[var_name]
        return float(hdi.sel(ci_bound="lower")), float(hdi.sel(ci_bound="upper"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/paper.toml"))
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("figures/mcmc_compare_posteriors_H0.pdf"),
    )
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument("--var-name", default="H0")
    parser.add_argument("--group", default="posterior")
    args, _ = parser.parse_known_args()
    return args


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


BASE_DIR = repo_root()
args = _parse_args()
config_path = _resolve_path(args.config, BASE_DIR)
paper = load_mapping(config_path)
figure = paper["figures"]["mcmc_compare_posteriors"]

configs = [
    PosteriorConfig(
        label=entry["label"],
        path=Path(entry["path"]),
        color=entry["color"],
        linestyle=entry["linestyle"],
    )
    for entry in figure["posteriors"]
]
VAR_NAME = args.var_name
OUT_FILE = _resolve_path(args.output_pdf, BASE_DIR)
FIGURE_DPI = args.figure_dpi
GROUP = args.group
ax_kwargs = figure.get("ax_kwargs", {})
legend_kwargs = figure.get("legend_kwargs", {})


# %% [markdown]
# Defining the plotting and summary helpers:


# %%
def plot_posteriors(
    configs: list[PosteriorConfig],
    var_name: str,
    base_dir: Path,
    *,
    group: str = "posterior",
    ax_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
):
    fig, ax = plt.subplots()
    for config in configs:
        x, prob = config.get_x_and_prob_arrays(base_dir, group, var_name)
        ax.plot(
            x, prob, label=config.label, color=config.color, linestyle=config.linestyle
        )

    ax_kwargs = ax_kwargs or {}
    legend_kwargs = legend_kwargs or {}
    ax.set(**ax_kwargs)
    ax.legend(**legend_kwargs)
    fig.tight_layout()
    return fig


def summarize_hdi(
    configs: list[PosteriorConfig],
    var_name: str,
    base_dir: Path,
    *,
    group: str = "posterior",
    prob: float = 0.6827,
) -> pd.DataFrame:
    rows = []
    for config in configs:
        lower, upper = config.get_hdi(base_dir, group, var_name, prob=prob)
        rows.append(
            {
                "label": config.label,
                "lower": lower,
                "upper": upper,
                "width": upper - lower,
                "sigma": (upper - lower) / 2,
            }
        )
    return (
        pd.DataFrame(rows).sort_values("sigma", ascending=True).reset_index(drop=True)
    )


# %% [markdown]
# Plotting:

# %%
fig = plot_posteriors(
    configs,
    VAR_NAME,
    BASE_DIR,
    group=GROUP,
    ax_kwargs=ax_kwargs,
    legend_kwargs=legend_kwargs,
)
fig.subplots_adjust(top=0.85)

if OUT_FILE is not None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=FIGURE_DPI, bbox_inches="tight")

# %% [markdown]
# 1σ HDI summary:

# %%
hdi_summary = summarize_hdi(
    configs,
    VAR_NAME,
    BASE_DIR,
    group=GROUP,
)
hdi_summary.style.format(precision=2)

# %%
