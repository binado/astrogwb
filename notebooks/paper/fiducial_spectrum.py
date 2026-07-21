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
# # Fiducial $\Omega_{\mathrm{GW}}$ and $S_h$ spectrum
#
# Plot the fiducial astrophysical SGWB spectral density $S_h(f, \Lambda_0)$ and
# the equivalent energy-density spectrum $\Omega_{\mathrm{GW}}(f, \Lambda_0)$ on
# a shared frequency axis with dual $y$-scales. The fiducial $S_h$ is computed
# with the same importance-weighted contraction used by the MCMC notebooks.

# %%
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes as MplAxes
from matplotlib.figure import Figure
from matplotlib.projections import register_projection
from pluscross import load_catalog

from _paper_style import CATEGORY, use_paper_style
from astrogwb.gwb import frequency_mask as make_frequency_mask
from astrogwb.gwb import omega_gw_from_spectral_density, spectral_density
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_proposal_logpdf,
    make_merger_rate_and_log_weights_fn,
)
from astrogwb.utils import repo_root
from astrogwb.waveform import polarization_power as compute_polarization_power

# gwpy (via gwmock-signal) replaces matplotlib's rectilinear axes. Restore the
# standard matplotlib projection for consistent notebook plotting.
register_projection(MplAxes)
jax.config.update("jax_enable_x64", True)

# %config InlineBackend.figure_format = "retina"

# %% [markdown]
# ## Defaults
#
# Direct notebook runs use the defaults below. Edit them in Jupyter or override
# with flags when running headless.

# %%
DEFAULT_CATALOG_PATH = Path("out/catalogs/bns-n16384-df1.h5")
DEFAULT_F_MIN = 2.0
DEFAULT_F_MAX = 4096.0
DEFAULT_Z_MIN = 0.0
DEFAULT_Z_MAX = 20.0
DEFAULT_N_GRID = 256
DEFAULT_H0 = 67.66
DEFAULT_OMEGA_M = 0.3096
DEFAULT_XI_0 = 1.0
DEFAULT_XI_N = 1.91
DEFAULT_GAMMA = 1.42
DEFAULT_KAPPA = 4.62
DEFAULT_Z_PEAK = 1.84
DEFAULT_LOCAL_MERGER_RATE = 161.0
DEFAULT_OMEGA_YMIN = 1e-15
DEFAULT_SH_YMIN = 1e-55
DEFAULT_OUTPUT_PDF = Path("figures/fiducial_spectrum.pdf")
DEFAULT_FIGURE_DPI = 300


# %%
def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def compute_fiducial_spectral_density(
    catalog_path: Path,
    fiducials: Mapping[str, float],
    *,
    f_min: float,
    f_max: float,
    z_min: float,
    z_max: float,
    n_grid: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``(frequencies, S_h, frequency_mask)`` at the fiducial point."""
    catalog = load_catalog(catalog_path)
    frequencies = jnp.asarray(catalog.frequencies)
    polarization_power = jnp.asarray(compute_polarization_power(catalog))
    samples = {
        name: jnp.asarray(values) for name, values in catalog.source_parameters.items()
    }
    del catalog

    missing = [
        name for name in ("redshift", "luminosity_distance") if name not in samples
    ]
    if missing:
        raise ValueError(
            "catalog samples are missing required parameter(s): " + ", ".join(missing)
        )

    z_grid = jnp.linspace(z_min, z_max, n_grid)
    proposal_log_pdf = compute_proposal_logpdf(
        samples["redshift"], z_grid=z_grid, fiducials=fiducials
    )
    merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
        z_grid=z_grid,
        proposal_log_pdf=proposal_log_pdf,
        fiducial_xi_0=fiducials["xi_0"],
        fiducial_xi_n=fiducials["xi_n"],
    )
    total_rate, log_weights = merger_rate_and_log_weights_fn(fiducials, samples)
    observed_spectral_density = spectral_density(
        polarization_power,
        jnp.exp(log_weights),
        total_rate,
        average_mode="analytic_inclination",
    )
    mask = make_frequency_mask(frequencies, fmin=f_min, fmax=f_max)
    return frequencies, observed_spectral_density, mask


def plot_omega_and_sh(
    frequencies: jax.Array,
    spectral_density_arr: jax.Array,
    mask: jax.Array,
    *,
    omega_ymin: float = DEFAULT_OMEGA_YMIN,
    sh_ymin: float = DEFAULT_SH_YMIN,
    omega_color: str | None = None,
    sh_color: str | None = None,
) -> Figure:
    """Plot $\\Omega_{\\mathrm{GW}}(f)$ and $S_h(f)$ on dual $y$-axes."""
    if omega_color is None:
        omega_color = CATEGORY["cosmology"]
    if sh_color is None:
        sh_color = CATEGORY["astrophysical"]

    omega_gw = omega_gw_from_spectral_density(spectral_density_arr, frequencies)
    pos = (omega_gw > 0.0) & (spectral_density_arr > 0.0) & mask
    freq = np.asarray(frequencies[pos])
    omega = np.asarray(omega_gw[pos])
    sh = np.asarray(spectral_density_arr[pos])

    fig, ax_omega = plt.subplots()
    ax_sh = ax_omega.twinx()

    (line_omega,) = ax_omega.loglog(
        freq, omega, color=omega_color, label=r"$\Omega_{\mathrm{GW}}$"
    )
    (line_sh,) = ax_sh.loglog(freq, sh, color=sh_color, label=r"$S_h$")

    ax_omega.set_xlabel(r"$f\ \mathrm{(Hz)}$")
    ax_omega.set_ylabel(r"$\Omega_{\mathrm{GW}}(f)$", color=omega_color)
    ax_sh.set_ylabel(r"$S_h(f)\ \mathrm{[Hz^{-1}]}$", color=sh_color)
    ax_omega.tick_params(axis="y", colors=omega_color)
    ax_sh.tick_params(axis="y", colors=sh_color)
    ax_omega.set_ylim(omega_ymin, None)
    ax_sh.set_ylim(sh_ymin, None)
    ax_omega.legend(
        handles=[line_omega, line_sh],
        loc="upper right",
        frameon=False,
    )
    return fig


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--f-min", type=float, default=DEFAULT_F_MIN)
    parser.add_argument("--f-max", type=float, default=DEFAULT_F_MAX)
    parser.add_argument("--z-min", type=float, default=DEFAULT_Z_MIN)
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX)
    parser.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    parser.add_argument("--h0", type=float, default=DEFAULT_H0)
    parser.add_argument("--omega-m", type=float, default=DEFAULT_OMEGA_M)
    parser.add_argument("--xi-0", type=float, default=DEFAULT_XI_0)
    parser.add_argument("--xi-n", type=float, default=DEFAULT_XI_N)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    parser.add_argument("--z-peak", type=float, default=DEFAULT_Z_PEAK)
    parser.add_argument(
        "--local-merger-rate", type=float, default=DEFAULT_LOCAL_MERGER_RATE
    )
    parser.add_argument("--omega-ymin", type=float, default=DEFAULT_OMEGA_YMIN)
    parser.add_argument("--sh-ymin", type=float, default=DEFAULT_SH_YMIN)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--figure-dpi", type=int, default=DEFAULT_FIGURE_DPI)
    args, _ = parser.parse_known_args(argv)
    return args


# %%
args = _parse_args()
root = repo_root()

fiducials = {
    "H0": args.h0,
    "Omega_m": args.omega_m,
    "xi_0": args.xi_0,
    "xi_n": args.xi_n,
    "gamma": args.gamma,
    "kappa": args.kappa,
    "z_peak": args.z_peak,
    "local_merger_rate": args.local_merger_rate,
}

use_paper_style()

# %% [markdown]
# ## Fiducial spectrum
#
# Load the waveform catalog, evaluate importance weights at $\Lambda_0$, and
# form $S_h(f, \Lambda_0)$. Convert to $\Omega_{\mathrm{GW}}(f)$ for the left
# axis.

# %%
catalog_path = _resolve_path(args.catalog, root)
frequencies, observed_spectral_density, mask = compute_fiducial_spectral_density(
    catalog_path,
    fiducials,
    f_min=args.f_min,
    f_max=args.f_max,
    z_min=args.z_min,
    z_max=args.z_max,
    n_grid=args.n_grid,
)

figure = plot_omega_and_sh(
    frequencies,
    observed_spectral_density,
    mask,
    omega_ymin=args.omega_ymin,
    sh_ymin=args.sh_ymin,
)

# %% [markdown]
# ## Save figure

# %%
output_path = _resolve_path(args.output_pdf, root)
output_path.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output_path, dpi=args.figure_dpi, bbox_inches="tight")
print("saved figure:", output_path)
