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
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from _paper_style import SPECTRUM, combo_colors, use_paper_style
from matplotlib.axes import Axes as MplAxes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection
from pluscross import load_catalog

from astrogwb.config.loading import load_mapping
from astrogwb.detector import effective_psd, load_sensitivity_map
from astrogwb.gwb import frequency_mask as make_frequency_mask
from astrogwb.gwb import (
    hubble_constant_si,
    omega_gw_from_spectral_density,
    spectral_density,
)
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
DEFAULT_CONFIG_PATH = Path("configs/paper.toml")
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
DEFAULT_OMEGA_GW_MIN = 1e-15
DEFAULT_OUTPUT_PDF = Path("figures/fiducial_spectrum.pdf")
DEFAULT_OUTPUT_EFFECTIVE_PSD_PDF = Path(
    "figures/fiducial_effective_psd_by_detector.pdf"
)
DEFAULT_FIGURE_DPI = 300


# %%
def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def detector_networks_from_config(
    config: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Return detector networks from ``configs/paper.toml``."""
    raw = config["detector_networks"]
    return {name: tuple(detectors) for name, detectors in raw.items()}


def detector_labels_from_config(
    config: Mapping[str, Any],
    networks: Mapping[str, tuple[str, ...]],
) -> list[str]:
    """Return display labels for ``networks`` from cosmology detector posteriors."""
    posteriors = config["figures"]["mcmc_cosmological_parameters"][
        "detector_posteriors"
    ]
    label_by_network = {entry["network"]: entry["label"] for entry in posteriors}
    missing = [name for name in networks if name not in label_by_network]
    if missing:
        raise ValueError(
            "missing detector labels in paper config for network(s): "
            + ", ".join(missing)
        )
    return [label_by_network[name] for name in networks]


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


def sh_ymin_matching_omega_floor(
    omega_gw: np.ndarray,
    spectral_density_arr: np.ndarray,
    omega_gw_min: float,
) -> float:
    """Infer $S_h$ ymin from the frequency where $\\Omega_{\\mathrm{GW}}$ hits its floor.

    Picks the bin whose $\\Omega_{\\mathrm{GW}}$ is closest (in log space) to
    ``omega_gw_min`` and returns $S_h$ there, so both axes show the same
    frequency band when clipped at their respective floors.
    """
    if omega_gw_min <= 0.0:
        raise ValueError(f"omega_gw_min must be positive, got {omega_gw_min}")
    if omega_gw.size == 0:
        raise ValueError("cannot infer S_h ymin from an empty spectrum")
    index = int(np.argmin(np.abs(np.log(omega_gw) - np.log(omega_gw_min))))
    return float(spectral_density_arr[index])


def plot_omega_and_sh(
    frequencies: jax.Array,
    spectral_density_arr: jax.Array,
    mask: jax.Array,
    *,
    h0: float,
    omega_gw_min: float = DEFAULT_OMEGA_GW_MIN,
    omega_color: str | None = None,
    sh_color: str | None = None,
) -> Figure:
    """Plot $\\Omega_{\\mathrm{GW}}(f)$ and $S_h(f)$ on dual $y$-axes."""
    if omega_color is None:
        omega_color = SPECTRUM["omega_gw"]
    if sh_color is None:
        sh_color = SPECTRUM["sh"]

    omega_gw = omega_gw_from_spectral_density(
        spectral_density_arr,
        frequencies,
        hubble_constant_si=hubble_constant_si(h0),
    )
    pos = (omega_gw > 0.0) & (spectral_density_arr > 0.0) & mask
    freq = np.asarray(frequencies[pos])
    omega = np.asarray(omega_gw[pos])
    sh = np.asarray(spectral_density_arr[pos])
    sh_ymin = sh_ymin_matching_omega_floor(omega, sh, omega_gw_min)

    fig, ax_sh = plt.subplots()
    ax_omega = ax_sh.twinx()

    (line_sh,) = ax_sh.loglog(freq, sh, color=sh_color, label=r"$S_h$")
    (line_omega,) = ax_omega.loglog(
        freq, omega, color=omega_color, label=r"$\Omega_{\mathrm{GW}}$"
    )

    ax_sh.set_xlabel(r"$f\ \mathrm{(Hz)}$")
    ax_sh.set_ylabel(r"$S_h(f)\ \mathrm{[Hz^{-1}]}$", color=sh_color)
    ax_omega.set_ylabel(r"$\Omega_{\mathrm{GW}}(f)$", color=omega_color)
    ax_sh.tick_params(axis="y", colors=sh_color)
    ax_omega.tick_params(axis="y", colors=omega_color)
    ax_sh.set_ylim(sh_ymin, None)
    ax_omega.set_ylim(omega_gw_min, None)
    ax_sh.set_axisbelow(True)
    ax_sh.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax_omega.grid(False)
    ax_sh.legend(
        handles=[line_sh, line_omega],
        loc="upper right",
        frameon=False,
    )
    return fig


def _base_network_name(name: str) -> str:
    """Map an ET+CE network name onto its ET-only counterpart."""
    suffix = "-CE-Hanford"
    return name.removesuffix(suffix)


def detector_network_styles(
    networks: Mapping[str, tuple[str, ...]],
) -> tuple[list[str], list[str]]:
    """Shared color per ET / ET+CE pair; dashed linestyle for CE companions."""
    names = list(networks)
    bases: list[str] = []
    for name in names:
        base = _base_network_name(name)
        if base not in bases:
            bases.append(base)
    palette = combo_colors(len(bases))
    color_by_base = dict(zip(bases, palette, strict=True))
    colors = [color_by_base[_base_network_name(name)] for name in names]
    linestyles = ["--" if name.endswith("-CE-Hanford") else "-" for name in names]
    return colors, linestyles


def plot_effective_psds(
    frequencies: jax.Array,
    psds_by_network: Mapping[str, jax.Array | np.ndarray],
    labels: Sequence[str],
    *,
    colors: Sequence[str],
    linestyles: Sequence[str],
    mask: jax.Array,
) -> Figure:
    """Overlay network effective PSDs on shared log–log axes."""
    names = list(psds_by_network)
    if len(names) != len(labels):
        raise ValueError("network count must match label count")
    if len(names) != len(colors) or len(names) != len(linestyles):
        raise ValueError("color and linestyle counts must match the networks")

    fig, ax = plt.subplots()
    freq = np.asarray(frequencies)
    band = np.asarray(mask, dtype=bool)
    for name, label, color, linestyle in zip(
        names, labels, colors, linestyles, strict=True
    ):
        psd = np.asarray(psds_by_network[name])
        pos = band & np.isfinite(psd) & (psd > 0.0) & (freq > 0.0)
        ax.loglog(
            freq[pos],
            psd[pos],
            color=color,
            linestyle=linestyle,
            label=label,
        )

    ax.set_xlabel(r"$f\ \mathrm{(Hz)}$")
    ax.set_ylabel(r"$S_{\mathrm{eff}}(f)\ \mathrm{[Hz^{-1}]}$")
    ax.set_axisbelow(True)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    handles = [
        Line2D([], [], color=color, linestyle=linestyle, label=label)
        for label, color, linestyle in zip(labels, colors, linestyles, strict=True)
    ]
    ax.legend(
        handles=handles,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        borderaxespad=0,
        handlelength=2.5,
    )
    fig.tight_layout()
    return fig


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
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
    parser.add_argument(
        "--omega-gw-min",
        type=float,
        default=DEFAULT_OMEGA_GW_MIN,
        help=(
            "Lower y-limit for Omega_GW; S_h ymin is taken from S_h at the "
            "frequency where Omega_GW is closest to this floor."
        ),
    )
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument(
        "--output-effective-psd-pdf",
        type=Path,
        default=DEFAULT_OUTPUT_EFFECTIVE_PSD_PDF,
    )
    parser.add_argument("--figure-dpi", type=int, default=DEFAULT_FIGURE_DPI)
    args, _ = parser.parse_known_args(argv)
    return args


# %%
args = _parse_args()
root = repo_root()
config = load_mapping(_resolve_path(args.config, root))

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
# form $S_h(f, \Lambda_0)$. Convert to $\Omega_{\mathrm{GW}}(f)$ for the right
# axis using the configured fiducial $H_0$ (not the package $H_0$ default).
# The $S_h$ floor is inferred from the frequency where $\Omega_{\mathrm{GW}}$
# meets `--omega-gw-min`, so both curves show the same frequency band.

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
    h0=args.h0,
    omega_gw_min=args.omega_gw_min,
)

# %% [markdown]
# ## Effective PSD by detector network
#
# Network effective noise PSDs for the detector combinations in
# ``configs/paper.toml``, overlaid on a shared log–log frequency axis. Colors
# and linestyles match the $H_0$ density comparison (shared color per ET /
# ET+CE pair; dashed for CE companions). Labels come from the cosmology
# ``detector_posteriors`` entries in the same config.

# %%
networks = detector_networks_from_config(config)
detector_labels = detector_labels_from_config(config, networks)
detector_colors, detector_linestyles = detector_network_styles(networks)

effective_psds = {}
for network, detectors in networks.items():
    sensitivities = load_sensitivity_map(detectors)
    effective_psds[network] = jnp.asarray(
        effective_psd(frequencies, list(detectors), sensitivities)
    )

effective_psd_figure = plot_effective_psds(
    frequencies,
    effective_psds,
    detector_labels,
    colors=detector_colors,
    linestyles=detector_linestyles,
    mask=mask,
)

# %% [markdown]
# ## Save figures

# %%
output_path = _resolve_path(args.output_pdf, root)
output_path.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output_path, dpi=args.figure_dpi, bbox_inches="tight")
print("saved figure:", output_path)

effective_psd_output_path = _resolve_path(args.output_effective_psd_pdf, root)
effective_psd_output_path.parent.mkdir(parents=True, exist_ok=True)
effective_psd_figure.savefig(
    effective_psd_output_path, dpi=args.figure_dpi, bbox_inches="tight"
)
print("saved figure:", effective_psd_output_path)
