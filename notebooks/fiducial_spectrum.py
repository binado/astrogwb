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
# # Fiducial injection spectrum, network PSDs, and SNR
#
# This notebook plots the independent fiducial-injection SGWB against the six
# detector networks of the `cosmological-parameters` experiment:
#
# - the strain power $S_h(f)$ and energy-density spectrum
#   $\Omega_{\mathrm{GW}}(f)$ on dual $y$-axes;
# - each network's effective noise PSD $S_{\mathrm{eff}}(f)$;
# - $S_h(f)$ against the per-bin Gaussian scale $\sigma$ of the three ET-only
#   networks (no Cosmic Explorer);
# - $\Omega_{\mathrm{GW}}(f)$ against the same $\sigma$ converted through the
#   $f^3$ map that takes $S_h$ to $\Omega_{\mathrm{GW}}$;
# - the matched-filter integrand $\Delta\mathrm{SNR}^{2}(f)$ and the
#   cumulative $\mathrm{SNR}(<f)$ and $\mathrm{SNR}(>f)$, overlaid for every
#   network with the same colors and linestyles as the $S_{\mathrm{eff}}$
#   comparison;
# - a stacked panel of the spectrum above both cumulative SNR curves for the
#   reference network only.
#
# Point `INJECTION_CATALOG_PATH` at the injection catalog used by `mcmc.py`.
# The fiducials, analysis grid, and detector networks the overlays follow are
# inlined in the configuration cell below as `FIDUCIALS`, `GRID`, `NETWORKS`,
# and `REFERENCE_NETWORK`.

# %% [markdown]
# ## Imports and JAX configuration

# %%
from collections.abc import Mapping, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes as MplAxes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection
from matplotlib.ticker import LogFormatterMathtext, ScalarFormatter

from astrogwb.detector import effective_psd, gaussian_bin_scale, load_sensitivity_map
from astrogwb.gwb import (
    omega_gw_from_spectral_density,
    spectral_snr_squared_per_bin,
)
from astrogwb.paper.catalogs import load_run_catalog
from astrogwb.paper.config.mcmc import AnalysisGrid
from astrogwb.paper.config.runs import FIGURES_DIR
from astrogwb.paper.inference import prepare_observation
from astrogwb.paper.plotting import (
    DETECTOR_COMPARISON_LEGEND,
    SPECTRUM,
    SPECTRUM_LINESTYLES,
    Network,
    detector_network_styles,
    save_figures,
    use_paper_style,
)
from astrogwb.utils import years_to_seconds

# gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes. Restore
# matplotlib axes so plotting behaves as expected after importing detector utilities.
register_projection(MplAxes)

jax.config.update("jax_enable_x64", True)
use_paper_style()
# %config InlineBackend.figure_format = 'retina'


# %% [markdown]
# ## Pipeline configuration
#
# The injection catalog is the observed SGWB. Everything else here is the
# `cosmological-parameters` experiment's configuration written out literally
# rather than merged from its config layers: `GRID` is the frequency band,
# observation time, and redshift grid the reference overlays use, `FIDUCIALS`
# is the point the non-sampled parameters are conditioned at, and `NETWORKS`
# carries each compared network's own detector list.
#
# These values mirror `config/analysis/base/parameters.toml`,
# `config/analysis/base/model.toml`, and
# `config/analysis/runs/cosmological-parameters/*.toml`. Editing those files
# does **not** update this notebook; the copies below are hand-maintained.

# %%
#: The notebook may be executed from the repository root (`jupytext --execute`,
#: whose kernel cwd is this directory) or from the root itself, so probe for
#: `notebooks/` the way `catalog_convergence.py` does. `Path().parent` is `.`,
#: not `..`, hence the literal.
ROOT_DIR = Path() if Path("notebooks").is_dir() else Path("..")
INJECTION_CATALOG_PATH = ROOT_DIR / "outputs/catalogs/md-imrphenom-s41-n32768.h5"
#: Where this notebook's figures go: under the one output root the workflow,
#: the figure scripts and `config/plotting.json` all agree on.
BASE_DIR = ROOT_DIR / FIGURES_DIR / "fiducial_spectrum"

# Inlined from config/analysis/base/parameters.toml [fiducials]. Only "H0" is
# read below; the rest are kept so this is the whole fiducial point.
FIDUCIALS: dict[str, float] = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "xi_0": 1.0,
    "xi_n": 1.91,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
    "minimum_mass": 1.0,
    "mass_width": 1.5,
}

# Inlined from config/analysis/base/model.toml plus its top-level
# observation_time -- what that experiment's RunConfig.analysis_grid assembled to.
GRID = AnalysisGrid(
    observation_time=1.0,
    f_min=2.0,
    f_max=2048.0,
    minimum_redshift=0.3,
    maximum_redshift=20.0,
    n_grid=256,
)

# Labels copied from astrogwb.paper.plotting.DETECTOR_NETWORKS, detector lists
# from config/analysis/runs/cosmological-parameters/<name>.toml. Order is
# load-bearing: detector_network_styles assigns a color by first appearance of
# each base network name, so reordering recolors the curves and breaks the match
# with the other network figures.
NETWORKS: tuple[Network, ...] = (
    Network("ET-triangular", r"ET-$\Delta$", ("E1", "E2", "E3")),
    Network(
        "ET-triangular-CE-Hanford", r"ET-$\Delta$ $+$ CE", ("E1", "E2", "E3", "C1")
    ),
    Network("ET-2L-aligned", "ET-2L-par", ("S1", "R1")),
    Network("ET-2L-aligned-CE-Hanford", r"ET-2L-par $+$ CE", ("S1", "R1", "C1")),
    Network("ET-2L-misaligned", "ET-2L", ("S2", "R2")),
    Network("ET-2L-misaligned-CE-Hanford", r"ET-2L $+$ CE", ("S2", "R2", "C1")),
)
REFERENCE_NETWORK = "ET-2L-aligned-CE-Hanford"
ET_ONLY_NETWORKS: tuple[Network, ...] = tuple(
    network for network in NETWORKS if not network.name.endswith("-CE-Hanford")
)

OMEGA_GW_MIN = 1.0e-15

# Cumulative-SNR curves on the stacked figure: Okabe-Ito blue / vermillion,
# distinct from the black dual-axis spectrum.
SNR_LT_COLOR = "#0072B2"
SNR_GT_COLOR = "#D55E00"
SNR_LT_LINESTYLE = "-"
SNR_GT_LINESTYLE = "--"


# %% [markdown]
# ## Plot helpers
#
# Shared band-limiting, SNR accumulation, and the dual-axis $S_h$ /
# $\Omega_{\mathrm{GW}}$ drawing used by the spectrum panels.


# %%
def sh_ymin_matching_omega_floor(
    omega_gw: np.ndarray,
    spectral_density_arr: np.ndarray,
    omega_gw_min: float,
) -> float:
    """Infer $S_h$ ymin from the frequency where $\\Omega_{\\mathrm{GW}}$ hits its floor."""
    if omega_gw_min <= 0.0:
        raise ValueError(f"omega_gw_min must be positive, got {omega_gw_min}")
    if omega_gw.size == 0:
        raise ValueError("cannot infer S_h ymin from an empty spectrum")
    index = int(np.argmin(np.abs(np.log(omega_gw) - np.log(omega_gw_min))))
    return float(spectral_density_arr[index])


def band_limited_spectrum(
    frequencies: jax.Array,
    spectral_density_arr: jax.Array,
    frequency_mask: jax.Array,
    *,
    h0: float,
    effective_psd_arr: jax.Array | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Restrict $S_h$, $\\Omega_{\\mathrm{GW}}$, and optional $S_{\\mathrm{eff}}$ to the band."""
    omega_gw = omega_gw_from_spectral_density(
        spectral_density_arr,
        frequencies,
        hubble_constant=h0,
    )
    mask = np.asarray(frequency_mask)
    freq = np.asarray(frequencies)[mask]
    omega = np.asarray(omega_gw)[mask]
    sh = np.asarray(spectral_density_arr)[mask]
    pos = (omega > 0.0) & (sh > 0.0) & (freq > 0.0)
    seff: np.ndarray | None = None
    if effective_psd_arr is not None:
        seff = np.asarray(effective_psd_arr)[mask]
        pos = pos & np.isfinite(seff) & (seff > 0.0)
        seff = seff[pos]
    return freq[pos], omega[pos], sh[pos], seff


def snr_integrand_and_cumulative(
    spectral_density: np.ndarray,
    effective_psd_arr: np.ndarray,
    observation_time_sec: float,
    df: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin $\\Delta\\mathrm{SNR}^2$ and cumulative SNR from the left and right.

    ``SNR(<f)`` and ``SNR(>f)`` both include the bin at ``f``, so the last
    left-hand value and the first right-hand value equal the total SNR.
    """
    snr_squared = np.asarray(
        spectral_snr_squared_per_bin(
            jnp.asarray(spectral_density),
            jnp.asarray(effective_psd_arr),
            observation_time_sec,
            df,
        )
    )
    snr_lt = np.sqrt(np.cumsum(snr_squared))
    snr_gt = np.sqrt(np.cumsum(snr_squared[::-1])[::-1])
    return snr_squared, snr_lt, snr_gt


def _network_legend_handles(
    networks: Sequence[Network],
    colors: Sequence[str],
    linestyles: Sequence[str],
) -> list[Line2D]:
    return [
        Line2D([], [], color=color, linestyle=linestyle, label=network.label)
        for network, color, linestyle in zip(networks, colors, linestyles, strict=True)
    ]


def _format_axis_ticks(axis: MplAxes) -> None:
    """Render tick labels in mathtext / TeX on log and linear axes."""
    if axis.get_xscale() == "log":
        axis.xaxis.set_major_formatter(LogFormatterMathtext())
    else:
        axis.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    if axis.get_yscale() == "log":
        axis.yaxis.set_major_formatter(LogFormatterMathtext())
    else:
        axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))


def _spectrum_line_styles(
    *,
    omega_color: str | None,
    sh_color: str | None,
    omega_linestyle: str | None,
    sh_linestyle: str | None,
) -> tuple[str, str, str, str]:
    if omega_color is None:
        omega_color = SPECTRUM["omega_gw"]
    if sh_color is None:
        sh_color = SPECTRUM["sh"]
    if omega_linestyle is None:
        omega_linestyle = SPECTRUM_LINESTYLES["omega_gw"]
    if sh_linestyle is None:
        sh_linestyle = SPECTRUM_LINESTYLES["sh"]
    return omega_color, sh_color, omega_linestyle, sh_linestyle


def _draw_omega_and_sh(
    ax_sh: MplAxes,
    frequency: np.ndarray,
    omega_gw: np.ndarray,
    spectral_density: np.ndarray,
    *,
    omega_gw_min: float,
    ax_omega: MplAxes | None = None,
    omega_color: str | None = None,
    sh_color: str | None = None,
    omega_linestyle: str | None = None,
    sh_linestyle: str | None = None,
    xlabel: bool = True,
    legend: bool = True,
) -> tuple[MplAxes, Line2D, Line2D]:
    """Draw dual-axis $S_h$ / $\\Omega_{\\mathrm{GW}}$ onto ``ax_sh``."""
    axis_color = "k"
    omega_color, sh_color, omega_linestyle, sh_linestyle = _spectrum_line_styles(
        omega_color=omega_color,
        sh_color=sh_color,
        omega_linestyle=omega_linestyle,
        sh_linestyle=sh_linestyle,
    )
    if ax_omega is None:
        ax_omega = ax_sh.twinx()

    (line_sh,) = ax_sh.loglog(
        frequency,
        spectral_density,
        color=sh_color,
        linestyle=sh_linestyle,
        label=r"$S_h$",
    )
    (line_omega,) = ax_omega.loglog(
        frequency,
        omega_gw,
        color=omega_color,
        linestyle=omega_linestyle,
        label=r"$\Omega_{\mathrm{GW}}$",
    )

    if xlabel:
        ax_sh.set_xlabel(r"$f\ \mathrm{(Hz)}$", color=axis_color)
    ax_sh.set_ylabel(r"$S_h(f)\ \mathrm{[Hz^{-1}]}$", color=axis_color)
    ax_omega.set_ylabel(r"$\Omega_{\mathrm{GW}}(f)$", color=axis_color)
    ax_sh.tick_params(axis="x", colors=axis_color)
    ax_sh.tick_params(axis="y", colors=axis_color)
    ax_omega.tick_params(axis="y", colors=axis_color)
    for axis in (ax_sh, ax_omega):
        for spine in axis.spines.values():
            spine.set_color(axis_color)
    sh_ymin = sh_ymin_matching_omega_floor(omega_gw, spectral_density, omega_gw_min)
    _, ymax = ax_sh.get_ylim()
    ax_sh.set_ylim(sh_ymin, ymax)
    ax_omega.set_ylim(omega_gw_min, None)
    ax_sh.set_axisbelow(True)
    ax_sh.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax_omega.grid(False)
    _format_axis_ticks(ax_sh)
    _format_axis_ticks(ax_omega)
    if legend:
        ax_sh.legend(
            handles=[line_sh, line_omega],
            loc="upper right",
            frameon=False,
            handlelength=2.5,
        )
    return ax_omega, line_sh, line_omega


# %% [markdown]
# ## Loading the waveform catalog
#
# `prepare_observation` builds the fiducial $S_h$ from the injection catalog on
# the inlined `GRID`; each compared network then gets its own
# $S_{\mathrm{eff}}$ from the detector list inlined in `NETWORKS`. The ET-only
# overlays use $\sigma = S_{\mathrm{eff}}/\sqrt{2 T \Delta f}$ on that same
# band, and $\sigma_\Omega$ is the $f^3$ conversion of $\sigma$.

# %%
catalog = load_run_catalog(INJECTION_CATALOG_PATH, label="injection")
observation = prepare_observation(catalog, grid=GRID)
frequencies = observation.frequencies
frequency_mask = observation.frequency_mask

reference_network = next(
    network for network in NETWORKS if network.name == REFERENCE_NETWORK
)
detector_colors, detector_linestyles = detector_network_styles(NETWORKS)
et_only_colors, et_only_linestyles = detector_network_styles(ET_ONLY_NETWORKS)

effective_psds: dict[str, jax.Array] = {}
for network in NETWORKS:
    sensitivities = load_sensitivity_map(network.detectors)
    effective_psds[network.name] = jnp.asarray(
        effective_psd(frequencies, list(network.detectors), sensitivities)
    )

observation_time_sec = years_to_seconds(GRID.observation_time)
frequency_by_network: dict[str, np.ndarray] = {}
snr_squared_by_network: dict[str, np.ndarray] = {}
snr_lt_by_network: dict[str, np.ndarray] = {}
snr_gt_by_network: dict[str, np.ndarray] = {}
sigma_by_network: dict[str, np.ndarray] = {}
omega_sigma_by_network: dict[str, np.ndarray] = {}
reference_band: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
for network in NETWORKS:
    band_freq, band_omega, band_sh, band_seff = band_limited_spectrum(
        frequencies,
        observation.spectral_density,
        frequency_mask,
        h0=FIDUCIALS["H0"],
        effective_psd_arr=effective_psds[network.name],
    )
    if band_seff is None:
        raise RuntimeError(f"{network.name} effective PSD was not restricted")
    snr_squared, snr_lt, snr_gt = snr_integrand_and_cumulative(
        band_sh,
        band_seff,
        observation_time_sec,
        observation.df,
    )
    sigma = np.asarray(
        gaussian_bin_scale(
            jnp.asarray(band_seff), GRID.observation_time, observation.df
        )
    )
    frequency_by_network[network.name] = band_freq
    snr_squared_by_network[network.name] = snr_squared
    snr_lt_by_network[network.name] = snr_lt
    snr_gt_by_network[network.name] = snr_gt
    sigma_by_network[network.name] = sigma
    omega_sigma_by_network[network.name] = np.asarray(
        omega_gw_from_spectral_density(
            jnp.asarray(sigma),
            jnp.asarray(band_freq),
            hubble_constant=FIDUCIALS["H0"],
        )
    )
    if network.name == reference_network.name:
        reference_band = (band_freq, band_omega, band_sh, band_seff)
if reference_band is None:
    raise RuntimeError("reference-network spectrum was not computed")
freq, omega, sh, _ = reference_band
fiducial_freq, fiducial_omega, fiducial_sh, _ = band_limited_spectrum(
    frequencies,
    observation.spectral_density,
    frequency_mask,
    h0=FIDUCIALS["H0"],
)
print(f"loaded injection: n_frequency_bins={frequencies.shape[0]}")
print("band bins:", int(np.sum(np.asarray(frequency_mask))), "of", frequencies.shape[0])
print("reference network:", reference_network.label)


# %% [markdown]
# ## Fiducial $S_h$ and $\Omega_{\mathrm{GW}}$
#
# Dual $y$-axes for the injection catalog's spectral density. The $S_h$ floor
# is taken from the bin whose $\Omega_{\mathrm{GW}}$ is closest to
# `OMEGA_GW_MIN`, so both axes show the same frequency band.


# %%
def plot_omega_and_sh(
    frequencies: jax.Array,
    spectral_density_arr: jax.Array,
    frequency_mask: jax.Array,
    *,
    h0: float,
    omega_gw_min: float,
) -> Figure:
    """Plot $\\Omega_{\\mathrm{GW}}(f)$ and $S_h(f)$ on dual $y$-axes."""
    freq, omega, sh, _ = band_limited_spectrum(
        frequencies,
        spectral_density_arr,
        frequency_mask,
        h0=h0,
    )
    fig, ax_sh = plt.subplots()
    _draw_omega_and_sh(
        ax_sh,
        freq,
        omega,
        sh,
        omega_gw_min=omega_gw_min,
    )
    return fig


# %%
fig = plot_omega_and_sh(
    frequencies,
    observation.spectral_density,
    frequency_mask,
    h0=FIDUCIALS["H0"],
    omega_gw_min=OMEGA_GW_MIN,
)
_ = save_figures({BASE_DIR / "omega_and_sh.pdf": fig})


# %% [markdown]
# ## Network effective PSDs
#
# $S_{\mathrm{eff}}(f)$ for each detector network in `NETWORKS`.
# `detector_network_styles` shares a color between each ET configuration and
# its ET+CE companion, and dashes the CE curves.


# %%
def plot_effective_psds(
    frequencies: jax.Array,
    networks: Sequence[Network],
    psds_by_network: Mapping[str, jax.Array | np.ndarray],
    *,
    colors: Sequence[str],
    linestyles: Sequence[str],
    frequency_mask: jax.Array,
) -> Figure:
    """Overlay network effective PSDs on shared log–log axes."""
    if len(networks) != len(colors) or len(networks) != len(linestyles):
        raise ValueError("color and linestyle counts must match the networks")

    fig, ax = plt.subplots()
    mask = np.asarray(frequency_mask)
    freq = np.asarray(frequencies)[mask]
    for network, color, linestyle in zip(networks, colors, linestyles, strict=True):
        psd = np.asarray(psds_by_network[network.name])[mask]
        pos = np.isfinite(psd) & (psd > 0.0) & (freq > 0.0)
        ax.loglog(
            freq[pos],
            psd[pos],
            color=color,
            linestyle=linestyle,
        )

    ax.set_xlabel(r"$f\ \mathrm{(Hz)}$")
    ax.set_ylabel(r"$S_{\mathrm{eff}}(f)\ \mathrm{[Hz^{-1}]}$")
    ax.set_axisbelow(True)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    _format_axis_ticks(ax)
    ax.legend(
        handles=_network_legend_handles(networks, colors, linestyles),
        **DETECTOR_COMPARISON_LEGEND,
    )
    fig.tight_layout()
    return fig


# %%
fig = plot_effective_psds(
    frequencies,
    NETWORKS,
    effective_psds,
    colors=detector_colors,
    linestyles=detector_linestyles,
    frequency_mask=frequency_mask,
)
_ = save_figures({BASE_DIR / "effective_psds.pdf": fig})


# %% [markdown]
# ## $S_h$ and ET-only $\sigma$
#
# The fiducial $S_h$ against $\sigma = S_{\mathrm{eff}} / \sqrt{2 T \Delta f}$
# for the three ET-only networks. $\sigma$ is the per-bin Gaussian scale of
# $S_h$, so it shares units and observation-time scaling. Colors follow
# `detector_network_styles`; labels sit above the axes.


# %%
def plot_spectrum_and_sensitivities(
    frequency: np.ndarray,
    spectrum: np.ndarray,
    networks: Sequence[Network],
    frequency_by_network: Mapping[str, np.ndarray],
    sensitivities_by_network: Mapping[str, np.ndarray],
    *,
    colors: Sequence[str],
    linestyles: Sequence[str],
    spectrum_label: str,
    spectrum_color: str,
    spectrum_linestyle: str,
    ylabel: str,
    ymin: float | None = None,
    include_spectrum_in_legend: bool = True,
    spectrum_legend_loc: str | None = None,
) -> Figure:
    """Overlay a fiducial spectrum with per-network Gaussian sensitivities.

    When ``spectrum_legend_loc`` is set, the spectrum gets its own legend inside
    the axes (e.g. ``"upper left"``). The network legend still uses
    ``DETECTOR_COMPARISON_LEGEND`` above the frame and remains ``ax.legend_``
    so ``tight_layout`` keeps reserving space for it; the inner legend is
    pinned with ``ax.add_artist``.
    """
    if len(networks) != len(colors) or len(networks) != len(linestyles):
        raise ValueError("color and linestyle counts must match the networks")
    if include_spectrum_in_legend and spectrum_legend_loc is not None:
        raise ValueError(
            "use at most one of include_spectrum_in_legend and spectrum_legend_loc"
        )

    fig, ax = plt.subplots()
    (line_spectrum,) = ax.loglog(
        frequency,
        spectrum,
        color=spectrum_color,
        linestyle=spectrum_linestyle,
        label=spectrum_label,
    )
    for network, color, linestyle in zip(networks, colors, linestyles, strict=True):
        network_frequency = np.asarray(frequency_by_network[network.name])
        sensitivity = np.asarray(sensitivities_by_network[network.name])
        pos = np.isfinite(sensitivity) & (sensitivity > 0.0) & (network_frequency > 0.0)
        ax.loglog(
            network_frequency[pos],
            sensitivity[pos],
            color=color,
            linestyle=linestyle,
        )

    ax.set_xlabel(r"$f\ \mathrm{(Hz)}$")
    ax.set_ylabel(ylabel)
    if ymin is not None:
        _, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
    ax.set_axisbelow(True)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    _format_axis_ticks(ax)
    network_handles = _network_legend_handles(networks, colors, linestyles)
    legend_handles = (
        [line_spectrum, *network_handles]
        if include_spectrum_in_legend
        else network_handles
    )
    if spectrum_legend_loc is not None:
        # Keep the detector legend as ax.legend_ so tight_layout still
        # accounts for the above-axes bbox; pin the spectrum entry separately.
        spectrum_legend = ax.legend(
            handles=[line_spectrum],
            loc=spectrum_legend_loc,
            frameon=False,
            handlelength=2.5,
        )
        ax.add_artist(spectrum_legend)
    ax.legend(handles=legend_handles, **DETECTOR_COMPARISON_LEGEND)
    fig.tight_layout()
    return fig


# %%
fig = plot_spectrum_and_sensitivities(
    fiducial_freq,
    fiducial_sh,
    ET_ONLY_NETWORKS,
    frequency_by_network,
    sigma_by_network,
    colors=et_only_colors,
    linestyles=et_only_linestyles,
    spectrum_label=r"$S_h$",
    spectrum_color=SPECTRUM["sh"],
    spectrum_linestyle=SPECTRUM_LINESTYLES["sh"],
    ylabel=r"$S_h(f), \, \sigma(f)\ \mathrm{[Hz^{-1}]}$",
    ymin=sh_ymin_matching_omega_floor(fiducial_omega, fiducial_sh, OMEGA_GW_MIN),
    include_spectrum_in_legend=False,
    spectrum_legend_loc="upper left",
)
_ = save_figures({BASE_DIR / "sh_and_sigma.pdf": fig})


# %% [markdown]
# ## $\Omega_{\mathrm{GW}}$ and ET-only $\sigma_\Omega$
#
# The same comparison in energy-density units: each $\sigma$ is converted with
# `omega_gw_from_spectral_density`, the $f^3$ map that takes $S_h$ to
# $\Omega_{\mathrm{GW}}$.

# %%
fig = plot_spectrum_and_sensitivities(
    fiducial_freq,
    fiducial_omega,
    ET_ONLY_NETWORKS,
    frequency_by_network,
    omega_sigma_by_network,
    colors=et_only_colors,
    linestyles=et_only_linestyles,
    spectrum_label=r"$\Omega_{\mathrm{GW}}$",
    spectrum_color=SPECTRUM["omega_gw"],
    spectrum_linestyle=":",
    ylabel=r"$\Omega_{\mathrm{GW}}(f), \, \sigma(f)$",
    ymin=OMEGA_GW_MIN,
    include_spectrum_in_legend=False,
    spectrum_legend_loc="upper left",
)
_ = save_figures({BASE_DIR / "omega_and_sigma.pdf": fig})


# %% [markdown]
# ## SNR integrand and cumulative SNR
#
# Per-bin $\Delta\mathrm{SNR}^{2}(f) = 2 T \Delta f (S_h / S_{\mathrm{eff}})^{2}$,
# with $\mathrm{SNR}(<f)$ accumulated from the left and $\mathrm{SNR}(>f)$ from
# the right. Every compared network is overlaid with the same colors and
# linestyles as the $S_{\mathrm{eff}}$ figure.


# %%
def plot_snr_cumulative(
    networks: Sequence[Network],
    frequency_by_network: Mapping[str, np.ndarray],
    snr_squared_by_network: Mapping[str, np.ndarray],
    snr_lt_by_network: Mapping[str, np.ndarray],
    snr_gt_by_network: Mapping[str, np.ndarray],
    *,
    colors: Sequence[str],
    linestyles: Sequence[str],
) -> Figure:
    """Overlay per-network SNR integrand and cumulative SNR from each side."""
    if len(networks) != len(colors) or len(networks) != len(linestyles):
        raise ValueError("color and linestyle counts must match the networks")

    fig, axes = plt.subplots(3, 1, sharex=True)
    ax_integrand, ax_lt, ax_gt = axes

    for network, color, linestyle in zip(networks, colors, linestyles, strict=True):
        freq = frequency_by_network[network.name]
        ax_integrand.loglog(
            freq,
            snr_squared_by_network[network.name],
            color=color,
            linestyle=linestyle,
        )
        ax_lt.semilogx(
            freq,
            snr_lt_by_network[network.name],
            color=color,
            linestyle=linestyle,
        )
        ax_gt.semilogx(
            freq,
            snr_gt_by_network[network.name],
            color=color,
            linestyle=linestyle,
        )

    ax_integrand.set_ylabel(r"$\Delta\mathrm{SNR}^{2}(f)$")
    ax_lt.set_ylabel(r"$\mathrm{SNR}(<f)$")
    ax_gt.set_ylabel(r"$\mathrm{SNR}(>f)$")
    ax_gt.set_xlabel(r"$f\ \mathrm{(Hz)}$")
    ax_integrand.legend(
        handles=_network_legend_handles(networks, colors, linestyles),
        **DETECTOR_COMPARISON_LEGEND,
    )

    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
        _format_axis_ticks(axis)
    fig.tight_layout()
    return fig


# %%
fig = plot_snr_cumulative(
    NETWORKS,
    frequency_by_network,
    snr_squared_by_network,
    snr_lt_by_network,
    snr_gt_by_network,
    colors=detector_colors,
    linestyles=detector_linestyles,
)
_ = save_figures({BASE_DIR / "snr_cumulative.pdf": fig})


# %% [markdown]
# ## Spectrum and cumulative SNR
#
# The fiducial spectrum stacked above both cumulative SNR curves, for the
# reference network only.


# %%
def plot_spectrum_and_cumulative_snr(
    frequency: np.ndarray,
    omega_gw: np.ndarray,
    spectral_density: np.ndarray,
    snr_lt: np.ndarray,
    snr_gt: np.ndarray,
    *,
    omega_gw_min: float,
) -> Figure:
    """Stack $S_h$ / $\\Omega_{\\mathrm{GW}}$ above both cumulative SNR curves."""
    fig, (ax_sh, ax_snr) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0]}
    )
    _draw_omega_and_sh(
        ax_sh,
        frequency,
        omega_gw,
        spectral_density,
        omega_gw_min=omega_gw_min,
        xlabel=False,
    )
    ax_sh.tick_params(axis="x", labelbottom=False)

    (line_lt,) = ax_snr.semilogx(
        frequency,
        snr_lt,
        color=SNR_LT_COLOR,
        linestyle=SNR_LT_LINESTYLE,
        label=r"$\mathrm{SNR}(<f)$",
    )
    (line_gt,) = ax_snr.semilogx(
        frequency,
        snr_gt,
        color=SNR_GT_COLOR,
        linestyle=SNR_GT_LINESTYLE,
        label=r"$\mathrm{SNR}(>f)$",
    )
    ax_snr.set_xlabel(r"$f\ \mathrm{(Hz)}$")
    ax_snr.set_ylabel(r"$\mathrm{SNR}$")
    ax_snr.set_axisbelow(True)
    ax_snr.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    _format_axis_ticks(ax_snr)
    ax_snr.legend(
        handles=[line_lt, line_gt],
        loc="best",
        frameon=False,
        handlelength=2.5,
    )
    fig.tight_layout()
    return fig


# %%
fig = plot_spectrum_and_cumulative_snr(
    freq,
    omega,
    sh,
    snr_lt_by_network[reference_network.name],
    snr_gt_by_network[reference_network.name],
    omega_gw_min=OMEGA_GW_MIN,
)
_ = save_figures({BASE_DIR / "spectrum_and_cumulative_snr.pdf": fig})
