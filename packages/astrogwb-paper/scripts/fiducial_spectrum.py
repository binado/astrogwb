"""Plot the independent fiducial-injection SGWB spectrum and effective PSDs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astrogwb.cosmology import hubble_constant_si
from astrogwb.detector import effective_psd, load_sensitivity_map
from astrogwb.gwb import (
    omega_gw_from_spectral_density,
)
from astrogwb_paper.catalogs import CatalogSource
from astrogwb_paper.config.figures import (
    Network,
    load_analysis_grid,
    load_fiducials,
    load_injection_spec,
    resolve_networks,
)
from astrogwb_paper.inference import prepare_observation
from astrogwb_paper.paths import paper_project_root, resolve_paper_path
from astrogwb_paper.plotting import (
    DETECTOR_COMPARISON_LEGEND,
    DETECTOR_NETWORKS,
    SPECTRUM,
    SPECTRUM_LINESTYLES,
    detector_network_styles,
    use_paper_style,
)
from matplotlib.axes import Axes as MplAxes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection

# gwpy (via gwmock-signal) replaces matplotlib's rectilinear axes. Restore the
# standard matplotlib projection for consistent plotting.
register_projection(MplAxes)
jax.config.update("jax_enable_x64", True)

# These panels compare the same six networks as the cosmological-parameters
# experiment, so they borrow its detector lists rather than restating them.
# This figure reads no chains, so there is no argv order to keep in step.
SPECTRUM_EXPERIMENT = "cosmological-parameters"
# Lower y-limit for Omega_GW; the S_h ymin is taken from S_h at the frequency
# where Omega_GW is closest to this floor.
OMEGA_GW_MIN = 1.0e-15


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
    frequency_slice: slice,
    *,
    h0: float,
    omega_gw_min: float,
    omega_color: str | None = None,
    sh_color: str | None = None,
    omega_linestyle: str | None = None,
    sh_linestyle: str | None = None,
) -> Figure:
    """Plot $\\Omega_{\\mathrm{GW}}(f)$ and $S_h(f)$ on dual $y$-axes."""
    axis_color = "k"
    if omega_color is None:
        omega_color = SPECTRUM["omega_gw"]
    if sh_color is None:
        sh_color = SPECTRUM["sh"]
    if omega_linestyle is None:
        omega_linestyle = SPECTRUM_LINESTYLES["omega_gw"]
    if sh_linestyle is None:
        sh_linestyle = SPECTRUM_LINESTYLES["sh"]

    omega_gw = omega_gw_from_spectral_density(
        spectral_density_arr,
        frequencies,
        hubble_constant_si=hubble_constant_si(h0),
    )
    band_frequencies = frequencies[frequency_slice]
    band_omega_gw = omega_gw[frequency_slice]
    band_spectral_density = spectral_density_arr[frequency_slice]
    pos = (band_omega_gw > 0.0) & (band_spectral_density > 0.0)
    freq = np.asarray(band_frequencies[pos])
    omega = np.asarray(band_omega_gw[pos])
    sh = np.asarray(band_spectral_density[pos])
    sh_ymin = sh_ymin_matching_omega_floor(omega, sh, omega_gw_min)

    fig, ax_sh = plt.subplots()
    ax_omega = ax_sh.twinx()

    (line_sh,) = ax_sh.loglog(
        freq, sh, color=sh_color, linestyle=sh_linestyle, label=r"$S_h$"
    )
    (line_omega,) = ax_omega.loglog(
        freq,
        omega,
        color=omega_color,
        linestyle=omega_linestyle,
        label=r"$\Omega_{\mathrm{GW}}$",
    )

    ax_sh.set_xlabel(r"$f\ \mathrm{(Hz)}$", color=axis_color)
    ax_sh.set_ylabel(r"$S_h(f)\ \mathrm{[Hz^{-1}]}$", color=axis_color)
    ax_omega.set_ylabel(r"$\Omega_{\mathrm{GW}}(f)$", color=axis_color)
    ax_sh.tick_params(axis="x", colors=axis_color)
    ax_sh.tick_params(axis="y", colors=axis_color)
    ax_omega.tick_params(axis="y", colors=axis_color)
    for axis in (ax_sh, ax_omega):
        for spine in axis.spines.values():
            spine.set_color(axis_color)
    ax_sh.set_ylim(sh_ymin, None)
    ax_omega.set_ylim(omega_gw_min, None)
    ax_sh.set_axisbelow(True)
    ax_sh.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax_omega.grid(False)
    ax_sh.legend(
        handles=[line_sh, line_omega],
        loc="upper right",
        frameon=False,
        handlelength=2.5,
    )
    return fig


def plot_effective_psds(
    frequencies: jax.Array,
    networks: Sequence[Network],
    psds_by_network: Mapping[str, jax.Array | np.ndarray],
    *,
    colors: Sequence[str],
    linestyles: Sequence[str],
    frequency_slice: slice,
) -> Figure:
    """Overlay network effective PSDs on shared log–log axes."""
    if len(networks) != len(colors) or len(networks) != len(linestyles):
        raise ValueError("color and linestyle counts must match the networks")

    labels = [network.label for network in networks]
    fig, ax = plt.subplots()
    freq = np.asarray(frequencies[frequency_slice])
    for network, label, color, linestyle in zip(
        networks, labels, colors, linestyles, strict=True
    ):
        psd = np.asarray(psds_by_network[network.name])[frequency_slice]
        pos = np.isfinite(psd) & (psd > 0.0) & (freq > 0.0)
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
    ax.legend(handles=handles, **DETECTOR_COMPARISON_LEGEND)
    fig.tight_layout()
    return fig


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-effective-psd-pdf", type=Path, required=True)
    parser.add_argument("--figure-dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    root = paper_project_root()
    fiducials = load_fiducials()
    grid = load_analysis_grid()
    networks = resolve_networks(SPECTRUM_EXPERIMENT, DETECTOR_NETWORKS)
    use_paper_style()

    catalog_path = resolve_paper_path(args.catalog, root)
    source = CatalogSource(catalog_path, None, load_injection_spec(), "injection")
    observation = prepare_observation(source, fiducials=fiducials, grid=grid)
    frequencies = observation.frequencies
    frequency_slice = observation.frequency_slice
    figure = plot_omega_and_sh(
        frequencies,
        observation.spectral_density,
        frequency_slice,
        h0=fiducials["H0"],
        omega_gw_min=OMEGA_GW_MIN,
    )

    detector_colors, detector_linestyles = detector_network_styles(networks)
    effective_psds = {}
    for network in networks:
        sensitivities = load_sensitivity_map(network.detectors)
        effective_psds[network.name] = jnp.asarray(
            effective_psd(frequencies, list(network.detectors), sensitivities)
        )
    effective_psd_figure = plot_effective_psds(
        frequencies,
        networks,
        effective_psds,
        colors=detector_colors,
        linestyles=detector_linestyles,
        frequency_slice=frequency_slice,
    )

    output_path = resolve_paper_path(args.output_pdf, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.figure_dpi, bbox_inches="tight")
    print("saved figure:", output_path)

    effective_psd_output_path = resolve_paper_path(args.output_effective_psd_pdf, root)
    effective_psd_output_path.parent.mkdir(parents=True, exist_ok=True)
    effective_psd_figure.savefig(
        effective_psd_output_path, dpi=args.figure_dpi, bbox_inches="tight"
    )
    print("saved figure:", effective_psd_output_path)


if __name__ == "__main__":
    main()
