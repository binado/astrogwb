"""Plot the fiducial SGWB spectrum and network effective PSDs.

The fiducial $S_h$ uses the same importance-weighted contraction as the MCMC
runs. $\\Omega_{\\mathrm{GW}}(f)$ shares the frequency axis on a dual $y$-scale.
"""

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
from astrogwb.frequency import frequency_mask as make_frequency_mask
from astrogwb.gwb import (
    omega_gw_from_spectral_density,
    spectral_density,
)
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
    make_merger_rate_and_log_weights_fn,
)
from astrogwb.waveform import polarization_power as compute_polarization_power
from astrogwb_paper.paths import paper_project_root
from astrogwb_paper.plotting import (
    DETECTOR_COMPARISON_LEGEND,
    SPECTRUM,
    combo_colors,
    use_paper_style,
)
from matplotlib.axes import Axes as MplAxes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection
from pluscross import load_catalog

# gwpy (via gwmock-signal) replaces matplotlib's rectilinear axes. Restore the
# standard matplotlib projection for consistent plotting.
register_projection(MplAxes)
jax.config.update("jax_enable_x64", True)


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _parse_network_definition(value: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in value:
        raise ValueError(
            f"invalid network definition {value!r}; expected NAME=DET1,DET2,..."
        )
    name, detector_list = value.split("=", 1)
    name = name.strip()
    detectors = tuple(detector.strip() for detector in detector_list.split(","))
    if not name or not detectors or any(not detector for detector in detectors):
        raise ValueError(
            f"invalid network definition {value!r}; expected NAME=DET1,DET2,..."
        )
    return name, detectors


def parse_networks(definitions: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Parse repeatable ``--network NAME=DET1,DET2`` flags in declaration order."""
    if not definitions:
        raise ValueError("at least one --network NAME=DET1,DET2,... is required")
    networks: dict[str, tuple[str, ...]] = {}
    for definition in definitions:
        name, detectors = _parse_network_definition(definition)
        if name in networks:
            raise ValueError(f"duplicate --network definition for {name!r}")
        networks[name] = detectors
    return networks


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
    _, _, proposal_logprob = compute_merger_rate_distance_and_logprob(
        fiducials, samples, redshift_grid=z_grid
    )
    merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
        fiducials=fiducials,
        redshift_grid=z_grid,
        proposal_logprob=proposal_logprob,
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
    omega_gw_min: float,
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
    ax.legend(handles=handles, **DETECTOR_COMPARISON_LEGEND)
    fig.tight_layout()
    return fig


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument(
        "--network",
        action="append",
        required=True,
        metavar="NAME=DET1,DET2,...",
        help="Detector-network definition (repeatable, declaration order).",
    )
    parser.add_argument("--detector-labels", nargs="+", required=True)
    parser.add_argument("--f-min", type=float, required=True)
    parser.add_argument("--f-max", type=float, required=True)
    parser.add_argument("--z-min", type=float, required=True)
    parser.add_argument("--z-max", type=float, required=True)
    parser.add_argument("--n-grid", type=int, required=True)
    parser.add_argument("--h0", type=float, required=True)
    parser.add_argument("--omega-m", type=float, required=True)
    parser.add_argument("--xi-0", type=float, required=True)
    parser.add_argument("--xi-n", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--z-peak", type=float, required=True)
    parser.add_argument("--local-merger-rate", type=float, required=True)
    parser.add_argument(
        "--omega-gw-min",
        type=float,
        required=True,
        help=(
            "Lower y-limit for Omega_GW; S_h ymin is taken from S_h at the "
            "frequency where Omega_GW is closest to this floor."
        ),
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-effective-psd-pdf", type=Path, required=True)
    parser.add_argument("--figure-dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    root = paper_project_root()
    try:
        networks = parse_networks(args.network)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if len(args.detector_labels) != len(networks):
        raise SystemExit("detector label count must match --network count")

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
        args.detector_labels,
        colors=detector_colors,
        linestyles=detector_linestyles,
        mask=mask,
    )

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


if __name__ == "__main__":
    main()
