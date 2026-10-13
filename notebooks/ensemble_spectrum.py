import marimo

__generated_with = "0.25.1"
app = marimo.App()

with app.setup(hide_code=True):
    import os
    from collections.abc import Sequence
    from pathlib import Path

    from astrogwb.paper.runtime import configure_runtime

    # Backend configuration precedes waveform construction and array creation.
    configure_runtime(num_chains=1)

    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.axes import Axes as MplAxes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.projections import register_projection
    from matplotlib.ticker import LogFormatterMathtext, ScalarFormatter
    from numpy.typing import NDArray
    from scipy.optimize import brentq

    from astrogwb.cosmology import luminosity_distance
    from astrogwb.detector import effective_psd, gaussian_bin_scale
    from astrogwb.frequency import frequency_mask
    from astrogwb.gwb import (
        characteristic_noise,
        characteristic_strain,
        spectral_snr,
        spectral_snr_squared_per_log_frequency,
    )
    from astrogwb.gwb.importance import reference_catalog, reference_catalog_stem
    from astrogwb.inference import (
        ImportanceGaussianLikelihood,
        amplitude_shot_noise_variance,
    )
    from astrogwb.paper.cache import default_cache_dir
    from astrogwb.paper.config import (
        detector_registry,
        fiducials,
        population_metadata,
        waveform_metadata,
    )
    from astrogwb.paper.config.detectors import DetectorRegistry
    from astrogwb.paper.config.runs import CATALOGS_ROOT, FIGURES_DIR
    from astrogwb.paper.plotting import (
        DETECTOR_COMPARISON_LEGEND,
        DETECTOR_NETWORKS,
        SPECTRUM,
        Network,
        detector_network_styles,
        save_figures,
        use_paper_style,
    )
    from astrogwb.populations import build_population
    from astrogwb.populations.metadata import PopulationMetadata
    from astrogwb.simulators.core import batch_keys, load, write
    from astrogwb.simulators.polarization_power import (
        CatalogMetadata,
        PolarizationPowerData,
    )
    from astrogwb.simulators.spectra import (
        BackgroundSpectralDensityMetadata,
        spectra_ensemble,
    )
    from astrogwb.utils import years_to_seconds

    # gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes.
    # Restore them so plotting behaves as expected after the imports above.
    register_projection(MplAxes)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ensemble spectra, SNR table and characteristic amplitudes

    The paper's spectrum figure and SNR tables, from one Poisson ensemble of
    background spectra drawn with `IMRPhenomXAS_NRTidalv3`. For each paper network and
    several values of the lower band edge $f_{\min}$ it reports the SNR of every
    draw, summarized as the mean over draws, as LaTeX
    tables (one with the ET-COBA PSD for ET-$\Delta$, one with the ET-D PSD). The
    spectrum is plotted as the dimensionless characteristic amplitudes of
    Belgacem et al. (arXiv:2411.04029, Eqs. 47-49), with percentile bands from
    the draws. The same draws give the shot noise of a single realization about
    the mean.

    The ensemble is cached one draw per file
    (`astrogwb.simulators.spectra.spectra_ensemble`) under
    `default_cache_dir() / "ensembles"`, shared with
    `notebooks/shot_noise_coverage.py`, which uses the same record and seed
    as its injections. A single interactive draw lives in
    `notebooks/fiducial_spectrum.py`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Physics background

    **The spectrum.** The background is a sum over compact-binary mergers;
    its spectral density $S_h(f)$ is the incoherent sum of the sources'
    contributions. With a Poisson count,
    $$N \sim \mathrm{Poisson}(R\,T),$$
    so each draw is one realization and the spread over draws is the scatter a
    single observation would show.

    **The SNR.** For a stochastic signal and a network with effective noise
    PSD $S_{\mathrm{eff}}$ (see `spectral_snr`),
    $$\rho^2 = 2T \sum_f \Delta f\,\frac{S_h(f)^2}{S_{\mathrm{eff}}(f)^2},$$
    summed over the analysis band. Each bin adds a positive term, so raising
    $f_{\min}$ can only lower $\rho$. The table reports the *mean over draws of
    each draw's* $\rho$, not the SNR of the mean spectrum. The norm is convex,
    so by Jensen's inequality the former is never below the latter; the
    difference is the shot noise. The SNR of the mean is printed below as a
    cross-check only.

    **Characteristic amplitudes.** With $T$ the observation time,
    $$h_c = \sqrt{2 f S_h}, \qquad
    h_n = \frac{\sqrt{2 f S_{\mathrm{eff}}}}{(2 T f)^{1/4}},$$
    and
    $$\rho^2 = \int d\ln f\,\left(\frac{h_c}{h_n}\right)^4
    \quad\Longrightarrow\quad
    \left(\frac{h_c}{h_n}\right)^4 = \frac{d\rho^2}{d\ln f}.$$
    The gap between the curves on a log plot shows where the SNR accrues.

    **The network PSD.** A stochastic background is detected by
    cross-correlating detectors, weighting each pair by the overlap reduction
    function and combining them by inverse variance. A single detector has no
    cross-correlation, hence an infinite effective PSD and zero SNR; this is
    why the rows are *networks*, not single detectors.

    **Why the cut matters.** In the inspiral
    $$S_h \propto f^{2/3},$$
    but third-generation detectors (ET, CE) have noise that falls steeply towards
    low frequency, so much of their SNR sits at a few Hz. Current-generation
    networks gain little below $\sim 10$ Hz.

    **How the cut is applied.** As a mask on the full frequency grid, so bin
    widths at the band edge are preserved.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Configuration

    Setting `ASTROGWB_NOTEBOOK_SMOKE=1` swaps in a tiny ensemble with fixed
    counts so the whole notebook runs in seconds.
    """)
    return


@app.cell
def _():
    ROOT_DIR = Path(__file__).resolve().parents[1]

    num_draws = 1000
    seed = 41
    chunk_size = 1024
    observation_time = 1.0  # years
    maximum_frequency = 2048.0
    minimum_frequencies = [2.0, 5.0, 10.0, 20.0]
    minimum_redshift = 0.35
    maximum_redshift = 20.0
    approximant = "IMRPhenomXAS_NRTidalv3"
    # Shared with shot_noise_coverage.py: same record, seed and directory.
    ensemble_dir = default_cache_dir() / "ensembles"
    #: Where this notebook's figures and tables go: one folder per notebook under
    #: the figures root, named after the notebook file.
    BASE_DIR = ROOT_DIR / FIGURES_DIR / Path(__file__).stem
    use_paper_style(root=ROOT_DIR)
    # Percentiles of the plotted bands (16-84%).
    percentiles = (16.0, 50.0, 84.0)

    # Execution smoke tests exercise the same code with tiny draws.
    SMOKE = os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") == "1"
    if SMOKE:
        num_draws = 3
        chunk_size = 8

    metadata = BackgroundSpectralDensityMetadata(
        count="fixed" if SMOKE else "poisson",
        num_events=64 if SMOKE else None,
        observation_time=observation_time,
        hyperparameters={**fiducials(root=ROOT_DIR)},
        # Keep fine spacing through the SNR peak, then use a log tail for a
        # 1024-point grid. Keep these settings in sync with shot_noise_coverage.py,
        # which reads the same cached draws.
        waveform=waveform_metadata(
            root=ROOT_DIR,
            approximant=approximant,
            frequency_resolution=0.125,
            turnover_frequency=100.0,
            dlogf=0.01266,
        ),
        population=population_metadata(
            root=ROOT_DIR,
            minimum_redshift=minimum_redshift,
            maximum_redshift=maximum_redshift,
        ),
    )
    registry = detector_registry(root=ROOT_DIR)

    # ET-triangular with the ET-D PSD instead of the shared one. The geometry
    # is copied from E1-E3; the overrides deep-merge, so ``registry`` is intact.
    etd_psd = "ET-000A-18_ETD_hlf_psd.txt"
    etd_registry = detector_registry(
        root=ROOT_DIR,
        detectors={
            f"E{i}D": {
                "geometry": registry.detectors[f"E{i}"].geometry.model_dump(),
                "psd_reference": etd_psd,
            }
            for i in (1, 2, 3)
        },
        networks={
            "ET-triangular-D-PSD": ["E1D", "E2D", "E3D"],
            "ET-triangular-D-PSD-CE-Hanford": ["E1D", "E2D", "E3D", "C1"],
        },
    )
    etd_swap = {
        "ET-triangular": "ET-triangular-D-PSD",
        "ET-triangular-CE-Hanford": "ET-triangular-D-PSD-CE-Hanford",
    }
    coba_rows = DETECTOR_NETWORKS
    etd_rows = tuple((etd_swap.get(name, name), label) for name, label in coba_rows)
    return (
        BASE_DIR,
        ROOT_DIR,
        SMOKE,
        approximant,
        chunk_size,
        coba_rows,
        ensemble_dir,
        etd_registry,
        etd_rows,
        maximum_frequency,
        metadata,
        minimum_frequencies,
        num_draws,
        observation_time,
        percentiles,
        registry,
        seed,
    )


@app.cell
def _(SMOKE):
    # A smoke run draws three spectra: never let it overwrite the paper files.
    write_figures_switch = mo.ui.switch(value=not SMOKE, label="Write figures")
    write_figures_switch
    return (write_figures_switch,)


@app.cell
def _(write_figures_switch):
    write_figures = write_figures_switch.value
    return (write_figures,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Toolbox
    """)
    return


@app.function(hide_code=True)
def shot_noise_table(
    registry: DetectorRegistry,
    rows: Sequence[tuple[str, str]],
    frequencies: NDArray[np.float64],
    draws: NDArray[np.float64],
    observation_time: float,
    minimum_frequency: float,
    maximum_frequency: float,
) -> pd.DataFrame:
    r"""The shot noise of one draw about the ensemble mean, per network.

    Each draw's relative amplitude offset is its projection on the mean,
    :math:`\epsilon_i = (m | S_i - m) / (m | m)`, with the per-bin Gaussian
    scale of the network as the inner product's weight, so
    :math:`\rho\,\mathrm{sd}(\epsilon)` is the shot noise in units of the
    detector's amplitude error. The correlation column is the SNR-weighted
    mean correlation of the relative residuals across bins: one when the shot
    noise is a single amplitude mode.

    Parameters
    ----------
    registry
        Source of the networks.
    rows
        ``(network name, row label)`` pairs, in row order.
    frequencies, draws
        Frequency grid ``(F,)`` and the ensemble ``(M, F)``.
    observation_time
        Observing time in years.
    minimum_frequency, maximum_frequency
        Analysis band in Hz.

    Returns
    -------
    pandas.DataFrame
        ``rho``, ``rho sd(eps)`` and its error, and the correlation, by row.
    """
    grid = jnp.asarray(frequencies)
    mean = draws.mean(axis=0)
    residual = (draws - mean) / np.where(mean > 0, mean, 1.0)
    count = draws.shape[0]
    records = {}
    for name, label in rows:
        geometry, sensitivities = registry.build_network(name)
        noise = jnp.asarray(effective_psd(grid, geometry, sensitivities))
        keep = np.asarray(
            frequency_mask(grid, fmin=minimum_frequency, fmax=maximum_frequency)
            & jnp.isfinite(noise)
            & (noise > 0.0)
        )
        sigma = np.asarray(gaussian_bin_scale(noise, observation_time, grid))
        weight = np.where(
            keep & (mean > 0), (mean / np.where(keep, sigma, 1.0)) ** 2, 0
        )
        rho = float(np.sqrt(weight.sum()))
        eps = residual @ weight / weight.sum()
        spread = float(np.std(eps, ddof=1))
        top = np.argsort(-weight)[: min(200, int(np.count_nonzero(weight)))]
        pair = weight[top][:, None] * weight[top][None, :]
        records[label] = {
            "rho": rho,
            "rho sd(eps)": rho * spread,
            "error": rho * spread / np.sqrt(2.0 * (count - 1)),
            "bin correlation": float(
                np.sum(pair * np.corrcoef(residual[:, top].T)) / pair.sum()
            ),
        }
    return pd.DataFrame.from_dict(records, orient="index")


@app.function(hide_code=True)
def snr_per_draw(
    registry: DetectorRegistry,
    rows: Sequence[tuple[str, str]],
    frequencies: NDArray[np.float64],
    draws: NDArray[np.float64],
    observation_time: float,
    minimum_frequencies: Sequence[float],
    maximum_frequency: float,
) -> dict[str, NDArray[np.float64]]:
    """SNR of every draw, per network and lower frequency cut.

    Parameters
    ----------
    registry
        Source of the networks.
    rows
        ``(network name, row label)`` pairs, in row order.
    frequencies, draws
        Frequency grid ``(F,)`` and spectral densities ``(M, F)``.
    observation_time
        Observing time in years.
    minimum_frequencies
        Lower band edges in Hz.
    maximum_frequency
        Upper band edge in Hz.

    Returns
    -------
    dict
        Row label to SNRs of shape ``(len(minimum_frequencies), M)``.
    """
    grid = jnp.asarray(frequencies)
    spectra = jnp.asarray(draws)
    seconds = years_to_seconds(observation_time)
    values: dict[str, NDArray[np.float64]] = {}
    for name, label in rows:
        geometry, sensitivities = registry.build_network(name)
        noise = jnp.asarray(effective_psd(grid, geometry, sensitivities))
        values[label] = np.stack(
            [
                np.asarray(
                    spectral_snr(
                        spectra,
                        noise,
                        seconds,
                        grid,
                        frequency_mask=frequency_mask(
                            grid, fmin=fmin, fmax=maximum_frequency
                        ),
                    )
                )
                for fmin in minimum_frequencies
            ]
        )
    return values


@app.function(hide_code=True)
def snr_table(
    snrs: dict[str, NDArray[np.float64]],
    minimum_frequencies: Sequence[float],
) -> pd.DataFrame:
    """Mean over draws of the SNR.

    Parameters
    ----------
    snrs
        Output of :func:`snr_per_draw`.
    minimum_frequencies
        Lower band edges in Hz, one column each.

    Returns
    -------
    pandas.DataFrame
        Mean SNR indexed by row label, one column per cut.
    """
    return pd.DataFrame.from_dict(
        {label: values.mean(axis=-1) for label, values in snrs.items()},
        orient="index",
        columns=[rf"$f_{{\min}} = {fmin:g}$ Hz" for fmin in minimum_frequencies],
    )


@app.function(hide_code=True)
def latex_table(table: pd.DataFrame, caption: str, label: str) -> str:
    """Render an SNR table as a LaTeX ``table`` environment."""
    return table.to_latex(
        column_format="l" + "r" * table.shape[1],
        caption=caption,
        label=label,
        float_format="%.2f",
    )


@app.function(hide_code=True)
def format_axis_ticks(axis: MplAxes) -> None:
    """Render tick labels in mathtext on log and linear axes."""
    for scale, ticks in (
        (axis.get_xscale(), axis.xaxis),
        (axis.get_yscale(), axis.yaxis),
    ):
        ticks.set_major_formatter(
            LogFormatterMathtext()
            if scale == "log"
            else ScalarFormatter(useMathText=True)
        )


@app.function(hide_code=True)
def amplitude_curves(
    registry: DetectorRegistry,
    rows: Sequence[tuple[str, str]],
    frequencies: NDArray[np.float64],
    draws: NDArray[np.float64],
    observation_time: float,
    minimum_frequency: float,
    maximum_frequency: float,
    percentiles: tuple[float, float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, dict[str, NDArray]]]:
    r"""Characteristic strain and per-network noise and SNR density, with bands.

    Parameters
    ----------
    registry
        Source of the networks.
    rows
        ``(network name, row label)`` pairs.
    frequencies, draws
        Frequency grid ``(F,)`` and spectral densities ``(M, F)``.
    observation_time
        Observing time in years.
    minimum_frequency, maximum_frequency
        Plotted band in Hz.
    percentiles
        ``(low, middle, high)`` percentiles over draws.

    Returns
    -------
    frequency, strain, curves
        The plotted frequencies, the percentiles of :math:`h_c` over draws
        ``(3, F')``, and per network name the frequencies of its finite bins
        (``"frequency"``), its :math:`h_n` (``"noise"``) and the percentiles
        of :math:`d\mathrm{SNR}^2/d\ln f` (``"density"``, ``(3, F'')``).
    """
    grid = jnp.asarray(frequencies)
    spectra = jnp.asarray(draws)
    seconds = years_to_seconds(observation_time)
    band = np.asarray(
        frequency_mask(grid, fmin=minimum_frequency, fmax=maximum_frequency)
    )
    strain = np.percentile(
        np.asarray(characteristic_strain(spectra, grid)), percentiles, axis=0
    )
    curves = {}
    for name, _ in rows:
        geometry, sensitivities = registry.build_network(name)
        noise = jnp.asarray(effective_psd(grid, geometry, sensitivities))
        keep = band & np.isfinite(np.asarray(noise)) & (np.asarray(noise) > 0.0)
        density = spectral_snr_squared_per_log_frequency(spectra, noise, grid, seconds)
        curves[name] = {
            "frequency": np.asarray(grid)[keep],
            "noise": np.asarray(characteristic_noise(noise, grid, observation_time))[
                keep
            ],
            "density": np.percentile(np.asarray(density)[:, keep], percentiles, axis=0),
        }
    return np.asarray(grid)[band], strain[:, band], curves


@app.function(hide_code=True)
def network_legend_handles(
    networks: Sequence[Network],
    colors: Sequence[str],
    linestyles: Sequence[str],
) -> list[Line2D]:
    """One legend line per network, in the colors and styles of its curves."""
    return [
        Line2D([], [], color=color, linestyle=linestyle, label=network.label)
        for network, color, linestyle in zip(networks, colors, linestyles, strict=True)
    ]


@app.function(hide_code=True)
def plot_characteristic_amplitudes(
    frequency: NDArray[np.float64],
    strain: NDArray[np.float64],
    curves: dict[str, dict[str, NDArray]],
    networks: Sequence[Network],
    colors: Sequence[str],
    linestyles: Sequence[str],
    observation_time: float,
) -> Figure:
    r"""Stack :math:`h_c` against each :math:`h_n` over :math:`d\mathrm{SNR}^2/d\ln f`.

    The top panel shows the median characteristic strain against each network's
    :math:`h_n`. The fourth power of the ratio of the two is that network's SNR
    density, shown in the bottom panel on the shared frequency axis, so the
    comparison lines up with where the SNR accrues. On a log frequency axis the
    area under a density curve is the network's :math:`\mathrm{SNR}^2`. The
    lines are the median over draws, the middle row of ``strain`` and of each
    ``"density"``. The scatter over draws of both is below the line width; that
    of :math:`h_c` is shown by :func:`plot_strain_scatter`.
    """
    width, height = plt.rcParams["figure.figsize"]
    fig, (ax, ax_density) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(width, 1.6 * height),
        gridspec_kw={"height_ratios": (3, 2)},
    )
    positive = strain[1] > 0.0
    ax.loglog(
        frequency[positive], strain[1][positive], color=SPECTRUM["sh"], linestyle=":"
    )
    for network, color, linestyle in zip(networks, colors, linestyles, strict=True):
        curve = curves[network.name]
        ax.loglog(curve["frequency"], curve["noise"], color=color, linestyle=linestyle)
        ax_density.semilogx(
            curve["frequency"],
            curve["density"][1],
            color=color,
            linestyle=linestyle,
        )
    ax.set_ylim(0.3 * strain[1][positive].min(), 20.0 * strain[1].max())
    ax.set_ylabel(r"$h_c(f), \, h_n(f)$")
    ax_density.set_xlabel(r"$f\ \mathrm{(Hz)}$")
    ax_density.set_ylabel(r"$d\mathrm{SNR}^{2}/d\ln f$")
    ax_density.text(
        0.98,
        0.92,
        rf"$T = {observation_time:g}\ \mathrm{{yr}}$",
        transform=ax_density.transAxes,
        ha="right",
        va="top",
    )
    for panel in (ax, ax_density):
        panel.set_axisbelow(True)
        panel.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
        format_axis_ticks(panel)
    signal_legend = ax.legend(
        handles=[Line2D([], [], color=SPECTRUM["sh"], linestyle=":")],
        labels=[r"$h_c$"],
        loc="upper right",
        frameon=False,
        handlelength=2.5,
    )
    ax.add_artist(signal_legend)
    ax.legend(
        handles=network_legend_handles(networks, colors, linestyles),
        **DETECTOR_COMPARISON_LEGEND,
    )
    fig.tight_layout()
    return fig


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Draw the ensemble

    The Poisson ensemble, one cached file per draw.
    """)
    return


@app.cell
def _(chunk_size, ensemble_dir, metadata, num_draws, seed):
    _ensemble = spectra_ensemble(
        metadata, seed, num_draws, ensemble_dir, chunk_size=chunk_size
    )
    frequencies = np.asarray(_ensemble["frequencies"], dtype=np.float64)
    draws = _ensemble["spectral_density"]
    mean = draws.mean(axis=0)
    return draws, frequencies, mean


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. SNR per network and cut

    Each draw's SNR, then the mean over draws. The
    SNR of the mean spectrum is shown below for comparison only.
    """)
    return


@app.cell
def _(
    coba_rows,
    draws,
    etd_registry,
    etd_rows,
    frequencies,
    maximum_frequency,
    minimum_frequencies,
    observation_time,
    registry,
):
    snrs_coba = snr_per_draw(
        registry,
        coba_rows,
        frequencies,
        draws,
        observation_time,
        minimum_frequencies,
        maximum_frequency,
    )
    snrs_etd = snr_per_draw(
        etd_registry,
        etd_rows,
        frequencies,
        draws,
        observation_time,
        minimum_frequencies,
        maximum_frequency,
    )
    table_coba = snr_table(snrs_coba, minimum_frequencies)
    table_etd = snr_table(snrs_etd, minimum_frequencies)
    mo.vstack(
        [
            mo.md("**ET-$\\Delta$ with the ET-COBA PSD**"),
            table_coba.style.format("{:.2f}"),
            mo.md("**ET-$\\Delta$ with the ET-D PSD**"),
            table_etd.style.format("{:.2f}"),
        ]
    )
    return snrs_coba, table_coba, table_etd


@app.cell
def _(
    coba_rows,
    frequencies,
    maximum_frequency,
    mean,
    minimum_frequencies,
    observation_time,
    registry,
    snrs_coba,
):
    # Cross-check only: the SNR of the mean spectrum, which Jensen's
    # inequality keeps at or below the mean of the SNRs.
    _of_mean = snr_per_draw(
        registry,
        coba_rows,
        frequencies,
        mean[None, :],
        observation_time,
        minimum_frequencies,
        maximum_frequency,
    )
    mo.vstack(
        [
            mo.md("**Cross-check (ET-COBA PSD): SNR of the mean spectrum**"),
            pd.DataFrame(
                {label: _of_mean[label][:, 0] for label in _of_mean},
                index=[f"fmin = {f:g} Hz" for f in minimum_frequencies],
            ).T,
            mo.md("**Mean of the SNRs minus the SNR of the mean**"),
            pd.DataFrame(
                {
                    label: snrs_coba[label].mean(axis=-1) - _of_mean[label][:, 0]
                    for label in _of_mean
                },
                index=[f"fmin = {f:g} Hz" for f in minimum_frequencies],
            ).T,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. LaTeX table
    """)
    return


@app.cell
def _(BASE_DIR, table_coba, table_etd, write_figures):
    _blocks = {
        tag: latex_table(
            table,
            "Mean over the draws of each draw's SNR "
            "for each detector network as the lower edge of the analysis band is "
            f"raised, with {psd} PSD for ET-$\\Delta$.",
            f"tab:mean-spectrum-snr-{tag}",
        )
        for table, psd, tag in (
            (table_coba, "the ET-COBA", "coba"),
            (table_etd, "the ET-D", "etd"),
        )
    }
    if write_figures:
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        for _tag, _block in _blocks.items():
            _path = BASE_DIR / f"ensemble_spectrum_snr_{_tag}.tex"
            _path.write_text(_block, encoding="utf-8")
    mo.md("\n\n".join(f"```latex\n{block}\n```" for block in _blocks.values()))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Reading the table

    Along each row the mean SNR is non-increasing in $f_{\min}$ (each draw's
    is). The rows differ in how fast they fall: networks whose sensitivity
    extends to a few Hz lose the most when the cut rises. The range is the
    scatter a single observed realization has about the mean. The two tables
    differ only in the ET-$\Delta$ rows (ET-COBA vs. ET-D noise curve); the
    2L rows are identical.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Characteristic amplitudes

    The paper's spectrum figure, in two stacked panels on a shared frequency
    axis. The top panel shows the median $h_c$ against each network's $h_n$;
    the bottom panel, each network's
    $$\left(\frac{h_c}{h_n}\right)^4 = \frac{d\mathrm{SNR}^2}{d\ln f}$$
    computed draw by draw, with the same band. The area under a median curve in
    $\ln f$ is approximately that network's $\mathrm{SNR}^2$. The colors and line styles are those of the other
    network figures.
    """)
    return


@app.cell
def _(
    coba_rows,
    draws,
    frequencies,
    maximum_frequency,
    minimum_frequencies,
    observation_time,
    percentiles,
    registry,
):
    # Order is load-bearing: detector_network_styles colors by first appearance.
    plotted_networks = tuple(
        Network(name, label, registry.networks[name], registry)
        for name, label in coba_rows
    )
    plotted_colors, plotted_linestyles = detector_network_styles(plotted_networks)
    band_frequency, band_strain, amplitude_curves_by_network = amplitude_curves(
        registry,
        coba_rows,
        frequencies,
        draws,
        observation_time,
        minimum_frequencies[0],
        maximum_frequency,
        percentiles,
    )
    return (
        amplitude_curves_by_network,
        band_frequency,
        band_strain,
        plotted_colors,
        plotted_linestyles,
        plotted_networks,
    )


@app.cell
def _(
    BASE_DIR,
    ROOT_DIR,
    amplitude_curves_by_network,
    band_frequency,
    band_strain,
    plotted_colors,
    plotted_linestyles,
    observation_time,
    plotted_networks,
    write_figures,
):
    _fig = plot_characteristic_amplitudes(
        band_frequency,
        band_strain,
        amplitude_curves_by_network,
        plotted_networks,
        plotted_colors,
        plotted_linestyles,
        observation_time,
    )
    if write_figures:
        save_figures(
            {BASE_DIR / "characteristic_amplitudes.pdf": _fig},
            root=ROOT_DIR,
        )
    _fig
    return


@app.function(hide_code=True)
def plot_strain_scatter(
    frequency: NDArray[np.float64],
    strain: NDArray[np.float64],
) -> Figure:
    r"""Plot the scatter of :math:`h_c` over draws relative to its median.

    The band is the ``(low, high)`` percentile rows of ``strain`` as a
    percentage deviation from the ``middle`` row, so scatter far below the
    line width of the amplitude figure stays readable.
    """
    fig, ax = plt.subplots()
    positive = strain[1] > 0.0
    deviation = 100.0 * (strain[:, positive] / strain[1, positive] - 1.0)
    ax.fill_between(
        frequency[positive],
        deviation[0],
        deviation[2],
        color=SPECTRUM["sh"],
        alpha=0.25,
        linewidth=0,
    )
    ax.plot(frequency[positive], deviation[0], color=SPECTRUM["sh"], linewidth=0.8)
    ax.plot(frequency[positive], deviation[2], color=SPECTRUM["sh"], linewidth=0.8)
    ax.axhline(0.0, color=SPECTRUM["sh"], linewidth=0.5, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$f\ \mathrm{(Hz)}$")
    ax.set_ylabel(r"$h_c / \mathrm{median}(h_c) - 1\ (\%)$")
    ax.set_axisbelow(True)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    format_axis_ticks(ax)
    fig.tight_layout()
    return fig


@app.cell
def _(BASE_DIR, ROOT_DIR, band_frequency, band_strain, write_figures):
    _fig = plot_strain_scatter(band_frequency, band_strain)
    if write_figures:
        save_figures(
            {BASE_DIR / "characteristic_strain_scatter.pdf": _fig},
            root=ROOT_DIR,
        )
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 8. Shot noise of one draw

    One realization of the background scatters about the mean: a finite
    Poisson sum over mergers. Projected on the mean spectrum, the scatter is
    an amplitude offset $\epsilon$; $\rho\,\mathrm{sd}(\epsilon)$ compares
    it with the detector's amplitude error $1/\rho$, over the widest band
    ($f_{\min}$ = the first cut). The likelihood models it with
    `astrogwb.inference.ImportanceGaussianLikelihood`;
    `notebooks/shot_noise_coverage.py` checks the two against each other on
    these draws. A bin correlation of one means the shot noise is a single
    amplitude mode, which is what the rank-one likelihood assumes.
    """)
    return


@app.cell
def _(
    coba_rows,
    draws,
    frequencies,
    maximum_frequency,
    minimum_frequencies,
    observation_time,
    registry,
):
    shot_noise = shot_noise_table(
        registry,
        coba_rows,
        frequencies,
        draws,
        observation_time,
        minimum_frequencies[0],
        maximum_frequency,
    )
    shot_noise
    return (shot_noise,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 9. Minimum redshift

    Foreground subtraction removes the nearby sources, which dominate the shot
    noise: its variance goes as $1/z_{\min}$, while the signal loses only a
    little. This section scans the window's lower edge $z_{\min}$ with the
    rescaled importance estimator of `astrogwb.gwb.importance`. One reference
    catalog is generated at the redshift where $d_L = 1$ Mpc, below every
    window, and rescaled to each window's nodes, so the scan costs no
    waveforms. Per network it computes $\rho$ and the relative shot-noise
    scatter $s$ of the amplitude at the fiducials, and from them the relative
    uncertainty of the amplitude, $\sigma_{\mathrm{tot}}/A = (\rho^{-2} + s^2)^{1/2}$.
    At $z_{\min}$ of the ensemble, $\rho s$ is checked against the measured
    $\rho\,\mathrm{sd}(\epsilon)$ of section 8.
    """)
    return


@app.cell
def _(SMOKE, ROOT_DIR, approximant, metadata, minimum_frequencies):
    #: Where the reference waveforms sit: the redshift at which the fiducial
    #: luminosity distance is this many Mpc. Below every window of the scan.
    reference_distance_mpc = 1.0
    scan_seed = 42
    scan_num_samples = 2**14  # Sobol net, as in cosmology_grid_posteriors.py
    scan_redshift_nodes = 32  # converged to 1e-4 down to z_min = 0.01
    scan_redshifts = np.geomspace(0.01, 1.5, 49)
    scan_chunk_size = 4096
    if SMOKE:
        scan_num_samples = 64
        scan_redshift_nodes = 8
        scan_redshifts = np.geomspace(0.05, 3.0, 5)
        scan_chunk_size = 64

    scan_hyperparameters = {**metadata.hyperparameters}
    reference_redshift = redshift_at_distance(
        reference_distance_mpc,
        scan_hyperparameters["H0"],
        scan_hyperparameters["Omega_m"],
    )
    # The reference grid runs from the lowest observed frequency to past the
    # detector-frame merger at the reference redshift.
    scan_catalog_metadata = CatalogMetadata(
        waveform=waveform_metadata(
            root=ROOT_DIR,
            approximant=approximant,
            frequency_spacing="log",
            minimum_frequency=minimum_frequencies[0],
            maximum_frequency=4000.0,
            frequency_resolution=0.02,
            turnover_frequency=None,
        ),
        population=metadata.population,
        fiducials=scan_hyperparameters,
        num_samples=scan_num_samples,
        sampling="sobol",
        reference_redshift=reference_redshift,
    )
    mo.md(
        f"Reference redshift {reference_redshift:.6g} "
        f"($d_L$ = {reference_distance_mpc:g} Mpc); window edges "
        f"{scan_redshifts[0]:.3g} to {scan_redshifts[-1]:.3g}."
    )
    return (
        scan_catalog_metadata,
        scan_chunk_size,
        scan_hyperparameters,
        scan_redshift_nodes,
        scan_redshifts,
        scan_seed,
    )


@app.function(hide_code=True)
def redshift_at_distance(
    distance: float, hubble_constant: float, omega_m: float
) -> float:
    """The redshift at which the luminosity distance is ``distance`` Mpc.

    Parameters
    ----------
    distance
        Luminosity distance in Mpc.
    hubble_constant, omega_m
        The flat LCDM cosmology.

    Returns
    -------
    float
        The redshift, found by root-finding on
        :func:`astrogwb.cosmology.luminosity_distance`.
    """
    return float(
        brentq(
            lambda z: (
                float(luminosity_distance(z, hubble_constant, omega_m)) - distance
            ),
            1e-9,
            1.0,
            xtol=1e-16,
        )
    )


@app.function(hide_code=True)
def ensure_reference_catalog(
    metadata: CatalogMetadata,
    seed: int,
    directory: Path,
    *,
    chunk_size: int | None = None,
) -> PolarizationPowerData:
    """The reference catalog of ``metadata`` at ``seed``, drawn on a miss.

    The same file ``cosmology_grid_posteriors.py`` and
    ``shot_noise_coverage.py`` read and write,
    ``<directory>/reference_catalog-<key>-<seed>.h5``.

    Parameters
    ----------
    metadata
        The catalog record.
    seed
        The draw's seed.
    directory
        Where catalogs are kept.
    chunk_size
        Waveforms per batch when drawing.

    Returns
    -------
    PolarizationPowerData
        The catalog's frequencies, power and source parameters.

    Raises
    ------
    ValueError
        If the cached file records another metadata or seed.
    """
    path = directory / f"{reference_catalog_stem(metadata, seed)}.h5"
    if path.is_file():
        data, recorded, attrs = load(path, CatalogMetadata)
        if recorded.key() != metadata.key() or attrs.get("seed") != seed:
            raise ValueError(
                f"{path} records {recorded.key()} at seed {attrs.get('seed')}, "
                f"not the requested {metadata.key()} at seed {seed}"
            )
        return data  # ty: ignore[invalid-return-type]
    data = reference_catalog(metadata, batch_keys(seed, 1)[0], chunk_size=chunk_size)
    write(path, data, metadata, seed=seed)
    return data


@app.function(hide_code=True)
def minimum_redshift_scan(
    data: PolarizationPowerData,
    catalog_metadata: CatalogMetadata,
    population: PopulationMetadata,
    registry: DetectorRegistry,
    rows: Sequence[tuple[str, str]],
    frequencies: NDArray[np.float64],
    redshifts: NDArray[np.float64],
    hyperparameters: dict[str, float],
    observation_time: float,
    minimum_frequency: float,
    maximum_frequency: float,
    num_redshift_nodes: int,
) -> pd.DataFrame:
    r"""The SNR and shot-noise scatter of the amplitude against the window edge.

    For each ``redshifts`` entry the catalog is bound with that lower edge and
    evaluated at ``hyperparameters``; then, per network, :math:`\rho` is the
    SNR of the expected spectrum and :math:`s` the SNR-weighted relative
    standard deviation of its shot noise
    (:func:`astrogwb.inference.amplitude_shot_noise_variance`). The relative
    uncertainty of the amplitude is :math:`(\rho^{-2} + s^2)^{1/2}`.

    Parameters
    ----------
    data, catalog_metadata
        The reference catalog and its record. Its reference redshift must not
        exceed any entry of ``redshifts``.
    population
        The target population's record.
    registry, rows
        Source of the networks and ``(network name, row label)`` pairs.
    frequencies
        Observed frequency grid ``(F,)``.
    redshifts
        Lower edges of the redshift window.
    hyperparameters
        The point the spectrum is evaluated at.
    observation_time
        Observing time in years.
    minimum_frequency, maximum_frequency
        Analysis band in Hz.
    num_redshift_nodes
        Quadrature nodes per window.

    Returns
    -------
    pandas.DataFrame
        One row per (network label, window edge), with ``snr``, ``scatter`` and
        ``relative_uncertainty`` columns.
    """
    target = build_population(population.model_name, **population.model_kwargs)
    grid = jnp.asarray(frequencies)
    seconds = years_to_seconds(observation_time)
    networks = {}
    for name, label in rows:
        geometry, sensitivities = registry.build_network(name)
        noise = jnp.asarray(effective_psd(grid, geometry, sensitivities))
        keep = (
            frequency_mask(grid, fmin=minimum_frequency, fmax=maximum_frequency)
            & jnp.isfinite(noise)
            & (noise > 0.0)
        )
        networks[label] = (
            noise,
            gaussian_bin_scale(noise, observation_time, grid),
            keep,
        )
    records = []
    for edge in redshifts:
        likelihood = ImportanceGaussianLikelihood.from_catalog(
            data,
            catalog_metadata,
            population=target,
            frequencies=frequencies,
            num_redshift_nodes=num_redshift_nodes,
            density_sites=(),
            shot_noise="amplitude",
            observation_time=observation_time,
            minimum_redshift=float(edge),
        )
        spectrum, variance, _ = likelihood.predict(hyperparameters)
        for label, (noise, scale, keep) in networks.items():
            snr = float(
                spectral_snr(spectrum, noise, seconds, grid, frequency_mask=keep)
            )
            scatter = float(
                jnp.sqrt(amplitude_shot_noise_variance(spectrum, variance, scale, keep))
            )
            records.append(
                {
                    "network": label,
                    "minimum_redshift": float(edge),
                    "snr": snr,
                    "scatter": scatter,
                    "relative_uncertainty": float(np.hypot(1.0 / snr, scatter)),
                }
            )
    return pd.DataFrame.from_records(records)


@app.function(hide_code=True)
def plot_minimum_redshift_scan(
    scan: pd.DataFrame,
    networks: Sequence[Network],
    colors: Sequence[str],
    linestyles: Sequence[str],
    window_edge: float,
) -> Figure:
    r"""Plot :math:`\sigma_{\mathrm{tot}}/A` against the window's lower edge.

    One curve per network, with a dot at its minimum. The dotted vertical line is
    the analysis' own window edge ``window_edge``.

    Parameters
    ----------
    scan
        Output of :func:`minimum_redshift_scan`.
    networks, colors, linestyles
        Plotted networks, in order, with their styles.
    window_edge
        The lower edge of the paper's analysis window.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, ax = plt.subplots()
    for network, color, linestyle in zip(networks, colors, linestyles, strict=True):
        rows = scan[scan["network"] == network.label].sort_values("minimum_redshift")
        edge = rows["minimum_redshift"].to_numpy()
        total = 100.0 * rows["relative_uncertainty"].to_numpy()
        ax.semilogy(edge, total, color=color, linestyle=linestyle)
        best = int(np.argmin(total))
        ax.plot(edge[best], total[best], "o", color=color, markersize=4)
    ax.axvline(window_edge, color="k", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$z_{\min}$")
    ax.set_ylabel(r"$\sigma_{\mathrm{tot}}/A\ (\%)$")
    ax.set_axisbelow(True)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    format_axis_ticks(ax)
    ax.legend(
        handles=network_legend_handles(networks, colors, linestyles),
        **DETECTOR_COMPARISON_LEGEND,
    )
    fig.tight_layout()
    return fig


@app.cell
def _(
    ROOT_DIR,
    frequencies,
    maximum_frequency,
    metadata,
    minimum_frequencies,
    observation_time,
    registry,
    scan_catalog_metadata,
    scan_chunk_size,
    scan_hyperparameters,
    scan_redshift_nodes,
    scan_redshifts,
    scan_seed,
    coba_rows,
):
    scan_data = ensure_reference_catalog(
        scan_catalog_metadata,
        scan_seed,
        ROOT_DIR / CATALOGS_ROOT,
        chunk_size=scan_chunk_size,
    )
    scan = minimum_redshift_scan(
        scan_data,
        scan_catalog_metadata,
        metadata.population,
        registry,
        coba_rows,
        frequencies,
        scan_redshifts,
        scan_hyperparameters,
        observation_time,
        minimum_frequencies[0],
        maximum_frequency,
        scan_redshift_nodes,
    )
    return (scan,)


@app.cell
def _(
    BASE_DIR,
    ROOT_DIR,
    metadata,
    plotted_colors,
    plotted_linestyles,
    plotted_networks,
    scan,
    write_figures,
):
    _fig = plot_minimum_redshift_scan(
        scan,
        plotted_networks,
        plotted_colors,
        plotted_linestyles,
        float(metadata.population.model_kwargs["minimum_redshift"]),
    )
    if write_figures:
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        scan.to_csv(BASE_DIR / "minimum_redshift_scan.csv", index=False)
        save_figures({BASE_DIR / "minimum_redshift_scan.pdf": _fig}, root=ROOT_DIR)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Check against the ensemble

    At the window edge of the ensemble, the model's $\rho s$ (a property of the
    expected spectrum) should match the $\rho\,\mathrm{sd}(\epsilon)$ measured
    on the draws in section 8, and the optimum shows how much the choice of
    edge matters for each network.
    """)
    return


@app.cell
def _(
    ROOT_DIR,
    frequencies,
    maximum_frequency,
    metadata,
    minimum_frequencies,
    observation_time,
    registry,
    coba_rows,
    scan,
    scan_catalog_metadata,
    scan_chunk_size,
    scan_hyperparameters,
    scan_redshift_nodes,
    scan_seed,
    shot_noise,
):
    _edge = float(metadata.population.model_kwargs["minimum_redshift"])
    _at_edge = minimum_redshift_scan(
        ensure_reference_catalog(
            scan_catalog_metadata,
            scan_seed,
            ROOT_DIR / CATALOGS_ROOT,
            chunk_size=scan_chunk_size,
        ),
        scan_catalog_metadata,
        metadata.population,
        registry,
        coba_rows,
        frequencies,
        np.asarray([_edge]),
        scan_hyperparameters,
        observation_time,
        minimum_frequencies[0],
        maximum_frequency,
        scan_redshift_nodes,
    ).set_index("network")
    _best = scan.loc[
        scan.groupby("network")["relative_uncertainty"].idxmin()
    ].set_index("network")
    comparison = pd.DataFrame(
        {
            "rho sd(eps), draws": shot_noise["rho sd(eps)"],
            "rho s, model": _at_edge["snr"] * _at_edge["scatter"],
            "sigma_tot/A at edge (%)": 100.0 * _at_edge["relative_uncertainty"],
            "best z_min": _best["minimum_redshift"],
            "sigma_tot/A at best (%)": 100.0 * _best["relative_uncertainty"],
        }
    )
    comparison
    return


if __name__ == "__main__":
    app.run()
