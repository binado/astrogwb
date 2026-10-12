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
    from matplotlib.patches import Patch
    from matplotlib.projections import register_projection
    from matplotlib.ticker import LogFormatterMathtext, ScalarFormatter
    from numpy.typing import NDArray

    from astrogwb.detector import effective_psd, gaussian_bin_scale
    from astrogwb.frequency import frequency_mask
    from astrogwb.gwb import (
        characteristic_noise,
        characteristic_strain,
        spectral_snr,
        spectral_snr_squared_per_log_frequency,
    )
    from astrogwb.paper.cache import default_cache_dir
    from astrogwb.paper.config import (
        detector_registry,
        fiducials,
        population_metadata,
        waveform_metadata,
    )
    from astrogwb.paper.config.detectors import DetectorRegistry
    from astrogwb.paper.config.runs import FIGURES_DIR
    from astrogwb.paper.plotting import (
        DETECTOR_COMPARISON_LEGEND,
        DETECTOR_NETWORKS,
        SPECTRUM,
        Network,
        detector_network_styles,
        save_figures,
        use_paper_style,
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
    background spectra drawn with `IMRPhenomXAS`. For each paper network and
    several values of the lower band edge $f_{\min}$ it reports the SNR of every
    draw, summarized as the mean over draws with its 16-84% range, as LaTeX
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

    num_draws = 200
    seed = 41
    chunk_size = 1024
    observation_time = 1.0  # years
    maximum_frequency = 2048.0
    minimum_frequencies = [2.0, 5.0, 10.0, 20.0]
    minimum_redshift = 0.35
    maximum_redshift = 20.0
    # Shared with shot_noise_coverage.py: same record, seed and directory.
    ensemble_dir = default_cache_dir() / "ensembles"
    #: Where this notebook's figures and tables go, under the one output root
    #: the workflow and `config/plotting.toml` agree on.
    BASE_DIR = ROOT_DIR / FIGURES_DIR / "fiducial_spectrum"
    use_paper_style(root=ROOT_DIR)
    # Percentiles of the plotted bands and of the table ranges (16-84%).
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
        waveform=waveform_metadata(root=ROOT_DIR, approximant="IMRPhenomXAS"),
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
    percentiles: tuple[float, float, float],
) -> pd.DataFrame:
    """Mean over draws of the SNR, with the outer percentiles as its range.

    Parameters
    ----------
    snrs
        Output of :func:`snr_per_draw`.
    minimum_frequencies
        Lower band edges in Hz, one column each.
    percentiles
        ``(low, middle, high)`` percentiles; the middle one is unused, the
        range is ``low`` to ``high``.

    Returns
    -------
    pandas.DataFrame
        ``mean (low--high)`` strings indexed by row label, one column per cut.
    """
    low, _, high = percentiles
    cells = {}
    for label, values in snrs.items():
        lo, hi = np.percentile(values, [low, high], axis=-1)
        cells[label] = [
            f"{mean:.1f} ({lo_i:.1f}--{hi_i:.1f})"
            for mean, lo_i, hi_i in zip(values.mean(axis=-1), lo, hi, strict=True)
        ]
    return pd.DataFrame.from_dict(
        cells,
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
def plot_amplitudes_and_density(
    frequency: NDArray[np.float64],
    strain: NDArray[np.float64],
    curves: dict[str, dict[str, NDArray]],
    networks: Sequence[Network],
    colors: Sequence[str],
    linestyles: Sequence[str],
    observation_time: float,
) -> Figure:
    r"""Stack :math:`h_c` against :math:`h_n` over :math:`d\mathrm{SNR}^2/d\ln f`.

    The fourth power of the gap between the signal band and a network's
    :math:`h_n` in the top panel is that network's SNR density below, so the
    shared frequency axis lines up where the SNR accrues. Bands and lines are
    the percentiles over draws; ``strain`` and each ``"density"`` carry
    ``(low, middle, high)`` rows.
    """
    width, height = plt.rcParams["figure.figsize"]
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(width, 1.6 * height),
        gridspec_kw={"height_ratios": (3, 2)},
    )
    positive = strain[1] > 0.0
    ax_top.fill_between(
        frequency[positive],
        np.maximum(strain[0][positive], np.finfo(float).tiny),
        strain[2][positive],
        color=SPECTRUM["sh"],
        alpha=0.25,
        linewidth=0,
    )
    ax_top.loglog(frequency[positive], strain[1][positive], color=SPECTRUM["sh"])
    for network, color, linestyle in zip(networks, colors, linestyles, strict=True):
        curve = curves[network.name]
        ax_top.loglog(
            curve["frequency"], curve["noise"], color=color, linestyle=linestyle
        )
        ax_bottom.semilogx(
            curve["frequency"],
            curve["density"][1],
            color=color,
            linestyle=linestyle,
        )
        ax_bottom.fill_between(
            curve["frequency"],
            curve["density"][0],
            curve["density"][2],
            color=color,
            alpha=0.2,
            linewidth=0,
        )
    ax_top.set_ylim(0.3 * strain[0][positive].min(), 20.0 * strain[2].max())
    ax_top.set_ylabel(r"$h_c(f), \, h_n(f)$")
    ax_bottom.set_xlabel(r"$f\ \mathrm{(Hz)}$")
    ax_bottom.set_ylabel(r"$d\mathrm{SNR}^{2}/d\ln f$")
    ax_bottom.text(
        0.98,
        0.92,
        rf"$T = {observation_time:g}\ \mathrm{{yr}}$",
        transform=ax_bottom.transAxes,
        ha="right",
        va="top",
    )
    for ax in (ax_top, ax_bottom):
        ax.set_axisbelow(True)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    signal_legend = ax_top.legend(
        handles=[
            (
                Patch(facecolor=SPECTRUM["sh"], alpha=0.25, linewidth=0),
                Line2D([], [], color=SPECTRUM["sh"]),
            )
        ],
        labels=[r"$h_c$ (median, 16--84\%)"],
        loc="upper left",
        frameon=False,
        handlelength=2.5,
    )
    ax_top.add_artist(signal_legend)
    ax_top.legend(
        handles=[
            Line2D([], [], color=color, linestyle=linestyle, label=network.label)
            for network, color, linestyle in zip(
                networks, colors, linestyles, strict=True
            )
        ],
        **DETECTOR_COMPARISON_LEGEND,
    )
    for ax in (ax_bottom, ax_top):
        format_axis_ticks(ax)
    fig.tight_layout(h_pad=0.4)
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

    Each draw's SNR, then the mean over draws with its 16-84% range. The
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
    percentiles,
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
    table_coba = snr_table(snrs_coba, minimum_frequencies, percentiles)
    table_etd = snr_table(snrs_etd, minimum_frequencies, percentiles)
    mo.vstack(
        [
            mo.md("**ET-$\\Delta$ with the ET-COBA PSD**"),
            table_coba,
            mo.md("**ET-$\\Delta$ with the ET-D PSD**"),
            table_etd,
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
            "Mean over the draws of each draw's SNR, with the 16--84\\% range, "
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

    The paper's spectrum figure. The median $h_c$ with its 16-84% band against
    each network's $h_n$ on top; below, each network's
    $$\left(\frac{h_c}{h_n}\right)^4 = \frac{d\mathrm{SNR}^2}{d\ln f}$$
    computed draw by draw, with the same band. The area under a median curve in
    $\ln f$ is approximately that network's $\mathrm{SNR}^2$. The colors and line styles are those of the other
    network figures.
    """)
    return


@app.cell
def _(
    BASE_DIR,
    ROOT_DIR,
    coba_rows,
    draws,
    frequencies,
    maximum_frequency,
    minimum_frequencies,
    observation_time,
    percentiles,
    registry,
    write_figures,
):
    # Order is load-bearing: detector_network_styles colors by first appearance.
    _networks = tuple(
        Network(name, label, registry.networks[name], registry)
        for name, label in coba_rows
    )
    _colors, _linestyles = detector_network_styles(_networks)
    _frequency, _strain, _curves = amplitude_curves(
        registry,
        coba_rows,
        frequencies,
        draws,
        observation_time,
        minimum_frequencies[0],
        maximum_frequency,
        percentiles,
    )
    _fig = plot_amplitudes_and_density(
        _frequency,
        _strain,
        _curves,
        _networks,
        _colors,
        _linestyles,
        observation_time,
    )
    if write_figures:
        save_figures(
            {BASE_DIR / "characteristic_amplitudes_and_snr_density.pdf": _fig},
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
    return


if __name__ == "__main__":
    app.run()
