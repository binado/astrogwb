import marimo

__generated_with = "0.25.1"
app = marimo.App()

with app.setup(hide_code=True):
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from astrogwb.runtime import configure_runtime

    # Backend configuration precedes array creation.
    configure_runtime()

    import jax
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

    from astrogwb.background import (
        characteristic_noise,
        characteristic_strain,
        omega_gw_from_spectral_density,
        spectral_snr,
        spectral_snr_squared_per_log_frequency,
    )
    from astrogwb.config import load_config
    from astrogwb.detector import (
        effective_psd,
        pair_bin_scale,
        pair_response_moment,
        pair_terms,
    )
    from astrogwb.detector.config import DetectorRegistry
    from astrogwb.frequency import frequency_grid, frequency_mask
    from astrogwb.importance.catalog import read_catalog
    from astrogwb.inference import CatalogModel, Prediction, shot_noise_variance
    from astrogwb.inference import Network as LikelihoodNetwork
    from astrogwb.paper.plotting import (
        DETECTOR_COMPARISON_LEGEND,
        DETECTOR_NETWORKS,
        SPECTRUM,
        SPECTRUM_LINESTYLES,
        Network,
        detector_network_styles,
        save_figures,
        use_paper_style,
    )
    from astrogwb.utils import years_to_seconds

    # gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes.
    # Restore them so plotting behaves as expected after the imports above.
    register_projection(MplAxes)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # The expected spectrum at the fiducial

    The paper's spectrum figures and SNR table, from the model mean spectrum
    $S_h(f) = \mathbb{E}[S_h](f)$ at the fiducial hyperparameters: the
    importance-sampled `CatalogModel` of the shared reference catalog, the
    same model the grid-posterior notebooks evaluate. It writes

    - `fiducial_spectrum.pdf`: $S_h(f)$ and $\Omega_{\mathrm{GW}}(f)$;
    - `characteristic_amplitudes.pdf`: $h_c$ against each network's $h_n$,
      over $d\mathrm{SNR}^2/d\ln f$, for $T = 1$ yr;
    - `mean_spectrum_snr.tex`: each network's SNR as the lower band edge
      $f_{\min}$ is raised, and the shot noise $\rho s$ of one realization.

    A single Poisson realization scatters about this spectrum by a relative
    $s \approx 0.2\%$, so its SNR differs from the SNR of the mean only at
    order $s^2$. The ensemble check lives in `notebooks/ensemble_spectrum.py`.

    The catalog is read, never simulated: `Snakefile` rule
    `reference_catalog` writes `outputs/catalogs/reference-fiducial.h5`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configuration
    """)
    return


@app.cell
def _():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    #: Where this notebook's figures and table go: one folder per notebook under
    #: the figures root, named after the notebook file.
    BASE_DIR = ROOT_DIR / "outputs" / "figures" / Path(__file__).stem
    use_paper_style(root=ROOT_DIR)

    observation_time = 1.0  # years
    minimum_frequencies = [2.0, 5.0, 10.0, 20.0]
    # The shared reference catalog, written by Snakefile: 2^14 Sobol draws at
    # the default approximant, generated at reference redshift 1e-4. Its
    # population holds the analysis window (z in [0.35, 20]).
    catalog_path = ROOT_DIR / "outputs" / "catalogs" / "reference-fiducial.h5"
    num_redshift_nodes = 32
    # The shot noise carries each network's per-source response, from these
    # isotropic draws, as in the grid and coverage notebooks.
    response_draws, response_seed = 32_768, 0

    # A geometric grid over the default waveform's band, about 2000 points
    # (d ln f = 0.0035). The likelihood's loglinear grid is linear at 1 Hz
    # below 100 Hz, too coarse at the low-frequency noise wall: there it
    # overestimates the SNR above 10 Hz by about 2.5%. This grid agrees with
    # one twice as fine to 0.1%.
    _waveform = load_config(ROOT_DIR / "config").waveforms["default"]
    maximum_frequency = float(_waveform.maximum_frequency)
    frequencies = frequency_grid(
        "log", _waveform.minimum_frequency, maximum_frequency, 0.007
    )

    registry = load_config(ROOT_DIR / "config").registry
    network_rows = DETECTOR_NETWORKS
    # Order is load-bearing: detector_network_styles colors by first appearance.
    plotted_networks = tuple(
        Network(name, label, registry.networks[name], registry)
        for name, label in network_rows
    )
    plotted_colors, plotted_linestyles = detector_network_styles(plotted_networks)
    return (
        BASE_DIR,
        ROOT_DIR,
        catalog_path,
        frequencies,
        maximum_frequency,
        minimum_frequencies,
        network_rows,
        num_redshift_nodes,
        observation_time,
        plotted_colors,
        plotted_linestyles,
        plotted_networks,
        registry,
        response_draws,
        response_seed,
    )


@app.cell
def _():
    write_figures_switch = mo.ui.switch(value=True, label="Write figures")
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
def snr_table(
    prediction: Prediction,
    registry: DetectorRegistry,
    rows: Sequence[tuple[str, str]],
    frequencies: NDArray[np.float64],
    observation_time: float,
    minimum_frequencies: Sequence[float],
    maximum_frequency: float,
    *,
    response_draws: int,
    response_seed: int,
) -> pd.DataFrame:
    r"""SNR of the mean spectrum per network and lower band edge, and its shot noise.

    The SNR columns are :math:`\rho` of ``prediction.mean`` over
    :math:`[f_{\min}, f_{\max}]` for each of ``minimum_frequencies``. The last
    column is :math:`\rho s` over the widest band: :math:`s` is the relative
    standard deviation of one realization's shot noise projected on the
    network's noise-weighted template
    (:func:`astrogwb.inference.shot_noise_variance`), with the network's
    per-source response.

    Parameters
    ----------
    prediction
        The model at the fiducial, with its power moment.
    registry
        Source of the networks.
    rows
        ``(network name, row label)`` pairs, in row order.
    frequencies
        Observed frequency grid ``(F,)``, that of ``prediction``.
    observation_time
        Observing time in years.
    minimum_frequencies
        Lower band edges in Hz; the first is the widest band.
    maximum_frequency
        Upper band edge in Hz.
    response_draws, response_seed
        Isotropic draws of each network's per-pair response
        (:func:`~astrogwb.detector.pair_response_moment`).

    Returns
    -------
    pandas.DataFrame
        One row per label, one SNR column per cut and a ``rho s`` column.
    """
    grid = jnp.asarray(frequencies)
    seconds = years_to_seconds(observation_time)
    columns = [rf"$f_{{\min}} = {fmin:g}$ Hz" for fmin in minimum_frequencies]
    records = {}
    for name, label in rows:
        geometry = registry.build_network(name)
        noise = jnp.asarray(effective_psd(grid, geometry))
        finite = jnp.isfinite(noise) & (noise > 0.0)
        snrs = [
            float(
                spectral_snr(
                    prediction.mean,
                    noise,
                    seconds,
                    grid,
                    frequency_mask=frequency_mask(
                        grid, fmin=fmin, fmax=maximum_frequency
                    )
                    & finite,
                )
            )
            for fmin in minimum_frequencies
        ]
        band = np.asarray(
            frequency_mask(grid, fmin=minimum_frequencies[0], fmax=maximum_frequency)
        )
        scale = np.asarray(pair_bin_scale(grid, geometry, observation_time))
        overlap = np.stack([term.overlap for term in pair_terms(grid, geometry)])
        network = LikelihoodNetwork.from_pairs(
            scale,
            overlap,
            band & np.isfinite(scale) & (scale > 0.0),
            response_moment=pair_response_moment(
                jax.random.key(response_seed), response_draws, grid, geometry
            ),
        )
        scatter = float(jnp.sqrt(shot_noise_variance(prediction, network)))
        records[label] = [*snrs, snrs[0] * scatter]
        del network
    return pd.DataFrame.from_dict(
        records, orient="index", columns=[*columns, r"$\rho s$"]
    )


@app.function(hide_code=True)
def plot_spectral_density(
    frequency: NDArray[np.float64],
    spectral_density: NDArray[np.float64],
    omega_gw: NDArray[np.float64],
) -> Figure:
    r""":math:`S_h(f)` and :math:`\Omega_{\mathrm{GW}}(f)` on twin log axes."""
    fig, ax_sh = plt.subplots()
    ax_omega = ax_sh.twinx()
    (line_sh,) = ax_sh.loglog(
        frequency,
        spectral_density,
        color=SPECTRUM["sh"],
        linestyle=SPECTRUM_LINESTYLES["sh"],
        label=r"$S_h$",
    )
    (line_omega,) = ax_omega.loglog(
        frequency,
        omega_gw,
        color=SPECTRUM["omega_gw"],
        linestyle=SPECTRUM_LINESTYLES["omega_gw"],
        label=r"$\Omega_{\mathrm{GW}}$",
    )
    ax_sh.set_xlabel(r"$f\ \mathrm{(Hz)}$")
    ax_sh.set_ylabel(r"$S_h(f)\ \mathrm{[Hz^{-1}]}$")
    ax_omega.set_ylabel(r"$\Omega_{\mathrm{GW}}(f)$")
    ax_sh.set_axisbelow(True)
    ax_sh.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    for axis in (ax_sh, ax_omega):
        format_axis_ticks(axis)
    ax_sh.legend(handles=[line_sh, line_omega], loc="lower center", frameon=False)
    fig.tight_layout()
    return fig


@app.function(hide_code=True)
def amplitude_curves(
    spectral_density: jax.Array,
    registry: DetectorRegistry,
    networks: Sequence[Network],
    frequencies: NDArray[np.float64],
    observation_time: float,
    band: NDArray[np.bool_],
) -> dict[str, dict[str, NDArray[np.float64]]]:
    r"""Each network's :math:`h_n` and :math:`d\mathrm{SNR}^2/d\ln f` on its finite bins.

    Parameters
    ----------
    spectral_density
        The spectrum ``(F,)``.
    registry, networks
        Source of the networks and the networks.
    frequencies
        Observed frequency grid ``(F,)``.
    observation_time
        Observing time in years.
    band
        Plotted bins ``(F,)``.

    Returns
    -------
    dict
        Per network name, the frequencies of its finite plotted bins
        (``"frequency"``), its :math:`h_n` (``"noise"``) and
        :math:`d\mathrm{SNR}^2/d\ln f` (``"density"``) there.
    """
    grid = jnp.asarray(frequencies)
    seconds = years_to_seconds(observation_time)
    curves = {}
    for network in networks:
        noise = jnp.asarray(effective_psd(grid, registry.build_network(network.name)))
        keep = band & np.isfinite(np.asarray(noise)) & (np.asarray(noise) > 0.0)
        curves[network.name] = {
            "frequency": np.asarray(grid)[keep],
            "noise": np.asarray(characteristic_noise(noise, grid, observation_time))[
                keep
            ],
            "density": np.asarray(
                spectral_snr_squared_per_log_frequency(
                    spectral_density, noise, grid, seconds
                )
            )[keep],
        }
    return curves


@app.function(hide_code=True)
def plot_characteristic_amplitudes(
    frequency: NDArray[np.float64],
    strain: NDArray[np.float64],
    curves: Mapping[str, Mapping[str, NDArray[np.float64]]],
    networks: Sequence[Network],
    colors: Sequence[str],
    linestyles: Sequence[str],
    observation_time: float,
) -> Figure:
    r"""Stack :math:`h_c` against each :math:`h_n` over :math:`d\mathrm{SNR}^2/d\ln f`.

    The fourth power of the ratio of the two top-panel curves is that
    network's SNR density, shown in the bottom panel on the shared frequency
    axis, so the comparison lines up with where the SNR accrues. On a log
    frequency axis the area under a density curve is the network's
    :math:`\mathrm{SNR}^2`.
    """
    width, height = plt.rcParams["figure.figsize"]
    fig, (ax, ax_density) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(width, 1.6 * height),
        gridspec_kw={"height_ratios": (3, 2)},
    )
    positive = strain > 0.0
    ax.loglog(
        frequency[positive], strain[positive], color=SPECTRUM["sh"], linestyle=":"
    )
    for network, color, linestyle in zip(networks, colors, linestyles, strict=True):
        curve = curves[network.name]
        ax.loglog(curve["frequency"], curve["noise"], color=color, linestyle=linestyle)
        ax_density.semilogx(
            curve["frequency"], curve["density"], color=color, linestyle=linestyle
        )
    ax.set_ylim(0.3 * strain[positive].min(), 20.0 * strain.max())
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
    ## The model at the fiducial

    The reference catalog bound to its own population on the observed grid,
    evaluated at the catalog's fiducials. The expected number of mergers in the
    window over $T$ is the observer-frame total rate times $T$.

    The reference grid stops at 4000 Hz, and a node at redshift $z$ reads it at
    $f(1+z)/(1+z_{\mathrm{ref}})$, so the top observed bins see only the low
    nodes; queries past the edge contribute zero. The BNS signal there is
    already negligible: dropping the last 100, 50 or 20 Hz of the reference
    grid changes the spectrum by at most $2 \times 10^{-4}$, $5 \times 10^{-5}$
    or $3 \times 10^{-6}$ (all at about 2 kHz), so the part past the edge is
    smaller still.
    """)
    return


@app.cell
def _(catalog_path, frequencies, num_redshift_nodes, observation_time):
    _data, _metadata, _ = read_catalog(catalog_path)
    _model = CatalogModel.from_catalog(
        _data,
        _metadata,
        population=_metadata.population.build(),
        frequencies=frequencies,
        num_redshift_nodes=num_redshift_nodes,
        observation_time=observation_time,
    )
    fiducials = dict(_metadata.fiducials)
    prediction = _model.predict(fiducials)
    expected_mergers = float(prediction.extras["total_merger_rate"]) * years_to_seconds(
        observation_time
    )
    print(
        f"Expected mergers in T = {observation_time:g} yr: {expected_mergers:.4g}; "
        f"importance relative ESS "
        f"{float(prediction.extras['importance_relative_ess']):.3f}"
    )
    return expected_mergers, fiducials, prediction


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Spectral density and energy density
    """)
    return


@app.cell
def _(
    BASE_DIR,
    ROOT_DIR,
    fiducials,
    frequencies,
    maximum_frequency,
    minimum_frequencies,
    prediction,
    write_figures,
):
    band = np.asarray(
        frequency_mask(
            jnp.asarray(frequencies),
            fmin=minimum_frequencies[0],
            fmax=maximum_frequency,
        )
    )
    _omega = np.asarray(
        omega_gw_from_spectral_density(
            prediction.mean, jnp.asarray(frequencies), hubble_constant=fiducials["H0"]
        )
    )
    _fig = plot_spectral_density(
        frequencies[band], np.asarray(prediction.mean)[band], _omega[band]
    )
    if write_figures:
        save_figures({BASE_DIR / "fiducial_spectrum.pdf": _fig}, root=ROOT_DIR)
    _fig
    return (band,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Characteristic amplitudes

    $h_c = \sqrt{2 f S_h}$ against each network's
    $h_n = \sqrt{2 f S_{\mathrm{eff}}}/(2 T f)^{1/4}$, and below them
    $(h_c/h_n)^4 = d\mathrm{SNR}^2/d\ln f$, whose area in $\ln f$ is the
    network's $\mathrm{SNR}^2$.
    """)
    return


@app.cell
def _(
    BASE_DIR,
    ROOT_DIR,
    band,
    frequencies,
    observation_time,
    plotted_colors,
    plotted_linestyles,
    plotted_networks,
    prediction,
    registry,
    write_figures,
):
    _curves = amplitude_curves(
        prediction.mean,
        registry,
        plotted_networks,
        frequencies,
        observation_time,
        band,
    )
    _strain = np.asarray(
        characteristic_strain(prediction.mean, jnp.asarray(frequencies))
    )
    _fig = plot_characteristic_amplitudes(
        frequencies[band],
        _strain[band],
        _curves,
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## SNR per network and lower band edge

    $\rho^2 = 2T \sum_f \Delta f\, S_h^2/S_{\mathrm{eff}}^2$ over
    $[f_{\min}, f_{\max}]$, applied as a mask on the full grid so the band-edge
    bins keep their widths. The last column is the shot noise of one
    realization in units of the detector's amplitude error, over the widest
    band.
    """)
    return


@app.cell
def _(
    BASE_DIR,
    frequencies,
    maximum_frequency,
    minimum_frequencies,
    network_rows,
    observation_time,
    prediction,
    registry,
    response_draws,
    response_seed,
    write_figures,
):
    table = snr_table(
        prediction,
        registry,
        network_rows,
        frequencies,
        observation_time,
        minimum_frequencies,
        maximum_frequency,
        response_draws=response_draws,
        response_seed=response_seed,
    )
    _block = table.to_latex(
        column_format="l" + "r" * table.shape[1],
        caption=(
            "Optimal SNR of the expected background spectrum at the fiducial "
            f"for {observation_time:g} year of observation, as the lower edge of "
            r"the analysis band is raised, and the shot noise $\rho s$ of one "
            f"realization for $f_{{\\min}} = {minimum_frequencies[0]:g}$ Hz."
        ),
        label="tab:mean-spectrum-snr-coba",
        formatters=[
            *(["{:.1f}".format] * len(minimum_frequencies)),
            "{:.2f}".format,
        ],
    )
    if write_figures:
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "mean_spectrum_snr.tex").write_text(_block, encoding="utf-8")
    mo.vstack([table, mo.md(f"```latex\n{_block}\n```")])
    return (table,)


if __name__ == "__main__":
    app.run()
