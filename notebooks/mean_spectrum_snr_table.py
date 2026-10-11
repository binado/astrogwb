import marimo

__generated_with = "0.25.0"
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
    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray

    from astrogwb.detector import effective_psd, gaussian_bin_scale
    from astrogwb.frequency import frequency_mask
    from astrogwb.gwb import spectral_snr
    from astrogwb.paper.cache import default_cache_dir
    from astrogwb.paper.config import (
        detector_registry,
        fiducials,
        population_metadata,
        waveform_metadata,
    )
    from astrogwb.paper.config.detectors import DetectorRegistry
    from astrogwb.paper.plotting import DETECTOR_NETWORKS
    from astrogwb.simulators.spectra import (
        BackgroundSpectralDensityMetadata,
        spectra_ensemble,
    )
    from astrogwb.utils import years_to_seconds


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Mean-spectrum SNR vs. low-frequency cut

    How much SNR does each detector network lose as the lower edge of the
    analysis band moves up? This notebook draws a Poisson ensemble of
    background spectra with `IMRPhenomXAS`, averages it, computes the mean
    spectrum's SNR for each paper network at several values of $f_{\min}$,
    and prints LaTeX tables (networks $\times$ cuts), one with the ET-COBA
    PSD for ET-$\Delta$ and one with the ET-D PSD. The same draws give the
    shot noise of a single realization about the mean.

    The ensemble is cached one draw per file
    (`astrogwb.simulators.spectra.spectra_ensemble`) under
    `default_cache_dir() / "ensembles"`, shared with
    `notebooks/shot_noise_coverage.py`, which uses the same record and seed
    as its injections.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Physics background

    **The spectrum.** The background is a sum over compact-binary mergers;
    its spectral density $S_h(f)$ is the incoherent sum of the sources'
    contributions. With a Poisson count, $N \sim \mathrm{Poisson}(R\,T)$, so
    the mean over many draws estimates the *expected* spectrum, count
    fluctuations included.

    **The SNR.** For a stochastic signal and a network with effective noise
    PSD $S_{\mathrm{eff}}$ (see `spectral_snr`),
    $$\rho^2 = 2T \sum_f \Delta f\,\frac{S_h(f)^2}{S_{\mathrm{eff}}(f)^2},$$
    summed over the analysis band. Each bin adds a positive term, so raising
    $f_{\min}$ can only lower $\rho$.

    **The network PSD.** A stochastic background is detected by
    cross-correlating detectors, weighting each pair by the overlap reduction
    function and combining them by inverse variance. A single detector has no
    cross-correlation, hence an infinite effective PSD and zero SNR; this is
    why the rows are *networks*, not single detectors.

    **Why the cut matters.** In the inspiral $S_h \propto f^{2/3}$, but
    third-generation detectors (ET, CE) have noise that falls steeply towards
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
        coba_rows,
        etd_registry,
        etd_rows,
        chunk_size,
        ensemble_dir,
        maximum_frequency,
        metadata,
        minimum_frequencies,
        num_draws,
        observation_time,
        registry,
        seed,
    )


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
def snr_table(
    registry: DetectorRegistry,
    rows: Sequence[tuple[str, str]],
    frequencies: NDArray[np.float64],
    mean: NDArray[np.float64],
    observation_time: float,
    minimum_frequencies: list[float],
    maximum_frequency: float,
) -> pd.DataFrame:
    """Tabulate the mean spectrum's SNR per network and lower frequency cut.

    Parameters
    ----------
    registry
        Source of the networks.
    rows
        ``(network name, row label)`` pairs, in row order.
    frequencies, mean
        Frequency grid and mean spectral density.
    observation_time
        Observing time in years.
    minimum_frequencies
        Lower band edges in Hz, one column each.
    maximum_frequency
        Upper band edge in Hz.

    Returns
    -------
    pandas.DataFrame
        SNRs indexed by row label, with one column per cut.
    """
    grid = jnp.asarray(frequencies)
    spectrum = jnp.asarray(mean)
    seconds = years_to_seconds(observation_time)
    values: dict[str, list[float]] = {}
    for name, label in rows:
        geometry, sensitivities = registry.build_network(name)
        noise = jnp.asarray(effective_psd(grid, geometry, sensitivities))
        values[label] = [
            float(
                spectral_snr(
                    spectrum,
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
    return pd.DataFrame.from_dict(
        values,
        orient="index",
        columns=[rf"$f_{{\min}} = {fmin:g}$ Hz" for fmin in minimum_frequencies],
    )


@app.function(hide_code=True)
def latex_table(table: pd.DataFrame, caption: str, label: str) -> str:
    """Render an SNR table as a LaTeX ``table`` environment."""
    return table.to_latex(
        float_format="%.1f",
        column_format="l" + "r" * table.shape[1],
        caption=caption,
        label=label,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Draw and average

    The Poisson ensemble, one cached file per draw, averaged over the draw
    axis.
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
    """)
    return


@app.cell
def _(
    coba_rows,
    etd_registry,
    etd_rows,
    frequencies,
    maximum_frequency,
    mean,
    minimum_frequencies,
    observation_time,
    registry,
):
    table_coba = snr_table(
        registry,
        coba_rows,
        frequencies,
        mean,
        observation_time,
        minimum_frequencies,
        maximum_frequency,
    )
    table_etd = snr_table(
        etd_registry,
        etd_rows,
        frequencies,
        mean,
        observation_time,
        minimum_frequencies,
        maximum_frequency,
    )
    mo.vstack(
        [
            mo.md("**ET-$\\Delta$ with the ET-COBA PSD**"),
            table_coba,
            mo.md("**ET-$\\Delta$ with the ET-D PSD**"),
            table_etd,
        ]
    )
    return table_coba, table_etd


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. LaTeX table
    """)
    return


@app.cell
def _(table_coba, table_etd):
    blocks = [
        latex_table(
            table,
            "Mean-spectrum SNR of each detector network as the lower edge of "
            f"the analysis band is raised, with {psd} PSD for ET-$\\Delta$.",
            f"tab:mean-spectrum-snr-{tag}",
        )
        for table, psd, tag in (
            (table_coba, "the ET-COBA", "coba"),
            (table_etd, "the ET-D", "etd"),
        )
    ]
    mo.md("\n\n".join(f"```latex\n{block}\n```" for block in blocks))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Reading the table

    Along each row the SNR is non-increasing in $f_{\min}$. The rows differ
    in how fast they fall: networks whose sensitivity extends to a few Hz
    lose the most when the cut rises. The two tables differ only in the
    ET-$\\Delta$ rows (ET-COBA vs. ET-D noise curve); the 2L rows are identical.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Shot noise of one draw

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
