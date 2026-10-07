import marimo

__generated_with = "0.25.0"
app = marimo.App()

with app.setup(hide_code=True):
    import os
    from pathlib import Path

    from astrogwb.paper.runtime import configure_runtime

    # Backend configuration precedes waveform construction and array creation.
    configure_runtime(num_chains=1)

    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray

    from astrogwb.detector import effective_psd
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
    from astrogwb.simulators.core import batch_keys
    from astrogwb.simulators.spectra import (
        BackgroundSpectralDensityMetadata,
        BackgroundSpectralDensitySimulator,
        stack_spectra,
    )
    from astrogwb.utils import years_to_seconds


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Mean-spectrum SNR vs. low-frequency cut

    How much SNR does each detector network lose as the lower edge of the
    analysis band moves up? This notebook draws a Poisson ensemble of
    background spectra with `IMRPhenomXAS`, averages it (cached on disk),
    computes the mean spectrum's SNR for every registry network at several
    values of $f_{\min}$, and prints a LaTeX table (networks $\times$ cuts).
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
    why the rows are the registry's *networks*, not single detectors.

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
    mean_spectra_outfile = (
        default_cache_dir() / "spectra" / "mean_spectra_IMRPhenomXAS.npz"
    )

    # Execution smoke tests exercise the same code with tiny draws.
    SMOKE = os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") == "1"
    if SMOKE:
        num_draws = 3
        chunk_size = 8
        mean_spectra_outfile = (
            default_cache_dir() / "spectra" / "smoke_mean_spectra_IMRPhenomXAS.npz"
        )

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
    return (
        chunk_size,
        mean_spectra_outfile,
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
def mean_spectrum(
    metadata: BackgroundSpectralDensityMetadata,
    num_draws: int,
    seed: int,
    chunk_size: int,
    outfile: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the frequency grid and the mean spectrum of an ensemble.

    The mean is served from ``outfile`` when it exists; otherwise the
    ensemble is drawn at ``batch_keys(seed, num_draws)``, averaged over the
    draw axis, and saved. The file records neither the metadata nor the seed,
    so delete it when either changes.

    Parameters
    ----------
    metadata
        Spectrum record to draw.
    num_draws
        Number of draws to average.
    seed
        Seed of the draw keys.
    chunk_size
        Events reduced per chunk; a cost setting only.
    outfile
        ``.npz`` cache holding ``frequencies`` and ``mean``.

    Returns
    -------
    tuple
        Frequencies ``(F,)`` and mean spectral density ``(F,)``.
    """
    if outfile.is_file():
        with np.load(outfile) as cached:
            return cached["frequencies"], cached["mean"]
    simulator = BackgroundSpectralDensitySimulator(metadata, chunk_size=chunk_size)
    data = stack_spectra([simulator(key) for key in batch_keys(seed, num_draws)])
    frequencies = np.asarray(data["frequencies"], dtype=np.float64)
    mean = np.mean(np.asarray(data["spectral_density"], dtype=np.float64), axis=0)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outfile, frequencies=frequencies, mean=mean)
    return frequencies, mean


@app.function(hide_code=True)
def snr_table(
    registry: DetectorRegistry,
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
        Source of the networks; every ``registry.networks`` entry is a row.
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
        SNRs indexed by network name, with one column per cut.
    """
    grid = jnp.asarray(frequencies)
    spectrum = jnp.asarray(mean)
    seconds = years_to_seconds(observation_time)
    rows: dict[str, list[float]] = {}
    for name in registry.networks:
        geometry, sensitivities = registry.build_network(name)
        noise = jnp.asarray(effective_psd(grid, geometry, sensitivities))
        rows[name] = [
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
        rows,
        orient="index",
        columns=[rf"$f_{{\min}} = {fmin:g}$ Hz" for fmin in minimum_frequencies],
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Draw and average

    The Poisson ensemble is averaged over the draw axis and cached next to
    the other spectra, together with the frequency grid the SNR needs.
    """)
    return


@app.cell
def _(chunk_size, mean_spectra_outfile, metadata, num_draws, seed):
    frequencies, mean = mean_spectrum(
        metadata, num_draws, seed, chunk_size, mean_spectra_outfile
    )
    return frequencies, mean


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. SNR per network and cut
    """)
    return


@app.cell
def _(
    frequencies,
    maximum_frequency,
    mean,
    minimum_frequencies,
    observation_time,
    registry,
):
    table = snr_table(
        registry,
        frequencies,
        mean,
        observation_time,
        minimum_frequencies,
        maximum_frequency,
    )
    table
    return (table,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. LaTeX table
    """)
    return


@app.cell
def _(table):
    latex = table.to_latex(
        float_format="%.1f",
        column_format="l" + "r" * table.shape[1],
        caption=(
            "Mean-spectrum SNR of each detector network as the lower edge of "
            "the analysis band is raised."
        ),
        label="tab:mean-spectrum-snr",
    )
    mo.md(f"```latex\n{latex}\n```")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Reading the table

    Along each row the SNR is non-increasing in $f_{\min}$. The rows differ
    in how fast they fall: networks whose sensitivity extends to a few Hz
    lose the most when the cut rises.
    """)
    return


if __name__ == "__main__":
    app.run()
