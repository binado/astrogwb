import marimo

__generated_with = "0.25.0"
app = marimo.App()

with app.setup(hide_code=True):
    import os
    from collections.abc import Callable, Mapping, Sequence
    from dataclasses import dataclass
    from pathlib import Path

    from astrogwb.paper.runtime import configure_runtime

    # Backend configuration precedes waveform construction and array creation.
    configure_runtime(num_chains=1)

    import arviz_plots as azp
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numpyro.distributions as dist
    import pandas as pd
    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.projections import register_projection
    from numpy.typing import ArrayLike, NDArray
    from numpyro import handlers
    from scipy.stats import gaussian_kde

    from astrogwb.detector import (
        effective_psd,
        gaussian_bin_scale,
        log_frequency_noise_scale,
    )
    from astrogwb.distributions.amplitude import amplitude_prior
    from astrogwb.frequency import frequency_mask
    from astrogwb.gwb import spectral_snr
    from astrogwb.inference import (
        amplitude_H0_transform,
        gwb_amplitude_marginalized_model,
    )
    from astrogwb.paper.cache import default_cache_dir
    from astrogwb.paper.config import (
        detector_registry,
        fiducials,
        population_metadata,
        priors,
        waveform_metadata,
    )
    from astrogwb.paper.config.detectors import DetectorRegistry
    from astrogwb.paper.config.runs import FIGURES_DIR
    from astrogwb.paper.plotting import save_figures, use_paper_style
    from astrogwb.simulators.core import batch_keys, load, write
    from astrogwb.simulators.spectra import (
        SpectralDensityCatalog,
        SpectraMetadata,
        SpectraSimulator,
    )
    from astrogwb.utils import years_to_seconds

    # Statistics of a residual sample; each takes (samples, axis) so that the
    # bootstrap can evaluate every resample in one vectorized call.
    RESIDUAL_STATISTICS: dict[
        str, Callable[[NDArray[np.float64], int], NDArray[np.float64]]
    ] = {
        "mean": lambda r, axis: np.mean(r, axis=axis),
        "sd": lambda r, axis: np.std(r, axis=axis, ddof=1),
        "rms": lambda r, axis: np.sqrt(np.mean(r**2, axis=axis)),
        "q95_abs": lambda r, axis: np.quantile(np.abs(r), 0.95, axis=axis),
        "frac_abs_gt_1": lambda r, axis: np.mean(np.abs(r) > 1, axis=axis),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # How many injections, and how close to $z=0$? Shot noise vs. $\sigma(H_0)$

    The appendix has to justify two choices of the catalog-based
    forward model: the number of injections $N$ (we recommend
    $N = 128\mathrm{k}$–$256\mathrm{k}$) and the minimum redshift
    $z_{\min}$. The criterion is that **catalog shot noise stays below the
    expected statistical uncertainty on $H_0$**.

    This notebook measures exactly that, in four steps: the scatter of the
    spectrum itself, the SNR of each draw, the $H_0$ offset that scatter
    induces, and how all of it depends on $N$ and $z_{\min}$.

    | Figure | Role | Section |
    |---|---|---|
    | **A1** `paper_shot_noise_vs_detector.pdf` | paper | spectrum-level view |
    | **A2** `paper_offset_scaling.pdf` | paper | from scatter to an $H_0$ offset |
    | **A3** `paper_offset_vs_min_redshift.pdf` | paper | minimum redshift |
    | everything else | supporting | saved under the same directory |

    Draws contain source fluctuations only; no detector-noise realization is
    added, because detector noise is what $\sigma(H_0)$ already describes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Physics background

    **A background built from finitely many sources.** The stochastic
    background we model is a sum over compact-binary mergers. A forward model
    approximates that sum with $N$ Monte Carlo sources. Two different draws
    of $N$ sources give two different spectra, and that difference is *shot
    noise*. Each draw is normalized by the population's merger rate, so
    changing $N$ never changes the underlying spectrum, only how well one
    draw samples it.

    **Why the scatter falls as $N^{-1/2}$.** The spectrum is a mean over
    sources, so as long as one source has a finite variance, the usual
    central-limit scaling applies: the scatter of the estimator is
    $\propto N^{-1/2}$.

    **Why nearby sources spoil that, and why $z_{\min}$ matters.** A source
    at distance $d$ contributes $\propto 1/d^2$ to the spectrum, and in a
    locally Euclidean universe the number of sources in a shell goes as
    $d^2\,\mathrm{d}d$. The mean, $\int d^2 \cdot d^{-2}\,\mathrm{d}d$,
    converges, but the variance, $\int d^2 \cdot d^{-4}\,\mathrm{d}d \propto
    1/d_{\min}$, diverges towards small $d$. So the variance grows roughly as
    $1/z_{\min}$, driven by rare very close events that dominate individual
    draws and fatten the tails of the SNR distribution.

    **The right yardstick is $\sigma(H_0)$.** A spectrum that scatters by 10%
    is harmless if the detector can only measure it to 30%. What matters is
    how far shot noise moves the inferred $H_0$ compared with the width of the
    posterior itself.

    **Amplitude-only Fisher picture.** With every other parameter fixed, the
    template scales as $A(H_0) = H_{0,\mathrm{fid}}/H_0$. For a template $t$
    fitted to data $d$, with $(a|b)=\sum_f a(f)b(f)/\sigma_f^2$,
    $$\hat A = \frac{(d|t)}{(t|t)},\qquad
    H_{0,\mathrm{MAP}} = \frac{H_{0,\mathrm{fid}}}{\hat A},\qquad
    \sigma(H_0) = \frac{H_{0,\mathrm{fid}}}{\rho},\quad \rho = \sqrt{(t|t)}.$$

    **The offset statistic.** Fit each Monte Carlo template to one common
    reference spectrum (the best available estimate of the population mean)
    and record
    $$ r_i = \frac{H_{0,\mathrm{MAP},i} - H_{0,\mathrm{fid}}}{\sigma_{\mathrm{ref}}},
    \qquad \sigma_{\mathrm{ref}} = \frac{H_{0,\mathrm{fid}}}{\rho_{\mathrm{ref}}}, $$
    where $\rho_{\mathrm{ref}}$ is the SNR of the reference spectrum. The
    scale is **fixed**, not that of each draw. Dividing by each draw's own
    width $\sigma_i = H_{0,\mathrm{MAP},i}^2/(H_{0,\mathrm{fid}}\rho_i)$
    would be wrong here: a draw that fits a higher MAP also gets a wider
    $\sigma_i$, so numerator and denominator move together, which compresses
    one tail and stretches the other. With a fixed $\sigma_{\mathrm{ref}}$ the
    offset is linear in the MAP shift, and $\mathrm{sd}(r)$ reads directly as
    *the fraction of $\sigma(H_0)$ that shot noise adds*.

    **The tolerance.** Shot noise and detector noise are independent, so they
    add in quadrature: the effective uncertainty is
    $\sigma_{\mathrm{ref}}\sqrt{1+\mathrm{sd}(r)^2}$. We require
    $\mathrm{sd}(r) \le 1$: the shot-noise scatter of the $H_0$ offset is at
    most the statistical error itself, which inflates $\sigma(H_0)$ by at
    most $\sqrt{2}-1 \approx 41\%$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Configuration

    The count sweep runs at the baseline $z_{\min}$ across seven powers of two.
    The $N \times z_{\min}$ grid reuses the cached count-sweep ensembles at the
    baseline cutoff. Setting `ASTROGWB_NOTEBOOK_SMOKE=1` swaps in tiny counts
    and three draws, so the whole notebook can be executed in seconds.
    """)
    return


@app.cell
def _():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT_DIR / FIGURES_DIR / "spectrum_snrs"
    FIDUCIALS = fiducials(root=ROOT_DIR)

    # Count sweep (baseline cutoff) and the N x z_min grid.
    baseline_minimum_redshift = 0.35
    minimum_redshift = [0.05, 0.15, 0.35]
    num_events = [2**k for k in range(12, 19)]
    grid_counts = [2**14, 2**16, 2**17, 2**18]
    # Source counts overlaid on every figure except the scaling plots.
    paper_counts = [2**14, 2**16, 2**18]
    recommended_num_events = 2**17
    maximum_redshift = 20.0

    # The tolerance applies to sd(r): 1 inflates sigma(H0) by up to 41%.
    tolerance = 1.0
    n_bootstrap = 500
    offset_seed = 7

    num_draws = 200
    seed = 41
    data_seed = 42  # used only for the independent Poisson option
    chunk_size = 1024
    observation_time = 1.0  # years; affects SNR, not fixed-count normalization
    minimum_frequency = 2.0
    maximum_frequency = 2048.0
    network = "ET-2L-aligned-CE-Hanford"

    cache_dir = default_cache_dir() / "spectra"
    cache_only = False
    write_figures = True

    # Execution smoke tests exercise the same loops and waveform with tiny draws.
    SMOKE = os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") == "1"
    if SMOKE:
        num_events = [8, 16, 32]
        grid_counts = [8, 16, 32]
        paper_counts = [8, 16, 32]
        recommended_num_events = 32
        num_draws = 3
        chunk_size = 8
        n_bootstrap = 20
        write_figures = False

    # Read the shared draw's waveform rather than copying its scientific settings.
    base_metadata = SpectraMetadata(
        count="fixed",
        num_events=grid_counts[-1],
        observation_time=observation_time,
        hyperparameters={**FIDUCIALS},
        waveform=waveform_metadata(root=ROOT_DIR),
        population=population_metadata(
            root=ROOT_DIR,
            minimum_redshift=baseline_minimum_redshift,
            maximum_redshift=maximum_redshift,
        ),
    )
    # Poisson forward-model ensemble: the count follows rate * observation_time.
    poisson_metadata = SpectraMetadata.model_validate(
        {
            **base_metadata.model_dump(),
            "count": "fixed" if SMOKE else "poisson",
            "num_events": 64 if SMOKE else None,
        }
    )
    # The optional reference is one draw of the Poisson ensemble's metadata at
    # an independent seed (settings.data_seed).
    data_metadata = poisson_metadata
    h0_prior = priors(root=ROOT_DIR)["H0"]
    # Inverting the MLE gives the MAP only for a uniform prior on H0.
    if not isinstance(h0_prior, dist.Uniform) or float(h0_prior.low) <= 0:
        raise ValueError("the H0 MAP diagnostic requires a positive Uniform H0 prior")
    settings = AnalysisSettings(
        registry=detector_registry(root=ROOT_DIR),
        network=network,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        chunk_size=chunk_size,
        seed=seed,
        num_draws=num_draws,
        data_seed=data_seed,
        cache_dir=cache_dir,
        cache_only=cache_only,
        h0_prior=h0_prior,
    )
    use_paper_style(root=ROOT_DIR)
    # gwpy registers replacement default axes; ArviZ needs matplotlib axes.
    register_projection(Axes)
    return (
        BASE_DIR,
        ROOT_DIR,
        base_metadata,
        baseline_minimum_redshift,
        data_metadata,
        grid_counts,
        minimum_redshift,
        n_bootstrap,
        num_events,
        offset_seed,
        paper_counts,
        poisson_metadata,
        recommended_num_events,
        settings,
        tolerance,
        write_figures,
    )


@app.cell
def _():
    data_reference = "largest_mean"  # mean spectrum at max(num_events), or "poisson"
    return (data_reference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Toolbox

    Everything below is a named, top-level function; the narrative cells that
    follow only call them. Code is hidden by default.
    """)
    return


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class AnalysisSettings:
    """Detector network, frequency band, and cache policy shared by every case."""

    registry: DetectorRegistry
    network: str
    minimum_frequency: float
    maximum_frequency: float
    chunk_size: int
    #: The ensemble is ``num_draws`` draws at ``batch_keys(seed, num_draws)``;
    #: the optional Poisson reference is one draw at ``batch_keys(data_seed, 1)``.
    seed: int
    num_draws: int
    data_seed: int
    cache_dir: Path
    cache_only: bool
    h0_prior: dist.Uniform


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class SNRCase:
    """Compact analysis results; spectra remain in the checked HDF5 cache."""

    metadata: SpectraMetadata
    snrs: NDArray[np.float64]
    sigma_h0: NDArray[np.float64]
    summary: dict[str, int | float]
    catalog: SpectralDensityCatalog


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class H0FisherPrediction:
    """Template fits to common data and Fisher widths evaluated at their MAPs."""

    map_h0: NDArray[np.float64]
    sigma_h0: NDArray[np.float64]
    amplitude_mle: NDArray[np.float64]
    template_optimal_snr: NDArray[np.float64]
    at_prior_boundary: NDArray[np.bool_]

    def normalized_residuals(
        self, *, fiducial_h0: float, reference_sigma: float | None = None
    ) -> NDArray[np.float64]:
        """Return template MAP offsets in units of a Fisher width.

        Parameters
        ----------
        fiducial_h0
            Fiducial value the offsets are measured from.
        reference_sigma
            A fixed width shared by every draw. ``None`` divides each offset
            by its own Fisher width instead, which mixes numerator and
            denominator and skews the tails.
        """
        scale = self.sigma_h0 if reference_sigma is None else reference_sigma
        return (self.map_h0 - fiducial_h0) / scale


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class ResidualSummary:
    """Residual statistics with seeded bootstrap errors over draws."""

    num_draws: int
    values: Mapping[str, float]
    errors: Mapping[str, float]

    def row(self) -> dict[str, float]:
        """Flatten to ``{name: value, name_err: error}`` for tables."""
        row: dict[str, float] = {}
        for name, value in self.values.items():
            row[name] = value
            row[f"{name}_err"] = self.errors[name]
        return row


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class Reference:
    """The common spectrum that every template is fitted to."""

    kind: str
    catalog: SpectralDensityCatalog
    spectrum: NDArray[np.float64]
    snr: float
    averaged_draws: int
    seed: int

    def row(self) -> dict[str, str | int | float]:
        """Describe the reference for a table."""
        return {
            "reference": self.kind,
            "source_key": self.catalog.metadata.key(),
            "count": self.catalog.metadata.count,
            "num_events": int(self.catalog.n_events[0]),
            "averaged_draws": self.averaged_draws,
            "snr": self.snr,
            "seed": self.seed,
        }


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class OffsetAnalysis:
    """Fixed-width H0 offsets of an ensemble sweep, with per-draw cross-checks."""

    fiducial_h0: float
    reference_snr: float
    reference_sigma: float
    predictions: Mapping[int, H0FisherPrediction]
    fixed: Mapping[int, NDArray[np.float64]]
    per_draw: Mapping[int, NDArray[np.float64]]
    fixed_summary: Mapping[int, ResidualSummary]
    per_draw_summary: Mapping[int, ResidualSummary]


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class SpectrumStatistics:
    """Pointwise ensemble scatter, rather than uncertainty on its mean."""

    frequencies: NDArray[np.float64]
    mean: NDArray[np.float64]
    variance: NDArray[np.float64]
    standard_deviation: NDArray[np.float64]
    relative_variance: NDArray[np.float64]


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class NetworkSensitivity:
    """Per-bin and per-e-fold detector uncertainty on the full frequency grid."""

    band: NDArray[np.bool_]
    per_bin: NDArray[np.float64]
    per_log_frequency: NDArray[np.float64]


@app.function(hide_code=True)
def count_label(count: int) -> str:
    """Return a legend label such as ``$N = 2^{17}$`` for a source count."""
    exponent = count.bit_length() - 1
    if count == 2**exponent:
        return rf"$N = 2^{{{exponent}}}$"
    return rf"$N = {count}$"


@app.function(hide_code=True)
def redshift_label(minimum_redshift: float) -> str:
    """Return a legend label for a minimum redshift."""
    return rf"$z_{{\min}} = {minimum_redshift:.2f}$"


@app.function(hide_code=True)
def analysis_band(settings: AnalysisSettings, frequencies: jax.Array) -> jax.Array:
    """Return the analysis-band mask on the full frequency grid."""
    band = frequency_mask(
        frequencies,
        fmin=settings.minimum_frequency,
        fmax=settings.maximum_frequency,
    )
    if not np.any(np.asarray(band)):
        raise ValueError("the analysis frequency band contains no spectrum bins")
    return band


@app.function(hide_code=True)
def network_noise(settings: AnalysisSettings, frequencies: jax.Array) -> jax.Array:
    """Return the network's effective noise PSD on the full frequency grid."""
    if settings.network not in settings.registry.networks:
        raise ValueError(f"unknown network {settings.network!r}")
    geometry, sensitivities = settings.registry.build_network(settings.network)
    return jnp.asarray(effective_psd(frequencies, geometry, sensitivities))


@app.function(hide_code=True)
def compute_spectrum_snrs(
    catalog: SpectralDensityCatalog, settings: AnalysisSettings
) -> tuple[NDArray[np.float64], float]:
    """Return per-realization SNRs and the mean spectrum's SNR for one network.

    The observing time comes from the artifact, in years. Both calculations
    use the full frequency grid and mask afterwards, preserving the widths
    of bins at the analysis band's edges and all within-row correlations.
    No source sampling or waveform generation is performed here.
    """
    if not (
        np.isfinite(settings.minimum_frequency)
        and np.isfinite(settings.maximum_frequency)
        and 0.0 <= settings.minimum_frequency < settings.maximum_frequency
    ):
        raise ValueError("frequency bounds must be finite and 0 <= minimum < maximum")
    if not np.isfinite(catalog.observation_time) or catalog.observation_time <= 0:
        raise ValueError("observation_time must be finite and positive")
    if catalog.frequencies.size < 2:
        raise ValueError("SNR calculation requires at least two frequency bins")

    frequencies = jnp.asarray(catalog.frequencies)
    band = analysis_band(settings, frequencies)
    noise = network_noise(settings, frequencies)
    spectra = jnp.asarray(catalog.spectral_density)
    seconds = years_to_seconds(catalog.observation_time)
    snrs = np.asarray(
        spectral_snr(spectra, noise, seconds, frequencies, frequency_mask=band),
        dtype=np.float64,
    )
    mean_spectrum_snr = float(
        spectral_snr(
            jnp.mean(spectra, axis=0),
            noise,
            seconds,
            frequencies,
            frequency_mask=band,
        )
    )
    if not np.all(np.isfinite(snrs)) or not np.isfinite(mean_spectrum_snr):
        raise ValueError("SNR calculation produced nonfinite results")
    return snrs, mean_spectrum_snr


@app.function(hide_code=True)
def summarize_spectrum_snrs(
    snrs: ArrayLike, *, mean_spectrum_snr: float
) -> dict[str, int | float]:
    """Summarize realization SNRs, keeping the mean-spectrum statistic separate.

    A single draw has undefined sample SD; relative scatter is undefined when
    the mean is zero or SD is undefined. These values are returned as NaN,
    which table writers can represent as missing values.
    """
    values = np.asarray(snrs, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("snrs must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("snrs must be finite and nonnegative")
    if not np.isfinite(mean_spectrum_snr) or mean_spectrum_snr < 0:
        raise ValueError("mean_spectrum_snr must be finite and nonnegative")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
    return {
        "num_draws": int(values.size),
        "mean": mean,
        "median": float(np.median(values)),
        "sd": sd,
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "relative_scatter": sd / mean if mean > 0 else float("nan"),
        "mean_spectrum_snr": mean_spectrum_snr,
    }


@app.function(hide_code=True)
def draw_catalog(
    metadata: SpectraMetadata,
    seed: int,
    num_draws: int,
    settings: AnalysisSettings,
) -> SpectralDensityCatalog:
    """Serve or generate the spectra ``metadata`` gives at ``batch_keys(seed, num_draws)``."""
    path = settings.cache_dir / f"spectra-{metadata.key()}-{seed}-{num_draws}.h5"
    if path.is_file():
        outputs, _, _ = load(path, SpectraMetadata)
    elif settings.cache_only:
        raise FileNotFoundError(f"no cached spectra at {path}")
    else:
        outputs = SpectraSimulator(
            metadata, chunk_size=settings.chunk_size
        ).simulate_batch(batch_keys(seed, num_draws))
        write(path, outputs, metadata, seed=seed)
    return SpectralDensityCatalog.from_arrays(outputs, metadata)


@app.function(hide_code=True)
def analyze_metadata(metadata: SpectraMetadata, settings: AnalysisSettings) -> SNRCase:
    """Serve or generate one spectrum ensemble and summarize its SNRs."""
    catalog = draw_catalog(metadata, settings.seed, settings.num_draws, settings)
    snrs, mean_spectrum_snr = compute_spectrum_snrs(catalog, settings)
    if np.any(snrs <= 0):
        raise ValueError("sigma(H0) requires strictly positive SNR draws")
    sigma_h0 = np.asarray(metadata.fixed["H0"] / snrs, dtype=np.float64)
    if not np.all(np.isfinite(sigma_h0)):
        raise ValueError("sigma(H0) calculation produced nonfinite results")
    return SNRCase(
        metadata=metadata,
        snrs=snrs,
        sigma_h0=sigma_h0,
        summary=summarize_spectrum_snrs(snrs, mean_spectrum_snr=mean_spectrum_snr),
        catalog=catalog,
    )


@app.function(hide_code=True)
def analyze_case(
    base_metadata: SpectraMetadata,
    num_events: int,
    minimum_redshift: float,
    settings: AnalysisSettings,
) -> SNRCase:
    """Analyze one count/cutoff choice using the notebook's common settings."""
    population = base_metadata.population.with_model_kwargs(
        minimum_redshift=minimum_redshift
    )
    return analyze_metadata(
        SpectraMetadata.model_validate(
            {
                **base_metadata.model_dump(),
                "num_events": num_events,
                "population": population,
            }
        ),
        settings,
    )


@app.function(hide_code=True)
def build_grid(
    base_metadata: SpectraMetadata,
    counts: Sequence[int],
    redshifts: Sequence[float],
    settings: AnalysisSettings,
    *,
    reuse: Mapping[tuple[float, int], SNRCase] | None = None,
) -> dict[float, dict[int, SNRCase]]:
    """Analyze the ``N x z_min`` grid, reusing ensembles already in memory.

    Parameters
    ----------
    reuse
        Cases keyed by ``(minimum_redshift, num_events)``, such as the count
        sweep at the baseline cutoff. Others are served from the checked
        spectrum cache or generated.

    Returns
    -------
    dict
        ``grid[minimum_redshift][num_events]``, ordered by ``counts``.
    """
    known = dict(reuse or {})
    return {
        z: {
            n: known.get((z, n)) or analyze_case(base_metadata, n, z, settings)
            for n in counts
        }
        for z in redshifts
    }


@app.function(hide_code=True)
def select_reference(
    cases: Mapping[int, SNRCase],
    kind: str,
    settings: AnalysisSettings,
    *,
    poisson_metadata: SpectraMetadata | None = None,
) -> Reference:
    """Pick the common spectrum that every template is fitted to.

    Parameters
    ----------
    cases
        Ensembles keyed by source count.
    kind
        ``"largest_mean"`` averages every draw at the largest count (a mean
        spectrum, not a mean SNR). ``"poisson"`` uses one independently seeded
        Poisson draw, which includes count fluctuations.
    poisson_metadata
        Required for ``"poisson"``.
    """
    if kind == "largest_mean":
        case = cases[max(cases)]
        return Reference(
            kind=kind,
            catalog=case.catalog,
            spectrum=np.mean(case.catalog.spectral_density, axis=0, dtype=np.float64),
            snr=float(case.summary["mean_spectrum_snr"]),
            averaged_draws=case.catalog.num_draws,
            seed=settings.seed,
        )
    if kind == "poisson":
        if poisson_metadata is None:
            raise ValueError("the poisson reference needs its metadata")
        catalog = draw_catalog(poisson_metadata, settings.data_seed, 1, settings)
        snrs, _ = compute_spectrum_snrs(catalog, settings)
        return Reference(
            kind=kind,
            catalog=catalog,
            spectrum=np.asarray(catalog.spectral_density[0], dtype=np.float64),
            snr=float(snrs[0]),
            averaged_draws=1,
            seed=settings.data_seed,
        )
    raise ValueError('data_reference must be "largest_mean" or "poisson"')


@app.function(hide_code=True)
def plot_distribution_overlay(
    distributions: Mapping[str, ArrayLike], *, xlabel: str
) -> Figure:
    """Overlay ordered case distributions; use ECDFs if any case is constant."""
    if not distributions:
        raise ValueError("at least one distribution is required")
    arrays = {
        label: np.asarray(values, dtype=np.float64)
        for label, values in distributions.items()
    }
    if any(
        values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values))
        for values in arrays.values()
    ):
        raise ValueError("distributions must be non-empty finite 1D arrays")
    if any(np.ptp(values) == 0 for values in arrays.values()):
        # Plot ECDFs directly: a single draw must not be padded to match
        # longer cases, which would introduce NaNs into the distribution.
        figure, axis = plt.subplots(
            figsize=(6.4, 4.8),
            layout="constrained",
            subplot_kw={"axes_class": Axes},
        )
        for label, values in arrays.items():
            axis.ecdf(values, label=label)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Cumulative probability")
        axis.set_ylim(0, 1.05)
        axis.legend()
        return figure
    if len({values.size for values in arrays.values()}) != 1:
        raise ValueError("KDE overlays require the same draw count for each case")
    datasets = {
        label: xr.DataTree.from_dict(
            {"simulations": xr.Dataset({"value": ("draw", values)})}
        )
        for label, values in arrays.items()
    }
    collection = azp.plot_dist(
        datasets,
        group="simulations",
        sample_dims=["draw"],
        kind="kde",
        backend="matplotlib",
        # Sample extrema are not physical distribution boundaries. Reflecting
        # there exaggerates edge modes, especially for small ensembles.
        # Extend the evaluation grid so the density's tails remain visible.
        stats={
            "dist": {
                "bound_correction": False,
                "extend": True,
                "extend_fct": 3,
            }
        },
        aes={"color": ["model"]},
        aes_by_visuals={"dist": ["color"]},
        visuals={
            "credible_interval": False,
            "point_estimate": False,
            "point_estimate_text": False,
            "title": False,
            "remove_axis": False,
        },
        figure_kwargs={
            "figsize": (6.4, 4.8),
            "layout": "constrained",
            "subplot_kws": {"axes_class": Axes},
        },
    )
    figure = collection.viz["figure"].item()
    axis = figure.axes[0]
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.set_ylim(bottom=0)
    # Explicit handles retain the supplied case order in both overlays.
    axis.legend(axis.get_lines(), list(arrays))
    return figure


@app.function(hide_code=True)
def fisher_h0_prediction(
    amplitude_mle: ArrayLike,
    template_optimal_snr: ArrayLike,
    *,
    fiducial_h0: float,
    h0_prior: dist.Uniform,
) -> H0FisherPrediction:
    """Invert the model's MLE amplitude under a uniform prior on H0.

    The model pins each template at H0_fid and uses A(H0) = H0_fid/H0.
    The Fisher width at the MAP is 1 / (rho * abs(A'(H0_MAP))).
    Prior-boundary MAPs are flagged: their Gaussians are only local Fisher
    approximations, not representations of the truncated posterior.
    """
    amplitudes = np.asarray(amplitude_mle, dtype=np.float64)
    snrs = np.asarray(template_optimal_snr, dtype=np.float64)
    if any(
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
        for values in (amplitudes, snrs)
    ):
        raise ValueError("amplitudes and SNRs must be positive finite 1D arrays")
    if amplitudes.shape != snrs.shape:
        raise ValueError("each amplitude requires one template SNR")
    if not np.isfinite(fiducial_h0) or fiducial_h0 <= 0:
        raise ValueError("fiducial_h0 must be finite and positive")
    low, high = float(h0_prior.low), float(h0_prior.high)
    if not 0 < low < high:
        raise ValueError("H0 prior bounds must be positive and increasing")
    unconstrained_map = fiducial_h0 / amplitudes
    map_h0 = np.clip(unconstrained_map, low, high)
    return H0FisherPrediction(
        map_h0=map_h0,
        sigma_h0=map_h0**2 / (fiducial_h0 * snrs),
        amplitude_mle=amplitudes,
        template_optimal_snr=snrs,
        at_prior_boundary=(unconstrained_map <= low) | (unconstrained_map >= high),
    )


@app.function(hide_code=True)
def gaussian_mixture_density(
    grid: ArrayLike, centres: ArrayLike, widths: ArrayLike
) -> NDArray[np.float64]:
    """Evaluate an equally weighted mixture of per-realization Gaussians."""
    x = np.asarray(grid, dtype=np.float64)
    means = np.asarray(centres, dtype=np.float64)
    sigmas = np.asarray(widths, dtype=np.float64)
    if any(
        values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values))
        for values in (x, means, sigmas)
    ):
        raise ValueError("grid, centres and widths must be non-empty finite 1D arrays")
    if means.shape != sigmas.shape or np.any(sigmas <= 0):
        raise ValueError("each centre requires one strictly positive width")
    standardized = (x[:, None] - means[None, :]) / sigmas[None, :]
    densities = np.exp(-0.5 * standardized**2) / (np.sqrt(2 * np.pi) * sigmas)
    return np.mean(densities, axis=1)


@app.function(hide_code=True)
def template_amplitude_statistics(
    templates: ArrayLike,
    data: ArrayLike,
    scale: ArrayLike,
    band: ArrayLike,
    *,
    fiducial_h0: float,
    h0_prior: dist.Uniform,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read sufficient statistics from the existing marginalized model.

    Only the deterministic statistics are consumed; the quadrature evidence
    is not used to compute either the analytic MAP or the Fisher width.
    """

    def statistics(template: jax.Array) -> tuple[jax.Array, jax.Array]:
        def spectrum(
            params: Mapping[str, ArrayLike],
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            return template, {}

        trace = handlers.trace(gwb_amplitude_marginalized_model).get_trace(
            spectral_density_fn=spectrum,
            observed_spectral_density=jnp.asarray(data),
            priors={},
            scale=jnp.asarray(scale),
            amplitude_prior=amplitude_prior(
                h0_prior, amplitude_H0_transform(fiducial_h0)
            ),
            frequency_mask=jnp.asarray(band),
        )
        return trace["amplitude_mle"]["value"], trace["template_optimal_snr"]["value"]

    amplitude, snr = jax.jit(jax.vmap(statistics))(jnp.asarray(templates))
    return np.asarray(amplitude, dtype=np.float64), np.asarray(snr, dtype=np.float64)


@app.function(hide_code=True)
def fit_templates(
    case: SNRCase, reference: Reference, settings: AnalysisSettings
) -> H0FisherPrediction:
    """Fit every template in ``case`` to the reference spectrum."""
    catalog = case.catalog
    if not np.array_equal(catalog.frequencies, reference.catalog.frequencies):
        raise ValueError("data and templates must use the same frequency grid")
    frequencies = jnp.asarray(catalog.frequencies)
    scale = gaussian_bin_scale(
        network_noise(settings, frequencies),
        catalog.observation_time,
        frequencies,
    )
    fiducial_h0 = case.metadata.fixed["H0"]
    amplitudes, snrs = template_amplitude_statistics(
        catalog.spectral_density,
        reference.spectrum,
        scale,
        analysis_band(settings, frequencies),
        fiducial_h0=fiducial_h0,
        h0_prior=settings.h0_prior,
    )
    return fisher_h0_prediction(
        amplitudes,
        snrs,
        fiducial_h0=fiducial_h0,
        h0_prior=settings.h0_prior,
    )


@app.function(hide_code=True)
def bootstrap_statistic(
    values: ArrayLike,
    statistic: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """Return a statistic and its bootstrap standard error over draws.

    Parameters
    ----------
    values
        One-dimensional sample; draws are resampled with replacement.
    statistic
        Reduces ``(samples, axis)``, so all resamples evaluate at once.
    n_bootstrap
        Number of resamples.
    seed
        Seed of the resampling generator; the same seed reuses the same
        resample indices for every statistic.

    Returns
    -------
    tuple
        The statistic on the original sample and the SD of its replicates.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2:
        raise ValueError("bootstrap needs a 1D sample of at least two draws")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least two")
    indices = np.random.default_rng(seed).integers(
        0, data.size, size=(n_bootstrap, data.size)
    )
    replicates = statistic(data[indices], -1)
    return float(statistic(data, -1)), float(np.std(replicates, ddof=1))


@app.function(hide_code=True)
def summarize_residuals(
    residuals: ArrayLike, *, n_bootstrap: int, seed: int
) -> ResidualSummary:
    """Summarize offsets: mean, sd, rms, q95 of ``|r|`` and ``P(|r| > 1)``."""
    data = np.asarray(residuals, dtype=np.float64)
    values: dict[str, float] = {}
    errors: dict[str, float] = {}
    for name, statistic in RESIDUAL_STATISTICS.items():
        values[name], errors[name] = bootstrap_statistic(
            data, statistic, n_bootstrap=n_bootstrap, seed=seed
        )
    return ResidualSummary(num_draws=int(data.size), values=values, errors=errors)


@app.function(hide_code=True)
def analyze_offsets(
    cases: Mapping[int, SNRCase],
    reference: Reference,
    settings: AnalysisSettings,
    *,
    n_bootstrap: int,
    seed: int,
) -> OffsetAnalysis:
    """Fit each ensemble to one reference and measure its H0 offsets.

    The headline residuals use the fixed width
    ``sigma_ref = H0_fid / rho_ref``; per-draw widths are kept as a
    cross-check.

    Parameters
    ----------
    cases
        Ensembles keyed by source count (the key labels the result).
    reference
        Common spectrum and its SNR ``rho_ref``.
    settings
        Detector network and band.
    n_bootstrap, seed
        Bootstrap resamples and seed for the summary errors.
    """
    if not cases:
        raise ValueError("at least one ensemble is required")
    if not np.isfinite(reference.snr) or reference.snr <= 0:
        raise ValueError("the reference SNR must be finite and positive")
    fiducial_h0 = next(iter(cases.values())).metadata.fixed["H0"]
    reference_sigma = fiducial_h0 / reference.snr
    predictions = {
        count: fit_templates(case, reference, settings) for count, case in cases.items()
    }
    fixed = {
        count: prediction.normalized_residuals(
            fiducial_h0=fiducial_h0, reference_sigma=reference_sigma
        )
        for count, prediction in predictions.items()
    }
    per_draw = {
        count: prediction.normalized_residuals(fiducial_h0=fiducial_h0)
        for count, prediction in predictions.items()
    }
    return OffsetAnalysis(
        fiducial_h0=fiducial_h0,
        reference_snr=reference.snr,
        reference_sigma=reference_sigma,
        predictions=predictions,
        fixed=fixed,
        per_draw=per_draw,
        fixed_summary={
            count: summarize_residuals(r, n_bootstrap=n_bootstrap, seed=seed)
            for count, r in fixed.items()
        },
        per_draw_summary={
            count: summarize_residuals(r, n_bootstrap=n_bootstrap, seed=seed)
            for count, r in per_draw.items()
        },
    )


@app.function(hide_code=True)
def analyze_grid(
    grid: Mapping[float, Mapping[int, SNRCase]],
    settings: AnalysisSettings,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[float, OffsetAnalysis]:
    """Measure offsets for every ``N`` at each cutoff against its own reference.

    A cutoff changes the population, so each ``z_min`` uses its own reference:
    the mean spectrum of its largest-``N`` ensemble, with its own
    ``sigma_ref``.
    """
    return {
        z: analyze_offsets(
            cases,
            select_reference(cases, "largest_mean", settings),
            settings,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for z, cases in grid.items()
    }


@app.function(hide_code=True)
def offsets_table(offsets: OffsetAnalysis, *, ensemble: str) -> pd.DataFrame:
    """Tabulate fixed-width statistics beside the per-draw cross-check."""
    rows = []
    for count, fixed in offsets.fixed_summary.items():
        per_draw = offsets.per_draw_summary[count]
        rows.append(
            {
                "ensemble": ensemble,
                "num_events": count,
                **fixed.row(),
                "sd_per_draw": per_draw.values["sd"],
                "q95_abs_per_draw": per_draw.values["q95_abs"],
                "prior_boundary_fraction": float(
                    np.mean(offsets.predictions[count].at_prior_boundary)
                ),
            }
        )
    return pd.DataFrame(rows)


@app.function(hide_code=True)
def verdict_table(
    summaries: Mapping[float, Mapping[int, ResidualSummary]],
    counts: Sequence[int],
    *,
    tolerance: float,
) -> pd.DataFrame:
    """Compare ``sd(r)`` at the recommended counts with the tolerance."""
    rows = []
    for z, by_count in summaries.items():
        for count in counts:
            summary = by_count[count]
            rows.append(
                {
                    "minimum_redshift": z,
                    "num_events": count,
                    "sd": summary.values["sd"],
                    "sd_err": summary.errors["sd"],
                    "q95_abs": summary.values["q95_abs"],
                    "q95_abs_err": summary.errors["q95_abs"],
                    "passes": bool(summary.values["sd"] <= tolerance),
                }
            )
    return pd.DataFrame(rows)


@app.function(hide_code=True)
def verdict_text(table: pd.DataFrame, *, tolerance: float) -> str:
    """Write the appendix conclusion with the table's numbers filled in."""
    inflation = 100 * (np.sqrt(1 + tolerance**2) - 1)
    lines = [
        (
            rf"Tolerance: $\mathrm{{sd}}(r) \le {tolerance:g}$, an inflation "
            rf"of $\sigma(H_0)$ by {inflation:.2g}%."
        ),
        "",
    ]
    by_redshift: dict[float, list[dict]] = {}
    for record in table.to_dict("records"):
        by_redshift.setdefault(float(record["minimum_redshift"]), []).append(record)
    passing: list[float] = []
    for z, records in by_redshift.items():
        cells = "; ".join(
            f"$N = {int(r['num_events'])}$: sd $= {r['sd']:.3f} \\pm "
            f"{r['sd_err']:.3f}$, $q_{{95}}|r| = {r['q95_abs']:.3f}$ "
            f"({'pass' if r['passes'] else 'fail'})"
            for r in records
        )
        lines.append(f"- $z_{{\\min}} = {z:.2f}$ — {cells}")
        if all(r["passes"] for r in records):
            passing.append(z)
    lines.append("")
    if passing:
        listed = ", ".join(rf"$z_{{\min}} = {z:.2f}$" for z in passing)
        lines.append(
            f"Every tested $N$ in the table meets the tolerance for {listed}; "
            rf"the smallest such cutoff is $z_{{\min}} = {min(passing):.2f}$."
        )
    else:
        lines.append("No tested cutoff meets the tolerance at every listed $N$.")
    return "\n".join(lines)


@app.function(hide_code=True)
def check_common_grid(catalogs: Sequence[SpectralDensityCatalog]) -> None:
    """Require one frequency grid and observation time across catalogs."""
    first = catalogs[0]
    if any(
        not np.array_equal(catalog.frequencies, first.frequencies)
        or catalog.observation_time != first.observation_time
        for catalog in catalogs
    ):
        raise ValueError(
            "spectrum comparisons require the same grid and observation time"
        )


@app.function(hide_code=True)
def plot_fisher_h0_mixtures(
    predictions: Mapping[str, H0FisherPrediction], *, fiducial_h0: float
) -> Figure:
    """Overlay exact Gaussian-mixture densities with no extra sampling or KDE."""
    if not predictions:
        raise ValueError("at least one H0 prediction is required")
    lower = min(
        float(np.min(prediction.map_h0 - 6 * prediction.sigma_h0))
        for prediction in predictions.values()
    )
    upper = max(
        float(np.max(prediction.map_h0 + 6 * prediction.sigma_h0))
        for prediction in predictions.values()
    )
    grid = np.linspace(lower, upper, 2048)
    figure, axis = plt.subplots(
        figsize=(6.4, 4.8),
        layout="constrained",
        subplot_kw={"axes_class": Axes},
    )
    for label, prediction in predictions.items():
        density = gaussian_mixture_density(grid, prediction.map_h0, prediction.sigma_h0)
        axis.plot(grid, density, label=label)
    axis.axvline(
        fiducial_h0,
        color="0.35",
        linestyle="--",
        linewidth=1,
        label=r"$H_{0,\mathrm{fid}}$",
    )
    axis.set_xlabel(r"$H_0\,[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$")
    axis.set_ylabel("Density")
    axis.set_ylim(bottom=0)
    axis.legend()
    return figure


@app.function(hide_code=True)
def compute_spectrum_statistics(
    catalog: SpectralDensityCatalog,
) -> SpectrumStatistics:
    """Compute unbiased pointwise statistics along the realization axis."""
    spectra = np.asarray(catalog.spectral_density, dtype=np.float64)
    if catalog.metadata.sampled:
        raise ValueError("spectrum scatter requires fixed hyperparameters")
    if spectra.shape[0] < 2:
        raise ValueError("spectrum scatter requires at least two realizations")
    if not np.all(np.isfinite(spectra)) or np.any(spectra < 0):
        raise ValueError("spectra must be finite and nonnegative")
    mean = np.mean(spectra, axis=0)
    variance = np.var(spectra, axis=0, ddof=1)
    relative_variance = np.full(mean.shape, np.nan)
    positive = mean > 0
    residuals = spectra[:, positive] / mean[positive] - 1
    relative_variance[positive] = np.var(residuals, axis=0, ddof=1)
    return SpectrumStatistics(
        np.asarray(catalog.frequencies, dtype=np.float64),
        mean,
        variance,
        np.sqrt(variance),
        relative_variance,
    )


@app.function(hide_code=True)
def compute_network_sensitivity(
    catalog: SpectralDensityCatalog, settings: AnalysisSettings
) -> NetworkSensitivity:
    """Evaluate both scales on the full grid; selection never changes widths."""
    frequencies = jnp.asarray(catalog.frequencies)
    band = np.asarray(analysis_band(settings, frequencies))
    noise = network_noise(settings, frequencies)
    return NetworkSensitivity(
        band,
        np.asarray(gaussian_bin_scale(noise, catalog.observation_time, frequencies)),
        np.asarray(
            log_frequency_noise_scale(noise, frequencies, catalog.observation_time)
        ),
    )


@app.function(hide_code=True)
def compute_frequency_correlation(
    catalog: SpectralDensityCatalog,
    *,
    minimum_frequency: float,
    maximum_frequency: float,
    max_bins: int = 64,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Correlate relative residuals, retaining all rows and masking undefined bins.

    Positive per-frequency normalization preserves Pearson correlations,
    while dimensionless residuals avoid squaring tiny spectral densities.
    """
    statistics = compute_spectrum_statistics(catalog)
    if max_bins < 2:
        raise ValueError("max_bins must be at least two")
    if not (0 < minimum_frequency < maximum_frequency < np.inf):
        raise ValueError("correlation frequency bounds must be positive and ordered")
    band = (statistics.frequencies >= minimum_frequency) & (
        statistics.frequencies <= maximum_frequency
    )
    indices = np.flatnonzero(band)
    if indices.size < 2:
        raise ValueError("correlation requires at least two frequency bins")
    frequencies = statistics.frequencies[indices]
    if indices.size > max_bins:
        targets = np.geomspace(frequencies[0], frequencies[-1], max_bins)
        nearest = np.unique(np.abs(frequencies[:, None] - targets).argmin(axis=0))
        indices = indices[nearest]
        frequencies = statistics.frequencies[indices]
    rows = np.asarray(catalog.spectral_density[:, indices], dtype=np.float64)
    mean = statistics.mean[indices]
    residuals = np.full(rows.shape, np.nan)
    np.divide(rows, mean, out=residuals, where=mean > 0)
    residuals -= 1
    centered = residuals - residuals.mean(axis=0)
    covariance = centered.T @ centered / (rows.shape[0] - 1)
    scale = np.sqrt(np.diag(covariance))
    denominator = scale[:, None] * scale[None, :]
    correlation = np.full(covariance.shape, np.nan)
    np.divide(covariance, denominator, out=correlation, where=denominator > 0)
    return frequencies, np.clip(correlation, -1, 1)


@app.function(hide_code=True)
def plot_positive_curve(
    axis: Axes,
    frequencies: NDArray[np.float64],
    values: NDArray[np.float64],
    band: NDArray[np.bool_],
    *,
    label: str,
    color: str | None = None,
    linestyle: str = "-",
) -> None:
    """Mask invalid log values without joining across missing bins."""
    valid = band & (frequencies > 0) & np.isfinite(values) & (values > 0)
    axis.plot(
        frequencies[band],
        np.where(valid, values, np.nan)[band],
        label=label,
        color=color,
        linestyle=linestyle,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Frequency $[\mathrm{Hz}]$")


@app.function(hide_code=True)
def plot_spectrum_means(
    groups: Mapping[str, Mapping[str, SpectrumStatistics]],
    band: NDArray[np.bool_],
) -> Figure:
    """Plot ensemble-mean spectra, one panel per group."""
    figure, axes = plt.subplots(
        1,
        len(groups),
        figsize=(12.8, 4.8),
        layout="constrained",
        squeeze=False,
        subplot_kw={"axes_class": Axes},
    )
    for axis, (title, cases) in zip(axes[0], groups.items(), strict=True):
        for label, statistics in cases.items():
            plot_positive_curve(
                axis,
                statistics.frequencies,
                statistics.mean,
                band,
                label=label,
            )
        axis.set_title(title)
        axis.set_ylabel(r"$\overline{S_h}\,[\mathrm{Hz}^{-1}]$")
        axis.legend()
    return figure


@app.function(hide_code=True)
def plot_relative_variance(
    groups: Mapping[str, Mapping[str, SpectrumStatistics]],
    band: NDArray[np.bool_],
) -> Figure:
    """Plot the relative variance of each spectrum, one panel per group."""
    figure, axes = plt.subplots(
        1,
        len(groups),
        figsize=(12.8, 4.8),
        layout="constrained",
        squeeze=False,
        subplot_kw={"axes_class": Axes},
    )
    for axis, (title, cases) in zip(axes[0], groups.items(), strict=True):
        for label, statistics in cases.items():
            plot_positive_curve(
                axis,
                statistics.frequencies,
                statistics.relative_variance,
                band,
                label=label,
            )
        axis.set_title(title)
        axis.set_ylabel(r"$\mathrm{Var}(S_h/\overline{S_h}-1)$")
        axis.legend()
    return figure


@app.function(hide_code=True)
def plot_spectrum_sensitivity(
    cases: Mapping[str, SpectrumStatistics],
    sensitivity: NetworkSensitivity,
    *,
    network: str,
    observation_time: float,
) -> Figure:
    """Compare scatter with both the per-bin and per-e-fold detector scales."""
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.8),
        layout="constrained",
        subplot_kw={"axes_class": Axes},
    )
    for axis, noise, title, noise_label in zip(
        axes,
        (sensitivity.per_bin, sensitivity.per_log_frequency),
        ("Exact per-bin uncertainty", "Per-e-fold presentation scale"),
        (r"$\sigma_i$", r"$\sigma_{\ln f}$"),
        strict=True,
    ):
        for label, statistics in cases.items():
            plot_positive_curve(
                axis,
                statistics.frequencies,
                statistics.standard_deviation,
                sensitivity.band,
                label=label,
            )
        frequencies = next(iter(cases.values())).frequencies
        plot_positive_curve(
            axis,
            frequencies,
            noise,
            sensitivity.band,
            label=f"{network}: {noise_label}",
            color="0.35",
            linestyle="--",
        )
        axis.set_title(title)
        axis.set_ylabel(r"Standard deviation / sensitivity $[\mathrm{Hz}^{-1}]$")
        axis.legend()
    figure.suptitle(f"Network sensitivity: T = {observation_time:g} yr")
    return figure


@app.function(hide_code=True)
def plot_shot_noise_vs_detector(
    cases: Mapping[str, SpectrumStatistics],
    sensitivity: NetworkSensitivity,
    *,
    network: str,
) -> Figure:
    """Paper figure A1: shot-noise scatter against the per-bin detector error.

    Parameters
    ----------
    cases
        Pointwise statistics per source count.
    sensitivity
        Network uncertainty on the same frequency grid.
    network
        Network name for the legend.
    """
    figure, axis = plt.subplots(
        figsize=(6.4, 4.8),
        layout="constrained",
        subplot_kw={"axes_class": Axes},
    )
    for label, statistics in cases.items():
        plot_positive_curve(
            axis,
            statistics.frequencies,
            statistics.standard_deviation,
            sensitivity.band,
            label=label,
        )
    plot_positive_curve(
        axis,
        next(iter(cases.values())).frequencies,
        sensitivity.per_bin,
        sensitivity.band,
        label=rf"{network}: $\sigma_i$",
        color="0.35",
        linestyle="--",
    )
    axis.set_ylabel(r"Shot-noise SD / detector uncertainty $[\mathrm{Hz}^{-1}]$")
    axis.legend()
    return figure


@app.function(hide_code=True)
def plot_frequency_correlations(
    cases: Mapping[str, SpectralDensityCatalog],
    *,
    minimum_frequency: float,
    maximum_frequency: float,
) -> Figure:
    """Show frequency-frequency correlations of relative residuals."""
    figure, axes = plt.subplots(
        1,
        len(cases),
        figsize=(4.8 * len(cases), 4.8),
        layout="constrained",
        squeeze=False,
        subplot_kw={"axes_class": Axes},
    )
    for axis, (label, catalog) in zip(axes[0], cases.items(), strict=True):
        frequencies, correlation = compute_frequency_correlation(
            catalog,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
        mesh = axis.pcolormesh(
            frequencies,
            frequencies,
            np.ma.masked_invalid(correlation),
            shading="nearest",
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"Frequency $[\mathrm{Hz}]$")
        axis.set_ylabel(r"Frequency $[\mathrm{Hz}]$")
        axis.set_title(label)
    figure.suptitle(r"Relative residuals $r(f)=S_h(f)/\overline{S_h}(f)-1$")
    figure.colorbar(mesh, ax=list(axes[0]), label="Relative-residual correlation")
    return figure


@app.function(hide_code=True)
def plot_offset_scaling(
    offsets: OffsetAnalysis,
    poisson: OffsetAnalysis | None,
    *,
    paper_counts: Sequence[int],
    tolerance: float,
) -> Figure:
    """Paper figure A2: the offset distribution and how its width scales.

    Left, KDEs of the fixed-width residuals for ``paper_counts`` with the
    tolerance band. Right, ``sd(r)`` against ``N`` with bootstrap errors, an
    ``N^(-1/2)`` guide through the smallest ``N``, the tolerance line and
    the Poisson ensemble as a separate marker.
    """
    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.4),
        layout="constrained",
        subplot_kw={"axes_class": Axes},
    )
    shown = {count: offsets.fixed[count] for count in paper_counts}
    pooled = np.concatenate(list(shown.values()))
    pad = 0.25 * float(np.ptp(pooled)) + 1e-9
    grid = np.linspace(pooled.min() - pad, pooled.max() + pad, 800)
    for count, residuals in shown.items():
        if residuals.size > 1 and np.ptp(residuals) > 0:
            left.plot(grid, gaussian_kde(residuals)(grid), label=count_label(count))
        else:  # a constant sample has no density; mark its location
            left.axvline(residuals[0], label=count_label(count))
    left.axvspan(
        -tolerance,
        tolerance,
        color="0.5",
        alpha=0.15,
        label=rf"$|r| \leq {tolerance:g}$",
    )
    left.set_xlabel(
        r"$r = (H_{0,\mathrm{MAP}}-H_{0,\mathrm{fid}})/\sigma_{\mathrm{ref}}$"
    )
    left.set_ylabel("Density")
    left.set_ylim(bottom=0)
    left.legend()

    counts = np.array(sorted(offsets.fixed_summary))
    sd = np.array([offsets.fixed_summary[n].values["sd"] for n in counts])
    err = np.array([offsets.fixed_summary[n].errors["sd"] for n in counts])
    right.errorbar(
        counts, sd, yerr=err, marker="o", capsize=3, label=r"$\mathrm{sd}(r)$"
    )
    right.plot(
        counts,
        sd[0] * (counts / counts[0]) ** -0.5,
        color="0.35",
        linestyle=":",
        label=r"$\propto N^{-1/2}$",
    )
    right.axhline(
        tolerance,
        color="0.35",
        linestyle="--",
        label=rf"tolerance $= {tolerance:g}$",
    )
    if poisson is not None:
        for count, summary in poisson.fixed_summary.items():
            right.errorbar(
                [count],
                [summary.values["sd"]],
                yerr=[summary.errors["sd"]],
                marker="s",
                linestyle="none",
                capsize=3,
                label="Poisson",
            )
    right.set_xscale("log", base=2)
    right.set_yscale("log")
    right.set_xlabel(r"Number of injections $N$")
    right.set_ylabel(r"$\mathrm{sd}(r)$")
    right.legend()
    return figure


@app.function(hide_code=True)
def plot_offset_vs_min_redshift(
    summaries: Mapping[float, Mapping[int, ResidualSummary]],
    *,
    tolerance: float,
) -> Figure:
    """Paper figure A3: offset width against ``N``, one curve per ``z_min``.

    Solid curves are ``sd(r)`` with bootstrap errors and dashed curves the
    95th percentile of ``|r|``. The dotted ``N^(-1/2)`` guide is anchored at
    the smallest ``N`` of the smallest cutoff, and the dashed horizontal line
    is the ``sd(r)`` tolerance.
    """
    figure, axis = plt.subplots(
        figsize=(6.4, 4.8),
        layout="constrained",
        subplot_kw={"axes_class": Axes},
    )
    for z, by_count in summaries.items():
        counts = np.array(sorted(by_count))
        sd = np.array([by_count[n].values["sd"] for n in counts])
        err = np.array([by_count[n].errors["sd"] for n in counts])
        q95 = np.array([by_count[n].values["q95_abs"] for n in counts])
        line = axis.errorbar(
            counts,
            sd,
            yerr=err,
            marker="o",
            capsize=3,
            label=redshift_label(z),
        )
        axis.plot(
            counts,
            q95,
            color=line[0].get_color(),
            linestyle="--",
            marker="^",
            markerfacecolor="none",
        )
    first = summaries[min(summaries)]
    counts0 = min(first)
    all_counts = np.array(sorted({n for by in summaries.values() for n in by}))
    axis.plot(
        all_counts,
        first[counts0].values["sd"] * (all_counts / counts0) ** -0.5,
        color="0.35",
        linestyle=":",
        label=r"$\propto N^{-1/2}$",
    )
    axis.axhline(
        tolerance,
        color="0.35",
        linestyle="--",
        label=rf"$\mathrm{{sd}}(r)$ tolerance $= {tolerance:g}$",
    )
    axis.plot([], [], color="k", marker="o", label=r"$\mathrm{sd}(r)$")
    axis.plot(
        [],
        [],
        color="k",
        linestyle="--",
        marker="^",
        markerfacecolor="none",
        label=r"$q_{95}(|r|)$",
    )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel(r"Number of injections $N$")
    axis.set_ylabel(r"Offset in units of $\sigma_{\mathrm{ref}}$")
    axis.legend()
    return figure


@app.cell(hide_code=True)
def _(baseline_minimum_redshift):
    mo.md(rf"""
    ## 3. Generating the ensembles

    Every ensemble below is read from the checked spectrum cache, or generated
    on a miss, and holds many independent draws. The **count sweep** holds
    $z_{{\min}} = {baseline_minimum_redshift}$ and varies $N$. The
    **grid** repeats that at every $z_{{\min}}$ on a coarser set of counts, and
    shares the cached ensembles of the sweep at the baseline cutoff. The
    **Poisson** ensemble draws the count from the population's rate instead of
    fixing it.
    """)
    return


@app.cell
def _(base_metadata, baseline_minimum_redshift, num_events, settings):
    num_events_cases = {
        _count: analyze_case(base_metadata, _count, baseline_minimum_redshift, settings)
        for _count in num_events
    }
    pd.DataFrame(
        [
            {"num_events": _count, **_case.summary}
            for _count, _case in num_events_cases.items()
        ]
    )
    return (num_events_cases,)


@app.cell
def _(poisson_metadata, settings):
    # Kept apart from num_events_cases: its count is random, not a sweep value.
    poisson_case = analyze_metadata(poisson_metadata, settings)
    pd.DataFrame(
        [
            {
                "count": poisson_case.metadata.count,
                "mean_num_events": float(np.mean(poisson_case.catalog.n_events)),
                **poisson_case.summary,
            }
        ]
    )
    return (poisson_case,)


@app.cell
def _(
    base_metadata,
    baseline_minimum_redshift,
    grid_counts,
    minimum_redshift,
    num_events_cases,
    settings,
):
    grid = build_grid(
        base_metadata,
        grid_counts,
        minimum_redshift,
        settings,
        reuse={
            (baseline_minimum_redshift, _count): _case
            for _count, _case in num_events_cases.items()
        },
    )
    pd.DataFrame(
        [
            {"minimum_redshift": _z, "num_events": _count, **_case.summary}
            for _z, _cases in grid.items()
            for _count, _case in _cases.items()
        ]
    )
    return (grid,)


@app.cell
def _(data_metadata, data_reference, num_events_cases, settings):
    reference = select_reference(
        num_events_cases,
        data_reference,
        settings,
        poisson_metadata=data_metadata,
    )
    pd.DataFrame([reference.row()])
    return (reference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Spectrum-level view

    Before any SNR, compare the scatter of $S_h(f)$ itself with the
    uncertainty with which the detector network can measure each bin,
    $\sigma_i=S_{\mathrm{eff},i}/\sqrt{2T\Delta f_i}$. Where the shot-noise
    standard deviation lies below $\sigma_i$, the detector cannot see the
    catalog's Monte Carlo scatter in that bin.

    **Figure A1** shows that comparison for the paper's source counts.
    Variances use all independent realizations with $\mathrm{ddof}=1$.
    The supporting figures below it show the mean spectra, the relative
    variance $\mathrm{Var}(S_h/\overline{S_h}-1)$, and the frequency
    correlations of those relative residuals, which separate coherent
    amplitude scatter (correlations near one) from changes of spectral shape.
    The standard deviation describes individual spectra, not the error of the
    ensemble mean. Changing $z_{\min}$ changes the population and its rate,
    so the $z_{\min}$ group is shown at the recommended $N$.
    """)
    return


@app.cell
def _(
    grid,
    minimum_redshift,
    num_events_cases,
    paper_counts,
    poisson_case,
    recommended_num_events,
    settings,
):
    spectrum_case_groups = {
        "Source count": {
            **{
                count_label(_count): num_events_cases[_count].catalog
                for _count in paper_counts
            },
            "Poisson": poisson_case.catalog,
        },
        "Minimum redshift": {
            redshift_label(_z): grid[_z][recommended_num_events].catalog
            for _z in minimum_redshift
        },
    }
    _catalogs = [
        _catalog
        for _cases in spectrum_case_groups.values()
        for _catalog in _cases.values()
    ]
    check_common_grid(_catalogs)
    spectrum_statistics = {
        _title: {
            _label: compute_spectrum_statistics(_catalog)
            for _label, _catalog in _cases.items()
        }
        for _title, _cases in spectrum_case_groups.items()
    }
    spectrum_sensitivity = compute_network_sensitivity(_catalogs[0], settings)
    return (
        spectrum_case_groups,
        spectrum_sensitivity,
        spectrum_statistics,
    )


@app.cell
def _(settings, spectrum_sensitivity, spectrum_statistics):
    shot_noise_figure = plot_shot_noise_vs_detector(
        spectrum_statistics["Source count"],
        spectrum_sensitivity,
        network=settings.network,
    )
    shot_noise_figure
    return (shot_noise_figure,)


@app.cell
def _(spectrum_sensitivity, spectrum_statistics):
    spectrum_mean_figure = plot_spectrum_means(
        spectrum_statistics, spectrum_sensitivity.band
    )
    spectrum_mean_figure
    return (spectrum_mean_figure,)


@app.cell
def _(spectrum_sensitivity, spectrum_statistics):
    spectrum_relative_variance_figure = plot_relative_variance(
        spectrum_statistics, spectrum_sensitivity.band
    )
    spectrum_relative_variance_figure
    return (spectrum_relative_variance_figure,)


@app.cell
def _(
    base_metadata,
    settings,
    spectrum_sensitivity,
    spectrum_statistics,
):
    num_events_spectrum_sensitivity_figure = plot_spectrum_sensitivity(
        spectrum_statistics["Source count"],
        spectrum_sensitivity,
        network=settings.network,
        observation_time=base_metadata.observation_time,
    )
    num_events_spectrum_sensitivity_figure
    return (num_events_spectrum_sensitivity_figure,)


@app.cell
def _(
    base_metadata,
    settings,
    spectrum_sensitivity,
    spectrum_statistics,
):
    minimum_redshift_spectrum_sensitivity_figure = plot_spectrum_sensitivity(
        spectrum_statistics["Minimum redshift"],
        spectrum_sensitivity,
        network=settings.network,
        observation_time=base_metadata.observation_time,
    )
    minimum_redshift_spectrum_sensitivity_figure
    return (minimum_redshift_spectrum_sensitivity_figure,)


@app.cell
def _(settings, spectrum_case_groups):
    num_events_frequency_correlation_figure = plot_frequency_correlations(
        spectrum_case_groups["Source count"],
        minimum_frequency=settings.minimum_frequency,
        maximum_frequency=settings.maximum_frequency,
    )
    num_events_frequency_correlation_figure
    return (num_events_frequency_correlation_figure,)


@app.cell
def _(settings, spectrum_case_groups):
    minimum_redshift_frequency_correlation_figure = plot_frequency_correlations(
        spectrum_case_groups["Minimum redshift"],
        minimum_frequency=settings.minimum_frequency,
        maximum_frequency=settings.maximum_frequency,
    )
    minimum_redshift_frequency_correlation_figure
    return (minimum_redshift_frequency_correlation_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. SNR distributions (supporting)

    Each draw has its own SNR, so the spread of that distribution is the
    shot noise in units of the detector's sensitivity. In the finite-variance
    regime it narrows as $N^{-1/2}$ and its centre stays fixed, because the
    rate normalization keeps the underlying spectrum unchanged.
    """)
    return


@app.cell
def _(num_events_cases, paper_counts, poisson_case):
    num_events_snr_figure = plot_distribution_overlay(
        {
            **{
                count_label(_count): num_events_cases[_count].snrs
                for _count in paper_counts
            },
            "Poisson": poisson_case.snrs,
        },
        xlabel="SNR",
    )
    num_events_snr_figure
    return (num_events_snr_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. From scatter to an $H_0$ offset

    **Reference spectrum and $\sigma_{\mathrm{ref}}$.** Every Monte Carlo
    template is fitted to one common spectrum. By default that is the mean of
    all draws at the largest $N$, a mean spectrum rather than a mean SNR. Set
    `data_reference = "poisson"` to use one independently seeded Poisson draw
    instead, which includes count fluctuations as well; smoke runs substitute
    one small fixed-count draw. The reference SNR $\rho_{\mathrm{ref}}$ fixes
    $\sigma_{\mathrm{ref}} = H_{0,\mathrm{fid}}/\rho_{\mathrm{ref}}$, the
    expected $\sigma(H_0)$ of the real analysis.

    The fit reuses the amplitude-marginalized model's sufficient statistics
    ($\hat A_i$ and $\rho_i$, with its Gaussian noise weights and band), so the
    MAP includes any shape mismatch between template and reference. The MAP is
    clipped to the prior's bounds, and the table flags how many draws reach
    them.

    **Reading the table.** `sd` is the headline statistic: the fixed-width
    scatter of the offsets $r_i$, with bootstrap errors from resampling
    draws. `sd_per_draw` repeats it with each draw's own $\sigma_i$ as a
    cross-check; the two differ because of the numerator–denominator mixing
    described in Section 1. The default mean reference shares its draws with
    the largest ensemble, so that row measures convergence relative to it
    and cannot reveal an error common to every draw; with 200 draws its own
    scatter is about 0.5% of one template's.
    """)
    return


@app.cell
def _(
    n_bootstrap,
    num_events_cases,
    offset_seed,
    poisson_case,
    reference,
    settings,
):
    offsets = analyze_offsets(
        num_events_cases,
        reference,
        settings,
        n_bootstrap=n_bootstrap,
        seed=offset_seed,
    )
    # The count of a Poisson ensemble is random; place it at its mean.
    poisson_offsets = analyze_offsets(
        {round(float(np.mean(poisson_case.catalog.n_events))): poisson_case},
        reference,
        settings,
        n_bootstrap=n_bootstrap,
        seed=offset_seed,
    )
    pd.concat(
        [
            offsets_table(offsets, ensemble="fixed count"),
            offsets_table(poisson_offsets, ensemble="Poisson"),
        ],
        ignore_index=True,
    )
    return offsets, poisson_offsets


@app.cell
def _(offsets, paper_counts, poisson_offsets, tolerance):
    offset_scaling_figure = plot_offset_scaling(
        offsets,
        poisson_offsets,
        paper_counts=paper_counts,
        tolerance=tolerance,
    )
    offset_scaling_figure
    return (offset_scaling_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Figure A2.** The left panel shows the offsets for the paper's source
    counts; the shaded band is the tolerance. The right panel shows how their
    width shrinks with $N$. A slope of $-1/2$ means the scatter is in the
    ordinary Monte Carlo regime; where the measured points leave the dotted
    guide, rare nearby sources dominate. The Poisson point is an independent
    forward-model draw with a random count.

    **Supporting views.** The overlays below are the per-draw Fisher widths
    $\sigma_i$ and the Gaussian mixture they define,
    $p(H_0) = M^{-1}\sum_i \mathcal{N}(H_0; H_{0,\mathrm{MAP},i}, \sigma_i^2)$.
    The mixture combines the scatter of the fitted centres with the
    conditional Fisher uncertainty; it is evaluated directly, with no extra
    draws or KDE.
    """)
    return


@app.cell
def _(offsets, paper_counts, poisson_offsets):
    num_events_sigma_h0_figure = plot_distribution_overlay(
        {
            **{
                count_label(_count): offsets.predictions[_count].sigma_h0
                for _count in paper_counts
            },
            "Poisson": next(iter(poisson_offsets.predictions.values())).sigma_h0,
        },
        xlabel=r"$\sigma_{H_0,\mathrm{MAP}}\,[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$",
    )
    num_events_sigma_h0_figure
    return (num_events_sigma_h0_figure,)


@app.cell
def _(offsets, paper_counts, poisson_offsets):
    num_events_h0_fisher_figure = plot_fisher_h0_mixtures(
        {
            **{
                count_label(_count): offsets.predictions[_count]
                for _count in paper_counts
            },
            "Poisson": next(iter(poisson_offsets.predictions.values())),
        },
        fiducial_h0=offsets.fiducial_h0,
    )
    num_events_h0_fisher_figure
    return (num_events_h0_fisher_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Minimum redshift

    A cutoff changes the population itself, so each $z_{\min}$ gets its own
    reference (the mean spectrum of that cutoff's largest-$N$ ensemble) and its
    own $\sigma_{\mathrm{ref}}$. Residuals are then measured for every $N$ of
    the grid. Lowering $z_{\min}$ admits closer sources, which raises the
    variance (Section 1) and moves the curves of **Figure A3** up.

    The $N_{\max}$ point of each curve shares its draws with its own
    reference, which biases it low by about 0.5% of one template's scatter
    with 200 draws; read it as the end of the convergence curve rather than an
    independent test.
    """)
    return


@app.cell
def _(grid, n_bootstrap, offset_seed, settings):
    grid_offsets = analyze_grid(
        grid, settings, n_bootstrap=n_bootstrap, seed=offset_seed
    )
    pd.concat(
        [
            offsets_table(_offsets, ensemble=redshift_label(_z))
            for _z, _offsets in grid_offsets.items()
        ],
        ignore_index=True,
    )
    return (grid_offsets,)


@app.cell
def _(grid_offsets, tolerance):
    grid_summaries = {
        _z: _offsets.fixed_summary for _z, _offsets in grid_offsets.items()
    }
    offset_vs_min_redshift_figure = plot_offset_vs_min_redshift(
        grid_summaries, tolerance=tolerance
    )
    offset_vs_min_redshift_figure
    return grid_summaries, offset_vs_min_redshift_figure


@app.cell(hide_code=True)
def _(recommended_num_events):
    mo.md(rf"""
    **Supporting views at the recommended $N = {recommended_num_events}$.** The
    SNR and $\sigma(H_0) = H_{{0,\mathrm{{fid}}}}/\mathrm{{SNR}}$ of each draw
    under each cutoff.
    """)
    return


@app.cell
def _(grid, minimum_redshift, recommended_num_events):
    minimum_redshift_snr_figure = plot_distribution_overlay(
        {
            redshift_label(_z): grid[_z][recommended_num_events].snrs
            for _z in minimum_redshift
        },
        xlabel="SNR",
    )
    minimum_redshift_snr_figure
    return (minimum_redshift_snr_figure,)


@app.cell
def _(grid, minimum_redshift, recommended_num_events):
    minimum_redshift_sigma_h0_figure = plot_distribution_overlay(
        {
            redshift_label(_z): grid[_z][recommended_num_events].sigma_h0
            for _z in minimum_redshift
        },
        xlabel=r"$\sigma_{H_0}\,[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$",
    )
    minimum_redshift_sigma_h0_figure
    return (minimum_redshift_sigma_h0_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 8. Verdict
    """)
    return


@app.cell
def _(grid_counts, grid_summaries, tolerance):
    verdict = verdict_table(grid_summaries, grid_counts[-2:], tolerance=tolerance)
    verdict
    return (verdict,)


@app.cell
def _(tolerance, verdict):
    mo.md(verdict_text(verdict, tolerance=tolerance))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 9. Save figures
    """)
    return


@app.cell(hide_code=True)
def _(
    BASE_DIR,
    ROOT_DIR,
    minimum_redshift_frequency_correlation_figure,
    minimum_redshift_sigma_h0_figure,
    minimum_redshift_snr_figure,
    minimum_redshift_spectrum_sensitivity_figure,
    num_events_frequency_correlation_figure,
    num_events_h0_fisher_figure,
    num_events_sigma_h0_figure,
    num_events_snr_figure,
    num_events_spectrum_sensitivity_figure,
    offset_scaling_figure,
    offset_vs_min_redshift_figure,
    shot_noise_figure,
    spectrum_mean_figure,
    spectrum_relative_variance_figure,
    write_figures,
):
    if write_figures:
        save_figures(
            {
                # Paper figures A1-A3.
                BASE_DIR / "paper_shot_noise_vs_detector.pdf": shot_noise_figure,
                BASE_DIR / "paper_offset_scaling.pdf": offset_scaling_figure,
                BASE_DIR
                / "paper_offset_vs_min_redshift.pdf": offset_vs_min_redshift_figure,
                # Supporting figures.
                BASE_DIR / "spectrum_mean.pdf": spectrum_mean_figure,
                BASE_DIR
                / "spectrum_relative_variance.pdf": spectrum_relative_variance_figure,
                BASE_DIR
                / "num_events_spectrum_sensitivity.pdf": num_events_spectrum_sensitivity_figure,
                BASE_DIR
                / "minimum_redshift_spectrum_sensitivity.pdf": minimum_redshift_spectrum_sensitivity_figure,
                BASE_DIR
                / "num_events_frequency_correlation.pdf": num_events_frequency_correlation_figure,
                BASE_DIR
                / "minimum_redshift_frequency_correlation.pdf": minimum_redshift_frequency_correlation_figure,
                BASE_DIR / "num_events_snr_distribution.pdf": num_events_snr_figure,
                BASE_DIR
                / "num_events_sigma_H0_distribution.pdf": num_events_sigma_h0_figure,
                BASE_DIR
                / "num_events_H0_fisher_distribution.pdf": num_events_h0_fisher_figure,
                BASE_DIR
                / "minimum_redshift_snr_distribution.pdf": minimum_redshift_snr_figure,
                BASE_DIR
                / "minimum_redshift_sigma_H0_distribution.pdf": minimum_redshift_sigma_h0_figure,
            },
            root=ROOT_DIR,
        )
    return


if __name__ == "__main__":
    app.run()
