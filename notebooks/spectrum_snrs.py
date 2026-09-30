import marimo

__generated_with = "0.25.0"
app = marimo.App()

with app.setup(hide_code=True):
    import os
    from collections.abc import Mapping
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

    from astrogwb.catalog import SpectralDensityCatalog, SpectrumGenerator, simulate
    from astrogwb.detector import effective_psd, gaussian_bin_scale
    from astrogwb.frequency import frequency_mask
    from astrogwb.gwb import spectral_snr
    from astrogwb.inference import gwb_amplitude_marginalized_model
    from astrogwb.metadata import SpectraMetadata
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
    from astrogwb.populations.bns_madau_dickinson import amplitude_H0_fn
    from astrogwb.utils import years_to_seconds


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # SNR distributions of fixed-count spectrum ensembles

    Compare finite-catalog estimator scatter as the source count or the minimum
    redshift changes. Each realization contains exactly $N$ sources and is
    normalized by the population's merger rate. Increasing $N$ tests convergence
    around the same underlying spectrum; changing $z_{\min}$ changes the
    population itself. These draws include source fluctuations, without detector
    noise realizations.

    We also compare amplitude-only Fisher uncertainties in $H_0$. The count
    sweep fits a common reference spectrum and evaluates widths at each MAP;
    the redshift sweep retains the fiducial approximation
    $\sigma(H_0) = H_0 / \mathrm{SNR}$. Other parameters remain fixed.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Notebook configuration
    """)
    return


@app.cell
def _():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    BASE_DIR = ROOT_DIR / FIGURES_DIR / "spectrum_snrs"
    FIDUCIALS = fiducials(root=ROOT_DIR)
    registry = detector_registry(root=ROOT_DIR)

    # Each section varies one parameter and holds the other at its baseline.
    num_events = [16384, 2 * 16384, 4 * 16384]
    minimum_redshift = [0.05, 0.15, 0.35]
    baseline_num_events = 16384
    baseline_minimum_redshift = 0.35
    maximum_redshift = 20.0

    num_draws = 1000
    seed = 41
    data_seed = 42  # used only for the independent Poisson option
    batch_size = 1024
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
        baseline_num_events = 8
        num_draws = 3
        batch_size = 8
        write_figures = False

    # Read the shared draw's waveform rather than copying its scientific settings.
    base_metadata = SpectraMetadata(
        count="fixed",
        num_events=baseline_num_events,
        num_draws=num_draws,
        observation_time=observation_time,
        hyperparameters={**FIDUCIALS},
        waveform=waveform_metadata(root=ROOT_DIR),
        population=population_metadata(
            root=ROOT_DIR,
            seed=seed,
            minimum_redshift=baseline_minimum_redshift,
            maximum_redshift=maximum_redshift,
        ),
    )
    data_metadata = SpectraMetadata.model_validate(
        {
            **base_metadata.model_dump(),
            "count": "fixed" if SMOKE else "poisson",
            "num_events": 64 if SMOKE else None,
            "num_draws": 1,
            "population": {**base_metadata.population.model_dump(), "seed": data_seed},
        }
    )
    h0_prior = priors(root=ROOT_DIR)["H0"]
    # Inverting the MLE gives the MAP only for a uniform prior on H0.
    if not isinstance(h0_prior, dist.Uniform) or float(h0_prior.low) <= 0:
        raise ValueError("the H0 MAP diagnostic requires a positive Uniform H0 prior")
    use_paper_style(root=ROOT_DIR)
    # gwpy registers replacement default axes; ArviZ needs matplotlib axes.
    register_projection(Axes)
    return (
        BASE_DIR,
        ROOT_DIR,
        base_metadata,
        baseline_minimum_redshift,
        baseline_num_events,
        batch_size,
        cache_dir,
        cache_only,
        data_metadata,
        h0_prior,
        maximum_frequency,
        minimum_frequency,
        minimum_redshift,
        network,
        num_events,
        registry,
        write_figures,
    )


@app.cell
def _():
    data_reference = "largest_mean"  # mean spectrum at max(num_events), or "poisson"
    return (data_reference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Analysis and plotting helpers
    """)
    return


@app.cell(hide_code=True)
def _():
    def compute_spectrum_snrs(
        catalog: SpectralDensityCatalog,
        detector_registry: DetectorRegistry,
        network: str,
        *,
        minimum_frequency: float,
        maximum_frequency: float,
    ) -> tuple[NDArray[np.float64], float]:
        """Return per-realization SNRs and the mean spectrum's SNR for one network.

        The observing time comes from the artifact, in years. Both calculations
        use the full frequency grid and mask afterwards, preserving the widths
        of bins at the analysis band's edges and all within-row correlations.
        No source sampling or waveform generation is performed here.
        """
        if not (
            np.isfinite(minimum_frequency)
            and np.isfinite(maximum_frequency)
            and 0.0 <= minimum_frequency < maximum_frequency
        ):
            raise ValueError(
                "frequency bounds must be finite and 0 <= minimum < maximum"
            )
        if not np.isfinite(catalog.observation_time) or catalog.observation_time <= 0:
            raise ValueError("observation_time must be finite and positive")
        if catalog.frequencies.size < 2:
            raise ValueError("SNR calculation requires at least two frequency bins")
        if network not in detector_registry.networks:
            raise ValueError(f"unknown network {network!r}")

        frequencies = jnp.asarray(catalog.frequencies)
        band = frequency_mask(
            frequencies, fmin=minimum_frequency, fmax=maximum_frequency
        )
        if not np.any(np.asarray(band)):
            raise ValueError("the analysis frequency band contains no spectrum bins")
        geometry, sensitivities = detector_registry.build_network(network)
        noise = jnp.asarray(effective_psd(frequencies, geometry, sensitivities))
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

    @dataclass(frozen=True)
    class SNRCase:
        """Compact analysis results; spectra remain in the checked HDF5 cache."""

        metadata: SpectraMetadata
        snrs: NDArray[np.float64]
        sigma_h0: NDArray[np.float64]
        summary: dict[str, int | float]
        catalog: SpectralDensityCatalog

    return SNRCase, compute_spectrum_snrs, summarize_spectrum_snrs


@app.cell(hide_code=True)
def _(
    SNRCase,
    base_metadata,
    batch_size,
    cache_dir,
    cache_only,
    compute_spectrum_snrs,
    maximum_frequency,
    minimum_frequency,
    network,
    registry,
    summarize_spectrum_snrs,
):
    def analyze_case(num_events: int, minimum_redshift: float) -> SNRCase:
        """Analyze one count/cutoff choice using this notebook's common settings."""
        population = base_metadata.population.with_model_kwargs(
            minimum_redshift=minimum_redshift
        )
        metadata = SpectraMetadata.model_validate(
            {
                **base_metadata.model_dump(),
                "num_events": num_events,
                "population": population,
            }
        )
        catalog = simulate(
            metadata,
            SpectrumGenerator(batch_size=batch_size),
            cache_dir,
            generate=not cache_only,
        )
        snrs, mean_spectrum_snr = compute_spectrum_snrs(
            catalog,
            registry,
            network,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
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

    return (analyze_case,)


@app.cell(hide_code=True)
def _():
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

    return (plot_distribution_overlay,)


@app.cell(hide_code=True)
def _():
    @dataclass(frozen=True)
    class H0FisherPrediction:
        """Template fits to common data and Fisher widths evaluated at their MAPs."""

        map_h0: NDArray[np.float64]
        sigma_h0: NDArray[np.float64]
        amplitude_mle: NDArray[np.float64]
        template_optimal_snr: NDArray[np.float64]
        at_prior_boundary: NDArray[np.bool_]

        def normalized_residuals(self, *, fiducial_h0: float) -> NDArray[np.float64]:
            """Return template MAP offsets in units of their local Fisher widths."""
            return (self.map_h0 - fiducial_h0) / self.sigma_h0

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
            raise ValueError(
                "grid, centres and widths must be non-empty finite 1D arrays"
            )
        if means.shape != sigmas.shape or np.any(sigmas <= 0):
            raise ValueError("each centre requires one strictly positive width")
        standardized = (x[:, None] - means[None, :]) / sigmas[None, :]
        densities = np.exp(-0.5 * standardized**2) / (np.sqrt(2 * np.pi) * sigmas)
        return np.mean(densities, axis=1)

    return H0FisherPrediction, fisher_h0_prediction, gaussian_mixture_density


@app.cell(hide_code=True)
def _(h0_prior):
    def template_amplitude_statistics(
        templates: ArrayLike,
        data: ArrayLike,
        scale: ArrayLike,
        band: ArrayLike,
        *,
        fiducial_h0: float,
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
                amplitude_parameter="H0",
                amplitude_fiducial=fiducial_h0,
                amplitude_fn=amplitude_H0_fn,
                amplitude_prior=h0_prior,
                frequency_mask=jnp.asarray(band),
            )
            return trace["amplitude_mle"]["value"], trace["template_optimal_snr"][
                "value"
            ]

        amplitude, snr = jax.jit(jax.vmap(statistics))(jnp.asarray(templates))
        return np.asarray(amplitude, dtype=np.float64), np.asarray(
            snr, dtype=np.float64
        )

    return (template_amplitude_statistics,)


@app.cell(hide_code=True)
def _(H0FisherPrediction, gaussian_mixture_density):
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
            figsize=(6.4, 4.8), layout="constrained", subplot_kw={"axes_class": Axes}
        )
        for label, prediction in predictions.items():
            density = gaussian_mixture_density(
                grid, prediction.map_h0, prediction.sigma_h0
            )
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

    return (plot_fisher_h0_mixtures,)


@app.cell(hide_code=True)
def _(baseline_minimum_redshift):
    mo.md(rf"""
    ## Source-count sweep

    Hold $z_{{\min}} = {baseline_minimum_redshift}$ fixed and vary $N$.
    In the finite-variance convergence regime, estimator scatter scales as
    $N^{{-1/2}}$; the rate normalization keeps the underlying spectrum fixed.
    """)
    return


@app.cell
def _(analyze_case, baseline_minimum_redshift, num_events):
    num_events_cases = {
        _count: analyze_case(_count, baseline_minimum_redshift) for _count in num_events
    }
    pd.DataFrame(
        [
            {"num_events": _count, **_case.summary}
            for _count, _case in num_events_cases.items()
        ]
    )
    return (num_events_cases,)


@app.cell
def _(num_events_cases, plot_distribution_overlay):
    num_events_snr_figure = plot_distribution_overlay(
        {rf"$N = {_count}$": _case.snrs for _count, _case in num_events_cases.items()},
        xlabel="SNR",
    )
    num_events_snr_figure
    return (num_events_snr_figure,)


@app.cell
def _(num_events_h0_predictions, plot_distribution_overlay):
    num_events_sigma_h0_figure = plot_distribution_overlay(
        {
            rf"$N = {_count}$": _prediction.sigma_h0
            for _count, _prediction in num_events_h0_predictions.items()
        },
        xlabel=r"$\sigma_{H_0,\mathrm{MAP}}\,[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$",
    )
    num_events_sigma_h0_figure
    return (num_events_sigma_h0_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fisher approximation to the inferred $H_0$ distribution

    Fit every Monte Carlo **template** to one common reference spectrum. By
    default, average the full spectra from all draws at the largest configured
    source count ($N=65536$, 1000 draws by default). This is a mean spectrum,
    not a mean SNR, and the same reference is used for every $N$.

    Set `data_reference = "poisson"` to use one independently seeded Poisson
    spectrum at the fiducial hyperparameters and observing time instead. It
    includes physical source and count fluctuations, without detector noise.
    Smoke tests substitute one small fixed-count draw for that option.

    Reuse the amplitude-marginalized model's sufficient statistics:
    $\hat A_i = (d|t_i)/(t_i|t_i)$ and $\rho_i = \sqrt{(t_i|t_i)}$, using
    its Gaussian noise weights and analysis band. Here
    $(a|b)=\sum_f a(f)b(f)/\sigma_f^2$ and $t_i$ is evaluated at
    $H_{0,\mathrm{fid}}$. With $A(H_0)=H_{0,\mathrm{fid}}/H_0$ and the shared
    uniform prior on $H_0$, the MAP is $H_{0,\mathrm{fid}}/\hat A_i$, clipped
    to the prior's bounds. This uses the full spectra, including shape mismatch.

    Evaluate the local Fisher widths at each MAP:
    $\sigma_i = H_{0,\mathrm{MAP},i}^2/(H_{0,\mathrm{fid}}\rho_i)$.
    The curve for each source count is the equally weighted mixture
    $p(H_0) = M^{-1}\sum_i \mathcal{N}(H_0;H_{0,\mathrm{MAP},i},\sigma_i^2)$.
    It combines variation of fitted centres with conditional Fisher uncertainty;
    we evaluate this density directly, without additional random draws or KDE.
    Other hyperparameters remain fixed. These are local Fisher Gaussians rather
    than the exact conditional posteriors; a MAP at a prior boundary needs a
    truncated posterior treatment. The table flags boundary cases.
    """)
    return


@app.cell
def _(
    batch_size,
    cache_dir,
    cache_only,
    compute_spectrum_snrs,
    data_metadata,
    data_reference,
    maximum_frequency,
    minimum_frequency,
    network,
    num_events_cases,
    registry,
):
    if data_reference == "largest_mean":
        _reference_case = num_events_cases[max(num_events_cases)]
        data_catalog = _reference_case.catalog
        data_spectrum = np.mean(data_catalog.spectral_density, axis=0, dtype=np.float64)
        data_snr = _reference_case.summary["mean_spectrum_snr"]
        _averaged_draws = data_catalog.metadata.num_draws
    elif data_reference == "poisson":
        data_catalog = simulate(
            data_metadata,
            SpectrumGenerator(batch_size=batch_size),
            cache_dir,
            generate=not cache_only,
        )
        data_spectrum = np.asarray(data_catalog.spectral_density[0], dtype=np.float64)
        _snrs, _ = compute_spectrum_snrs(
            data_catalog,
            registry,
            network,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
        data_snr = float(_snrs[0])
        _averaged_draws = 1
    else:
        raise ValueError('data_reference must be "largest_mean" or "poisson"')
    pd.DataFrame(
        [
            {
                "reference": data_reference,
                "source_key": data_catalog.metadata.key(),
                "count": data_catalog.metadata.count,
                "num_events": int(data_catalog.n_events[0]),
                "averaged_draws": _averaged_draws,
                "snr": data_snr,
                "seed": data_catalog.metadata.population.seed,
            }
        ]
    )
    return data_catalog, data_snr, data_spectrum


@app.cell
def _(
    data_catalog,
    data_spectrum,
    fisher_h0_prediction,
    h0_prior,
    maximum_frequency,
    minimum_frequency,
    network,
    num_events_cases,
    registry,
    template_amplitude_statistics,
):
    _frequencies = jnp.asarray(data_catalog.frequencies)
    _band = frequency_mask(_frequencies, fmin=minimum_frequency, fmax=maximum_frequency)
    _geometry, _sensitivities = registry.build_network(network)
    _noise = jnp.asarray(effective_psd(_frequencies, _geometry, _sensitivities))
    _scale = gaussian_bin_scale(_noise, data_catalog.observation_time, _frequencies)
    num_events_h0_predictions = {}
    for _count, _case in num_events_cases.items():
        if not np.array_equal(_case.catalog.frequencies, data_catalog.frequencies):
            raise ValueError("data and templates must use the same frequency grid")
        _amplitudes, _snrs = template_amplitude_statistics(
            _case.catalog.spectral_density,
            data_spectrum,
            _scale,
            _band,
            fiducial_h0=_case.metadata.fixed["H0"],
        )
        num_events_h0_predictions[_count] = fisher_h0_prediction(
            _amplitudes,
            _snrs,
            fiducial_h0=_case.metadata.fixed["H0"],
            h0_prior=h0_prior,
        )
    pd.DataFrame(
        [
            {
                "num_events": _count,
                "mean_map_h0": float(np.mean(_prediction.map_h0)),
                "sd_map_h0": float(np.std(_prediction.map_h0, ddof=1)),
                "prior_boundary_fraction": float(
                    np.mean(_prediction.at_prior_boundary)
                ),
            }
            for _count, _prediction in num_events_h0_predictions.items()
        ]
    )
    return (num_events_h0_predictions,)


@app.cell
def _(base_metadata, num_events_h0_predictions, plot_fisher_h0_mixtures):
    num_events_h0_fisher_figure = plot_fisher_h0_mixtures(
        {
            rf"$N = {_count}$": _prediction
            for _count, _prediction in num_events_h0_predictions.items()
        },
        fiducial_h0=base_metadata.fixed["H0"],
    )
    num_events_h0_fisher_figure
    return (num_events_h0_fisher_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### MAP residuals in units of Fisher uncertainty

    Plot the per-template offsets
    $r_i = (H_{0,\mathrm{MAP},i}-H_{0,\mathrm{fid}})/\sigma_i$.
    These are MAP residuals, without additional posterior or detector-noise
    draws. The dashed standard normal is a unit-width reference, not a required
    distribution for this experiment. Approximately Gaussian, small SNR
    fluctuations give approximately Gaussian residuals; finite-template scatter
    can make their width exceed one.

    A residual SD above one means template-induced MAP scatter exceeds the
    Fisher uncertainty. The table also shows the mean, RMS offset, and fraction
    with $|r_i|>1$. All counts use the same reference; we do not recenter each
    ensemble separately. The default mean reference shares its draws with the
    largest template ensemble, so this measures convergence relative to that
    ensemble and cannot reveal systematic errors shared by all draws. With 1000
    draws, its random reference error is much smaller than individual template
    scatter. The Poisson option instead includes fluctuations in one physical
    data realization; repeated realizations would separate those from template
    error.
    """)
    return


@app.cell
def _(base_metadata, num_events_h0_predictions):
    num_events_h0_residuals = {
        _count: _prediction.normalized_residuals(fiducial_h0=base_metadata.fixed["H0"])
        for _count, _prediction in num_events_h0_predictions.items()
    }
    num_events_h0_residual_summary = pd.DataFrame(
        [
            {
                "num_events": _count,
                "mean": float(np.mean(_residuals)),
                "sd": float(np.std(_residuals, ddof=1)),
                "rms": float(np.sqrt(np.mean(_residuals**2))),
                "fraction_abs_gt_1": float(np.mean(np.abs(_residuals) > 1)),
            }
            for _count, _residuals in num_events_h0_residuals.items()
        ]
    )
    num_events_h0_residual_summary
    return num_events_h0_residual_summary, num_events_h0_residuals


@app.cell
def _(num_events_h0_residuals, plot_distribution_overlay):
    num_events_h0_residual_figure = plot_distribution_overlay(
        {
            rf"$N = {_count}$": _residuals
            for _count, _residuals in num_events_h0_residuals.items()
        },
        xlabel=r"$(H_{0,\mathrm{MAP}}-H_{0,\mathrm{fid}})/\sigma_{H_0}$",
    )
    _axis = num_events_h0_residual_figure.axes[0]
    _grid = np.linspace(-5, 5, 1000)
    # Keep the constant-ensemble ECDF fallback on the same probability scale.
    if _axis.get_ylabel() == "Density":
        _reference = np.exp(-0.5 * _grid**2) / np.sqrt(2 * np.pi)
    else:
        from scipy.special import ndtr

        _reference = ndtr(_grid)
    _axis.plot(
        _grid, _reference, color="0.35", linestyle="--", label=r"$\mathcal{N}(0,1)$"
    )
    if _axis.get_ylabel() == "Density":
        _axis.set_ylim(
            0,
            1.05 * max(float(np.max(_line.get_ydata())) for _line in _axis.get_lines()),
        )
    _axis.axvline(0, color="0.6", linewidth=0.8)
    _axis.axvspan(-1, 1, color="0.5", alpha=0.08)
    _axis.legend(
        _axis.get_lines()[: len(num_events_h0_residuals) + 1],
        [rf"$N = {_count}$" for _count in num_events_h0_residuals]
        + [r"$\mathcal{N}(0,1)$"],
    )
    num_events_h0_residual_figure
    return (num_events_h0_residual_figure,)


@app.cell(hide_code=True)
def _(baseline_num_events):
    mo.md(rf"""
    ## Minimum-redshift sweep

    Hold $N = {baseline_num_events}$ fixed and vary $z_{{\min}}$. Both the
    underlying spectrum and estimator scatter can change. Nearby sources can
    strongly affect the distribution tails, so interpreting those tails requires
    enough independent realizations.
    """)
    return


@app.cell
def _(analyze_case, baseline_num_events, minimum_redshift):
    minimum_redshift_cases = {
        _cutoff: analyze_case(baseline_num_events, _cutoff)
        for _cutoff in minimum_redshift
    }
    pd.DataFrame(
        [
            {"minimum_redshift": _cutoff, **_case.summary}
            for _cutoff, _case in minimum_redshift_cases.items()
        ]
    )
    return (minimum_redshift_cases,)


@app.cell
def _(minimum_redshift_cases, plot_distribution_overlay):
    minimum_redshift_snr_figure = plot_distribution_overlay(
        {
            rf"$z_{{\min}} = {_cutoff:.2f}$": _case.snrs
            for _cutoff, _case in minimum_redshift_cases.items()
        },
        xlabel="SNR",
    )
    minimum_redshift_snr_figure
    return (minimum_redshift_snr_figure,)


@app.cell
def _(minimum_redshift_cases, plot_distribution_overlay):
    minimum_redshift_sigma_h0_figure = plot_distribution_overlay(
        {
            rf"$z_{{\min}} = {_cutoff:.2f}$": _case.sigma_h0
            for _cutoff, _case in minimum_redshift_cases.items()
        },
        xlabel=r"$\sigma_{H_0}\,[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$",
    )
    minimum_redshift_sigma_h0_figure
    return (minimum_redshift_sigma_h0_figure,)


@app.cell(hide_code=True)
def _(
    BASE_DIR,
    ROOT_DIR,
    minimum_redshift_sigma_h0_figure,
    minimum_redshift_snr_figure,
    num_events_h0_fisher_figure,
    num_events_h0_residual_figure,
    num_events_sigma_h0_figure,
    num_events_snr_figure,
    write_figures,
):
    if write_figures:
        save_figures(
            {
                BASE_DIR / "num_events_snr_distribution.pdf": num_events_snr_figure,
                BASE_DIR
                / "num_events_sigma_H0_distribution.pdf": num_events_sigma_h0_figure,
                BASE_DIR
                / "num_events_H0_fisher_distribution.pdf": num_events_h0_fisher_figure,
                BASE_DIR
                / "num_events_H0_normalized_residual_distribution.pdf": num_events_h0_residual_figure,
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
