import marimo

__generated_with = "0.25.1"
app = marimo.App()

with app.setup(hide_code=True):
    import os
    from pathlib import Path

    from astrogwb.paper.runtime import configure_runtime

    # Backend configuration precedes waveform construction and array creation.
    configure_runtime(num_chains=1)

    import arviz as az
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.projections import register_projection
    from numpy.typing import NDArray
    from scipy import stats

    from astrogwb.detector import effective_psd, gaussian_bin_scale
    from astrogwb.frequency import frequency_mask
    from astrogwb.gwb.importance import (
        reference_catalog,
        reference_catalog_stem,
    )
    from astrogwb.inference import (
        ImportanceGaussianLikelihood,
        amplitude_direction,
        rank_one_gaussian_log_likelihood,
    )
    from astrogwb.inference import Network as LikelihoodNetwork
    from astrogwb.paper.cache import default_cache_dir
    from astrogwb.paper.config import (
        detector_registry,
        fiducials,
        population_metadata,
        priors,
        waveform_metadata,
    )
    from astrogwb.paper.config.detectors import DetectorRegistry
    from astrogwb.paper.config.runs import CATALOGS_ROOT, FIGURES_DIR
    from astrogwb.paper.plotting import (
        DETECTOR_NETWORKS,
        parameter_label,
        save_figures,
        use_paper_style,
    )
    from astrogwb.populations import build_population
    from astrogwb.simulators.core import batch_keys, load, write
    from astrogwb.simulators.polarization_power import (
        CatalogMetadata,
        PolarizationPowerData,
    )
    from astrogwb.simulators.spectra import (
        BackgroundSpectralDensityMetadata,
        spectra_ensemble,
    )

    # Log-likelihood of every injection at every grid point, (M, G): data is
    # (M, F), prediction and direction (G, F). One injection at a time, so
    # memory stays (G, F); compiled once for every network and variant.
    grid_log_likelihoods = jax.jit(
        lambda data, prediction, scale, direction, mask: jax.lax.map(
            lambda observed: rank_one_gaussian_log_likelihood(
                observed, prediction, scale, direction, mask
            ),
            data,
        )
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Coverage of the shot-noise likelihood for $H_0$

    `notebooks/cosmology_grid_posteriors.py` models the physical shot noise of its
    one injection as a rank-one term in the covariance of the spectral density.
    This notebook checks that the term is calibrated: over $M$ independent Poisson
    injections at the true values $\Lambda_t$, does the $H_0$ posterior cover
    $H_0^t$ at the nominal rate, and does the detector-only likelihood fail to?
    The notation follows the draft paper.

    ## Main concepts

    ### Data model

    The observable is the single-sided spectral density $S_h(f)$ of the
    background. Given hyperparameters $\Lambda$, its expectation over catalogs is
    $S_h(f|\Lambda)$, which the code estimates by importance sampling
    ($\hat S_h$). One observation is a single Poisson catalog realization,
    $N_{\rm ev} \sim \mathrm{Poisson}[\mathcal{R}(\Lambda)\,T]$ events over the
    observation time $T$,

    $$
    S_h^{\rm cat}(f) = \frac{1}{T} \sum_{i=1}^{N_{\rm ev}}
    \left[ |\tilde h_{+,i}(f)|^2 + |\tilde h_{\times,i}(f)|^2 \right],
    \qquad
    \left\langle S_h^{\rm cat}(f) \right\rangle = S_h(f|\Lambda).
    $$

    The data add detector noise, and the finite catalog adds shot noise
    $n_{\rm shot} \equiv S_h^{\rm cat} - S_h(f|\Lambda)$:

    $$
    S_h^{\rm data}(f) = S_h(f|\Lambda) + n_{\rm shot}(f|\Lambda) + n_{\rm det}(f).
    $$

    Frequencies are discretized in bins $k = 1, \dots, \mathcal{N}_f$ with centers
    $f_k$ and widths $\Delta f_k$. Bins outside $[f_{\rm min}, f_{\rm max}]$, or
    where the network has no sensitivity, are masked out.

    ### Detector noise, inner product and SNR

    The network's effective variance combines the detector pairs $(a,b)$ through
    their noise spectra $S_{n,a}$ and overlap reduction functions $\Gamma_{ab}$,

    $$
    \frac{1}{\sigma^2(f)} = \sum_{a<b} \frac{\Gamma_{ab}^2(f)}{S_{n,a}(f)\,S_{n,b}(f)},
    $$

    and the detector noise in a bin is Gaussian and independent across bins,
    $n_{\rm det}(f_k) \sim \mathcal{N}\big(0, \Delta S_h^2(f_k)\big)$, with

    $$
    \Delta S_h(f_k) = \frac{\sigma(f_k)}{\sqrt{2 T\, \Delta f_k}}.
    $$

    This is the `scale` of the code (`gaussian_bin_scale`). The noise-weighted
    inner product of two functions of frequency, and its discrete form, are

    $$
    \langle F | G \rangle = \mathrm{Re} \int df\, \frac{\tilde F^*(f)\, \tilde G(f)}{\sigma^2(f)}
    \quad\Longrightarrow\quad
    2T \langle F | G \rangle = \sum_k \frac{F_k\, G_k}{\Delta S_h^2(f_k)}.
    $$

    The optimal SNR of a background, bin by bin and in total, is

    $$
    \rho_k(\Lambda) = \frac{S_h(f_k|\Lambda)}{\Delta S_h(f_k)},
    \qquad
    \rho^2(\Lambda) = 2T \langle S_h(\Lambda) | S_h(\Lambda) \rangle = \sum_k \rho_k^2(\Lambda).
    $$

    Below, $\rho$ is always the SNR of the template $S_h^t(f) \equiv S_h(f|\Lambda_t)$,
    the expected spectrum at the true values.

    ### Likelihood without shot noise

    The Gaussian likelihood of the paper treats the detector noise as the only
    noise,

    $$
    \chi^2(\Lambda) = 2T \langle S_h^{\rm data} - S_h(\Lambda) | S_h^{\rm data} - S_h(\Lambda) \rangle
    = \sum_k \left[ \frac{S_h^{\rm data}(f_k) - S_h(f_k|\Lambda)}{\Delta S_h(f_k)} \right]^2,
    \qquad
    \mathcal{L}(S_h^{\rm data}|\Lambda) \propto e^{-\chi^2/2}.
    $$

    The posterior follows from Bayes' theorem with the prior $\pi(\Lambda)$; here
    only $H_0$ varies, with a uniform prior on a grid,

    $$
    p(\Lambda | S_h^{\rm data}) = \frac{\pi(\Lambda)\, \mathcal{L}(S_h^{\rm data}|\Lambda)}
    {\int d\Lambda\, \pi(\Lambda)\, \mathcal{L}(S_h^{\rm data}|\Lambda)}.
    $$

    This is the `detector` variant below.

    ### The amplitude of the background

    If the spectrum is rescaled as $S_h(f|\Lambda) = (A/A_t)\, S_h^t(f)$, the
    likelihood is Gaussian in $A$,

    $$
    \chi^2(A) = \left( \frac{A - \hat A_{\rm MLE}}{\sigma_A} \right)^2 + \mathrm{const},
    \qquad
    \hat A_{\rm MLE} = \frac{\langle S_h^{\rm data} | S_h^t \rangle}{\langle S_h^t | S_h^t \rangle}\, A_t,
    \qquad
    \sigma_A = \frac{A_t}{\rho}.
    $$

    Detector noise alone therefore fixes the amplitude to a relative accuracy
    $1/\rho$. This is the unit in which shot noise is compared with detector
    noise below.

    ### Shot-noise amplitude model

    Shot noise is almost perfectly correlated across the frequencies that carry
    the signal, so its main effect is to rescale the spectrum,

    $$
    S_h^{\rm cat}(f) \simeq S_h(f|\Lambda)\,\big[1 + \epsilon(\Lambda)\big],
    \qquad
    \epsilon \sim \mathcal{N}\big(0, \sigma_\epsilon^2(\Lambda)\big).
    $$

    ($\sigma_\epsilon$ is called `s` in the code.) It is set by the per-bin
    shot-noise variance $V_k = \mathrm{Var}[n_{\rm shot}(f_k)]$ as the SNR-weighted
    mean relative scatter,

    $$
    \sigma_\epsilon = \frac{\sum_k \rho_k^2\, \sqrt{V_k} / S_h(f_k|\Lambda)}{\sum_k \rho_k^2}.
    $$

    This is exact when the shot noise is perfectly correlated across bins, and an
    upper bound otherwise. For an injection without detector noise, the amplitude
    estimator above gives $\hat\epsilon = \hat A_{\rm MLE}/A_t - 1$. With
    detector noise too, the relative amplitude has variance
    $\rho^{-2} + \sigma_\epsilon^2$, and the ratio of shot to detector scatter is
    $\rho\,\sigma_\epsilon$.

    ### Likelihood with shot noise

    Marginalizing $\epsilon$ leaves a Gaussian in $S_h^{\rm data}$ with mean
    $S_h(\Lambda)$ and a rank-one correction to the diagonal detector covariance
    $D = \mathrm{diag}\big(\Delta S_h^2(f_k)\big)$,

    $$
    C = D + \mathbf{u}\,\mathbf{u}^T,
    \qquad
    \mathbf{u} = \sigma_\epsilon\, S_h(\Lambda).
    $$

    With the whitened residual $x_k = [S_h^{\rm data}(f_k) - S_h(f_k|\Lambda)] / \Delta S_h(f_k)$
    and mode $v_k = u_k / \Delta S_h(f_k)$, the Sherman-Morrison identity gives

    $$
    \chi^2 = x^T x - \frac{(x^T v)^2}{1 + v^T v},
    \qquad
    \ln\det C = \sum_k \ln \Delta S_h^2(f_k) + \ln\left(1 + v^T v\right),
    \qquad
    v^T v = (\rho\,\sigma_\epsilon)^2.
    $$

    The three likelihood variants differ only in $\mathbf{u}$:

    | Variant | $\mathbf{u}$ |
    |---|---|
    | `detector` | $0$ |
    | `amplitude` | $\sigma_\epsilon(\Lambda)\, S_h(\Lambda)$, with $\sigma_\epsilon$ re-evaluated at every grid point |
    | `fixed` | $\sigma_\epsilon(\Lambda_t)\, S_h(\Lambda)$, with $\sigma_\epsilon$ frozen at the true values |

    ### Calibration statistics

    For every posterior the notebook computes the posterior mean $\bar H_0$ and
    standard deviation, and two statistics of the truth $H_0^t$,

    $$
    z = \frac{\bar H_0 - H_0^t}{\mathrm{sd}(H_0)},
    \qquad
    \mathrm{PIT} = \int_{-\infty}^{H_0^t} p(H_0 | S_h^{\rm data})\, dH_0.
    $$

    A calibrated posterior has $\mathrm{PIT} \sim \mathcal{U}(0, 1)$ and
    $z \sim \mathcal{N}(0, 1)$. Linearize $S_h(\Lambda)$ about $\Lambda_t$ and take
    its response to $H_0$ along the amplitude direction. The posterior mean then
    moves with the data deviation $\epsilon\, S_h^t + n_{\rm det}$, and without the
    shot-noise term

    $$
    z = \pm\rho\,\epsilon + \xi, \quad \xi \sim \mathcal{N}(0, 1)
    \qquad\Longrightarrow\qquad
    \mathrm{Var}(z) = 1 + (\rho\,\sigma_\epsilon)^2.
    $$

    The shot-noise likelihood widens the posterior by $\sqrt{1 + (\rho\sigma_\epsilon)^2}$
    and restores $\mathrm{Var}(z) = 1$. With $\rho\sigma_\epsilon \approx 0.48$ the
    detector-only value is about $1.23$, and its 68% interval covers about 63%.

    ### What the notebook does

    - $M$ independent Poisson catalogs at $\Lambda_t$ are the injections
      $S_h^{\rm cat}$, shared with `notebooks/ensemble_spectrum.py`.
    - Each gets $R$ detector-noise draws $n_{\rm det}$. Detector noise is free next
      to an injection, and the error on $\mathrm{Var}(z)$ is then set by the $M$
      shot-noise draws rather than by $\sqrt{2/M}$ of a single detector draw.
    - The model $S_h(f|\Lambda)$ is the rescaled reference catalog of the grid
      notebook, at a different seed, so it is independent of the injections.
    - Only $H_0$ is swept. Its predictions on the grid do not depend on the data,
      so they are computed once.
    - Every error is a bootstrap over injections, since the $R$ draws of one
      injection share its shot noise.

    Setting `ASTROGWB_NOTEBOOK_SMOKE=1` swaps in a tiny catalog, few injections and
    fixed-count spectra.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configuration
    """)


@app.cell
def _():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    FIGURES = ROOT_DIR / FIGURES_DIR / "shot_noise_coverage"

    # The injections are the Poisson ensemble of ensemble_spectrum.py:
    # the same record, seed and cache, so the draws are made once for both.
    num_injections = 1000
    ensemble_seed = 41
    noise_seed = 2000  # detector noise, one stream per network
    # Detector-noise draws per injection: noise is free next to an injection,
    # and one draw each leaves var(z) dominated by its own sampling error.
    noise_draws = 20
    # The approximant of the ensemble draws, which this notebook reads.
    approximant = "IMRPhenomXAS_NRTidalv3"
    minimum_redshift, maximum_redshift = 0.35, 20.0
    catalog_seed = 42  # independent of the injections' seed
    num_samples = 2**14  # on a scrambled Sobol net; see importance_convergence.py
    redshift_nodes = 32
    catalog_chunk_size = 4096
    chunk_size = 16_384  # injection waveforms per reduced chunk
    grid_chunk_size = 8

    observation_time = 1.0  # years
    minimum_frequency = 2.0
    maximum_frequency = 2048.0
    fiducial_network = "ET-2L-aligned-CE-Hanford"
    network_names = [name for name, _ in DETECTOR_NETWORKS]
    h0_window, h0_size = (60.0, 76.0), 321

    SMOKE = os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") == "1"
    write_figures_default = True
    if SMOKE:
        num_injections = 4
        noise_draws = 2
        num_samples = 64
        redshift_nodes = 8
        catalog_chunk_size = 512
        chunk_size = 512
        grid_chunk_size = 4
        h0_window, h0_size = (40.0, 100.0), 31
        write_figures_default = False

    FIDUCIALS = fiducials(root=ROOT_DIR)
    PRIORS = priors(root=ROOT_DIR)
    registry = detector_registry(root=ROOT_DIR)
    # Same 1024-point grid as ensemble_spectrum.py, whose draws these injections
    # read. Keep this waveform record in sync with that notebook.
    waveform = waveform_metadata(
        root=ROOT_DIR,
        approximant=approximant,
        frequency_resolution=0.125,
        turnover_frequency=100.0,
        dlogf=0.01266,
    )
    frequencies = np.asarray(waveform.build().frequencies)
    population = population_metadata(
        root=ROOT_DIR,
        minimum_redshift=minimum_redshift,
        maximum_redshift=maximum_redshift,
    )
    catalog_metadata = CatalogMetadata(
        waveform=waveform_metadata(
            root=ROOT_DIR,
            approximant=approximant,
            frequency_spacing="log",
            minimum_frequency=minimum_frequency,
            maximum_frequency=4000.0,
            frequency_resolution=0.02,
            turnover_frequency=None,
        ),
        population=population,
        fiducials=FIDUCIALS,
        num_samples=num_samples,
        sampling="sobol",
    )
    injection_metadata = BackgroundSpectralDensityMetadata(
        count="fixed" if SMOKE else "poisson",
        num_events=64 if SMOKE else None,
        observation_time=observation_time,
        hyperparameters={**FIDUCIALS},
        waveform=waveform,
        population=population,
    )
    ensemble_dir = default_cache_dir() / "ensembles"

    write_figures = mo.ui.switch(value=write_figures_default, label="Write figures")
    write_figures
    use_paper_style(root=ROOT_DIR)
    # gwpy registers replacement default axes; keep matplotlib's.
    register_projection(Axes)
    return (
        FIDUCIALS,
        FIGURES,
        PRIORS,
        catalog_chunk_size,
        catalog_metadata,
        catalog_seed,
        chunk_size,
        ensemble_dir,
        ensemble_seed,
        fiducial_network,
        frequencies,
        grid_chunk_size,
        h0_size,
        h0_window,
        injection_metadata,
        maximum_frequency,
        minimum_frequency,
        network_names,
        noise_draws,
        noise_seed,
        num_injections,
        observation_time,
        population,
        redshift_nodes,
        registry,
        write_figures,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Toolbox
    """)


@app.function(hide_code=True)
def ensure_reference_catalog(
    metadata: CatalogMetadata,
    seed: int,
    directory: Path,
    *,
    chunk_size: int | None = None,
) -> PolarizationPowerData:
    """The reference catalog of ``metadata`` at ``seed``, drawn on a miss.

    The same file the grid notebook reads and writes,
    ``<directory>/reference_catalog-<key>-<seed>.h5``.

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
def network_data(
    registry: DetectorRegistry,
    name: str,
    frequencies: NDArray[np.float64],
    *,
    observation_time: float,
    minimum_frequency: float,
    maximum_frequency: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """The per-bin Gaussian scale and usable-bin mask of one network.

    Unusable bins get a finite scale of one; the mask drops them.
    """
    grid = jnp.asarray(frequencies)
    geometry, sensitivities = registry.build_network(name)
    psd = jnp.asarray(effective_psd(grid, geometry, sensitivities))
    mask = (
        frequency_mask(grid, fmin=minimum_frequency, fmax=maximum_frequency)
        & jnp.isfinite(psd)
        & (psd > 0.0)
    )
    scale = gaussian_bin_scale(psd, observation_time, grid)
    return np.asarray(jnp.where(mask, scale, 1.0)), np.asarray(mask)


@app.function(hide_code=True)
def posterior_statistics(
    grid: NDArray[np.float64],
    log_density: NDArray[np.float64],
    truth: float,
) -> dict[str, NDArray[np.float64]]:
    """PIT of the truth, mean and sd of each grid posterior.

    ``log_density`` is ``(M, G)``. The PIT is the posterior CDF at the truth,
    by the trapezoid rule; a calibrated posterior gives a uniform PIT.
    """
    density = np.exp(log_density - log_density.max(axis=-1, keepdims=True))
    density /= np.trapezoid(density, grid, axis=-1)[:, None]
    cell = np.diff(grid)
    cdf = np.concatenate(
        [
            np.zeros((density.shape[0], 1)),
            np.cumsum(0.5 * (density[:, 1:] + density[:, :-1]) * cell, axis=-1),
        ],
        axis=-1,
    )
    pit = np.array([np.interp(truth, grid, row) for row in cdf])
    mean = np.trapezoid(grid * density, grid, axis=-1)
    sd = np.sqrt(np.trapezoid((grid - mean[:, None]) ** 2 * density, grid, axis=-1))
    return {"pit": pit, "mean": mean, "sd": sd, "z": (mean - truth) / sd}


@app.function(hide_code=True)
def coverage_row(
    statistics: dict[str, NDArray[np.float64]],
    num_injections: int,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Interval coverage and the mean and variance of ``z``, with errors.

    The statistics are ``(num_injections * R,)``, injection-major: ``R``
    detector-noise draws per injection. Those draws share their injection's
    shot noise, so they are not independent, and every error is a bootstrap
    over injections, each resampled with all of its noise draws.
    """

    def summary(pit: NDArray[np.float64], z: NDArray[np.float64]) -> dict:
        axes = (-2, -1)
        values = {
            f"cover {mass:.0%}": np.mean(
                (pit > (1.0 - mass) / 2.0) & (pit < (1.0 + mass) / 2.0), axis=axes
            )
            for mass in (0.68, 0.95)
        }
        values["mean z"] = np.mean(z, axis=axes)
        values["var z"] = np.var(z, axis=axes)
        return values

    pit = statistics["pit"].reshape(num_injections, -1)
    z = statistics["z"].reshape(num_injections, -1)
    index = np.random.default_rng(seed).integers(
        0, num_injections, size=(n_bootstrap, num_injections)
    )
    point, resampled = summary(pit, z), summary(pit[index], z[index])
    row: dict[str, float] = {}
    for name, value in point.items():
        row[name] = float(value)
        row[f"{name} err"] = float(np.std(resampled[name]))
    return row


@app.function(hide_code=True)
def plot_eps_density(ax: Axes, samples: NDArray[np.float64], labels: list[str]) -> None:
    """Plot a Gaussian KDE of each column of ``samples`` on ``ax``.

    Every KDE is evaluated on one window, four standard deviations either side
    of the column means, so the tails are not cut at the data range and the
    widths and offsets of the columns are comparable. A dashed normal density,
    with the average mean and kernel-smoothed standard deviation of the columns,
    is overlaid.

    Parameters
    ----------
    ax
        Axes to draw on.
    samples
        Draws of shape ``(M, K)``, one column per label.
    labels
        The ``K`` legend labels.
    """
    sd = samples.std(axis=0, ddof=1)
    mean = samples.mean(axis=0)
    window = ((mean - 4.0 * sd).min(), (mean + 4.0 * sd).max())
    grid, density, bandwidth = az.kde(
        samples, axis=0, custom_lims=window, bound_correction=False
    )
    for label, x, y in zip(labels, grid, density, strict=True):
        ax.plot(x, y, label=label)
    # The columns share nearly one width, so a single reference normal with
    # their mean location and scale stands for all of them. A Gaussian KDE of
    # normal draws is a normal of variance sd^2 + bandwidth^2, so the reference
    # is widened by the kernel to be comparable with the curves.
    x = np.linspace(*window, 512)
    ax.plot(
        x,
        stats.norm.pdf(x, loc=mean.mean(), scale=np.hypot(sd, bandwidth).mean()),
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Normal",
    )
    ax.set(xlabel=r"$\epsilon$", ylabel="Density", xlim=window)
    ax.legend(fontsize="small")


@app.function(hide_code=True)
def plot_eps_matrix(
    ax: Axes,
    frequencies: NDArray[np.float64],
    matrix: NDArray[np.float64],
    *,
    label: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    extend: str = "neither",
) -> None:
    """Draw a matrix over frequency on ``ax``, with a colorbar.

    Parameters
    ----------
    ax
        Axes to draw on.
    frequencies
        Bin centers in Hz, shape ``(F,)``.
    matrix
        Matrix of shape ``(F, F)``, for example a covariance or correlation.
    label
        Colorbar label.
    vmin, vmax
        Color limits.
    cmap
        Matplotlib colormap name.
    extend
        Which ends of the colorbar get an arrow for clipped values.
    """
    mesh = ax.pcolormesh(
        frequencies,
        frequencies,
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"$f$ [Hz]",
        ylabel=r"$f'$ [Hz]",
        aspect="equal",
    )
    # In axes coordinates, so the colorbar is as tall as the drawn (square) axes,
    # which `colorbar(ax=ax)` is not once the aspect is equal.
    cax = ax.inset_axes((1.04, 0.0, 0.05, 1.0))
    ax.figure.colorbar(mesh, cax=cax, label=label, extend=extend)


@app.function(hide_code=True)
def plot_eps_eigenvalues(
    ax: Axes, eigenvalues: NDArray[np.float64], num_modes: int = 20
) -> None:
    """Plot the leading eigenvalues of the whitened shot-noise covariance.

    The shot noise is whitened by the detector scale, so a detector-noise
    variance is 1 and the dashed line marks it. An eigenvalue far above the
    line is a mode that detector noise cannot hide, and one far below it is
    irrelevant to the likelihood.

    Parameters
    ----------
    ax
        Axes to draw on.
    eigenvalues
        Eigenvalues in descending order.
    num_modes
        Number of leading modes to show.
    """
    modes = np.arange(1, num_modes + 1)
    ax.semilogy(modes, eigenvalues[:num_modes], "o")
    ax.axhline(1.0, color="0.5", ls="--", lw=1, label="detector noise")
    ax.set(xlabel="Mode", ylabel=r"Eigenvalue $\lambda_j$")
    ax.set_xticks([1, *range(5, num_modes + 1, 5)])
    ax.legend(fontsize="small")


@app.function(hide_code=True)
def common_shape_metrics(
    covariance: NDArray[np.float64],
    template: NDArray[np.float64],
    scale: NDArray[np.float64],
    shape_derivatives: dict[str, NDArray[np.float64]],
) -> tuple[dict[str, float], NDArray[np.float64]]:
    r"""How much of a catalog covariance the common-shape model drops.

    Everything is whitened by the detector noise,
    :math:`C = N^{-1/2} \mathrm{Cov} N^{-1/2}`, with the unit template
    direction :math:`\mathbf{e} = N^{-1/2}\mathbf{S} / \rho`. The common-shape
    (rank-one) model keeps :math:`\kappa^2 \mathbf{e}\mathbf{e}^T`, with
    :math:`\kappa^2 = \mathbf{e}^T C \mathbf{e} = \rho^2 \sigma_\epsilon^2`, and
    drops :math:`\Delta = C - \kappa^2 \mathbf{e}\mathbf{e}^T`: the
    perpendicular block :math:`P C P`, :math:`P = I - \mathbf{e}\mathbf{e}^T`,
    and its coupling :math:`P C \mathbf{e}` to the amplitude.

    The Kullback-Leibler divergence of the common-shape Gaussian
    :math:`\Sigma_{\rm cs} = I + \kappa^2 \mathbf{e}\mathbf{e}^T` from the
    true one :math:`\Sigma = I + C`, with Sherman-Morrison for
    :math:`\Sigma_{\rm cs}^{-1}`, is

    .. math::

        \mathrm{KL} = \tfrac12 \left[ \mathrm{tr}\, C - \kappa^2
            - \ln\det(I + C) + \ln(1 + \kappa^2) \right],

    and its leading term is
    :math:`\tfrac14 \| \Sigma_{\rm cs}^{-1/2} \Delta \Sigma_{\rm cs}^{-1/2} \|_F^2`.

    Parameters
    ----------
    covariance
        Catalog covariance over the counted bins, ``(F, F)``.
    template
        Expected spectrum over the counted bins, ``(F,)``.
    scale
        Detector standard deviation per counted bin, ``(F,)``.
    shape_derivatives
        Derivatives of the spectrum over the counted bins, ``(F,)`` each,
        keyed by parameter; each is whitened and projected off
        :math:`\mathbf{e}` for its posterior-variance inflation
        :math:`\mathbf{g}^T P C P \mathbf{g} / \|\mathbf{g}\|^2`.

    Returns
    -------
    metrics
        ``rho``, ``kappa^2``, the extreme eigenvalues of :math:`\Delta`, the
        largest of :math:`P C P`, :math:`\|P C \mathbf{e}\|`, the KL divergence
        and its leading term in nats, and one ``inflation - 1`` per derivative.
    delta_eigenvalues
        Eigenvalues of :math:`\Delta`, descending, ``(F,)``.
    """
    signal = template / scale
    rho = float(np.sqrt(signal @ signal))
    e = signal / rho
    whitened = covariance / np.outer(scale, scale)
    kappa2 = float(e @ whitened @ e)
    delta = whitened - kappa2 * np.outer(e, e)
    projector = np.eye(e.size) - np.outer(e, e)
    perpendicular = projector @ whitened @ projector
    delta_eigenvalues = np.linalg.eigvalsh(delta)[::-1]
    covariance_eigenvalues = np.linalg.eigvalsh(whitened)
    kl = 0.5 * (
        np.trace(whitened)
        - kappa2
        - np.sum(np.log1p(covariance_eigenvalues))
        + np.log1p(kappa2)
    )
    # Sigma_cs^{-1/2} = I - (1 - 1/sqrt(1 + kappa^2)) e e^T.
    root = np.eye(e.size) - (1.0 - 1.0 / np.sqrt(1.0 + kappa2)) * np.outer(e, e)
    metrics = {
        "rho": rho,
        "kappa^2": kappa2,
        "lambda_max(Delta)": float(delta_eigenvalues[0]),
        "lambda_min(Delta)": float(delta_eigenvalues[-1]),
        "lambda_max(PCP)": float(np.linalg.eigvalsh(perpendicular)[-1]),
        "|PCe|": float(np.linalg.norm(projector @ whitened @ e)),
        "KL [nats]": float(kl),
        "KL leading [nats]": float(0.25 * np.sum((root @ delta @ root) ** 2)),
    }
    for name, derivative in shape_derivatives.items():
        g = projector @ (derivative / scale)
        metrics[f"inflation - 1 ({name})"] = float(g @ perpendicular @ g / (g @ g))
    return metrics, delta_eigenvalues


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model on the grid

    The rescaled spectrum and its shot-noise variance at every $H_0$ grid
    point, once; then $s^2$ at the fiducial per network (the adopted `fixed`
    variant) and at every grid point (`amplitude`).
    """)


@app.cell
def _(
    FIDUCIALS,
    catalog_chunk_size,
    catalog_metadata,
    catalog_seed,
    frequencies,
    grid_chunk_size,
    h0_size,
    h0_window,
    observation_time,
    per_network,
    population,
    redshift_nodes,
):
    catalog_data = ensure_reference_catalog(
        catalog_metadata, catalog_seed, CATALOGS_ROOT, chunk_size=catalog_chunk_size
    )
    target = build_population(population.model_name, **population.model_kwargs)
    likelihood = ImportanceGaussianLikelihood.from_catalog(
        catalog_data,
        catalog_metadata,
        population=target,
        frequencies=frequencies,
        num_redshift_nodes=redshift_nodes,
        density_sites=(),
        shot_noise="amplitude",
        observation_time=observation_time,
    )
    h0_grid = np.linspace(*h0_window, h0_size)

    @jax.jit
    def _predict(likelihood, h0):
        def point(value):
            return likelihood.predict({**FIDUCIALS, "H0": value})[0]

        return jax.lax.map(point, h0, batch_size=grid_chunk_size)

    grid_prediction = _predict(likelihood, jnp.asarray(h0_grid))
    _fiducial = int(np.argmin(np.abs(h0_grid - FIDUCIALS["H0"])))
    print(f"grid: {h0_size} H0 points; fiducial node H0 = {h0_grid[_fiducial]:.3f}")
    fiducial_prediction, fiducial_variance, _ = likelihood.predict(FIDUCIALS)

    # s^2 of each network: the catalog projected on its noise-weighted template.
    @jax.jit
    def _relative_variance(likelihood, h0, scale, mask):
        network = LikelihoodNetwork(scale, mask)
        return jax.lax.map(
            lambda value: likelihood.relative_variance(
                {**FIDUCIALS, "H0": value}, network
            ),
            h0,
            batch_size=grid_chunk_size,
        )

    grid_relative_variance = {
        _name: _relative_variance(
            likelihood, jnp.asarray(h0_grid), jnp.asarray(_scale), jnp.asarray(_mask)
        )
        for _name, (_scale, _mask) in per_network.items()
    }
    fiducial_relative_variance = {
        _name: float(
            likelihood.relative_variance(
                FIDUCIALS, LikelihoodNetwork(jnp.asarray(_scale), jnp.asarray(_mask))
            )
        )
        for _name, (_scale, _mask) in per_network.items()
    }
    return (
        fiducial_prediction,
        fiducial_relative_variance,
        fiducial_variance,
        grid_prediction,
        grid_relative_variance,
        h0_grid,
        likelihood,
    )


@app.cell
def _(
    frequencies,
    maximum_frequency,
    minimum_frequency,
    network_names,
    observation_time,
    registry,
):
    per_network = {
        name: network_data(
            registry,
            name,
            frequencies,
            observation_time=observation_time,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
        for name in network_names
    }
    return (per_network,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Injections

    $M$ Poisson spectra at the fiducials: the ensemble of
    `notebooks/ensemble_spectrum.py`, read from the shared cache
    (`spectra_ensemble`, one file per draw).
    """)


@app.cell
def _(
    chunk_size,
    ensemble_dir,
    ensemble_seed,
    frequencies,
    injection_metadata,
    num_injections,
):
    _ensemble = spectra_ensemble(
        injection_metadata,
        ensemble_seed,
        num_injections,
        ensemble_dir,
        chunk_size=chunk_size,
    )
    injections = _ensemble["spectral_density"]
    injected_events = np.asarray(_ensemble["n_events"])
    print(f"injection record {injection_metadata.key()}, seed {ensemble_seed}")
    assert injections.shape[1] == frequencies.size
    print(
        f"{num_injections} injections, events {injected_events.min():,} to "
        f"{injected_events.max():,}"
    )
    return (injections,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model against the injections

    **Table.** For each detector network it compares the shot-noise scatter the
    model predicts with the scatter the Poisson injections realize, in units of
    the detector error $1/\rho$. No posterior is involved, and no detector noise is
    added to the injections.

    The model side is $\rho\,\sigma_\epsilon$, with $\sigma_\epsilon$ evaluated at
    $\Lambda_t$ from the per-bin variance $V_k$ of the reference catalog. The
    injection side projects each injection $S^{\rm cat}_{h,i}$ on the template
    $S_h^t$,

    $$
    \hat\epsilon_i = \frac{\langle S_h^t | S^{\rm cat}_{h,i} - S_h^t \rangle}{\langle S_h^t | S_h^t \rangle}
    = \frac{\hat A_{\rm MLE}}{A_t} - 1,
    $$

    and reports $\rho\,\mathrm{sd}(\hat\epsilon)$. Multiplying by $\rho$ expresses
    the scatter in detector widths, the unit in which a coverage test is
    sensitive.

    | Column | Quantity |
    |---|---|
    | `rho` | $\rho$, the SNR of the template |
    | `model rho s` | $\rho\,\sigma_\epsilon$, predicted |
    | `empirical rho sd(eps)` | $\rho\,\mathrm{sd}(\hat\epsilon)$ over the $M$ injections |
    | `error` | the sampling error of a standard deviation, $\mathrm{sd}/\sqrt{2(M-1)}$ |
    | `empirical / model` | the ratio, which should be the same for every network |

    The shot noise is one amplitude mode, so $\rho$ cancels in the ratio. Expect
    it at most 1: it equals 1 if the shot noise is perfectly correlated across
    bins, and falls below 1 by the decorrelated part, where the model is
    conservative. A ratio above 1 beyond `error` would mean the model understates
    the shot noise.
    """)


@app.cell
def _(fiducial_prediction, fiducial_relative_variance, injections, per_network):
    _label = dict(DETECTOR_NETWORKS)
    _template = np.asarray(fiducial_prediction)
    _rows = []
    for _name, (_scale, _mask) in per_network.items():
        _weight = np.where(_mask, 1.0 / _scale**2, 0.0)
        _norm = np.sum(_template**2 * _weight)
        _rho = np.sqrt(_norm)
        _model = _rho * np.sqrt(fiducial_relative_variance[_name])
        _eps = (injections - _template) @ (_template * _weight) / _norm
        _empirical = _rho * np.std(_eps, ddof=1)
        _rows.append(
            {
                "network": _label[_name],
                "rho": _rho,
                "model rho s": _model,
                "empirical rho sd(eps)": _empirical,
                "error": _empirical / np.sqrt(2.0 * (_eps.size - 1)),
                "empirical / model": _empirical / _model,
            }
        )
    model_against_injections = pd.DataFrame(_rows)
    model_against_injections
    return (model_against_injections,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Shot-noise covariance of the injections

    The relative shot-noise fluctuation per bin, with no detector noise, is

    $$
    \epsilon(f_k) = \frac{S^{\rm cat}_h(f_k) - S_h^t(f_k)}{S_h^t(f_k)},
    $$

    and its sample covariance $\hat C_{kl}$ over the $M$ injections, with
    correlation $\hat R_{kl} = \hat C_{kl} / \sqrt{\hat C_{kk}\hat C_{ll}}$.

    **Figure (variance).** The diagonal $\hat C_{kk}$ against frequency, over the
    bins the fiducial network can use, with the model's per-bin
    $V_k / S_h^t(f_k)^2$ from the reference catalog for comparison.

    **Figure (correlation).** The heatmap of $\hat R_{kl}$. A block of
    correlation close to 1 is the band where shot noise acts as a single
    amplitude mode, which is the assumption behind the rank-one model.
    """)


@app.cell
def _(
    fiducial_network,
    fiducial_prediction,
    fiducial_variance,
    frequencies,
    injections,
    per_network,
):
    _template = np.asarray(fiducial_prediction)
    _mask = per_network[fiducial_network][1] & (_template > 0.0)
    eps_frequencies = frequencies[_mask]
    # Squared SNR of the template in each usable bin: the weight of a bin in the
    # likelihood, and the whitening of the shot noise by the detector noise.
    eps_snr_squared = (_template[_mask] / per_network[fiducial_network][0][_mask]) ** 2
    eps_samples = (injections[:, _mask] - _template[_mask]) / _template[_mask]
    eps_covariance = np.cov(eps_samples, rowvar=False)
    eps_variance = np.diag(eps_covariance)
    model_variance = np.asarray(fiducial_variance)[_mask] / _template[_mask] ** 2
    eps_correlation = eps_covariance / np.sqrt(np.outer(eps_variance, eps_variance))

    variance_figure, (_ax, _ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(5.0, 4.6),
        sharex=True,
        height_ratios=(3, 1),
    )
    _ax.loglog(eps_frequencies, eps_variance, label="injections")
    _ax.loglog(eps_frequencies, model_variance, "--", label="model")
    _ax.set(ylabel=r"$\mathrm{Var}[\epsilon(f)]$", ylim=(1e-6, 1e-5))
    _ax.legend(fontsize="small")
    _ax_ratio.semilogx(eps_frequencies, model_variance / eps_variance)
    _ax_ratio.axhline(1.0, color="0.5", lw=1)
    _ax_ratio.set(xlabel=r"$f$ [Hz]", ylabel="model / injections")
    variance_figure.tight_layout()
    variance_figure
    return (
        eps_correlation,
        eps_covariance,
        eps_frequencies,
        eps_samples,
        eps_snr_squared,
        variance_figure,
    )


@app.cell
def _(eps_correlation, eps_frequencies):
    correlation_figure, _ax = plt.subplots(figsize=(5.0, 4.2))
    plot_eps_matrix(
        _ax,
        eps_frequencies,
        eps_correlation,
        label=r"$\mathrm{Corr}[\epsilon_k, \epsilon_{k'}]$",
        # The correlation is close to 1 almost everywhere; a [-1, 1] range would
        # hide the decorrelation at high frequency.
        vmin=np.floor(20.0 * np.nanmin(eps_correlation)) / 20.0,
        vmax=1.0,
    )
    correlation_figure.tight_layout()
    correlation_figure
    return (correlation_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Excess of the covariance over a constant.** The shot-noise covariance is almost
    constant, so it is shown relative to the paper's rank-one model, with
    $\sigma_\epsilon^2 = \sum_{kk'} w_k w_{k'} \mathrm{Cov}[\epsilon_k, \epsilon_{k'}]$
    the variance of the fractional amplitude, with weights $w_k = \rho_k^2 /
    \sum_j \rho_j^2$ from the squared SNR of each bin: $\mathrm{Cov}[\epsilon_k, \epsilon_{k'}] /
    \sigma_\epsilon^2 - 1$ on a diverging scale clipped at $\pm 0.3$, which shows
    where and with which sign the covariance departs from a constant.
    """)


@app.cell(hide_code=True)
def _(eps_covariance, eps_snr_squared):
    # Effective shot-noise amplitude: the variance of epsilon projected on the
    # template, with bin weights rho_k^2, as in the paper's scalar approximation.
    _weights = eps_snr_squared / np.sum(eps_snr_squared)
    sigma_eps = float(np.sqrt(_weights @ eps_covariance @ _weights))
    # Fractional excess of the covariance over the rank-one model sigma_eps^2.
    eps_covariance_excess = eps_covariance / sigma_eps**2 - 1.0
    return (eps_covariance_excess,)


@app.cell(hide_code=True)
def _(eps_covariance_excess, eps_frequencies):
    covariance_deviation_figure, _ax = plt.subplots(figsize=(5.0, 4.2))
    plot_eps_matrix(
        _ax,
        eps_frequencies,
        eps_covariance_excess,
        label=r"$\mathrm{Cov}[\epsilon_k, \epsilon_{k'}] / \sigma_\epsilon^2 - 1$",
        vmin=-0.3,
        vmax=0.3,
        cmap="RdBu_r",
    )
    covariance_deviation_figure.tight_layout()
    covariance_deviation_figure
    return (covariance_deviation_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Figure (distribution).** Correlation close to 1 shows a single mode, not that
    the mode is Gaussian, which the shot-noise likelihood assumes. At a few
    frequencies in the inspiral band it plots the standardized
    $\epsilon / \mathrm{sd}(\epsilon)$ across the $M$ injections: *left*, a
    histogram against $\mathcal{N}(0, 1)$; *right*, a Q-Q plot against the normal
    quantiles, where a Gaussian lies on the diagonal. With $M$ draws only the
    bulk is tested: the tails are not resolved beyond about $\pm\sqrt{2\ln M}$.
    """)


@app.cell
def _(eps_frequencies, eps_samples):
    _probe = (5.0, 10.0, 20.0, 50.0, 100.0)
    _index = [int(np.argmin(np.abs(eps_frequencies - f))) for f in _probe]
    probe_labels = [f"{eps_frequencies[_i]:.0f} Hz" for _i in _index]
    probe_samples = eps_samples[:, _index]
    probe_standardized = (
        probe_samples - probe_samples.mean(axis=0)
    ) / probe_samples.std(axis=0, ddof=1)

    distribution_figure, _ax = plt.subplots(figsize=(4.5, 3.6))
    plot_eps_density(_ax, probe_samples, probe_labels)
    distribution_figure.tight_layout()

    eps_normality = pd.DataFrame(
        {
            "f [Hz]": eps_frequencies[_index],
            "sd(eps)": probe_samples.std(axis=0, ddof=1),
            "skew": stats.skew(probe_samples, axis=0),
            "excess kurtosis": stats.kurtosis(probe_samples, axis=0),
            "Shapiro p": [stats.shapiro(_column).pvalue for _column in probe_samples.T],
        }
    )
    mo.vstack([distribution_figure, eps_normality])
    return distribution_figure, probe_labels, probe_samples, probe_standardized


@app.cell(hide_code=True)
def _(probe_labels, probe_standardized):
    _count = probe_standardized.shape[0]
    _theory = stats.norm.ppf((np.arange(1, _count + 1) - 0.5) / _count)
    qq_figure, _ax = plt.subplots(figsize=(4.0, 3.6))
    for _label, _column in zip(probe_labels, probe_standardized.T, strict=True):
        _ax.plot(_theory, np.sort(_column), ".", ms=3, label=_label)
    _ax.plot([-3, 3], [-3, 3], color="0.5", lw=1)
    _ax.set(xlabel="Normal quantile", ylabel=r"$\epsilon / \mathrm{sd}(\epsilon)$")
    _ax.legend(fontsize="small")
    qq_figure.tight_layout()
    qq_figure
    return (qq_figure,)


@app.cell(hide_code=True)
def _(eps_covariance_excess, eps_frequencies, probe_labels, probe_samples):
    combined_figure, (_ax_density, _ax_excess) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    plot_eps_density(_ax_density, probe_samples, probe_labels)
    plot_eps_matrix(
        _ax_excess,
        eps_frequencies,
        eps_covariance_excess,
        label=r"$\mathrm{Cov}[\epsilon_k, \epsilon_{k'}] / \sigma_\epsilon^2 - 1$",
        vmin=-0.3,
        vmax=0.3,
        cmap="RdBu_r",
    )
    combined_figure.tight_layout()
    combined_figure
    return (combined_figure,)


@app.cell(hide_code=True)
def _(eps_samples, eps_snr_squared):
    # Shot noise in units of the detector noise: detector variance is 1.
    _whitened = eps_samples * np.sqrt(eps_snr_squared)
    whitened_eigenvalues = np.linalg.eigvalsh(np.cov(_whitened, rowvar=False))[::-1]
    eigenvalue_figure, _ax = plt.subplots(figsize=(4.0, 3.6))
    plot_eps_eigenvalues(
        _ax, whitened_eigenvalues, num_modes=min(20, eps_samples.shape[0] - 1)
    )
    eigenvalue_figure.tight_layout()
    eigenvalue_figure
    return (whitened_eigenvalues,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Accuracy of the common-shape approximation

    The excess above is a fraction of $\sigma_\epsilon^2$ per bin and ignores
    the detector noise, so it overstates the problem in bins the detector
    noise dominates. The likelihood sees the shot noise only in units of the
    detector noise. Whiten by $N = \mathrm{diag}(\Delta S_h^2)$:

    $$
    C = N^{-1/2}\, \mathrm{Cov}[S_h^{\rm cat}]\, N^{-1/2},
    \qquad
    \mathbf{e} = \frac{N^{-1/2} S_h^t}{\rho},
    \qquad
    \kappa^2 = \mathbf{e}^T C\, \mathbf{e} = \rho^2 \sigma_\epsilon^2.
    $$

    The common-shape model keeps $\kappa^2 \mathbf{e}\mathbf{e}^T$ and drops
    $\Delta = C - \kappa^2 \mathbf{e}\mathbf{e}^T$. With
    $P = I - \mathbf{e}\mathbf{e}^T$, $\Delta$ is the perpendicular shot noise
    $P C P$ plus its coupling $P C \mathbf{e}$ to the amplitude. Its eigenvalues
    are variances relative to one detector-noise variance, so
    $\lambda_{\max}(\Delta)$ is the worst mode the model misses, next to the
    $\kappa^2$ it keeps. A coupling-dominated $\Delta$ has a spectrum symmetric
    about zero, $\pm\|P C \mathbf{e}\|$.

    The covariance is analytic, from Campbell's theorem on the reference
    catalog (`ImportanceGaussianLikelihood.catalog_covariance`), with no Poisson
    realizations:
    $\mathrm{Cov}[S_k, S_{k'}] = (\mathcal{R}/T)\,\langle P_k P_{k'} \rangle$,
    on the redshift nodes, importance weights and inclination factor of the
    per-bin variance $V_k$. Its diagonal is $V_k$ up to the interpolation order
    of the square.

    **Table.** Per network, $\kappa^2$, the extreme eigenvalues of $\Delta$,
    $\lambda_{\max}(P C P)$, $\|P C \mathbf{e}\|$ and the Kullback-Leibler
    divergence, in nats, of the common-shape Gaussian
    $\mathcal{N}(0, I + \kappa^2 \mathbf{e}\mathbf{e}^T)$ from the true
    $\mathcal{N}(0, I + C)$,

    $$
    \mathrm{KL} = \tfrac12\left[\mathrm{tr}\, C - \kappa^2 - \ln\det(I + C)
    + \ln(1 + \kappa^2)\right]
    \simeq \tfrac14 \big\| \Sigma_{\rm cs}^{-1/2} \Delta\, \Sigma_{\rm cs}^{-1/2} \big\|_F^2,
    $$

    on the scale of the $< 0.02$ nats the two-cumulant (Gaussian) approximation
    costs. The `inflation - 1` columns are the fractional posterior-variance
    increase along the shape derivative $\mathbf{g} = P\, N^{-1/2}\,
    \partial S_h / \partial \Lambda$ that the model misses,
    $\mathbf{g}^T P C P\, \mathbf{g} / \|\mathbf{g}\|^2$. $H_0$ is left out: its
    derivative is almost along $\mathbf{e}$, and the small remainder is
    dominated by round-off.

    **Figure.** *Left*, at the fiducial network: the eigenvalues of the
    analytic $C$, the positive and negative eigenvalues of $\Delta$ (in
    magnitude), and those of the injections' sample covariance, against the
    detector noise at 1. The injections are independent of the reference
    catalog, so their agreement checks the analytic covariance. An isotropic
    spectrum sampled by $M$ draws would have a noise floor of about
    $\mathrm{tr}\, C / M$. This one falls by orders of magnitude per mode, so
    the error on each sample eigenvalue scales with the eigenvalue instead,
    about $\sqrt{2/M}$, and the injections resolve the leading modes far below
    $\mathrm{tr}\, C / M$. $\Delta$ has one negative eigenvalue: the rest are
    round-off and fall below the axis. *Right*, per network: $\kappa^2$,
    $\lambda_{\max}(\Delta)$ and the KL divergence, with the detector noise
    (dashed) and the 0.02 nats of the two-cumulant approximation (dotted).
    """)


@app.cell
def _(FIDUCIALS, fiducial_prediction, fiducial_variance, likelihood, per_network):
    _template = np.asarray(fiducial_prediction)
    catalog_covariance = np.asarray(likelihood.catalog_covariance(FIDUCIALS))
    # Campbell's diagonal squares the interpolated source spectrum; V_k
    # interpolates the squared one, which can only be larger (Jensen).
    _positive = _template > 0.0
    _diagonal_ratio = (
        np.diag(catalog_covariance)[_positive]
        / np.asarray(fiducial_variance)[_positive]
    )
    print(f"diag Cov / V_k: {_diagonal_ratio.min():.5f} to {_diagonal_ratio.max():.5f}")
    assert np.all(_diagonal_ratio <= 1.0 + 1e-10)

    _shape_parameters = ("z_peak", "gamma", "kappa")
    _derivatives = {
        _name: np.asarray(
            jax.jvp(
                lambda value, name=_name: likelihood.spectrum(
                    {**FIDUCIALS, name: value}
                ),
                (jnp.asarray(FIDUCIALS[_name]),),
                (jnp.asarray(1.0),),
            )[1]
        )
        for _name in _shape_parameters
    }
    _label = dict(DETECTOR_NETWORKS)
    _rows = []
    delta_eigenvalues = {}
    covariance_eigenvalues = {}
    for _name, (_scale, _mask) in per_network.items():
        _keep = _mask & _positive
        _covariance = catalog_covariance[np.ix_(_keep, _keep)]
        _metrics, delta_eigenvalues[_name] = common_shape_metrics(
            _covariance,
            _template[_keep],
            _scale[_keep],
            {_key: _value[_keep] for _key, _value in _derivatives.items()},
        )
        _whitened = _covariance / np.outer(_scale[_keep], _scale[_keep])
        covariance_eigenvalues[_name] = np.linalg.eigvalsh(_whitened)[::-1]
        _relative = float(
            likelihood.relative_variance(
                FIDUCIALS, LikelihoodNetwork(jnp.asarray(_scale), jnp.asarray(_mask))
            )
        )
        np.testing.assert_allclose(
            _metrics["kappa^2"], _metrics["rho"] ** 2 * _relative, rtol=1e-8
        )
        _rows.append({"network": _label[_name], "bins": int(_keep.sum()), **_metrics})
    common_shape = pd.DataFrame(_rows)
    common_shape
    return common_shape, covariance_eigenvalues, delta_eigenvalues


@app.cell(hide_code=True)
def _(
    common_shape,
    covariance_eigenvalues,
    delta_eigenvalues,
    eps_samples,
    fiducial_network,
    whitened_eigenvalues,
):
    _modes = 20
    _index = np.arange(1, _modes + 1)
    _delta = delta_eigenvalues[fiducial_network]
    _covariance = covariance_eigenvalues[fiducial_network]
    common_shape_figure, (_ax_spectrum, _ax_networks) = plt.subplots(
        1, 2, figsize=(9.0, 3.8)
    )
    _ax_spectrum.semilogy(_index, _covariance[:_modes], "o", label=r"$C$")
    _ax_spectrum.semilogy(_index, _delta[:_modes], "^", label=r"$\Delta > 0$")
    _ax_spectrum.semilogy(_index, -_delta[::-1][:_modes], "v", label=r"$-\Delta > 0$")
    _sampled = min(_modes, eps_samples.shape[0] - 1)
    _ax_spectrum.semilogy(
        _index[:_sampled],
        whitened_eigenvalues[:_sampled],
        "+",
        color="black",
        label="injections",
    )
    _ax_spectrum.axhline(1.0, color="0.5", ls="--", lw=1, label="detector noise")
    # Below this, eigenvalues of Delta are round-off of a matrix of order kappa^2.
    _floor = 1e-12 * _covariance[0]
    _ax_spectrum.set(
        xlabel="Mode", ylabel=r"$|\lambda_j|$", ylim=(_floor, 10.0 * _covariance[0])
    )
    _ax_spectrum.set_xticks([1, *range(5, _modes + 1, 5)])
    _ax_spectrum.legend(fontsize="small", ncols=2, loc="upper right")

    _position = np.arange(len(common_shape))
    for _column, _marker, _label in (
        ("kappa^2", "o", r"$\kappa^2$"),
        ("lambda_max(Delta)", "^", r"$\lambda_{\max}(\Delta)$"),
        ("KL [nats]", "s", "KL [nats]"),
    ):
        _ax_networks.semilogy(_position, common_shape[_column], _marker, label=_label)
    _ax_networks.axhline(1.0, color="0.5", ls="--", lw=1)
    _ax_networks.axhline(0.02, color="0.5", ls=":", lw=1)
    _ax_networks.set_xticks(_position, common_shape["network"], rotation=30, ha="right")
    _ax_networks.set_xlim(-0.5, len(common_shape) - 0.5)
    _ax_networks.legend(
        fontsize="small", ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.0)
    )
    common_shape_figure.tight_layout()
    common_shape_figure
    return (common_shape_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Coverage

    **Table.** For each network and likelihood variant, it summarizes the
    $M \times R$ posteriors of $H_0$ against the true value $H_0^t$. Each posterior
    is $p(H_0 | S_h^{\rm data}) \propto \pi(H_0)\,\mathcal{L}(S_h^{\rm data}|H_0)$ on the grid,
    with the uniform prior and the likelihood of the variant, and
    $S_h^{\rm data} = S_h^{\rm cat} + n_{\rm det}$ (one injection plus one noise
    draw).

    | Column | Quantity |
    |---|---|
    | `cover 68%`, `cover 95%` | the fraction of posteriors with $\mathrm{PIT} \in \big(\tfrac{1-m}{2}, \tfrac{1+m}{2}\big)$ for mass $m$, which should equal $m$ |
    | `mean z` | the mean of $z$, which should be 0 |
    | `var z` | the variance of $z$, which should be 1 |
    | `... err` | bootstrap standard error of the column to its left |

    The expected values are

    $$
    \mathrm{Var}(z) = 1 + (\rho\,\sigma_\epsilon)^2 \ \text{(detector)},
    \qquad
    \mathrm{Var}(z) = 1 \ \text{(amplitude, fixed)}.
    $$

    The errors resample whole injections, each with all of its $R$ noise draws,
    because those draws share one realization of $n_{\rm shot}$ and are not
    independent.
    """)


@app.cell
def _(
    FIDUCIALS,
    PRIORS,
    fiducial_relative_variance,
    grid_prediction,
    grid_relative_variance,
    h0_grid,
    injections,
    noise_draws,
    noise_seed,
    per_network,
):
    _label = dict(DETECTOR_NETWORKS)
    _log_prior = np.asarray(PRIORS["H0"].log_prob(jnp.asarray(h0_grid)))
    _count, _bins = injections.shape
    statistics: dict[tuple[str, str], dict[str, NDArray[np.float64]]] = {}
    _rows = []
    for _k, (_name, (_scale, _mask)) in enumerate(per_network.items()):
        # (M, R, F) -> (M R, F), injection-major: R noise draws per injection.
        _noise = np.asarray(
            jax.random.normal(
                jax.random.fold_in(jax.random.key(noise_seed), _k),
                (_count, noise_draws, _bins),
            )
        )
        _data = jnp.asarray(
            (injections[:, None, :] + _noise * _scale).reshape(-1, _bins)
        )
        _s2_fixed = jnp.asarray(fiducial_relative_variance[_name])
        _s2_grid = grid_relative_variance[_name]
        _directions = {
            "detector": jnp.zeros_like(grid_prediction),
            "amplitude": amplitude_direction(grid_prediction, _s2_grid),
            "fixed": amplitude_direction(grid_prediction, _s2_fixed),
        }
        for _variant, _direction in _directions.items():
            _log_density = (
                np.asarray(
                    grid_log_likelihoods(
                        _data,
                        grid_prediction,
                        jnp.asarray(_scale),
                        _direction,
                        jnp.asarray(_mask),
                    )
                )
                + _log_prior
            )
            statistics[_name, _variant] = posterior_statistics(
                h0_grid, _log_density, FIDUCIALS["H0"]
            )
            _rows.append(
                {
                    "network": _label[_name],
                    "likelihood": _variant,
                    **coverage_row(statistics[_name, _variant], _count),
                }
            )
    coverage = pd.DataFrame(_rows)
    coverage
    return coverage, statistics


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Figure.** At the fiducial network it plots the distributions of the two
    calibration statistics over the $M \times R$ posteriors, for `detector`
    (solid) and `fixed` (dashed). The `amplitude` variant is in the table but not
    plotted.

    - *Left:* a histogram of $\mathrm{PIT}$ against the uniform density,
      $\mathcal{U}(0, 1)$. Piling at both ends means the posterior is too narrow,
      a central hump means too wide, and a slope means a biased mean.
    - *Right:* a histogram of $z = (\bar H_0 - H_0^t)/\mathrm{sd}(H_0)$ against
      the standard normal. Without the shot-noise term the posterior is too narrow
      by $\sqrt{1 + (\rho\sigma_\epsilon)^2}$, and $z$ is wider than the curve by
      the same factor.

    The histograms hold $M R$ entries but only $M$ independent shot-noise draws,
    so they look smoother than their real precision. The bootstrap errors in the
    table are the honest uncertainty.
    """)


@app.cell
def _(
    fiducial_network,
    statistics: dict[tuple[str, str], dict[str, NDArray[np.float64]]],
):
    coverage_figure, (_ax_pit, _ax_z) = plt.subplots(1, 2, figsize=(9, 3.6))
    _bins = np.linspace(0.0, 1.0, 11)
    _z_grid = np.linspace(-4.0, 4.0, 201)
    for _variant, _style in (("detector", "-"), ("fixed", "--")):
        _stats = statistics[fiducial_network, _variant]
        _ax_pit.hist(
            _stats["pit"],
            bins=_bins,
            density=True,
            histtype="step",
            linestyle=_style,
            label=_variant,
        )
        _ax_z.hist(
            _stats["z"],
            bins=np.linspace(-4.0, 4.0, 17),
            density=True,
            histtype="step",
            linestyle=_style,
            label=_variant,
        )
    _ax_pit.axhline(1.0, color="0.5", lw=1)
    _ax_z.plot(
        _z_grid, np.exp(-0.5 * _z_grid**2) / np.sqrt(2 * np.pi), color="0.5", lw=1
    )
    _ax_pit.set(xlabel=f"PIT of {parameter_label('H0')}", ylabel="Density")
    _ax_z.set(xlabel=r"$z = (\bar H_0 - H_0^{\mathrm{true}}) / \mathrm{sd}$")
    _ax_pit.legend(title="likelihood", fontsize="small")
    coverage_figure.tight_layout()
    coverage_figure
    return (coverage_figure,)


@app.cell
def _(
    FIGURES,
    combined_figure,
    common_shape,
    common_shape_figure,
    correlation_figure,
    covariance_deviation_figure,
    coverage,
    coverage_figure,
    distribution_figure,
    model_against_injections,
    qq_figure,
    variance_figure,
    write_figures,
):
    if write_figures.value:
        save_figures(
            {
                FIGURES / "coverage_H0.pdf": coverage_figure,
                FIGURES / "shot_noise_variance.pdf": variance_figure,
                FIGURES / "shot_noise_correlation.pdf": correlation_figure,
                FIGURES / "shot_noise_distribution.pdf": distribution_figure,
                FIGURES / "shot_noise_qq.pdf": qq_figure,
                FIGURES / "shot_noise_deviation.pdf": covariance_deviation_figure,
                FIGURES / "shot_noise_density_excess.pdf": combined_figure,
                FIGURES / "shot_noise_common_shape.pdf": common_shape_figure,
            }
        )
        FIGURES.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(FIGURES / "coverage_H0.csv", index=False)
        model_against_injections.to_csv(
            FIGURES / "shot_noise_model_vs_injections.csv", index=False
        )
        common_shape.to_csv(FIGURES / "shot_noise_common_shape.csv", index=False)


if __name__ == "__main__":
    app.run()
