import marimo

__generated_with = "0.25.0"
app = marimo.App()

with app.setup(hide_code=True):
    import os
    from collections.abc import Callable, Mapping, Sequence
    from functools import partial
    from pathlib import Path

    from astrogwb.paper.runtime import configure_runtime

    # Backend configuration precedes waveform construction and array creation.
    configure_runtime(num_chains=1)

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numpyro.distributions as dist
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.projections import register_projection
    from numpy.typing import NDArray

    from astrogwb import __version__
    from astrogwb.detector import sample_pair_response
    from astrogwb.inference import (
        ImportanceGaussianLikelihood,
        grid_log_posterior,
    )
    from astrogwb.inference import Network as LikelihoodNetwork
    from astrogwb.paper.cache import default_cache_dir
    from astrogwb.paper.catalogs import validate_matching_frequency_grids
    from astrogwb.paper.config import (
        detector_registry,
        fiducials,
        population_metadata,
        priors,
        waveform_metadata,
    )
    from astrogwb.paper.config.runs import CATALOGS_ROOT, FIGURES_DIR
    from astrogwb.paper.plotting import (
        DETECTOR_NETWORKS,
        TRUTH,
        convert_grids_to_samples,
        parameter_label,
        plot_corner,
        plot_network_marginals,
        save_figures,
        use_paper_style,
    )
    from astrogwb.paper.posteriors import (
        ensure_reference_catalog,
        network_injection_metadata,
        network_pair_data,
        noise_digest,
        pad_pairs,
        settings_digest,
        shot_noise_variant,
    )
    from astrogwb.populations import build_population
    from astrogwb.simulators.polarization_power import (
        CatalogMetadata,
    )
    from astrogwb.simulators.spectra import (
        BackgroundSpectralDensityMetadata,
        network_spectra_ensemble,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Grid posteriors for cosmology and modified propagation

    Posteriors on a **grid**, with no sampler: the log density of the
    Gaussian spectrum likelihood is evaluated at every grid point and cached,
    so the figures can be iterated on without recomputing anything.

    The data are a **zero-noise Poisson injection**: one Poisson catalog drawn
    at the fiducials, with no Gaussian noise added on top, as each network
    records it. Every source reaches a network through its own antenna
    patterns at an isotropic sky position, and the detector pairs are combined
    with the optimal cross-correlation weights
    (`NetworkSpectralDensitySimulator`), so the six networks see one catalog
    but record six spectra. Each is a catalog realization
    $S_h^{\mathrm{cat}}$ that carries physical shot noise about the expected
    spectrum, which each network's likelihood models with its own response. The
    model spectrum is a **rescaled reference-redshift catalog**: intrinsic
    draws at the same fiducials, each generated once at the window's lower
    edge and rescaled to the Gauss-Legendre redshift nodes. The two are drawn
    at different seeds, so the injection is not a subset of the catalog.

    Three problems, each over all six detector networks, each in its own
    section below:

    1. $H_0$ alone (*Inferring cosmological parameters*, H0).
    2. $H_0$ and $\Omega_m$ (*Inferring cosmological parameters*, H0 and Omega_m).
    3. $\Xi_0$ and $n$ (*Inferring modified propagation parameters*), with
       $H_0$ fixed.

    Setting `ASTROGWB_NOTEBOOK_SMOKE=1` swaps in a tiny catalog and coarse
    grids so the whole notebook runs in a minute or two.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    **Likelihood.** In each bin the observed spectrum is Gaussian about the
    predicted one with scale $\sigma_i = S_{\mathrm{eff},i}/\sqrt{2T\Delta f_i}$.
    The network changes $\sigma$, which bins are usable, the data it recorded
    and the response its shot noise carries, but not the expected spectrum, so
    each grid point is predicted once and all six networks' likelihoods are
    evaluated on that one prediction.

    **Shot noise.** The data are one finite catalog, so they also scatter
    about the expected spectrum $S$ with the per-bin variance $V_f$ of a
    Poisson sum (Campbell's theorem), computed from the same reference catalog
    with every factor of the quadrature squared and the inclination factor
    replaced by the network's response correlation
    $\langle r_f r_{f'} \rangle$, from isotropic response draws. Within the
    band almost all of
    it is one flat amplitude mode, $S^{\mathrm{cat}} \approx S(1 + \epsilon)$,
    so it enters as $\epsilon \sim \mathcal{N}(0, s^2)$ marginalized: a
    rank-one covariance $D + s^2 S S^T$, inverted in closed form, with $s$ the
    SNR-weighted mean of $\sqrt{V_f}/S_f$. Along the template this is a
    Gaussian in the amplitude with variance $\rho^{-2} + A^2 s^2$. The
    variance comes from the model at each grid point, never from the data.
    Variants: `detector` (no shot noise), `amplitude` ($s^2$ at each point),
    `fixed` ($s^2$ at the fiducial), and `per_frequency` ($u_f =
    \sqrt{V_f}$, a cross-check where $\Xi_0$ and $n$ tilt the template).

    **Prediction.** Every draw $\theta_k$ is generated once, at the window's
    lower edge $z_{\min}$. A $(2,2)$-mode aligned-spin waveform scales
    exactly with detector-frame mass, so with $s_j = (1+z_j)/(1+z_{\min})$
    the mean reference power $\bar P_{\mathrm{ref}}$ is rescaled to the
    Gauss-Legendre nodes $z_j$ (in $a = 1/(1+z)$, weights $w_j$),
    $S(f;\Lambda) = R(\Lambda) \sum_j w_j\, p(z_j\mid\Lambda)\,
    [d^{\mathrm{ref}} / d_L(z_j\mid\Lambda)]^2\, s_j^4\,
    \bar P_{\mathrm{ref}}(f s_j)$,
    interpolating $\bar P_{\mathrm{ref}}$ in $\ln f$. No swept parameter is
    intrinsic, so the draws are not reweighted: $\Xi_0$ and $n$ enter through
    $d_L$, and $H_0$, $\Omega_m$ through both $p(z)$ and $d_L$. Redshift
    carries no Monte Carlo noise, and the node count costs no waveforms;
    `notebooks/importance_convergence.py` sizes $N$ and $Z$.

    **Posterior.** The model samples every parameter in `[priors]`, so the
    prior is in the log density. Parameters that are not swept are pinned at
    their fiducial, which adds a constant. Plots normalize on the grid.
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
    FIGURES = ROOT_DIR / FIGURES_DIR / "cosmology_grid_posteriors"

    # Different seeds: the injection must not be a subset of the catalog.
    data_seed = 41
    catalog_seed = 42
    # Intrinsic draws on a scrambled Sobol net, one waveform each: at 2^14 the
    # template error is ~5e-5 sigma (importance_convergence.py), where i.i.d.
    # draws would need ~3e5 for 0.1 sigma.
    num_samples = 2**14
    marginal_seed = 43  # cell and jitter draws behind the plots
    num_marginal_samples = 2**18  # grid draws per plot, for the marginals and corners
    redshift_nodes = 32  # Gauss-Legendre nodes in 1/(1 + z); costs no waveforms
    catalog_chunk_size = 4096  # reference waveforms per lax.map batch
    # Injection waveforms per reduced chunk: peak memory is (pairs, F, chunk)
    # for the 12 detector pairs of the six networks.
    chunk_size = 4096
    grid_chunk_size = 8  # grid points per lax.map batch; bounds peak memory

    observation_time = 1.0  # years
    minimum_frequency = 2.0
    maximum_frequency = 2048.0
    fiducial_network = "ET-2L-aligned-CE-Hanford"
    network_names = [name for name, _ in DETECTOR_NETWORKS]

    # Grid knobs: each window is (low, high), each size a point count. Uniform
    # grids: the corner plot reads each density value as proportional to the
    # mass of its cell. Windows should hold the posterior comfortably; the
    # best network's sigma is ~0.2 for H0 and ~0.003 for xi_0, the worst's
    # (ET-2L, at 45 deg from local North) ~1.5 and ~0.02. xi_n is flat at
    # xi_0 = 1, so it takes its prior range.
    h0_window, h0_size = (58.5, 77.0), 926  # H0 alone: 0.02 step
    h0_omega_window, h0_omega_size = (59.0, 76.5), 351  # H0 in the 2D problem
    omega_m_window, omega_m_size = (0.2856, 0.3336), 41  # +/- 4 prior sigma
    xi_0_window, xi_0_size = (0.86, 1.14), 281
    xi_n_window, xi_n_size = (0.3, 3.0), 55  # prior range

    # Likelihood variants evaluated per problem, and the one every figure
    # shows. "modes" keeps the leading eigenmodes of the dense catalog
    # covariance of the stacked per-pair data, the paper's likelihood;
    # "amplitude" keeps only the rank-one amplitude term with s^2 evaluated at
    # each grid point, and "fixed" pins s^2 at the fiducial (the A/B section
    # compares). Every variant's shot noise carries the per-source network
    # response, from `response_draws` isotropic draws, as the injections do.
    likelihood_variant = "modes"
    problem_variants = {
        "H0": ("detector", "amplitude", "fixed", "modes"),
        "H0_Omega_m": ("detector", "amplitude", "fixed", "modes"),
        # No per_frequency: it reads the per-pair response draws at every
        # point, (L, P, F) per network, which do not fit stacked.
        "xi_0_xi_n": ("detector", "amplitude", "fixed", "modes"),
    }
    response_draws, response_seed = 32_768, 0
    modes_tolerance, max_modes = 1e-3, 32  # dropped eigenvalue sum, noise units
    ab_tolerance = 0.05

    SMOKE = os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") == "1"
    write_figures_default = True
    cache_name_prefix = ""
    if SMOKE:
        num_samples = 64
        redshift_nodes = 8
        catalog_chunk_size = 512
        chunk_size = 512
        grid_chunk_size = 4
        num_marginal_samples = 2000
        response_draws, max_modes = 64, 4
        write_figures_default = False
        cache_name_prefix = "smoke_"
        h0_window, h0_size = (40.0, 100.0), 7
        h0_omega_window, h0_omega_size = (55.0, 80.0), 4
        omega_m_window, omega_m_size = (0.30, 0.32), 3
        xi_0_window, xi_0_size = (0.8, 1.4), 3
        xi_n_window, xi_n_size = (1.5, 2.5), 3

    # (low, high, size) per swept parameter, per problem.
    grid_specs = {
        "H0": {"H0": (*h0_window, h0_size)},
        "H0_Omega_m": {
            "H0": (*h0_omega_window, h0_omega_size),
            "Omega_m": (*omega_m_window, omega_m_size),
        },
        "xi_0_xi_n": {
            "xi_0": (*xi_0_window, xi_0_size),
            "xi_n": (*xi_n_window, xi_n_size),
        },
    }

    cache_dir = default_cache_dir() / "posteriors"
    FIDUCIALS = fiducials(root=ROOT_DIR)
    PRIORS = priors(root=ROOT_DIR)
    registry = detector_registry(root=ROOT_DIR)

    # The injection and the reference catalog share the approximant and the
    # cosmological population. The injection is on the observed grid; the
    # reference catalog on geometric bins 1% wide, from the observed f_min to
    # past the detector-frame BNS merger at the window's lower edge.
    waveform = waveform_metadata(root=ROOT_DIR)
    frequencies = np.asarray(waveform.build().frequencies)
    population = population_metadata(root=ROOT_DIR)
    catalog_metadata = CatalogMetadata(
        waveform=waveform_metadata(
            root=ROOT_DIR,
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
    # The one catalog as each network records it.
    network_injections = {
        name: network_injection_metadata(injection_metadata, registry, name)
        for name in network_names
    }

    write_figures = mo.ui.switch(value=write_figures_default, label="Write figures")
    write_figures
    use_paper_style(root=ROOT_DIR)
    # gwpy registers replacement default axes; keep matplotlib's.
    register_projection(Axes)
    return (
        FIDUCIALS,
        FIGURES,
        PRIORS,
        ab_tolerance,
        cache_dir,
        cache_name_prefix,
        catalog_chunk_size,
        catalog_metadata,
        catalog_seed,
        chunk_size,
        data_seed,
        fiducial_network,
        frequencies,
        grid_chunk_size,
        grid_specs,
        likelihood_variant,
        marginal_seed,
        max_modes,
        maximum_frequency,
        minimum_frequency,
        modes_tolerance,
        network_injections,
        network_names,
        num_marginal_samples,
        observation_time,
        population,
        problem_variants,
        redshift_nodes,
        registry,
        response_draws,
        response_seed,
        write_figures,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Toolbox
    """)
    return


@app.function(hide_code=True)
def cached_log_densities(
    compute: Callable[[], tuple[dict[str, NDArray], dict[str, NDArray]]],
    outfile: Path,
) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
    """Return ``(grids, log_densities)``, computed once and cached in ``outfile``.

    The file stores ``grid__<parameter>`` and ``log_density__<network>``. It
    records none of the settings behind it, so the caller puts a
    :func:`settings_digest` in the file name.

    Parameters
    ----------
    compute
        Called on a miss; returns the grids by parameter and the log density
        by network.
    outfile
        ``.npz`` path.

    Returns
    -------
    tuple
        Grids by parameter name, and log densities by network name.
    """
    if not outfile.is_file():
        grids, log_densities = compute()
        outfile.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"grid__{name}": grid for name, grid in grids.items()}
        arrays |= {f"log_density__{name}": ld for name, ld in log_densities.items()}
        np.savez(outfile, **arrays)  # ty: ignore[invalid-argument-type]
        return grids, log_densities
    with np.load(outfile) as cached:
        grids = {
            k.removeprefix("grid__"): cached[k]
            for k in cached.files
            if k.startswith("grid__")
        }
        log_densities = {
            k.removeprefix("log_density__"): cached[k]
            for k in cached.files
            if k.startswith("log_density__")
        }
    return grids, log_densities


@app.function(hide_code=True)
def evaluate_grid(
    likelihood: ImportanceGaussianLikelihood,
    priors: Mapping[str, dist.Distribution],
    chunk_size: int | None,
    axes: Mapping[str, tuple[float, float, int]],
    fixed: Mapping[str, float],
    per_network: Mapping[str, dict[str, object]],
    relative_variance: NDArray[np.float64] | None,
) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
    """Evaluate one problem's log posterior on its grid, for every network.

    Parameters
    ----------
    likelihood
        The problem's likelihood, with its shot-noise mode set. One prediction
        per grid point, shared by all networks, which differ only in
        :class:`~astrogwb.inference.Network`.
    priors
        Prior of every parameter.
    chunk_size
        Grid points per :func:`jax.lax.map` batch.
    axes
        ``(low, high, size)`` of each swept parameter.
    fixed
        Every other parameter, pinned at its fiducial.
    per_network
        :func:`network_pair_data` by network name, with the network's stacked
        per-pair ``observed`` data and held ``response_moments``, and
        ``modes`` and ``mode_blocks`` for the ``"modes"`` mode.
    relative_variance
        :math:`s^2` per network, in ``per_network`` order, for the ``"fixed"``
        mode; ``None`` otherwise.

    Returns
    -------
    tuple
        Grids by parameter, and log densities by network with one axis per
        parameter in ``axes`` order.
    """
    grids = {name: np.linspace(lo, hi, n) for name, (lo, hi, n) in axes.items()}
    jax_grids = {name: jnp.asarray(grid) for name, grid in grids.items()}
    names = list(per_network)
    # Every network scores its own per-pair data through its own response,
    # held as per-node moments (the draws themselves are too large to stack);
    # the "modes" mode also reads each network's basis and node blocks.
    fields = ("observed", "response_moments", "pair_overlap")
    if likelihood.shot_noise == "modes":
        fields += ("modes", "mode_blocks")
    leaves = {
        field: jnp.stack([jnp.asarray(per_network[n][field]) for n in names])
        for field in fields
    }
    networks = LikelihoodNetwork(
        scale=jnp.stack([jnp.asarray(per_network[n]["scale"]) for n in names]),
        mask=jnp.stack([jnp.asarray(per_network[n]["frequency_mask"]) for n in names]),
        relative_variance=(
            None if relative_variance is None else jnp.asarray(relative_variance)
        ),
        **leaves,
    )
    batched = np.asarray(
        grid_log_posterior(
            likelihood,
            priors,
            jax_grids,
            fixed=fixed,
            networks=networks,
            chunk_size=chunk_size,
        )
    )
    log_densities = {network: batched[i] for i, network in enumerate(names)}
    return grids, log_densities


@app.function(hide_code=True)
def grid_posterior(
    run: str,
    axes: Mapping[str, tuple[float, float, int]],
    likelihood: ImportanceGaussianLikelihood,
    relative_variance: NDArray[np.float64] | None,
    *,
    priors: Mapping[str, dist.Distribution],
    chunk_size: int | None,
    fiducials: Mapping[str, float],
    per_network: Mapping[str, dict[str, object]],
    settings: Mapping[str, object],
    cache_dir: Path,
    prefix: str,
) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
    """One problem's grid log densities, cached under a settings digest.

    Parameters
    ----------
    run
        Problem name, also in the cache file name.
    axes
        ``(low, high, size)`` of each swept parameter.
    likelihood
        The problem's :class:`ImportanceGaussianLikelihood`. Its catalog stays
        a traced input of the compiled evaluator, not a constant baked into it.
    relative_variance
        :math:`s^2` per network for the ``"fixed"`` mode, else ``None``.
    priors
        Prior of every parameter.
    chunk_size
        Grid points per batch.
    fiducials
        Every parameter; those not in ``axes`` are pinned at their value.
    per_network
        :func:`network_data` by network name, with each network's
        ``observed`` spectrum and ``response``.
    settings
        Everything else the cached values depend on, for the digest.
    cache_dir
        Directory of the ``.npz`` file.
    prefix
        File name prefix, to keep smoke-run caches apart.

    Returns
    -------
    tuple
        See :func:`cached_log_densities`.
    """
    digest = settings_digest(run=run, axes=axes, **settings)
    fixed = {k: v for k, v in fiducials.items() if k not in axes}
    return cached_log_densities(
        partial(
            evaluate_grid,
            likelihood,
            priors,
            chunk_size,
            axes,
            fixed,
            per_network,
            relative_variance,
        ),
        cache_dir / f"{prefix}{run}_{digest}.npz",
    )


@app.function(hide_code=True)
def problem_posteriors(
    run: str,
    axes: Mapping[str, tuple[float, float, int]],
    variants: Sequence[str],
    likelihood: ImportanceGaussianLikelihood,
    fiducial_relative_variance: NDArray[np.float64],
    settings: Mapping[str, object],
    **kwargs: object,
) -> dict[str, tuple[dict[str, NDArray], dict[str, NDArray]]]:
    """:func:`grid_posterior` of one problem for every likelihood variant.

    ``kwargs`` are :func:`grid_posterior`'s remaining arguments. One catalog
    serves every variant; each variant compiles once.
    """
    posteriors = {}
    for variant in variants:
        variant_likelihood, relative_variance, variant_settings = shot_noise_variant(
            variant, likelihood, fiducial_relative_variance
        )
        # The basis settings key only the "modes" caches, so the others stay valid.
        shared = {
            k: v for k, v in settings.items() if k != "modes" or variant == "modes"
        }
        posteriors[variant] = grid_posterior(
            run,
            axes,
            variant_likelihood,
            relative_variance,
            settings={**shared, **variant_settings},
            **kwargs,  # ty: ignore[invalid-argument-type]
        )
    return posteriors


@app.function(hide_code=True)
def marginal_density(
    grid: NDArray[np.float64], log_density: NDArray[np.float64], axis: int
) -> NDArray[np.float64]:
    """Normalized 1D marginal of a grid log density along ``axis``.

    The other axes are integrated out by the trapezoid rule on a uniform
    grid; ``grid`` is the coordinate of ``axis``, and the other axes are
    assumed to share one cell size per axis, which cancels in the
    normalization.

    Returns
    -------
    numpy.ndarray
        Density over ``grid``, integrating to one.
    """
    density = np.exp(log_density - np.max(log_density))
    other = tuple(i for i in range(density.ndim) if i != axis)
    marginal = density.sum(axis=other) if other else density
    return marginal / np.trapezoid(marginal, grid)


@app.function(hide_code=True)
def median_and_hdi(
    grid: NDArray[np.float64], density: NDArray[np.float64], mass: float = 0.6827
) -> tuple[float, float, float]:
    """Median and highest-density interval of a unimodal 1D grid density.

    Parameters
    ----------
    grid
        Uniform coordinates.
    density
        Density over ``grid``.
    mass
        Enclosed probability of the interval.

    Returns
    -------
    tuple
        ``(median, lower, upper)``; the interval is where the density exceeds
        the level that encloses ``mass``, with its edges interpolated linearly
        between grid points, so it is not quantized to the grid cell.
    """
    cdf = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))]
    )
    cdf /= cdf[-1]
    median = float(np.interp(0.5, cdf, grid))
    density = density / np.trapezoid(density, grid)

    def crossing(level: float, inside: int, outside: int) -> float:
        slope = (grid[outside] - grid[inside]) / (density[outside] - density[inside])
        return float(grid[inside] + (level - density[inside]) * slope)

    def interval(level: float) -> tuple[float, float]:
        above = np.flatnonzero(density >= level)
        first, last = int(above[0]), int(above[-1])
        lower = crossing(level, first, first - 1) if first > 0 else float(grid[0])
        upper = (
            crossing(level, last, last + 1) if last < len(grid) - 1 else float(grid[-1])
        )
        return lower, upper

    # Bisect the density level whose superlevel interval encloses `mass`.
    low, high = 0.0, float(density.max())
    for _ in range(60):
        level = 0.5 * (low + high)
        lower, upper = interval(level)
        enclosed = np.interp(upper, grid, cdf) - np.interp(lower, grid, cdf)
        low, high = (level, high) if enclosed > mass else (low, level)
    lower, upper = interval(0.5 * (low + high))
    return median, lower, upper


@app.function(hide_code=True)
def summary_rows(
    run: str,
    grids: Mapping[str, NDArray[np.float64]],
    log_densities: Mapping[str, NDArray[np.float64]],
    axes: Mapping[str, tuple[float, float, int]],
) -> list[dict[str, object]]:
    """Median and 68% HDI of each swept parameter, for every network.

    Parameters
    ----------
    run
        Problem name, recorded in each row.
    grids
        Grid by parameter.
    log_densities
        Log density by network, with one axis per parameter in ``axes`` order.
    axes
        ``(low, high, size)`` of each swept parameter; only the keys are used.

    Returns
    -------
    list of dict
        One row per (network, parameter), ready for :class:`pandas.DataFrame`.
    """
    label = dict(DETECTOR_NETWORKS)
    rows: list[dict[str, object]] = []
    for network, log_density in log_densities.items():
        for axis, name in enumerate(axes):
            median, lo, hi = median_and_hdi(
                grids[name], marginal_density(grids[name], log_density, axis=axis)
            )
            rows.append(
                {
                    "problem": run,
                    "network": label[network],
                    "parameter": name,
                    "median": median,
                    "lower": lo,
                    "upper": hi,
                    "half width": (hi - lo) / 2,
                }
            )
    return rows


@app.function(hide_code=True)
def marginal_window(
    grid: NDArray[np.float64],
    log_densities: Mapping[str, NDArray[np.float64]],
    axis: int,
    num_sd: float = 5.0,
) -> tuple[float, float]:
    """Plot window of ``num_sd`` standard deviations around the broadest marginal.

    The grid window is far wider than any posterior, so the marginal plots zoom
    to the network with the largest sd, clipped to the grid.

    Parameters
    ----------
    grid
        Coordinate of ``axis``.
    log_densities
        Unnormalized log density by network name.
    axis
        Axis of ``grid`` in each log density.
    num_sd
        Half-width of the window in standard deviations.

    Returns
    -------
    tuple of float
        ``(low, high)`` limits of the window.
    """
    moments = []
    for log_density in log_densities.values():
        density = marginal_density(grid, log_density, axis=axis)
        mean = np.trapezoid(grid * density, grid)
        moments.append(
            (mean, np.sqrt(np.trapezoid((grid - mean) ** 2 * density, grid)))
        )
    mean, sd = max(moments, key=lambda moment: moment[1])
    return max(grid[0], mean - num_sd * sd), min(grid[-1], mean + num_sd * sd)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reference catalog and injection

    The reference catalog generates every intrinsic draw once, at the
    window's lower edge, so its power is `(F_ref, N)`. It is generated in
    `lax.map` batches on a miss and read from `outputs/catalogs/` afterwards.
    The injection is one Poisson catalog recorded by every network, drawn in
    one waveform pass shared by the networks (`network_spectra_ensemble`) and
    cached beside the posteriors, one file per network.

    The rescaled spectrum is built in the same cell as the catalog, once.
    The population is built a single time because it hashes by identity. Each
    problem evaluates its grid with `grid_log_posterior`, which predicts every
    grid point once for all six networks. The likelihood is a pytree passed to
    it per call, so the catalog is one traced input shared by all three
    problems rather than a constant compiled into each. The NumPy catalog is dropped there, so only the device copy stays
    resident.
    """)
    return


@app.cell
def _(
    catalog_chunk_size,
    catalog_metadata,
    catalog_seed,
    frequencies,
    observation_time,
    population,
    redshift_nodes,
):
    # The NumPy catalog is local to this cell, so it is freed once the device
    # copy made by the likelihood exists: the power and its square.
    catalog_data = ensure_reference_catalog(
        catalog_metadata, catalog_seed, CATALOGS_ROOT, chunk_size=catalog_chunk_size
    )
    power = catalog_data["polarization_power"]
    print(
        f"reference catalog: {catalog_metadata.num_samples:,} draws, "
        f"{power.shape[0]} reference bins, {power.nbytes / 1e9:.2f} GB of power; "
        f"{redshift_nodes} redshift nodes"
    )
    target = build_population(population.model_name, **population.model_kwargs)
    # No swept parameter is intrinsic, so no density factor enters a weight.
    # Built with the amplitude mode, so the squared power is held for every
    # variant `shot_noise_variant` switches to.
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
    return (likelihood,)


@app.cell
def _(
    cache_dir,
    cache_name_prefix,
    chunk_size,
    data_seed,
    frequencies,
    network_injections,
):
    # Draw 0 of `data_seed`: the key of the sky-averaged injection, so the
    # networks record the same sources it summed.
    _drawn = network_spectra_ensemble(
        list(network_injections.values()),
        data_seed,
        1,
        cache_dir / f"{cache_name_prefix}network_injections",
        chunk_size=chunk_size,
    )
    observed = {}
    for _name, _data in zip(network_injections, _drawn, strict=True):
        validate_matching_frequency_grids(_data["frequencies"], frequencies)
        observed[_name] = pad_pairs(_data["pair_spectral_density"][0])
    print(
        f"injection: {int(_drawn[0]['n_events'][0]):,} Poisson events, "
        f"recorded by {len(observed)} networks"
    )
    return (observed,)


@app.cell
def _(
    FIDUCIALS,
    catalog_metadata,
    catalog_seed,
    data_seed,
    frequencies,
    likelihood,
    max_modes,
    maximum_frequency,
    minimum_frequency,
    modes_tolerance,
    network_injections,
    network_names,
    observation_time,
    observed,
    problem_variants,
    redshift_nodes,
    registry,
    response_draws,
    response_seed,
):
    per_network = {
        name: network_pair_data(
            registry,
            name,
            frequencies,
            observation_time=observation_time,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
        for name in network_names
    }
    # Every network's per-pair data and the per-node moments under its per-pair
    # response, held so s^2 is not projected once per draw at every point and
    # the (L, P, F) draws are dropped; for the "modes" mode, also the basis
    # and node blocks of the catalog covariance at the fiducials, padded to one
    # k so networks stack. No density site varies, so the held moments are
    # exact.
    _uses_modes = any("modes" in v for v in problem_variants.values())
    for _name, _data in per_network.items():
        _detectors, _sensitivities = registry.build_network(_name)
        _network = LikelihoodNetwork(
            jnp.asarray(_data["scale"]),
            jnp.asarray(_data["frequency_mask"]),
            response=jnp.asarray(
                pad_pairs(
                    sample_pair_response(
                        jax.random.key(response_seed),
                        response_draws,
                        frequencies,
                        _detectors,
                    )
                )
            ),
            pair_overlap=jnp.asarray(_data["pair_overlap"]),
        )
        _data.update(
            observed=observed[_name],
            response_moments=likelihood.response_moments(FIDUCIALS, _network),
        )
        if not _uses_modes:
            continue
        _network = likelihood.with_shot_noise_modes(
            FIDUCIALS,
            _network,
            tolerance=modes_tolerance,
            max_modes=max_modes,
            num_modes=max_modes,
        )
        _data.update(modes=_network.modes, mode_blocks=_network.mode_blocks)
    del _network
    # Everything a cached log density depends on besides its own axes.
    cache_settings = {
        "version": __version__,
        "networks": network_names,
        "noise": noise_digest(per_network),
        "seeds": (data_seed, catalog_seed),
        "band": (minimum_frequency, maximum_frequency),
        "observation_time": observation_time,
        "density_sites": (),
        "injection": {n: m.key() for n, m in network_injections.items()},
        "reference_catalog": catalog_metadata.key(),
        "redshift_nodes": redshift_nodes,
        # The amplitude mode's s^2: the catalog projected on each template.
        "relative_variance": "projected",
        "response": (response_draws, response_seed),
        "data": "stacked_pairs",
        "modes": (modes_tolerance, max_modes),
    }
    labelled_networks = [(n, l) for n, l in DETECTOR_NETWORKS if n in network_names]
    return cache_settings, labelled_networks, per_network


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Shot noise at the fiducials

    The relative amplitude scatter $s$ of one catalog realization, with the
    network's response, and the optimal SNR $\rho$ of the expected spectrum,
    per network. $\rho s$ is the shot noise in units of the detector's
    amplitude error; it does not depend on the observation time. `notebooks/spectrum_snrs.py` measures it
    empirically from Poisson ensembles, about 0.48 for the fiducial network.
    """)
    return


@app.cell
def _(FIDUCIALS, likelihood, per_network):
    _spectrum, _, _ = likelihood.predict(FIDUCIALS)
    _label = dict(DETECTOR_NETWORKS)
    _rows = []
    for _name, _data in per_network.items():
        _keep = np.asarray(_data["frequency_mask"])
        _sigma = np.asarray(_data["scale"])
        _s2 = float(
            likelihood.relative_variance(
                FIDUCIALS,
                LikelihoodNetwork(
                    _data["scale"],
                    _data["frequency_mask"],
                    response_moments=jnp.asarray(_data["response_moments"]),
                    pair_overlap=jnp.asarray(_data["pair_overlap"]),
                ),
            )
        )
        _expected = np.asarray(_data["pair_overlap"]) * np.asarray(_spectrum)
        _snr = float(np.sqrt(np.sum(np.where(_keep, (_expected / _sigma) ** 2, 0))))
        _rows.append(
            {
                "network": _label[_name],
                "rho": _snr,
                "s": np.sqrt(_s2),
                "rho s": _snr * np.sqrt(_s2),
                "s2": _s2,
            }
        )
    shot_noise_table = pd.DataFrame(_rows)
    fiducial_relative_variance = shot_noise_table["s2"].to_numpy()
    shot_noise_table.drop(columns="s2")
    return fiducial_relative_variance, shot_noise_table


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Inferring cosmological parameters

    The two cosmological problems, with $H_0$ alone first and then $H_0$
    together with $\Omega_m$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## H0

    The posterior on $H_0$ alone, every other parameter pinned at its
    fiducial. $H_0$ alone has one parameter, so its corner plot is its
    marginal below.
    """)
    return


@app.cell
def _(
    FIDUCIALS,
    PRIORS,
    cache_dir,
    cache_name_prefix,
    cache_settings,
    fiducial_relative_variance,
    grid_chunk_size,
    grid_specs,
    likelihood_variant,
    per_network,
    problem_variants,
    likelihood,
):
    h0_posteriors = problem_posteriors(
        "H0",
        grid_specs["H0"],
        problem_variants["H0"],
        likelihood,
        fiducial_relative_variance,
        cache_settings,
        priors=PRIORS,
        chunk_size=grid_chunk_size,
        fiducials=FIDUCIALS,
        per_network=per_network,
        cache_dir=cache_dir,
        prefix=cache_name_prefix,
    )
    h0_posterior = h0_posteriors[likelihood_variant]
    return h0_posterior, h0_posteriors


@app.cell
def _(FIDUCIALS, h0_posterior, labelled_networks, marginal_seed, num_marginal_samples):
    _grids, _log_densities = h0_posterior
    _rng = np.random.default_rng(marginal_seed)
    h0_marginal = plot_network_marginals(
        "H0",
        {
            name: convert_grids_to_samples(_grids, ld, num_marginal_samples, rng=_rng)
            for name, ld in _log_densities.items()
        },
        labelled_networks,
        FIDUCIALS["H0"],
        xlim=marginal_window(_grids["H0"], _log_densities, axis=0),
    )
    mo.as_html(h0_marginal)
    return (h0_marginal,)


@app.cell
def _(grid_specs, h0_posterior):
    _grids, _log_densities = h0_posterior
    h0_summary_rows = summary_rows("H0", _grids, _log_densities, grid_specs["H0"])
    return (h0_summary_rows,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## H0 and Omega_m

    $H_0$ and $\Omega_m$ together, $\Xi_0$ and $n$ pinned at their fiducials.
    The $H_0$ marginal integrates over $\Omega_m$.
    """)
    return


@app.cell
def _(
    FIDUCIALS,
    PRIORS,
    cache_dir,
    cache_name_prefix,
    cache_settings,
    fiducial_relative_variance,
    grid_chunk_size,
    grid_specs,
    likelihood_variant,
    per_network,
    problem_variants,
    likelihood,
):
    h0_omega_m_posteriors = problem_posteriors(
        "H0_Omega_m",
        grid_specs["H0_Omega_m"],
        problem_variants["H0_Omega_m"],
        likelihood,
        fiducial_relative_variance,
        cache_settings,
        priors=PRIORS,
        chunk_size=grid_chunk_size,
        fiducials=FIDUCIALS,
        per_network=per_network,
        cache_dir=cache_dir,
        prefix=cache_name_prefix,
    )
    h0_omega_m_posterior = h0_omega_m_posteriors[likelihood_variant]
    return h0_omega_m_posterior, h0_omega_m_posteriors


@app.cell
def _(
    FIDUCIALS,
    h0_omega_m_posterior,
    labelled_networks,
    marginal_seed,
    num_marginal_samples,
):
    _grids, _log_densities = h0_omega_m_posterior
    _rng = np.random.default_rng(marginal_seed)
    h0_omega_m_marginal = plot_network_marginals(
        "H0",
        {
            name: convert_grids_to_samples(_grids, ld, num_marginal_samples, rng=_rng)
            for name, ld in _log_densities.items()
        },
        labelled_networks,
        FIDUCIALS["H0"],
        xlim=marginal_window(_grids["H0"], _log_densities, axis=0),
    )
    mo.as_html(h0_omega_m_marginal)
    return (h0_omega_m_marginal,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Corner plot at the fiducial network.
    """)
    return


@app.cell
def _(
    FIDUCIALS,
    fiducial_network,
    grid_specs,
    h0_omega_m_posterior,
    marginal_seed,
    num_marginal_samples,
):
    _grids, _log_densities = h0_omega_m_posterior
    h0_omega_m_corner = plot_corner(
        convert_grids_to_samples(
            _grids,
            _log_densities[fiducial_network],
            num_marginal_samples,
            rng=np.random.default_rng(marginal_seed),
        ),
        {name: FIDUCIALS[name] for name in grid_specs["H0_Omega_m"]},
    )
    mo.as_html(h0_omega_m_corner)
    return (h0_omega_m_corner,)


@app.cell
def _(grid_specs, h0_omega_m_posterior):
    _grids, _log_densities = h0_omega_m_posterior
    h0_omega_m_summary_rows = summary_rows(
        "H0_Omega_m", _grids, _log_densities, grid_specs["H0_Omega_m"]
    )
    return (h0_omega_m_summary_rows,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Inferring modified propagation parameters

    $\Xi_0$ and $n$, with $H_0$ fixed at its fiducial. The marginal is in
    $\Xi_0$, which integrates over $n$.
    """)
    return


@app.cell
def _(
    FIDUCIALS,
    PRIORS,
    cache_dir,
    cache_name_prefix,
    cache_settings,
    fiducial_relative_variance,
    grid_chunk_size,
    grid_specs,
    likelihood_variant,
    per_network,
    problem_variants,
    likelihood,
):
    xi_posteriors = problem_posteriors(
        "xi_0_xi_n",
        grid_specs["xi_0_xi_n"],
        problem_variants["xi_0_xi_n"],
        likelihood,
        fiducial_relative_variance,
        cache_settings,
        priors=PRIORS,
        chunk_size=grid_chunk_size,
        fiducials=FIDUCIALS,
        per_network=per_network,
        cache_dir=cache_dir,
        prefix=cache_name_prefix,
    )
    xi_posterior = xi_posteriors[likelihood_variant]
    return xi_posterior, xi_posteriors


@app.cell
def _(FIDUCIALS, labelled_networks, marginal_seed, num_marginal_samples, xi_posterior):
    _grids, _log_densities = xi_posterior
    _rng = np.random.default_rng(marginal_seed)
    xi_marginal = plot_network_marginals(
        "xi_0",
        {
            name: convert_grids_to_samples(_grids, ld, num_marginal_samples, rng=_rng)
            for name, ld in _log_densities.items()
        },
        labelled_networks,
        FIDUCIALS["xi_0"],
        xlim=marginal_window(_grids["xi_0"], _log_densities, axis=0),
    )
    mo.as_html(xi_marginal)
    return (xi_marginal,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Corner plot at the fiducial network. At $\Xi_0 = 1$ the $n$ axis is
    degenerate: the valley along it is expected.
    """)
    return


@app.cell
def _(
    FIDUCIALS,
    fiducial_network,
    grid_specs,
    marginal_seed,
    num_marginal_samples,
    xi_posterior,
):
    _grids, _log_densities = xi_posterior
    xi_corner = plot_corner(
        convert_grids_to_samples(
            _grids,
            _log_densities[fiducial_network],
            num_marginal_samples,
            rng=np.random.default_rng(marginal_seed),
        ),
        {name: FIDUCIALS[name] for name in grid_specs["xi_0_xi_n"]},
    )
    mo.as_html(xi_corner)
    return (xi_corner,)


@app.cell
def _(grid_specs, xi_posterior):
    _grids, _log_densities = xi_posterior
    xi_summary_rows = summary_rows(
        "xi_0_xi_n", _grids, _log_densities, grid_specs["xi_0_xi_n"]
    )
    return (xi_summary_rows,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Shot-noise likelihood variants

    Every problem was evaluated with each of its likelihood variants. The
    comparison uses the marginal posterior mean and standard deviation, which
    are not quantized to the grid cell as the HDI is: on the $H_0$ grid one
    cell is already about $0.17\sigma$.

    - **Effect of shot noise**: the shift of the mean and the ratio of the
      standard deviations of each variant against `detector`.
    - **A/B, fixed vs per-point $s^2$**: a covariance that depends on the
      parameters adds its $\ln\det$ to the likelihood, which can carry
      spurious information. If pinning $s^2$ at the fiducial moves no mean or
      standard deviation by more than `ab_tolerance` detector $\sigma$, the
      fixed value is the safer choice.
    - **Leakage**: for $(\Xi_0, n)$, which tilt the template, the
      per-frequency direction against the amplitude one.
    """)
    return


@app.function(hide_code=True)
def moment_rows(
    run: str,
    likelihood: str,
    grids: Mapping[str, NDArray[np.float64]],
    log_densities: Mapping[str, NDArray[np.float64]],
    axes: Mapping[str, tuple[float, float, int]],
) -> list[dict[str, object]]:
    """Marginal posterior mean and standard deviation, per network and parameter.

    Trapezoid moments of :func:`marginal_density`; unlike the HDI they resolve
    changes far below one grid cell.
    """
    label = dict(DETECTOR_NETWORKS)
    rows: list[dict[str, object]] = []
    for network, log_density in log_densities.items():
        for axis, name in enumerate(axes):
            grid = grids[name]
            density = marginal_density(grid, log_density, axis=axis)
            mean = float(np.trapezoid(grid * density, grid))
            variance = float(np.trapezoid((grid - mean) ** 2 * density, grid))
            rows.append(
                {
                    "problem": run,
                    "likelihood": likelihood,
                    "network": label[network],
                    "parameter": name,
                    "mean": mean,
                    "sd": np.sqrt(variance),
                    "cell": float(grid[1] - grid[0]),
                }
            )
    return rows


@app.function(hide_code=True)
def compare_moments(
    moments: pd.DataFrame, variant: str, reference: str, scale: str = "detector"
) -> pd.DataFrame:
    """``variant`` against ``reference``, in units of the ``scale`` sd.

    Rows are the (problem, network, parameter) both were evaluated for. A
    posterior narrower than one grid cell is not resolved, so its rows are NaN
    rather than a ratio of discretization noise.
    """
    keys = ["problem", "network", "parameter"]
    by = {
        name: moments[moments["likelihood"] == name].set_index(keys)
        for name in (variant, reference, scale)
    }
    index = by[variant].index.intersection(by[reference].index)
    resolved = by[scale].loc[index, "sd"] >= by[scale].loc[index, "cell"]
    sigma = by[scale].loc[index, "sd"].where(resolved)
    return pd.DataFrame(
        {
            "mean shift / sigma": (
                by[variant].loc[index, "mean"] - by[reference].loc[index, "mean"]
            )
            / sigma,
            "sd change / sigma": (
                by[variant].loc[index, "sd"] - by[reference].loc[index, "sd"]
            )
            / sigma,
            "sd ratio": (
                by[variant].loc[index, "sd"] / by[reference].loc[index, "sd"]
            ).where(resolved),
        }
    ).reset_index()


@app.cell
def _(grid_specs, h0_omega_m_posteriors, h0_posteriors, xi_posteriors):
    shot_noise_moments = pd.DataFrame(
        [
            row
            for _run, _posteriors in (
                ("H0", h0_posteriors),
                ("H0_Omega_m", h0_omega_m_posteriors),
                ("xi_0_xi_n", xi_posteriors),
            )
            for _variant, (_grids, _log_densities) in _posteriors.items()
            for row in moment_rows(
                _run, _variant, _grids, _log_densities, grid_specs[_run]
            )
        ]
    )
    return (shot_noise_moments,)


@app.cell
def _(shot_noise_moments):
    shot_noise_effect = pd.concat(
        [
            compare_moments(shot_noise_moments, _variant, "detector").assign(
                likelihood=_variant
            )
            for _variant in ("amplitude", "fixed", "per_frequency")
            if (shot_noise_moments["likelihood"] == _variant).any()
        ]
    )
    shot_noise_effect
    return


@app.cell
def _(ab_tolerance, shot_noise_moments):
    ab_comparison = compare_moments(shot_noise_moments, "fixed", "amplitude")
    _worst = float(
        ab_comparison[["mean shift / sigma", "sd change / sigma"]].abs().max().max()
    )
    if np.isnan(_worst):
        _verdict = "no posterior is resolved by its grid: no A/B verdict"
    elif _worst < ab_tolerance:
        _verdict = (
            f"fixed s^2 is within {ab_tolerance} sigma of per-point s^2 "
            f"everywhere (worst {_worst:.3f}): adopt likelihood_variant = 'fixed'"
        )
    else:
        _verdict = (
            f"fixed s^2 moves a moment by {_worst:.3f} sigma > {ab_tolerance}: "
            "keep the per-point 'amplitude' variant"
        )
    print(_verdict)
    ab_comparison
    return


@app.cell
def _(shot_noise_moments):
    leakage = compare_moments(shot_noise_moments, "per_frequency", "amplitude")
    leakage
    return


@app.cell
def _(FIDUCIALS, fiducial_network, h0_posteriors):
    shot_noise_variants_figure, _ax = plt.subplots()
    # Dashed on top: "fixed" lies on "amplitude" to well below a line width.
    _styles = {"detector": "-", "amplitude": "-", "fixed": "--"}
    for _variant, (_grids, _log_densities) in h0_posteriors.items():
        _ax.plot(
            _grids["H0"],
            marginal_density(_grids["H0"], _log_densities[fiducial_network], axis=0),
            linestyle=_styles.get(_variant, ":"),
            label=_variant.replace("_", " "),
        )
    # Zoom to +/- 5 detector-only sd about its mean: the grid window is wider.
    _grid = h0_posteriors["detector"][0]["H0"]
    _density = marginal_density(
        _grid, h0_posteriors["detector"][1][fiducial_network], axis=0
    )
    _mean = np.trapezoid(_grid * _density, _grid)
    _sd = np.sqrt(np.trapezoid((_grid - _mean) ** 2 * _density, _grid))
    _ax.set_xlim(_mean - 5 * _sd, _mean + 5 * _sd)
    _ax.axvline(FIDUCIALS["H0"], **TRUTH)  # ty: ignore[invalid-argument-type]
    _ax.set(xlabel=parameter_label("H0"), ylabel="Posterior density")
    _ax.legend(title="likelihood")
    shot_noise_variants_figure.tight_layout()
    mo.as_html(shot_noise_variants_figure)
    return (shot_noise_variants_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Summary and output

    Median and 68% HDI of every swept parameter, by problem and network.
    Figures are written when the switch in *Configuration* is on.
    """)
    return


@app.cell
def _(h0_omega_m_summary_rows, h0_summary_rows, xi_summary_rows):
    summary = pd.DataFrame(
        [*h0_summary_rows, *h0_omega_m_summary_rows, *xi_summary_rows]
    )
    summary
    return (summary,)


@app.cell
def _(
    FIGURES,
    h0_marginal,
    h0_omega_m_corner,
    h0_omega_m_marginal,
    shot_noise_table,
    shot_noise_variants_figure,
    summary,
    write_figures,
    xi_corner,
    xi_marginal,
):
    if write_figures.value:
        FIGURES.mkdir(parents=True, exist_ok=True)
        # The tables' inputs: intervals, and rho and s per network.
        summary.to_csv(FIGURES / "summary.csv", index=False)
        shot_noise_table.to_csv(FIGURES / "shot_noise.csv", index=False)
        save_figures(
            {
                FIGURES / "marginals_H0.pdf": h0_marginal,
                FIGURES / "shot_noise_variants_H0.pdf": shot_noise_variants_figure,
                FIGURES / "marginals_H0_Omega_m.pdf": h0_omega_m_marginal,
                FIGURES / "corner_H0_Omega_m.pdf": h0_omega_m_corner,
                FIGURES / "marginals_xi_0_xi_n.pdf": xi_marginal,
                FIGURES / "corner_xi_0_xi_n.pdf": xi_corner,
            }
        )
    return


if __name__ == "__main__":
    app.run()
