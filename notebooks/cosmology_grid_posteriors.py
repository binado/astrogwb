import marimo

__generated_with = "0.25.0"
app = marimo.App()

with app.setup(hide_code=True):
    import hashlib
    import json
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
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.projections import register_projection
    from numpy.typing import NDArray

    from astrogwb import __version__
    from astrogwb.detector import effective_psd, gaussian_bin_scale
    from astrogwb.frequency import frequency_mask
    from astrogwb.gwb.importance import (
        build_rescaled_shot_noise,
        build_rescaled_spectrum,
        reference_catalog,
        reference_catalog_stem,
    )
    from astrogwb.inference import (
        GaussianGWBBatchedLikelihood,
        SpectralVarianceFn,
        amplitude_shot_noise_variance,
    )
    from astrogwb.paper.cache import default_cache_dir
    from astrogwb.paper.catalogs import validate_matching_frequency_grids
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
        CORNER_LEVELS,
        DETECTOR_COMPARISON_LEGEND,
        DETECTOR_NETWORKS,
        TRUTH,
        Network,
        detector_network_styles,
        get_corner_kwargs,
        parameter_label,
        plot_corner_for_posterior_grid,
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
        BackgroundSpectralDensitySimulator,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Grid posteriors for cosmology and modified propagation

    Posteriors on a **grid**, with no sampler: the log density of the
    Gaussian spectrum likelihood is evaluated at every grid point and cached,
    so the figures can be iterated on without recomputing anything.

    The data are a **zero-noise Poisson injection**: one Poisson-count
    spectrum drawn at the fiducials, with no Gaussian noise added on top. It is
    a catalog realization $S_h^{\mathrm{cat}}$, so it carries physical shot
    noise about the expected spectrum, which the likelihood models. The
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
    The network only changes $\sigma$ (and which bins are usable), so each grid
    point is predicted once and all six networks' likelihoods are evaluated on
    that one prediction.

    **Shot noise.** The data are one finite catalog, so they also scatter
    about the expected spectrum $S$ with the per-bin variance $V_f$ of a
    Poisson sum (Campbell's theorem), computed from the same reference catalog
    with every factor of the quadrature squared. Within the band almost all of
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
    redshift_nodes = 32  # Gauss-Legendre nodes in 1/(1 + z); costs no waveforms
    catalog_chunk_size = 4096  # reference waveforms per lax.map batch
    chunk_size = 16_384  # injection waveforms per reduced chunk
    grid_chunk_size = 8  # grid points per lax.map batch; bounds peak memory

    observation_time = 1.0  # years
    minimum_frequency = 2.0
    maximum_frequency = 2048.0
    fiducial_network = "ET-2L-aligned-CE-Hanford"
    network_names = [name for name, _ in DETECTOR_NETWORKS]

    # Grid knobs: each window is (low, high), each size a point count. Uniform
    # grids: the corner plot reads each density value as proportional to the
    # mass of its cell. Windows should hold the posterior comfortably; the
    # best network's sigma is ~0.3 for H0 and ~0.004 for xi_0, the worst's ~1
    # and ~0.013. xi_n is flat at xi_0 = 1, so it takes its prior range.
    h0_window, h0_size = (60.0, 76.0), 321  # H0 alone: 0.05 step
    h0_omega_window, h0_omega_size = (65.0, 70.5), 111  # H0 in the 2D problem
    omega_m_window, omega_m_size = (0.2856, 0.3336), 41  # +/- 4 prior sigma
    xi_0_window, xi_0_size = (0.94, 1.06), 121
    xi_n_window, xi_n_size = (0.3, 3.0), 55  # prior range

    # Likelihood variants evaluated per problem, and the one every figure
    # shows. "fixed" pins s^2 at the fiducial; the A/B section adopts it if it
    # moves no posterior mean or sd by more than ab_tolerance detector sigma,
    # which it does not (worst 0.003 sigma).
    likelihood_variant = "fixed"
    problem_variants = {
        "H0": ("detector", "amplitude", "fixed"),
        "H0_Omega_m": ("detector", "amplitude", "fixed"),
        "xi_0_xi_n": ("detector", "amplitude", "fixed", "per_frequency"),
    }
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
        injection_metadata,
        likelihood_variant,
        maximum_frequency,
        minimum_frequency,
        network_names,
        observation_time,
        population,
        problem_variants,
        redshift_nodes,
        registry,
        write_figures,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Toolbox
    """)
    return


@app.function(hide_code=True)
def settings_digest(**settings: object) -> str:
    """Short hash of everything a cached log density depends on.

    Parameters
    ----------
    **settings
        JSON-serializable (or ``str``-able) values; key order is irrelevant.

    Returns
    -------
    str
        Twelve hex characters, for use in a cache file name.
    """
    text = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


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
def ensure_reference_catalog(
    metadata: CatalogMetadata,
    seed: int,
    directory: Path,
    *,
    chunk_size: int | None = None,
) -> PolarizationPowerData:
    """The reference catalog of ``metadata`` at ``seed``, drawn on a miss.

    The file is ``<directory>/reference_catalog-<key>-<seed>.h5``. A hit is
    checked against the request it records; a miss is drawn at ``batch_keys(seed, 1)[0]`` and written.

    Parameters
    ----------
    metadata
        The catalog's record.
    seed
        The draw's seed.
    directory
        Cache directory.
    chunk_size
        Waveforms per ``lax.map`` batch on a miss; cost only.

    Returns
    -------
    PolarizationPowerData
        Power ``(F_ref, N)``, every draw at the window's lower edge.

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
) -> dict[str, object]:
    """The per-bin Gaussian scale and usable-bin mask of one network.

    Bins no detector pair measures have infinite effective PSD; they are
    dropped from the mask and given a finite scale of one, as
    ``InferenceInputs.model_kwargs`` does.

    Returns
    -------
    dict
        ``scale`` and ``frequency_mask``, shape ``(F,)``, as model keyword
        arguments.
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
    return {"scale": jnp.where(mask, scale, 1.0), "frequency_mask": mask}


@app.function(hide_code=True)
def evaluate_grid(
    log_density_fn: Callable[..., jax.Array],
    axes: Mapping[str, tuple[float, float, int]],
    fixed: Mapping[str, float],
    observed: NDArray[np.float64],
    per_network: Mapping[str, dict[str, object]],
) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
    """Evaluate one problem's log density on its grid, for every network.

    Parameters
    ----------
    log_density_fn
        The problem's evaluator: a :class:`GaussianGWBBatchedLikelihood` with
        its ``spectral_density_fn`` bound. One prediction per grid point,
        shared by all networks, which differ only in ``scale`` and
        ``frequency_mask``.
    axes
        ``(low, high, size)`` of each swept parameter.
    fixed
        Every other parameter, pinned at its fiducial.
    observed
        Observed spectrum ``(F,)``.
    per_network
        :func:`network_data` by network name.

    Returns
    -------
    tuple
        Grids by parameter, and log densities by network with one axis per
        parameter in ``axes`` order.
    """
    grids = {name: np.linspace(lo, hi, n) for name, (lo, hi, n) in axes.items()}
    jax_grids = {name: jnp.asarray(grid) for name, grid in grids.items()}
    names = list(per_network)
    batched = np.asarray(
        log_density_fn(
            jax_grids,
            fixed=fixed,
            observed_spectral_density=jnp.asarray(observed),
            scale=jnp.stack([jnp.asarray(per_network[n]["scale"]) for n in names]),
            frequency_mask=jnp.stack(
                [jnp.asarray(per_network[n]["frequency_mask"]) for n in names]
            ),
        )
    )
    log_densities = {network: batched[i] for i, network in enumerate(names)}
    return grids, log_densities


@app.function(hide_code=True)
def grid_posterior(
    run: str,
    axes: Mapping[str, tuple[float, float, int]],
    evaluator: Callable[..., jax.Array],
    *,
    fiducials: Mapping[str, float],
    observed: NDArray[np.float64],
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
    evaluator
        The problem's :class:`GaussianGWBBatchedLikelihood`, its
        ``spectral_density_fn`` bound with :func:`functools.partial`: bound
        per call, so the catalog stays a traced input of the compiled
        evaluator rather than a constant baked into it.
    fiducials
        Every parameter; those not in ``axes`` are pinned at their value.
    observed
        Observed spectrum ``(F,)``.
    per_network
        :func:`network_data` by network name.
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
        partial(evaluate_grid, evaluator, axes, fixed, observed, per_network),
        cache_dir / f"{prefix}{run}_{digest}.npz",
    )


@app.function(hide_code=True)
def shot_noise_variant(
    variant: str,
    variance_fn: SpectralVarianceFn,
    fiducial_relative_variance: NDArray[np.float64],
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluator keyword arguments and cache settings of one likelihood variant.

    Parameters
    ----------
    variant
        ``"detector"``, ``"amplitude"``, ``"fixed"`` or ``"per_frequency"``.
    variance_fn
        The per-bin shot-noise variance, a pytree bound to the catalog.
    fiducial_relative_variance
        :math:`s^2` at the fiducials per network, in ``per_network`` order;
        used by ``"fixed"`` only.

    Returns
    -------
    tuple
        Keyword arguments for :class:`GaussianGWBBatchedLikelihood`, and the
        settings that set the variant apart in a cache digest. The detector
        variant adds none, so its caches predate shot noise and stay valid.
    """
    if variant == "detector":
        return {}, {}
    if variant == "fixed":
        values = [float(value) for value in fiducial_relative_variance]
        return (
            {"amplitude_shot_noise_variance": jnp.asarray(values)},
            {"shot_noise": variant, "relative_variance": values},
        )
    return (
        {"shot_noise_variance_fn": variance_fn, "shot_noise_direction": variant},
        {"shot_noise": variant},
    )


@app.function(hide_code=True)
def problem_posteriors(
    run: str,
    axes: Mapping[str, tuple[float, float, int]],
    variants: Sequence[str],
    evaluator: GaussianGWBBatchedLikelihood,
    spectral_density_fn: Callable[..., object],
    variance_fn: SpectralVarianceFn,
    fiducial_relative_variance: NDArray[np.float64],
    settings: Mapping[str, object],
    **kwargs: object,
) -> dict[str, tuple[dict[str, NDArray], dict[str, NDArray]]]:
    """:func:`grid_posterior` of one problem for every likelihood variant.

    ``kwargs`` are :func:`grid_posterior`'s remaining arguments. One evaluator
    serves every variant; each variant compiles once.
    """
    posteriors = {}
    for variant in variants:
        evaluator_kwargs, variant_settings = shot_noise_variant(
            variant, variance_fn, fiducial_relative_variance
        )
        posteriors[variant] = grid_posterior(
            run,
            axes,
            partial(
                evaluator,
                spectral_density_fn=spectral_density_fn,
                **evaluator_kwargs,
            ),
            settings={**settings, **variant_settings},
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
        ``(median, lower, upper)``; the interval is the span of the cells
        above the density threshold that encloses ``mass``.
    """
    cell = density / density.sum()
    median = float(np.interp(0.5, np.cumsum(cell) - cell / 2, grid))
    order = np.argsort(cell)[::-1]
    kept = order[: int(np.searchsorted(np.cumsum(cell[order]), mass)) + 1]
    return median, float(grid[kept].min()), float(grid[kept].max())


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
def plot_marginals(
    parameter: str,
    grid: NDArray[np.float64],
    densities: Mapping[str, NDArray[np.float64]],
    networks: Sequence[tuple[str, str]],
    fiducial: float,
) -> Figure:
    """Overlay one parameter's marginal posterior for every network.

    Colors and line styles follow the detector-comparison convention of
    ``scripts/mcmc_cosmological_parameters.py``.

    Parameters
    ----------
    parameter
        Parameter name, for the axis label.
    grid
        Parameter grid.
    densities
        Marginal density by network name.
    networks
        ``(name, LaTeX label)`` in legend order.
    fiducial
        Truth marker position.

    Returns
    -------
    matplotlib.figure.Figure
    """
    styled = [Network(name, label, ()) for name, label in networks]
    colors, linestyles = detector_network_styles(styled)
    fig, ax = plt.subplots()
    handles = []
    for network, color, linestyle in zip(styled, colors, linestyles, strict=True):
        ax.plot(grid, densities[network.name], color=color, linestyle=linestyle)
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                linestyle=linestyle,  # ty: ignore[invalid-argument-type]
                label=network.label,
            )
        )
    ax.axvline(fiducial, **TRUTH)  # ty: ignore[invalid-argument-type]
    ax.set(xlabel=parameter_label(parameter), ylabel="Posterior density")
    ax.legend(handles=handles, **DETECTOR_COMPARISON_LEGEND)  # ty: ignore[no-matching-overload]
    fig.tight_layout()
    return fig


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reference catalog and injection

    The reference catalog generates every intrinsic draw once, at the
    window's lower edge, so its power is `(F_ref, N)`. It is generated in
    `lax.map` batches on a miss and read from `outputs/catalogs/` afterwards.
    The injection is one Poisson spectrum, cached beside the posteriors.

    The rescaled spectrum is built in the same cell as the catalog, once.
    The population is built a single time because it hashes by identity. Each
    problem builds its own `GaussianGWBBatchedLikelihood` in its section, which
    predicts every grid point once for all six networks. The spectral density
    function is a pytree passed to it per call, so the catalog is one traced
    input shared by all three problems rather than a constant compiled into
    each. The NumPy catalog is dropped there, so only the device copy stays
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
    # copies made by the two builders exist: the power and its square.
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
    spectral_density_fn, _ = build_rescaled_spectrum(
        catalog_data,
        catalog_metadata,
        population=target,
        frequencies=frequencies,
        num_redshift_nodes=redshift_nodes,
        density_sites=(),
    )
    shot_noise_variance_fn = build_rescaled_shot_noise(
        catalog_data,
        catalog_metadata,
        population=target,
        frequencies=frequencies,
        num_redshift_nodes=redshift_nodes,
        density_sites=(),
        observation_time=observation_time,
    )
    return shot_noise_variance_fn, spectral_density_fn


@app.cell
def _(
    cache_dir,
    cache_name_prefix,
    chunk_size,
    data_seed,
    frequencies,
    injection_metadata,
):
    injection_file = (
        cache_dir
        / f"{cache_name_prefix}injection_{injection_metadata.key()}_{data_seed}.npz"
    )
    if injection_file.is_file():
        with np.load(injection_file) as cached:
            observed = cached["spectral_density"]
            injection_frequencies = cached["frequencies"]
            num_injected = int(cached["n_events"])
    else:
        drawn = BackgroundSpectralDensitySimulator(
            injection_metadata, chunk_size=chunk_size
        )(batch_keys(data_seed, 1)[0])
        # Zero noise: the Poisson spectrum itself is the data. The draw axis
        # is leading even for one draw.
        observed = np.asarray(drawn["spectral_density"][0], dtype=np.float64)
        injection_frequencies = np.asarray(drawn["frequencies"], dtype=np.float64)
        num_injected = int(drawn["n_events"][0])
        injection_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            injection_file,
            spectral_density=observed,
            frequencies=injection_frequencies,
            n_events=num_injected,
        )
    validate_matching_frequency_grids(injection_frequencies, frequencies)
    print(f"injection: {num_injected:,} Poisson events")
    return (observed,)


@app.cell
def _(
    catalog_metadata,
    catalog_seed,
    data_seed,
    frequencies,
    injection_metadata,
    maximum_frequency,
    minimum_frequency,
    network_names,
    observation_time,
    redshift_nodes,
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
    # Everything a cached log density depends on besides its own axes.
    cache_settings = {
        "version": __version__,
        "networks": network_names,
        "seeds": (data_seed, catalog_seed),
        "band": (minimum_frequency, maximum_frequency),
        "observation_time": observation_time,
        "density_sites": (),
        "injection": injection_metadata.key(),
        "reference_catalog": catalog_metadata.key(),
        "redshift_nodes": redshift_nodes,
    }
    labelled_networks = [(n, l) for n, l in DETECTOR_NETWORKS if n in network_names]
    return cache_settings, labelled_networks, per_network


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Shot noise at the fiducials

    The relative amplitude scatter $s$ of one catalog realization and the
    optimal SNR $\rho$ of the expected spectrum, per network. $\rho s$ is the
    shot noise in units of the detector's amplitude error; it does not depend
    on the observation time. `notebooks/spectrum_snrs.py` measures it
    empirically from Poisson ensembles, about 0.48 for the fiducial network.
    """)
    return


@app.cell
def _(
    FIDUCIALS,
    per_network,
    shot_noise_variance_fn,
    spectral_density_fn,
):
    _spectrum = spectral_density_fn(FIDUCIALS)[0]
    _variance = shot_noise_variance_fn(FIDUCIALS)
    _label = dict(DETECTOR_NETWORKS)
    _rows = []
    for _name, _data in per_network.items():
        _keep = np.asarray(_data["frequency_mask"])
        _sigma = np.asarray(_data["scale"])
        _s2 = float(
            amplitude_shot_noise_variance(
                _spectrum, _variance, _data["scale"], _data["frequency_mask"]
            )
        )
        _snr = float(
            np.sqrt(np.sum(np.where(_keep, (np.asarray(_spectrum) / _sigma) ** 2, 0)))
        )
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
    return (fiducial_relative_variance,)


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
    observed,
    per_network,
    problem_variants,
    shot_noise_variance_fn,
    spectral_density_fn,
):
    h0_posteriors = problem_posteriors(
        "H0",
        grid_specs["H0"],
        problem_variants["H0"],
        GaussianGWBBatchedLikelihood(PRIORS, chunk_size=grid_chunk_size),
        spectral_density_fn,
        shot_noise_variance_fn,
        fiducial_relative_variance,
        cache_settings,
        fiducials=FIDUCIALS,
        observed=observed,
        per_network=per_network,
        cache_dir=cache_dir,
        prefix=cache_name_prefix,
    )
    h0_posterior = h0_posteriors[likelihood_variant]
    return h0_posterior, h0_posteriors


@app.cell
def _(FIDUCIALS, h0_posterior, labelled_networks):
    _grids, _log_densities = h0_posterior
    h0_marginal = plot_marginals(
        "H0",
        _grids["H0"],
        {
            name: marginal_density(_grids["H0"], ld, axis=0)
            for name, ld in _log_densities.items()
        },
        labelled_networks,
        FIDUCIALS["H0"],
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
    observed,
    per_network,
    problem_variants,
    shot_noise_variance_fn,
    spectral_density_fn,
):
    h0_omega_m_posteriors = problem_posteriors(
        "H0_Omega_m",
        grid_specs["H0_Omega_m"],
        problem_variants["H0_Omega_m"],
        GaussianGWBBatchedLikelihood(PRIORS, chunk_size=grid_chunk_size),
        spectral_density_fn,
        shot_noise_variance_fn,
        fiducial_relative_variance,
        cache_settings,
        fiducials=FIDUCIALS,
        observed=observed,
        per_network=per_network,
        cache_dir=cache_dir,
        prefix=cache_name_prefix,
    )
    h0_omega_m_posterior = h0_omega_m_posteriors[likelihood_variant]
    return h0_omega_m_posterior, h0_omega_m_posteriors


@app.cell
def _(FIDUCIALS, h0_omega_m_posterior, labelled_networks):
    _grids, _log_densities = h0_omega_m_posterior
    h0_omega_m_marginal = plot_marginals(
        "H0",
        _grids["H0"],
        {
            name: marginal_density(_grids["H0"], ld, axis=0)
            for name, ld in _log_densities.items()
        },
        labelled_networks,
        FIDUCIALS["H0"],
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
def _(FIDUCIALS, fiducial_network, grid_specs, h0_omega_m_posterior):
    _grids, _log_densities = h0_omega_m_posterior
    _names = list(grid_specs["H0_Omega_m"])
    h0_omega_m_corner = plot_corner_for_posterior_grid(
        [_grids[name] for name in _names],
        _log_densities[fiducial_network],
        labels=[parameter_label(name) for name in _names],
        truths=[FIDUCIALS[name] for name in _names],
        **get_corner_kwargs(levels=CORNER_LEVELS),  # ty: ignore[invalid-argument-type]
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
    observed,
    per_network,
    problem_variants,
    shot_noise_variance_fn,
    spectral_density_fn,
):
    xi_posteriors = problem_posteriors(
        "xi_0_xi_n",
        grid_specs["xi_0_xi_n"],
        problem_variants["xi_0_xi_n"],
        GaussianGWBBatchedLikelihood(PRIORS, chunk_size=grid_chunk_size),
        spectral_density_fn,
        shot_noise_variance_fn,
        fiducial_relative_variance,
        cache_settings,
        fiducials=FIDUCIALS,
        observed=observed,
        per_network=per_network,
        cache_dir=cache_dir,
        prefix=cache_name_prefix,
    )
    xi_posterior = xi_posteriors[likelihood_variant]
    return xi_posterior, xi_posteriors


@app.cell
def _(FIDUCIALS, labelled_networks, xi_posterior):
    _grids, _log_densities = xi_posterior
    xi_marginal = plot_marginals(
        "xi_0",
        _grids["xi_0"],
        {
            name: marginal_density(_grids["xi_0"], ld, axis=0)
            for name, ld in _log_densities.items()
        },
        labelled_networks,
        FIDUCIALS["xi_0"],
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
def _(FIDUCIALS, fiducial_network, grid_specs, xi_posterior):
    _grids, _log_densities = xi_posterior
    _names = list(grid_specs["xi_0_xi_n"])
    xi_corner = plot_corner_for_posterior_grid(
        [_grids[name] for name in _names],
        _log_densities[fiducial_network],
        labels=[parameter_label(name) for name in _names],
        truths=[FIDUCIALS[name] for name in _names],
        **get_corner_kwargs(levels=CORNER_LEVELS),  # ty: ignore[invalid-argument-type]
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
    return


@app.cell
def _(
    FIGURES,
    h0_marginal,
    h0_omega_m_corner,
    h0_omega_m_marginal,
    shot_noise_variants_figure,
    write_figures,
    xi_corner,
    xi_marginal,
):
    if write_figures.value:
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
