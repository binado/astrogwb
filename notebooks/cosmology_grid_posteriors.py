import marimo

__generated_with = "0.25.0"
app = marimo.App()

with app.setup(hide_code=True):
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from astrogwb.runtime import configure_runtime

    # Backend configuration precedes waveform construction and array creation.
    configure_runtime()

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

    from astrogwb.config import load_config
    from astrogwb.detector import pair_response_moment
    from astrogwb.importance.catalog import (
        read_catalog,
        validate_matching_frequency_grids,
    )
    from astrogwb.inference import (
        CatalogModel,
        DenseGaussianLikelihood,
        amplitude_template,
        grid_log_likelihood,
        grid_log_prior,
        shot_noise_variance,
    )
    from astrogwb.inference.gaussian import Network
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
    from astrogwb.paper.posteriors import pair_network
    from astrogwb.simulators import (
        BackgroundArtifact,
        network_spectra_ensemble,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Grid posteriors for cosmology and modified propagation

    Posteriors on a **grid**, with no sampler: the log density of the
    Gaussian spectrum likelihood is evaluated at every grid point.

    The notebook reads its inputs and never simulates them:
    `Snakefile` writes the injection ensemble, its detector
    noise and the reference catalog (see the configuration cell for the
    paths). A missing input is a `FileNotFoundError`.

    The data are a **zero-noise Poisson injection**: one Poisson catalog drawn
    at the fiducials, with no Gaussian noise added on top, as each network
    records it. Every source reaches a network through its own antenna
    patterns at an isotropic sky position, and the detector pairs are combined
    with the optimal cross-correlation weights
    (`astrogwb.simulators.simulate_background`), so the six networks see one catalog
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
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    **Likelihood.** Every detector pair's cross power in every bin is Gaussian
    about $\gamma_{ab} S$, with detector-noise scale
    $\sigma_{ab,k} \propto \sqrt{S_a S_b}/\sqrt{2T\Delta f_k}$. The network
    changes $\sigma$, which entries are usable, the data it recorded and the
    response its shot noise carries, but not the expected spectrum; each
    network is its own likelihood, evaluated over the grid.

    **Shot noise.** The data are one finite catalog, so the per-pair cross
    powers also scatter about their mean with the catalog covariance of
    Sec. 2.7, $M_{kl}(\Lambda) R_{ab,k;\,cd,l}$: the power moment $M$ of a
    Poisson sum (Campbell's theorem), computed from the same reference catalog
    at every grid point, times the network's response moment $R$, the second
    moment of the per-source ORF of every pair over `response_draws` isotropic
    draws, computed once. The likelihood is the dense Gaussian with covariance
    $\Sigma + M \circ R$, factorized once per grid point and network. The
    covariance comes from the model at each grid point, never from the data.
    Variants: `detector` (no shot noise) and `dense`, which the figures and
    tables show.

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
    FIGURES = ROOT_DIR / "outputs" / "figures" / "cosmology_grid_posteriors"

    # The injection's seed; the reference catalog is drawn at another, 42, so
    # the injection is not a subset of the catalog.
    data_seed = 41
    marginal_seed = 43  # cell and jitter draws behind the plots
    num_marginal_samples = 2**18  # grid draws per plot, for the marginals and corners
    redshift_nodes = 32  # Gauss-Legendre nodes in 1/(1 + z); costs no waveforms
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
    # (ET-2L-mis, at 45 deg from local North) ~1.5 and ~0.02. xi_n is flat at
    # xi_0 = 1, so it takes its prior range.
    h0_window, h0_size = (58.5, 77.0), 926  # H0 alone: 0.02 step
    h0_omega_window, h0_omega_size = (59.0, 76.5), 351  # H0 in the 2D problem
    omega_m_window, omega_m_size = (0.2856, 0.3336), 41  # +/- 4 prior sigma
    xi_0_window, xi_0_size = (0.86, 1.14), 281
    xi_n_window, xi_n_size = (0.3, 5.0), 95  # prior range, spacing 0.05

    # Likelihood variants evaluated per problem, and the one every figure
    # shows: "dense" is the paper's likelihood, the dense catalog covariance of
    # the stacked per-pair data, with the per-source network response from
    # `response_draws` isotropic draws, as the injections carry it; "detector"
    # drops the shot noise.
    likelihood_variant = "dense"
    problem_variants = {
        "H0": ("detector", "dense"),
        "H0_Omega_m": ("detector", "dense"),
        "xi_0_xi_n": ("detector", "dense"),
    }
    response_draws, response_seed = 32_768, 0

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

    FIDUCIALS = load_config(ROOT_DIR / "config").fiducials
    PRIORS = {
        name: spec.build()
        for name, spec in (load_config(ROOT_DIR / "config").priors).items()
    }
    registry = load_config(ROOT_DIR / "config").registry

    # The inputs, written by Snakefile and only read here:
    # the injection ensemble (draw 0 of `data_seed` is the injection, the
    # one Poisson catalog as each network records it), its detector noise,
    # and the reference catalog, 2^14 intrinsic draws on a scrambled Sobol
    # net at seed 42 on geometric bins 1% wide past the BNS merger
    # (importance_convergence.py sizes it). Both are drawn from the shared
    # population and approximant.
    ensemble_root = ROOT_DIR / "outputs" / "ensembles" / "fiducial"
    reference_catalog_path = ROOT_DIR / "outputs" / "catalogs" / "reference-fiducial.h5"
    background = BackgroundArtifact.open(ensemble_root / f"{data_seed}-0.h5")
    noise = background.noise()
    frequencies = background.frequencies

    write_figures = mo.ui.switch(value=True, label="Write figures")
    write_figures
    use_paper_style(root=ROOT_DIR)
    # gwpy registers replacement default axes; keep matplotlib's.
    register_projection(Axes)
    return (
        FIDUCIALS,
        FIGURES,
        PRIORS,
        data_seed,
        fiducial_network,
        frequencies,
        grid_chunk_size,
        grid_specs,
        likelihood_variant,
        marginal_seed,
        maximum_frequency,
        minimum_frequency,
        background,
        ensemble_root,
        network_names,
        noise,
        num_marginal_samples,
        observation_time,
        problem_variants,
        reference_catalog_path,
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
def evaluate_grid(
    model: CatalogModel,
    priors: Mapping[str, dist.Distribution],
    chunk_size: int,
    axes: Mapping[str, tuple[float, float, int]],
    fixed: Mapping[str, float],
    per_network: Mapping[str, tuple[Network, jax.Array]],
    shot_noise: bool,
) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
    """Evaluate one problem's log posterior on its grid, for every network.

    Parameters
    ----------
    model
        The catalog model, shared by every network.
    priors
        Prior of every parameter.
    chunk_size
        Grid points per :func:`jax.lax.map` batch.
    axes
        ``(low, high, size)`` of each swept parameter.
    fixed
        Every other parameter, pinned at its fiducial.
    per_network
        Each network's :class:`~astrogwb.inference.gaussian.Network` and its
        observed per-pair data on the counted entries, by network name.
    shot_noise
        Whether the covariance includes the catalog's shot noise.

    Returns
    -------
    tuple
        Grids by parameter, and log densities by network with one axis per
        parameter in ``axes`` order.
    """
    grids = {name: np.linspace(lo, hi, n) for name, (lo, hi, n) in axes.items()}
    jax_grids = {name: jnp.asarray(grid) for name, grid in grids.items()}
    log_prior = np.asarray(grid_log_prior(priors, jax_grids, fixed=fixed))
    # One likelihood per network: each compiles at its own number of entries.
    log_densities = {
        name: log_prior
        + np.asarray(
            grid_log_likelihood(
                DenseGaussianLikelihood(model, network, observed, shot_noise),
                jax_grids,
                fixed=fixed,
                chunk_size=chunk_size,
            )
        )
        for name, (network, observed) in per_network.items()
    }
    return grids, log_densities


@app.function(hide_code=True)
def problem_posteriors(
    axes: Mapping[str, tuple[float, float, int]],
    variants: Sequence[str],
    model: CatalogModel,
    *,
    priors: Mapping[str, dist.Distribution],
    chunk_size: int,
    fiducials: Mapping[str, float],
    per_network: Mapping[str, tuple[Network, jax.Array]],
) -> dict[str, tuple[dict[str, NDArray], dict[str, NDArray]]]:
    """:func:`evaluate_grid` of one problem for every likelihood variant.

    Parameters
    ----------
    axes
        ``(low, high, size)`` of each swept parameter.
    variants
        ``"detector"`` (no shot noise) or ``"dense"``.
    model
        The catalog model. Its catalog stays a traced input of the compiled
        evaluator, not a constant baked into it.
    priors
        Prior of every parameter.
    chunk_size
        Grid points per batch.
    fiducials
        Every parameter; those not in ``axes`` are pinned at their value.
    per_network
        See :func:`evaluate_grid`.

    Returns
    -------
    dict
        Per variant, grids by parameter and log densities by network.
    """
    fixed = {k: v for k, v in fiducials.items() if k not in axes}
    return {
        variant: evaluate_grid(
            model,
            priors,
            chunk_size,
            axes,
            fixed,
            per_network,
            shot_noise=variant == "dense",
        )
        for variant in variants
    }


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
    reference redshift $10^{-4}$, so its power is `(F_ref, N)`; it is read from
    `reference_catalog_path` (`astrogwb model-catalog`). The
    injection is one Poisson catalog recorded by every network, read from the
    ensemble's draw 0 (`network_spectra_ensemble`): each network's pair
    spectra were projected in one waveform pass shared by the networks.

    The catalog model is built in the same cell as the catalog, once, with
    its per-node power moments. The population is built a single time because
    it hashes by identity. Each problem evaluates its grid with
    `grid_log_likelihood`, one network at a time. The likelihood is a pytree
    passed to it per call, so the catalog is one traced input shared by all
    three problems rather than a constant compiled into each. The NumPy catalog is dropped there, so only the device copy stays
    resident.
    """)
    return


@app.cell
def _(frequencies, observation_time, redshift_nodes, reference_catalog_path):
    # The NumPy catalog is local to this cell, so it is freed once the device
    # copy made by the likelihood exists: the power and its square.
    catalog_data, catalog_metadata, _seed = read_catalog(reference_catalog_path)
    population = catalog_metadata.population
    power = catalog_data["intensity"]
    print(
        f"reference catalog: {catalog_metadata.num_samples:,} draws at seed "
        f"{_seed}, "
        f"{power.shape[0]} reference bins, {power.nbytes / 1e9:.2f} GB of power; "
        f"{redshift_nodes} redshift nodes"
    )
    target = population.build()
    # No swept parameter is intrinsic, so no density factor enters a weight
    # and the per-node power moments are formed here, once.
    model = CatalogModel.from_catalog(
        catalog_data,
        catalog_metadata,
        population=target,
        frequencies=frequencies,
        num_redshift_nodes=redshift_nodes,
        observation_time=observation_time,
    )
    return (model,)


@app.cell
def _(background, data_seed, ensemble_root, frequencies, network_names):
    # Draw 0 of `data_seed`: the key of the sky-averaged injection, so the
    # networks record the same sources it summed.
    _drawn = network_spectra_ensemble(ensemble_root, data_seed, 1, network_names)
    observed = {}
    for _name, _data in zip(network_names, _drawn, strict=True):
        validate_matching_frequency_grids(_data["frequencies"], frequencies)
        observed[_name] = _data["pair_spectral_density"][0]
    print(
        f"injection: {int(_drawn[0]['n_events'][0]):,} Poisson events, "
        f"recorded by {len(observed)} networks over "
        f"{background.config.population.observation_time:g} yr"
    )
    return (observed,)


@app.cell
def _(
    frequencies,
    maximum_frequency,
    minimum_frequency,
    network_names,
    noise,
    observed,
    registry,
    response_draws,
    response_seed,
):
    # Every network on its counted entries, with the response moment of its
    # per-pair response over the isotropic draws, and its per-pair data.
    per_network = {}
    for _name in network_names:
        _network = pair_network(
            frequencies,
            noise[_name],
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            response_moment=pair_response_moment(
                jax.random.key(response_seed),
                response_draws,
                frequencies,
                registry.build_network(_name),
            ),
        )
        per_network[_name] = (_network, _network.compress(observed[_name]))
    labelled_networks = [(n, l) for n, l in DETECTOR_NETWORKS if n in network_names]
    return labelled_networks, per_network


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Shot noise at the fiducials

    The relative amplitude scatter $s$ of one catalog realization, with the
    network's response, and the optimal SNR $\rho$ of the expected spectrum,
    per network. $\rho s$ is the shot noise in units of the detector's
    amplitude error; it does not depend on the observation time. Measured
    empirically from Poisson ensembles it is about 0.48 for the fiducial network.
    """)
    return


@app.cell
def _(FIDUCIALS, model, per_network):
    _prediction = model.predict(FIDUCIALS)
    _label = dict(DETECTOR_NETWORKS)
    _rows = []
    for _name, (_network, _) in per_network.items():
        _, _squared_snr = amplitude_template(_prediction, _network)
        _snr = float(np.sqrt(_squared_snr))
        _s = float(np.sqrt(shot_noise_variance(_prediction, _network)))
        _rows.append(
            {"network": _label[_name], "rho": _snr, "s": _s, "rho s": _snr * _s}
        )
    shot_noise_table = pd.DataFrame(_rows)
    shot_noise_table
    return (shot_noise_table,)


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
    grid_chunk_size,
    grid_specs,
    likelihood_variant,
    per_network,
    problem_variants,
    model,
):
    h0_posteriors = problem_posteriors(
        grid_specs["H0"],
        problem_variants["H0"],
        model,
        priors=PRIORS,
        chunk_size=grid_chunk_size,
        fiducials=FIDUCIALS,
        per_network=per_network,
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
    grid_chunk_size,
    grid_specs,
    likelihood_variant,
    per_network,
    problem_variants,
    model,
):
    h0_omega_m_posteriors = problem_posteriors(
        grid_specs["H0_Omega_m"],
        problem_variants["H0_Omega_m"],
        model,
        priors=PRIORS,
        chunk_size=grid_chunk_size,
        fiducials=FIDUCIALS,
        per_network=per_network,
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
    grid_chunk_size,
    grid_specs,
    likelihood_variant,
    per_network,
    problem_variants,
    model,
):
    xi_posteriors = problem_posteriors(
        grid_specs["xi_0_xi_n"],
        problem_variants["xi_0_xi_n"],
        model,
        priors=PRIORS,
        chunk_size=grid_chunk_size,
        fiducials=FIDUCIALS,
        per_network=per_network,
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
      standard deviations of the `dense` likelihood against `detector`.
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
            for _variant in ("dense",)
            if (shot_noise_moments["likelihood"] == _variant).any()
        ]
    )
    shot_noise_effect
    return


@app.cell
def _(FIDUCIALS, fiducial_network, h0_posteriors):
    shot_noise_variants_figure, _ax = plt.subplots()
    _styles = {"detector": "--", "dense": "-"}
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
