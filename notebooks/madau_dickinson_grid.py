import marimo

__generated_with = "0.25.0"
app = marimo.App()

with app.setup(hide_code=True):
    import dataclasses
    import time
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from astrogwb.paper.runtime import configure_runtime

    # A grid has no chains, so one host device. This precedes waveform
    # construction and array creation: the XLA backend is fixed once it starts.
    configure_runtime(num_chains=1)

    import arviz_stats
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    import pandas as pd
    from arviz_base import from_dict
    from jax.typing import ArrayLike
    from matplotlib.axes import Axes
    from matplotlib.projections import register_projection
    from numpy.typing import NDArray
    from xarray import DataTree

    from astrogwb.detector import sample_pair_response
    from astrogwb.inference import ImportanceGaussianLikelihood
    from astrogwb.inference import Network as LikelihoodNetwork
    from astrogwb.paper.catalogs import (
        read_reference_catalog,
        validate_matching_frequency_grids,
    )
    from astrogwb.paper.config import detector_registry, fiducials, priors
    from astrogwb.paper.config.runs import CATALOGS_ROOT, ENSEMBLES_ROOT, FIGURES_DIR
    from astrogwb.paper.plotting import (
        DETECTOR_NETWORKS,
        plot_corner,
        plot_network_marginals,
        save_figures,
        use_paper_style,
    )
    from astrogwb.paper.posteriors import (
        network_pair_data,
        pad_pairs,
        shot_noise_variant,
    )
    from astrogwb.populations import build_population
    from astrogwb.simulators import DetectorNoise, network_spectra_ensemble

    # Points of the coarse scan that locates the ridge, Newton iterations that
    # polish it, points of the integration window, and its half width in
    # standard deviations of the ridge.
    NUM_SCAN, NUM_NEWTON, NUM_WINDOW, WINDOW_SIGMAS = 64, 12, 81, 10.0
    # Gaussian width used where the curvature along the ridge is not negative.
    FALLBACK_SIGMA = 1e-2
    # Largest Newton step, in units of the parameter.
    MAX_NEWTON_STEP = 0.05
    # Margin kept between the solved ridge and the edges of the prior support.
    SUPPORT_MARGIN = 1e-4
    # Residual |gradient| * sigma above which a column counts as unconverged.
    CONVERGED = 1e-3
    QUANTILES = (0.025, 0.16, 0.5, 0.84, 0.975)

    @jax.jit
    def ridge_columns(
        likelihood: ImportanceGaussianLikelihood,
        fiducial: Mapping[str, ArrayLike],
        gamma: jax.Array,
        kappa: jax.Array,
        z_bounds: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Solve and integrate the thin direction at every ``(gamma, kappa)``.

        The likelihood is an argument, so one compilation serves every network of
        one structure; the grid points are mapped in batches to bound the memory.

        Returns
        -------
        tuple of jax.Array
            ``log_density``, ``z_star``, ``sigma`` and ``gradient_scale``, each
            ``(num_points,)``: the log of the integral over the window, the ridge
            position and width, and ``|d ln L / dz| * sigma`` at the ridge.
        """

        def log_likelihood(z: jax.Array, g: jax.Array, k: jax.Array) -> jax.Array:
            params = {**fiducial, "z_peak": z, "gamma": g, "kappa": k}
            return likelihood.log_likelihood(params, likelihood.network)[0]

        slope = jax.grad(log_likelihood)
        curvature = jax.grad(slope)
        low, high = z_bounds

        def column(point: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, ...]:
            g, k = point
            scan = jnp.linspace(low, high, NUM_SCAN)
            z = scan[jnp.argmax(jax.vmap(lambda zz: log_likelihood(zz, g, k))(scan))]

            def newton(_: int, z: jax.Array) -> jax.Array:
                h = curvature(z, g, k)
                step = jnp.where(h < 0, -slope(z, g, k) / h, 0.0)
                return jnp.clip(
                    z + jnp.clip(step, -MAX_NEWTON_STEP, MAX_NEWTON_STEP), low, high
                )

            z = jax.lax.fori_loop(0, NUM_NEWTON, newton, z)
            h = curvature(z, g, k)
            sigma = jnp.where(h < 0, 1.0 / jnp.sqrt(-h), FALLBACK_SIGMA)
            lower = jnp.maximum(low, z - WINDOW_SIGMAS * sigma)
            upper = jnp.minimum(high, z + WINDOW_SIGMAS * sigma)
            window = jnp.linspace(lower, upper, NUM_WINDOW)
            values = jax.vmap(lambda zz: log_likelihood(zz, g, k))(window)
            peak = jnp.max(values)
            weights = jnp.exp(values - peak)
            spacing = (upper - lower) / (NUM_WINDOW - 1)
            integral = spacing * (jnp.sum(weights) - 0.5 * (weights[0] + weights[-1]))
            return (
                jnp.log(integral) + peak,
                z,
                sigma,
                jnp.abs(slope(z, g, k)) * sigma,
            )

        return jax.lax.map(column, (gamma, kappa), batch_size=256)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Grid posteriors on the Madau-Dickinson parameters

    Posteriors on the shape of the merger-rate history,
    $(z_{\mathrm{peak}}, \gamma, \kappa)$, from a **ridge-aligned grid** and not
    from sampling: the posterior is too thin and curved for NUTS to sample
    reliably, which showed as divergences and chains that froze.

    The data and the model spectrum are those of `cosmology_grid_posteriors.py`:
    the zero-noise Poisson injection as each network records it, through the
    per-source antenna patterns, and the rescaled reference-redshift catalog
    drawn at a different seed.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    **Why not a plain grid.** The data constrain one combination of the three
    parameters, the amplitude of the spectrum, to a small fraction of the prior
    range: the posterior is a thin curved sheet, of width of order
    $10^{-4}$ to $10^{-3}$ of the prior range across it, whose normal is mostly along
    $z_{\mathrm{peak}}$. A uniform grid at the spacing of the soft directions
    steps over it; one fine enough to resolve it is out of reach.

    **Solving for the ridge.** The grid runs over $(\gamma, \kappa)$ only.
    At each point the likelihood is a thin spike in $z_{\mathrm{peak}}$. A coarse
    scan of `NUM_SCAN` values of $z_{\mathrm{peak}}$ locates the basin, Newton
    iterations locate the maximum $z^*$, and the curvature there gives the width
    $\sigma = (-\partial_z^2 \ln \mathcal{L})^{-1/2}$. The marginal density of
    $(\gamma, \kappa)$ is the trapezoid integral of the likelihood over
    $z^* \pm$ `WINDOW_SIGMAS` $\sigma$, clipped to the prior. Priors are uniform,
    so the likelihood is the posterior up to a constant. Draws take
    $z_{\mathrm{peak}}$ from a Gaussian of width $\sigma$ about $z^*$ of the
    chosen cell.

    **Likelihood.** `likelihood_variant`: `"detector"`, `"amplitude"`,
    `"fixed"` or `"modes"`. `"fixed"` holds the shot-noise variance at its
    fiducial value; `"modes"` keeps the leading eigenmodes of the catalog
    covariance with the per-source network response.

    **Checks.** Per network: the posterior mass in columns where Newton did not
    converge or the ridge sits on the support edge, and how far the quantiles
    move when the grid is refined by a factor of two.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configuration
    """)


@app.cell
def _():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    FIGURES = ROOT_DIR / FIGURES_DIR / "madau_dickinson_grid"

    # The injection's seed; the reference catalog is drawn at another, so the
    # injection is not a subset of the catalog.
    data_seed = 41
    redshift_nodes = 32  # Gauss-Legendre nodes in 1/(1 + z); costs no waveforms

    observation_time = 1.0  # years
    minimum_frequency = 2.0
    maximum_frequency = 2048.0
    fiducial_network = "ET-2L-aligned-CE-Hanford"
    network_names = [name for name, _ in DETECTOR_NETWORKS]

    # "detector" | "amplitude" | "fixed" | "modes": the likelihood every grid
    # uses. The shape parameters tilt the template, so the shot noise keeps its
    # perpendicular eigenmodes as well as the amplitude.
    likelihood_variant = "modes"
    # Isotropic response draws, which every variant's shot noise carries as
    # the network injections do, and the "modes" basis at the fiducials.
    response_draws, response_seed = 32_768, 0
    modes_tolerance, max_modes = 1e-3, 32  # dropped eigenvalue sum, noise units
    sampled = ("z_peak", "gamma", "kappa")
    num_gamma, num_kappa = 200, 120  # grid cells over (gamma, kappa)
    num_draws = 20_000  # draws of the posterior, for the plots and summaries
    draw_seed = 44

    FIDUCIALS = fiducials(root=ROOT_DIR)
    # The shared U(-10, 10) priors on gamma and kappa are restricted to positive
    # values; every other prior is the shared one.
    positive = {"dist": "Uniform", "kwargs": {"low": 0.0, "high": 10.0}}
    PRIORS = priors(root=ROOT_DIR, gamma=positive, kappa=positive)
    registry = detector_registry(root=ROOT_DIR)

    # The same inputs as cosmology_grid_posteriors.py, written by
    # workflow/coverage_draws.smk and only read here: the injection ensemble,
    # its detector noise and the reference catalog.
    ensemble_root = ROOT_DIR / ENSEMBLES_ROOT / "fiducial"
    reference_catalog_path = ROOT_DIR / CATALOGS_ROOT / "reference-fiducial.h5"
    noise = DetectorNoise.open(ensemble_root / "detector_noise.h5")
    frequencies = noise.frequencies

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
        draw_seed,
        fiducial_network,
        frequencies,
        likelihood_variant,
        max_modes,
        maximum_frequency,
        minimum_frequency,
        ensemble_root,
        modes_tolerance,
        network_names,
        noise,
        num_draws,
        num_gamma,
        num_kappa,
        observation_time,
        redshift_nodes,
        reference_catalog_path,
        registry,
        response_draws,
        response_seed,
        sampled,
        write_figures,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Toolbox
    """)


@app.function(hide_code=True)
def cell_centers(low: float, high: float, num: int) -> NDArray[np.float64]:
    """Centers of ``num`` equal cells spanning ``[low, high]``."""
    half = (high - low) / (2 * num)
    return np.linspace(low + half, high - half, num)


@app.function(hide_code=True)
def ridge_grid(
    likelihood: ImportanceGaussianLikelihood,
    fiducial: Mapping[str, ArrayLike],
    gamma_range: tuple[float, float],
    kappa_range: tuple[float, float],
    z_range: tuple[float, float],
    num_gamma: int,
    num_kappa: int,
) -> dict[str, NDArray[np.float64]]:
    """The marginal posterior on a ``(gamma, kappa)`` grid, with ``z_peak`` solved.

    Parameters
    ----------
    likelihood
        The likelihood with its data and network attached.
    fiducial
        Value of every parameter; the grid overrides ``z_peak``, ``gamma`` and
        ``kappa``.
    gamma_range, kappa_range
        Support of the two gridded parameters; the cells span it.
    z_range
        Support of ``z_peak``; the solved ridge and its window stay inside.
    num_gamma, num_kappa
        Number of cells along each gridded axis.

    Returns
    -------
    dict
        ``gamma`` and ``kappa`` (cell centers), and ``log_density``, ``z_star``,
        ``sigma``, ``gradient_scale`` of shape ``(num_gamma, num_kappa)``, plus
        ``seconds`` (the run time, compile included).
    """
    gamma = cell_centers(*gamma_range, num_gamma)
    kappa = cell_centers(*kappa_range, num_kappa)
    mesh_gamma, mesh_kappa = np.meshgrid(gamma, kappa, indexing="ij")
    bounds = jnp.asarray(
        [z_range[0] + SUPPORT_MARGIN, z_range[1] - SUPPORT_MARGIN], dtype=jnp.float64
    )
    start = time.perf_counter()
    solved = jax.block_until_ready(
        ridge_columns(
            likelihood,
            {name: jnp.asarray(value) for name, value in fiducial.items()},
            jnp.asarray(mesh_gamma.ravel()),
            jnp.asarray(mesh_kappa.ravel()),
            bounds,
        )
    )
    seconds = time.perf_counter() - start
    shape = (num_gamma, num_kappa)
    log_density, z_star, sigma, gradient_scale = (
        np.asarray(array, dtype=np.float64).reshape(shape) for array in solved
    )
    return {
        "gamma": gamma,
        "kappa": kappa,
        "log_density": log_density,
        "z_star": z_star,
        "sigma": sigma,
        "gradient_scale": gradient_scale,
        "seconds": np.asarray(seconds),
    }


@app.function(hide_code=True)
def cell_weights(grid: Mapping[str, NDArray[np.float64]]) -> NDArray[np.float64]:
    """Posterior mass of every ``(gamma, kappa)`` cell, summing to one."""
    log_density = grid["log_density"]
    weights = np.exp(log_density - np.max(log_density))
    return weights / weights.sum()


@app.function(hide_code=True)
def ridge_samples(
    grid: Mapping[str, NDArray[np.float64]],
    z_range: tuple[float, float],
    num_draws: int,
    *,
    rng: np.random.Generator,
) -> DataTree:
    """Draws from the grid posterior, in the layout of one MCMC chain.

    A cell is chosen with probability equal to its mass and the draw is
    jittered uniformly within it; ``z_peak`` is Gaussian about the cell's solved
    ridge with its width, clipped to ``z_range``.

    Returns
    -------
    xarray.DataTree
        ``posterior`` group with ``z_peak``, ``gamma`` and ``kappa``, dims
        ``(chain, draw)`` of sizes ``(1, num_draws)``.
    """
    weights = cell_weights(grid)
    cells = np.unravel_index(
        rng.choice(weights.size, size=num_draws, p=weights.ravel()), weights.shape
    )
    draws = {}
    for axis, name in enumerate(("gamma", "kappa")):
        centers = grid[name]
        half = (centers[1] - centers[0]) / 2
        draws[name] = (centers[cells[axis]] + rng.uniform(-half, half, num_draws))[None]
    z = grid["z_star"][cells] + grid["sigma"][cells] * rng.standard_normal(num_draws)
    draws["z_peak"] = np.clip(z, *z_range)[None]
    return from_dict({"posterior": draws})


@app.function(hide_code=True)
def weighted_quantiles(
    grid: Mapping[str, NDArray[np.float64]], parameter: str
) -> NDArray[np.float64]:
    """Quantiles :data:`QUANTILES` of ``parameter`` under the grid's cell masses.

    ``z_peak`` is read at the solved ridge of every cell.
    """
    weights = cell_weights(grid).ravel()
    if parameter == "z_peak":
        values = grid["z_star"].ravel()
    else:
        mesh = np.meshgrid(grid["gamma"], grid["kappa"], indexing="ij")
        values = mesh[("gamma", "kappa").index(parameter)].ravel()
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return np.interp(QUANTILES, cumulative / cumulative[-1], values[order])


@app.function(hide_code=True)
def grid_checks(
    coarse: Mapping[str, NDArray[np.float64]],
    fine: Mapping[str, NDArray[np.float64]],
    z_range: tuple[float, float],
    sampled: Sequence[str],
) -> dict[str, float]:
    """How trustworthy one grid is: solver health and sensitivity to resolution.

    Parameters
    ----------
    coarse, fine
        The grid and a refinement of it by a factor of two in each axis.
    z_range
        Support of ``z_peak``.
    sampled
        Parameters whose quantiles are compared.

    Returns
    -------
    dict
        ``unconverged_mass``, the posterior mass in columns whose Newton
        iteration left ``|gradient| * sigma`` above :data:`CONVERGED`;
        ``edge_mass``, the mass whose solved ridge is within 1% of the
        support edge; and ``max_shift``, the largest change of any quantile
        under refinement, in units of the parameter's prior width.
    """
    weights = cell_weights(coarse)
    width = z_range[1] - z_range[0]
    near_edge = (coarse["z_star"] < z_range[0] + 0.01 * width) | (
        coarse["z_star"] > z_range[1] - 0.01 * width
    )
    shifts = [
        np.max(
            np.abs(weighted_quantiles(coarse, name) - weighted_quantiles(fine, name))
        )
        / (width if name == "z_peak" else 10.0)
        for name in sampled
    ]
    return {
        "unconverged_mass": float(weights[coarse["gradient_scale"] > CONVERGED].sum()),
        "edge_mass": float(weights[near_edge].sum()),
        "max_shift": float(max(shifts)),
    }


@app.function(hide_code=True)
def summary_rows(
    posteriors: Mapping[str, DataTree], sampled: Sequence[str], prob: float = 0.6827
) -> list[dict[str, object]]:
    """Median and highest-density interval of each parameter, per network.

    Returns
    -------
    list of dict
        One row per (network, parameter), ready for :class:`pandas.DataFrame`.
    """
    label = dict(DETECTOR_NETWORKS)
    rows: list[dict[str, object]] = []
    for network, posterior in posteriors.items():
        hdi = arviz_stats.hdi(posterior, prob=prob, var_names=list(sampled))
        for name in sampled:
            lower, upper = (float(v) for v in hdi[name].values)
            draws = posterior["posterior"].dataset[name].values
            rows.append(
                {
                    "network": label[network],
                    "parameter": name,
                    "median": float(np.median(draws)),
                    "lower": lower,
                    "upper": upper,
                    "half width": (upper - lower) / 2,
                }
            )
    return rows


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reference catalog and injection

    The reference catalog is built once with
    `density_sites=()` and the amplitude mode, which holds the squared power
    and the Gram matrix of the catalog so that every variant can be switched
    to afterwards and no evaluation contracts the catalog. The injection is one
    Poisson catalog recorded by every network, the ensemble's draw 0.
    """)


@app.cell
def _(frequencies, observation_time, redshift_nodes, reference_catalog_path):
    # The NumPy catalog is local to this cell, so it is freed once the device
    # copy made by the likelihood exists.
    catalog_data, catalog_metadata, _seed = read_reference_catalog(
        reference_catalog_path
    )
    population = catalog_metadata.population
    print(
        f"reference catalog: {catalog_metadata.num_samples:,} draws at seed "
        f"{_seed}, "
        f"{redshift_nodes} redshift nodes"
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
    return (likelihood,)


@app.cell
def _(data_seed, ensemble_root, frequencies, network_names, registry):
    # Draw 0 of `data_seed`, shared with cosmology_grid_posteriors.py.
    _drawn = network_spectra_ensemble(ensemble_root, data_seed, 1, network_names)
    observed = {}
    for _name, _data in zip(network_names, _drawn, strict=True):
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
    frequencies,
    likelihood,
    likelihood_variant,
    max_modes,
    maximum_frequency,
    minimum_frequency,
    modes_tolerance,
    network_names,
    noise,
    observed,
    registry,
    response_draws,
    response_seed,
):
    per_network = {
        name: network_pair_data(
            noise,
            name,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
        for name in network_names
    }
    # Each network's per-pair response draws, which its shot noise carries,
    # reduced to what the likelihood holds: the per-node moments (so s^2 is
    # not projected once per draw at every point; exact, as no density site
    # varies) and, for the "modes" variant, the basis and node blocks. The
    # (L, P, F) draws are then dropped.
    base_networks = {}
    for _name, _data in per_network.items():
        _detectors, _ = registry.build_network(_name)
        _base = LikelihoodNetwork(
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
        _base = _base._replace(
            response_moments=likelihood.response_moments(FIDUCIALS, _base)
        )
        if likelihood_variant == "modes":
            _base = likelihood.with_shot_noise_modes(
                FIDUCIALS, _base, tolerance=modes_tolerance, max_modes=max_modes
            )
        base_networks[_name] = _base._replace(response=None)
    # The fiducial s^2 per network; only the "fixed" variant uses it.
    fiducial_relative_variance = np.array(
        [
            float(likelihood.relative_variance(FIDUCIALS, network))
            for network in base_networks.values()
        ]
    )
    variant_likelihood, _s2 = shot_noise_variant(
        likelihood_variant, likelihood, fiducial_relative_variance
    )

    def _network(index: int, name: str) -> LikelihoodNetwork:
        if _s2 is None:
            return base_networks[name]
        return base_networks[name]._replace(relative_variance=jnp.asarray(_s2[index]))

    # Each network scores the spectrum it recorded.
    network_likelihoods = {
        name: dataclasses.replace(
            variant_likelihood,
            observed=jnp.asarray(observed[name]),
            network=_network(index, name),
        )
        for index, name in enumerate(per_network)
    }
    labelled_networks = [(n, l) for n, l in DETECTOR_NETWORKS if n in network_names]
    return labelled_networks, network_likelihoods


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Grids

    One ridge-aligned grid per network, and a refinement of it by a factor of
    two for the resolution check. The first call compiles; the others reuse it.
    """)


@app.cell
def _(FIDUCIALS, PRIORS, network_likelihoods, num_gamma, num_kappa):
    z_range = (float(PRIORS["z_peak"].low), float(PRIORS["z_peak"].high))
    _gamma_range = (float(PRIORS["gamma"].low), float(PRIORS["gamma"].high))
    _kappa_range = (float(PRIORS["kappa"].low), float(PRIORS["kappa"].high))
    grids = {}
    fine_grids = {}
    for _name, _lk in network_likelihoods.items():
        _arguments = (FIDUCIALS, _gamma_range, _kappa_range, z_range)
        grids[_name] = ridge_grid(_lk, *_arguments, num_gamma, num_kappa)
        fine_grids[_name] = ridge_grid(_lk, *_arguments, 2 * num_gamma, 2 * num_kappa)
        print(
            f"{_name}: {num_gamma}x{num_kappa} in {float(grids[_name]['seconds']):.1f} s"
        )
    return fine_grids, grids, z_range


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Checks

    Per network: the posterior mass in columns where Newton did not converge
    (`unconverged_mass`) or the ridge sits at the edge of the support
    (`edge_mass`), and the largest change of any quantile under refinement, as
    a fraction of the prior width (`max_shift`). All three should be small.
    """)


@app.cell
def _(fine_grids, grids, sampled, z_range):
    label = dict(DETECTOR_NETWORKS)
    checks = pd.DataFrame(
        [
            {
                "network": label[name],
                **grid_checks(grids[name], fine_grids[name], z_range, sampled),
            }
            for name in grids
        ]
    )
    checks
    return (checks,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Posteriors
    """)


@app.cell
def _(draw_seed, grids, num_draws, z_range):
    _rng = np.random.default_rng(draw_seed)
    posteriors = {
        name: ridge_samples(grid, z_range, num_draws, rng=_rng)
        for name, grid in grids.items()
    }
    return (posteriors,)


@app.cell
def _(FIDUCIALS, PRIORS, labelled_networks, posteriors, sampled):
    marginals = {
        name: plot_network_marginals(
            name,
            posteriors,
            labelled_networks,
            FIDUCIALS[name],
            bounds=(float(PRIORS[name].low), float(PRIORS[name].high)),
        )
        for name in sampled
    }
    mo.vstack([mo.as_html(figure) for figure in marginals.values()])
    return (marginals,)


@app.cell
def _(FIDUCIALS, fiducial_network, posteriors, sampled):
    corner = plot_corner(
        posteriors[fiducial_network], {name: FIDUCIALS[name] for name in sampled}
    )
    mo.as_html(corner)
    return (corner,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Summary and output

    Median and 68% HDI of every sampled parameter, by network. Figures and the
    tables are written when the switch in *Configuration* is on.
    """)


@app.cell
def _(posteriors, sampled):
    summary = pd.DataFrame(summary_rows(posteriors, sampled))
    summary
    return (summary,)


@app.cell
def _(posteriors, sampled):
    # Pairwise correlation coefficients of the draws, by network.
    _label = dict(DETECTOR_NETWORKS)
    _rows = []
    for _name, _tree in posteriors.items():
        _draws = np.stack(
            [np.asarray(_tree["posterior"][p].values).ravel() for p in sampled]
        )
        _matrix = np.corrcoef(_draws)
        for _i, _first in enumerate(sampled):
            for _j in range(_i + 1, len(sampled)):
                _rows.append(
                    {
                        "network": _label[_name],
                        "pair": f"{_first}-{sampled[_j]}",
                        "correlation": float(_matrix[_i, _j]),
                    }
                )
    correlations = pd.DataFrame(_rows)
    correlations
    return (correlations,)


@app.cell
def _(FIGURES, checks, corner, correlations, marginals, summary, write_figures):
    if write_figures.value:
        save_figures(
            {
                **{
                    FIGURES / f"marginals_{name}.pdf": fig
                    for name, fig in marginals.items()
                },
                FIGURES / "corner_z_peak_gamma_kappa.pdf": corner,
            }
        )
        summary.to_csv(FIGURES / "summary.csv", index=False)
        summary.style.format(precision=3).hide(axis="index").to_latex(
            FIGURES / "summary.tex", hrules=True
        )
        checks.to_csv(FIGURES / "checks.csv", index=False)
        correlations.to_csv(FIGURES / "correlations.csv", index=False)


if __name__ == "__main__":
    app.run()
