import marimo

__generated_with = "0.25.0"
app = marimo.App()

with app.setup(hide_code=True):
    import time
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from astrogwb.runtime import configure_runtime

    # A grid has no chains, so one host device. This precedes waveform
    # construction and array creation: the XLA backend is fixed once it starts.
    configure_runtime()

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
    from scipy.interpolate import RegularGridInterpolator
    from xarray import DataTree

    from astrogwb.config import load_config
    from astrogwb.detector import pair_response_moment
    from astrogwb.distributions.config import DistributionConfig
    from astrogwb.importance.catalog import (
        read_catalog,
        validate_matching_frequency_grids,
    )
    from astrogwb.inference import CatalogModel, DenseGaussianLikelihood
    from astrogwb.paper.plotting import (
        DETECTOR_NETWORKS,
        plot_corner,
        plot_network_marginals,
        save_figures,
        use_paper_style,
    )
    from astrogwb.paper.posteriors import pair_network
    from astrogwb.simulators import BackgroundArtifact, network_spectra_ensemble

    # Grid points evaluated together by lax.map: each holds one dense covariance
    # and its Cholesky factor, so this bounds the memory.
    CHUNK = 16

    @jax.jit
    def grid_log_likelihood(
        likelihood: DenseGaussianLikelihood,
        fiducial: Mapping[str, ArrayLike],
        points: jax.Array,
    ) -> jax.Array:
        """The log likelihood at every ``(z_peak, gamma, kappa)`` row of ``points``.

        The likelihood is an argument, so one compilation serves every network of
        one structure; the points are mapped in chunks of :data:`CHUNK`.

        Returns
        -------
        jax.Array
            ``(num_points,)`` log likelihoods.
        """

        def one(point: jax.Array) -> jax.Array:
            z, g, k = point
            params = {**fiducial, "z_peak": z, "gamma": g, "kappa": k}
            return likelihood.log_likelihood(params)[0]

        return jax.lax.map(one, points, batch_size=CHUNK)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Grid posteriors on the Madau-Dickinson parameters

    Posteriors on the shape of the merger-rate history,
    $(z_{\mathrm{peak}}, \gamma, \kappa)$, from a **uniform grid** and not
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

    **The ridge.** The data constrain one combination of the three
    parameters, the amplitude of the spectrum, to a small fraction of the prior
    range: the posterior is a thin curved sheet whose normal is mostly along
    $z_{\mathrm{peak}}$, of width $\sigma \sim 10^{-3}$ to $10^{-2}$ near the
    fiducial and narrower at large $\gamma$.

    **Grid.** The likelihood is evaluated at the center of every cell of a
    uniform grid over the priors, fine along $z_{\mathrm{peak}}$
    (`num_z` cells) and coarse along $\gamma$ and $\kappa$. Priors are uniform,
    so the likelihood is the posterior up to a constant. Summing a
    $(\gamma, \kappa)$ column over $z_{\mathrm{peak}}$ gives its marginal mass,
    which needs the ridge resolved along $z_{\mathrm{peak}}$ only.

    **Draws along the ridge.** The ridge moves in $z_{\mathrm{peak}}$ by about
    one unit per unit of $\gamma$, much more than its width across one
    $\gamma$ cell. Jittering a draw uniformly within its cell would smear it
    into stripes in $z_{\mathrm{peak}}$, one per $\gamma$ row. Instead the draw's
    $z_{\mathrm{peak}}$ is shifted by the change of the ridge, the column's
    posterior-mean $z_{\mathrm{peak}}$ interpolated in $(\gamma, \kappa)$,
    between the cell center and the jittered point.

    **Likelihood.** `likelihood_variant`: `"detector"` (no shot noise) or
    `"dense"`, the dense Gaussian with the catalog covariance of the stacked
    per-pair data under the per-source network response.

    **Checks.** Per network: the posterior mass in columns whose ridge is
    narrower than a $z_{\mathrm{peak}}$ cell, where the column mass is
    unreliable, and in columns whose ridge sits at the support edge.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configuration
    """)


@app.cell
def _():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    FIGURES = ROOT_DIR / "outputs" / "figures" / "madau_dickinson_grid"

    # The injection's seed; the reference catalog is drawn at another, so the
    # injection is not a subset of the catalog.
    data_seed = 41
    redshift_nodes = 32  # Gauss-Legendre nodes in 1/(1 + z); costs no waveforms

    observation_time = 1.0  # years
    minimum_frequency = 2.0
    maximum_frequency = 2048.0
    fiducial_network = "ET-2L-aligned-CE-Hanford"
    # One fiducial-network grid takes about 2.5 h on a laptop at the sizes
    # below; list more of DETECTOR_NETWORKS to compare networks.
    network_names = [fiducial_network]

    # "detector" | "dense": the likelihood every grid uses.
    likelihood_variant = "dense"
    # Isotropic response draws, which the shot noise carries as the network
    # injections do, reduced once to each network's response moment.
    response_draws, response_seed = 32_768, 0
    sampled = ("z_peak", "gamma", "kappa")
    # Grid cells over (z_peak, gamma, kappa): fine enough along z_peak to resolve
    # the ridge near the fiducial.
    num_z, num_gamma, num_kappa = 1024, 60, 30
    num_draws = 100_000  # draws of the posterior, for the plots and summaries
    draw_seed = 44

    FIDUCIALS = load_config(ROOT_DIR / "config").fiducials
    # The shared U(-10, 10) priors on gamma and kappa are restricted to positive
    # values; every other prior is the shared one.
    positive = {"dist": "Uniform", "kwargs": {"low": 0.0, "high": 10.0}}
    PRIORS = {
        name: spec.build()
        for name, spec in (
            {
                **load_config(ROOT_DIR / "config").priors,
                "gamma": DistributionConfig.model_validate(positive),
                "kappa": DistributionConfig.model_validate(positive),
            }
        ).items()
    }
    registry = load_config(ROOT_DIR / "config").registry

    # The same inputs as cosmology_grid_posteriors.py, written by
    # Snakefile and only read here: the injection ensemble,
    # its detector noise and the reference catalog.
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
        draw_seed,
        fiducial_network,
        frequencies,
        likelihood_variant,
        maximum_frequency,
        minimum_frequency,
        ensemble_root,
        network_names,
        noise,
        num_draws,
        num_gamma,
        num_kappa,
        num_z,
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
def naive_grid(
    likelihood: DenseGaussianLikelihood,
    fiducial: Mapping[str, ArrayLike],
    ranges: Mapping[str, tuple[float, float]],
    shape: Sequence[int],
) -> dict[str, NDArray[np.float64]]:
    """The log likelihood at the cell centers of a ``(z_peak, gamma, kappa)`` grid.

    Parameters
    ----------
    likelihood
        The likelihood with its data and network attached.
    fiducial
        Value of every parameter; the grid overrides ``z_peak``, ``gamma`` and
        ``kappa``.
    ranges
        Support of ``z_peak``, ``gamma`` and ``kappa``, in that order; the cells
        span it.
    shape
        Number of cells along each axis.

    Returns
    -------
    dict
        ``z_peak``, ``gamma`` and ``kappa`` (cell centers), ``log_likelihood``
        of shape ``shape``, and ``seconds`` (the run time, compile included).
    """
    axes = {
        name: cell_centers(*bounds, num)
        for (name, bounds), num in zip(ranges.items(), shape, strict=True)
    }
    mesh = np.meshgrid(*axes.values(), indexing="ij")
    points = jnp.stack([jnp.asarray(m.ravel()) for m in mesh], axis=-1)
    start = time.perf_counter()
    log_likelihood = jax.block_until_ready(
        grid_log_likelihood(
            likelihood,
            {name: jnp.asarray(value) for name, value in fiducial.items()},
            points,
        )
    )
    return {
        **axes,
        "log_likelihood": np.asarray(log_likelihood, dtype=np.float64).reshape(shape),
        "seconds": np.asarray(time.perf_counter() - start),
    }


@app.function(hide_code=True)
def cell_weights(grid: Mapping[str, NDArray[np.float64]]) -> NDArray[np.float64]:
    """Posterior mass of every grid cell, summing to one."""
    log_likelihood = grid["log_likelihood"]
    weights = np.exp(log_likelihood - np.max(log_likelihood))
    return weights / weights.sum()


@app.function(hide_code=True)
def ridge_moments(
    grid: Mapping[str, NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Mass, mean and width of ``z_peak`` in every ``(gamma, kappa)`` column.

    Returns
    -------
    tuple of numpy.ndarray
        ``mass``, ``ridge`` and ``width``, each ``(num_gamma, num_kappa)``: the
        column's posterior mass and the mean and standard deviation of
        ``z_peak`` within it.
    """
    weights = cell_weights(grid)
    z = grid["z_peak"][:, None, None]
    mass = weights.sum(axis=0)
    safe = np.where(mass > 0, mass, 1.0)
    ridge = (z * weights).sum(axis=0) / safe
    width = np.sqrt(np.maximum(((z - ridge) ** 2 * weights).sum(axis=0) / safe, 0.0))
    return mass, ridge, width


@app.function(hide_code=True)
def ridge_samples(
    grid: Mapping[str, NDArray[np.float64]],
    num_draws: int,
    *,
    rng: np.random.Generator,
) -> DataTree:
    """Draws from the grid posterior, carried along the ridge, as one MCMC chain.

    A cell is chosen with probability equal to its mass and the draw is jittered
    uniformly within it. ``z_peak`` is then shifted by the change of the ridge,
    the column mean of ``z_peak`` interpolated in ``(gamma, kappa)``, between
    the cell center and the jittered point, and clipped to the grid's support.

    Returns
    -------
    xarray.DataTree
        ``posterior`` group with ``z_peak``, ``gamma`` and ``kappa``, dims
        ``(chain, draw)`` of sizes ``(1, num_draws)``.
    """
    names = ("z_peak", "gamma", "kappa")
    weights = cell_weights(grid)
    cells = np.unravel_index(
        rng.choice(weights.size, size=num_draws, p=weights.ravel()), weights.shape
    )
    halves = [(grid[name][1] - grid[name][0]) / 2 for name in names]
    draws = {
        name: grid[name][cells[axis]] + rng.uniform(-half, half, num_draws)
        for axis, (name, half) in enumerate(zip(names, halves, strict=True))
        if name != "z_peak"
    }
    _, ridge, _ = ridge_moments(grid)
    ridge_at = RegularGridInterpolator(
        (grid["gamma"], grid["kappa"]), ridge, bounds_error=False, fill_value=None
    )
    shift = ridge_at(np.c_[draws["gamma"], draws["kappa"]]) - ridge[cells[1:]]
    z = grid["z_peak"][cells[0]] + rng.uniform(-halves[0], halves[0], num_draws)
    low, high = grid["z_peak"][0] - halves[0], grid["z_peak"][-1] + halves[0]
    draws["z_peak"] = np.clip(z + shift, low, high)
    return from_dict({"posterior": {name: draws[name][None] for name in names}})


@app.function(hide_code=True)
def grid_checks(grid: Mapping[str, NDArray[np.float64]]) -> dict[str, float]:
    """How trustworthy one grid is along ``z_peak``.

    Returns
    -------
    dict
        ``unresolved_mass``, the posterior mass in columns whose ``z_peak``
        width is below one cell, so their mass rests on a cell or two; and
        ``edge_mass``, the mass in columns whose ridge is within 1% of the
        support edge.
    """
    mass, ridge, width = ridge_moments(grid)
    z = grid["z_peak"]
    spacing = z[1] - z[0]
    low, high = z[0] - spacing / 2, z[-1] + spacing / 2
    margin = 0.01 * (high - low)
    near_edge = (ridge < low + margin) | (ridge > high - margin)
    return {
        "unresolved_mass": float(mass[width < spacing].sum()),
        "edge_mass": float(mass[near_edge].sum()),
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

    The catalog model is built once with `density_sites=()`, which forms the
    per-node power moments up front, so no evaluation contracts the catalog. The injection is one
    Poisson catalog recorded by every network, the ensemble's draw 0.
    """)


@app.cell
def _(frequencies, observation_time, redshift_nodes, reference_catalog_path):
    # The NumPy catalog is local to this cell, so it is freed once the device
    # copy made by the likelihood exists.
    catalog_data, catalog_metadata, _seed = read_catalog(reference_catalog_path)
    population = catalog_metadata.population
    print(
        f"reference catalog: {catalog_metadata.num_samples:,} draws at seed "
        f"{_seed}, "
        f"{redshift_nodes} redshift nodes"
    )
    target = population.build()
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
def _(data_seed, ensemble_root, frequencies, network_names):
    # Draw 0 of `data_seed`, shared with cosmology_grid_posteriors.py.
    _drawn = network_spectra_ensemble(ensemble_root, data_seed, 1, network_names)
    observed = {}
    for _name, _data in zip(network_names, _drawn, strict=True):
        validate_matching_frequency_grids(_data["frequencies"], frequencies)
        observed[_name] = _data["pair_spectral_density"][0]
    print(
        f"injection: {int(_drawn[0]['n_events'][0]):,} Poisson events, "
        f"recorded by {len(observed)} networks"
    )
    return (observed,)


@app.cell
def _(
    frequencies,
    likelihood_variant,
    maximum_frequency,
    minimum_frequency,
    model,
    network_names,
    noise,
    observed,
    registry,
    response_draws,
    response_seed,
):
    # Each network scores the per-pair spectra it recorded, on its counted
    # entries, with the response moment of its per-pair response draws.
    network_likelihoods = {}
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
        network_likelihoods[_name] = DenseGaussianLikelihood(
            model,
            _network,
            _network.compress(observed[_name]),
            shot_noise=likelihood_variant == "dense",
        )
    labelled_networks = [(n, l) for n, l in DETECTOR_NETWORKS if n in network_names]
    return labelled_networks, network_likelihoods


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Grids

    One grid per network. The first call compiles; the others reuse it.
    """)


@app.cell
def _(FIDUCIALS, PRIORS, network_likelihoods, num_gamma, num_kappa, num_z, sampled):
    _ranges = {
        name: (float(PRIORS[name].low), float(PRIORS[name].high)) for name in sampled
    }
    grids = {}
    for _name, _lk in network_likelihoods.items():
        grids[_name] = naive_grid(
            _lk, FIDUCIALS, _ranges, (num_z, num_gamma, num_kappa)
        )
        print(
            f"{_name}: {num_z}x{num_gamma}x{num_kappa} in "
            f"{float(grids[_name]['seconds']):.0f} s"
        )
    return (grids,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Checks

    Per network: the posterior mass in columns whose ridge is narrower than a
    $z_{\mathrm{peak}}$ cell (`unresolved_mass`) or sits at the edge of the
    support (`edge_mass`). Both should be small.
    """)


@app.cell
def _(grids):
    label = dict(DETECTOR_NETWORKS)
    checks = pd.DataFrame(
        [{"network": label[name], **grid_checks(grid)} for name, grid in grids.items()]
    )
    checks
    return (checks,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Posteriors
    """)


@app.cell
def _(draw_seed, grids, num_draws):
    _rng = np.random.default_rng(draw_seed)
    posteriors = {
        name: ridge_samples(grid, num_draws, rng=_rng) for name, grid in grids.items()
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
def _(FIGURES, checks, corner, correlations, grids, marginals, summary, write_figures):
    if write_figures.value:
        FIGURES.mkdir(parents=True, exist_ok=True)
        for _name, _grid in grids.items():
            np.savez(FIGURES / f"grid_{_name}.npz", **_grid)
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
