import marimo

__generated_with = "0.25.0"
app = marimo.App()

with app.setup(hide_code=True):
    # ruff: noqa: PLR1711
    # Marimo generates bare returns when saving output-only cells.
    from functools import partial
    from pathlib import Path

    from astrogwb.paper.runtime import configure_runtime

    # Editable sampler settings. Restart the kernel after changing chain count.
    SMOKE_RUN = False  # True: two chains, 50 warmup, 50 retained draws.
    NUM_CHAINS = 2 if SMOKE_RUN else 4
    NUM_WARMUP = 50 if SMOKE_RUN else 200
    NUM_SAMPLES = 50 if SMOKE_RUN else 1000
    TARGET_ACCEPT = 0.9
    DENSE_MASS = True
    SEED = 42
    jax, CHAIN_METHOD = configure_runtime(num_chains=NUM_CHAINS)

    import arviz_base as azb
    import arviz_plots as azp
    import arviz_stats as azs
    import h5py
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numpyro
    import numpyro.distributions as dist
    from jax.typing import ArrayLike
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from numpyro.infer import MCMC, NUTS

    from astrogwb.constants import SECONDS_PER_YEAR
    from astrogwb.distributions.redshift import MadauDickinsonRedshiftDistribution
    from astrogwb.distributions.redshift.base import RedshiftDistribution


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Madau-Dickinson redshift fit

    Fit independent, exact redshifts from a **complete, all-sky catalog without
    selection effects**. The cosmology is fixed at $H_0=67.66$ and
    $\Omega_m=0.3096$. Only `gamma`, `kappa`, and `z_peak` are latent.
    The likelihood uses every individual observation, with the library's
    grid-normalized density; no histogram likelihood or minibatching is used.

    Enter the catalog's **stated** support below; do not use observed extrema
    as catalog bounds. Duration is optional and is in observer-frame years.
    Click **Run fit** after editing inputs. Histogram changes reuse the draws.
    Set `SMOKE_RUN = True` in setup for the 2-chain, 50-warmup, 50-draw smoke
    run on the **full** catalog; restart the kernel when changing chain count.
    Smoke results are for execution checks, not scientific inference.
    """)
    return


@app.cell
def _():
    # Editable catalog settings; support bounds were supplied by the catalog owner.
    catalog_path: Path = Path(
        "/Users/binado/work/research/phd/astrogwb/data/18321_1yrCatalogBNS.h5"
    )
    redshift_dataset = "z"  # This catalog uses "z"; standard catalogs use "redshift".
    minimum_redshift: float = 0.0
    maximum_redshift: float = 15.0
    # The accompanying ET-0197A-23 catalog description states a one-year catalog.
    observation_years: float | None = 1.0
    n_grid = 1000
    # Editable, notebook-local priors, ordered gamma, kappa, z_peak.
    prior_bounds = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0))
    return (
        catalog_path,
        maximum_redshift,
        minimum_redshift,
        n_grid,
        observation_years,
        prior_bounds,
        redshift_dataset,
    )


@app.function
def load_redshifts(
    path: Path, minimum: float, maximum: float, *, dataset_name: str = "redshift"
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Read and validate every observation, retaining the file metadata.

    Raises
    ------
    ValueError
        If support or observations are invalid. No observations are removed.
    """
    if not np.isfinite([minimum, maximum]).all() or not 0 <= minimum < maximum:
        raise ValueError("Require finite catalog bounds with 0 <= minimum < maximum.")
    with h5py.File(path.expanduser(), "r") as handle:
        if (
            "/" in dataset_name
            or dataset_name not in handle
            or not isinstance(handle[dataset_name], h5py.Dataset)
        ):
            raise ValueError(f"Expected an HDF5 root dataset named {dataset_name!r}.")
        dataset = handle[dataset_name]
        if dataset.ndim != 1 or dataset.size == 0 or dataset.dtype.kind not in "iuf":
            raise ValueError(
                f"Require a nonempty 1D real numeric array; got "
                f"shape={dataset.shape}, dtype={dataset.dtype}."
            )
        redshifts = np.asarray(dataset[...], dtype=np.float64)
        metadata = {
            "path": str(path.expanduser().resolve()),
            "dataset": dataset_name,
            "shape": dataset.shape,
            "count": dataset.size,
            "dtype": str(dataset.dtype),
            "range": (float(redshifts.min()), float(redshifts.max())),
            "root_attributes": dict(handle.attrs),
            "redshift_attributes": dict(dataset.attrs),
            "root_keys": list(handle.keys()),
        }
    invalid = (~np.isfinite(redshifts)) | (redshifts < minimum) | (redshifts > maximum)
    if invalid.any():
        indices = np.flatnonzero(invalid)
        raise ValueError(
            f"{indices.size} invalid observations outside finite support "
            f"[{minimum}, {maximum}]; first indices={indices[:10].tolist()}, "
            f"values={redshifts[indices[:10]].tolist()}. Nothing was removed."
        )
    # Zero-density endpoints cannot be observed under the proposed continuous model.
    if np.any(redshifts == 0):
        raise ValueError("Observations at z=0 have zero density under this model.")
    return redshifts, metadata


@app.function
def shape_distribution(
    theta: ArrayLike, minimum: float, maximum: float, n_grid: int
) -> RedshiftDistribution:
    """Build the unit-local-rate distribution on the stated catalog support."""
    theta = jnp.asarray(theta)
    return MadauDickinsonRedshiftDistribution(
        params={
            "H0": 67.66,
            "Omega_m": 0.3096,
            "local_merger_rate": 1.0,
            "gamma": theta[0],
            "kappa": theta[1],
            "z_peak": theta[2],
        },
        minimum_redshift=minimum,
        maximum_redshift=maximum,
        n_grid=n_grid,
    )


@app.function
def redshift_model(
    observed: ArrayLike,
    minimum: float,
    maximum: float,
    n_grid: int,
    prior_bounds: ArrayLike,
) -> None:
    """Observe the full catalog under three independent Uniform shape priors."""
    bounds = jnp.asarray(prior_bounds)
    theta = jnp.stack(
        [
            numpyro.sample(name, dist.Uniform(bounds[i, 0], bounds[i, 1]))
            for i, name in enumerate(("gamma", "kappa", "z_peak"))
        ]
    )
    distribution = shape_distribution(theta, minimum, maximum, n_grid)
    observed = jnp.asarray(observed)
    with numpyro.plate("catalog", observed.shape[0]):
        numpyro.sample("redshift", distribution, obs=observed)


@app.function
def fit_catalog(
    observed: ArrayLike,
    minimum: float,
    maximum: float,
    n_grid: int,
    prior_bounds: ArrayLike,
    *,
    num_chains: int = NUM_CHAINS,
    num_warmup: int = NUM_WARMUP,
    num_samples: int = NUM_SAMPLES,
    seed: int = SEED,
) -> MCMC:
    """Run NUTS on every observation, passing data as a model argument."""
    bounds = np.asarray(prior_bounds)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
        raise ValueError("Require three finite pairs of Uniform prior bounds.")
    if np.any(bounds[:, 0] < 0) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Require nonnegative, ordered Uniform prior bounds.")
    if n_grid < 2:
        raise ValueError("n_grid must be at least two.")
    # Only grid size is static; observations and support remain model arguments.
    kernel = NUTS(
        partial(redshift_model, n_grid=n_grid),
        target_accept_prob=TARGET_ACCEPT,
        dense_mass=DENSE_MASS,
    )
    sampler = MCMC(
        kernel,
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples,
        chain_method=CHAIN_METHOD,
        # Parallel tqdm callbacks cannot write to Marimo's cell-scoped stream.
        progress_bar=False,
    )
    sampler.run(
        jax.random.key(seed),
        observed=jnp.asarray(observed),
        minimum=minimum,
        maximum=maximum,
        prior_bounds=jnp.asarray(bounds),
    )
    return sampler


@app.function
@partial(jax.jit, static_argnames=("n_grid",))
def posterior_products(
    theta: ArrayLike,
    observed: ArrayLike,
    minimum: float,
    maximum: float,
    n_grid: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate full-data log likelihoods, grid densities and unit total rates."""

    def evaluate(draw: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        distribution = shape_distribution(draw, minimum, maximum, n_grid)
        return (
            jnp.sum(distribution.log_prob(observed)),
            distribution.normalized_y,
            distribution.total_merger_rate(),
        )

    # Map avoids a draws-by-observations allocation for large catalogs.
    return jax.lax.map(evaluate, jnp.asarray(theta))


@app.function
def density_figure(
    observed: ArrayLike,
    grid: ArrayLike,
    densities: ArrayLike,
    best_index: int,
    bins: int,
    smoke: bool,
) -> Figure:
    """Overlay a normalized histogram, best sampled fit and pointwise 90% band."""
    grid = np.asarray(grid)
    density = np.asarray(densities)
    lower, upper = np.quantile(density, [0.05, 0.95], axis=0)
    figure, axis = plt.subplots()
    axis.hist(
        np.asarray(observed), bins=bins, density=True, alpha=0.35, label="Catalog"
    )
    axis.fill_between(
        grid, lower, upper, alpha=0.25, label="Pointwise 90% density band"
    )
    axis.plot(grid, density[best_index], label="Best sampled fit (highest likelihood)")
    axis.set(xlabel="Redshift", ylabel="Probability density", xlim=(grid[0], grid[-1]))
    axis.set_title("SMOKE RUN: execution check only" if smoke else "Redshift shape fit")
    axis.legend()
    figure.tight_layout()
    return figure


@app.cell(hide_code=True)
def _(
    catalog_path: Path,
    maximum_redshift: float,
    minimum_redshift: float,
    redshift_dataset,
):
    mo.stop(
        catalog_path is None or minimum_redshift is None or maximum_redshift is None,
        mo.md(
            "**Required:** set `catalog_path`, `minimum_redshift`, and `maximum_redshift`."
        ),
    )
    observed, catalog_metadata = load_redshifts(
        catalog_path, minimum_redshift, maximum_redshift, dataset_name=redshift_dataset
    )
    mo.vstack([mo.md("### Catalog inspection"), mo.plain(catalog_metadata)])
    return (observed,)


@app.cell(hide_code=True)
def _():
    run_fit = mo.ui.run_button(label="Run fit")
    run_fit
    return (run_fit,)


@app.cell
def _(
    maximum_redshift: float,
    minimum_redshift: float,
    n_grid,
    observed,
    prior_bounds,
    run_fit,
):
    mo.stop(not run_fit.value, mo.md("Click **Run fit** to start NUTS."))
    sampler = fit_catalog(
        observed, minimum_redshift, maximum_redshift, n_grid, prior_bounds
    )
    return (sampler,)


@app.cell(hide_code=True)
def _(sampler):
    chain_samples = sampler.get_samples(group_by_chain=True)
    inference_data = azb.from_numpyro(sampler, log_likelihood=False)
    summary_table = azs.summary(
        inference_data,
        var_names=["gamma", "kappa", "z_peak"],
        ci_prob=0.9,
        ci_kind="eti",
        round_to="none",
    )
    divergences = int(np.asarray(sampler.get_extra_fields()["diverging"]).sum())
    trace_plots = azp.plot_trace(
        inference_data,
        var_names=["gamma", "kappa", "z_peak"],
        backend="matplotlib",
        figure_kwargs={"figsize": (9, 7), "layout": "constrained"},
    )
    trace_figure = trace_plots.viz["figure"].item()
    trace_figure.suptitle("SMOKE RUN traces" if SMOKE_RUN else "Posterior traces")
    pair_plots = azp.plot_pair(
        inference_data,
        var_names=["gamma", "kappa", "z_peak"],
        triangle="lower",
        marginal=True,
        visuals={"divergence": True},
        backend="matplotlib",
        figure_kwargs={"figsize": (9, 9), "layout": "constrained"},
    )
    pair_figure = pair_plots.viz["figure"].item()
    pair_figure.suptitle("SMOKE RUN pair plot" if SMOKE_RUN else "Posterior pair plot")
    mo.vstack(
        [
            mo.md(
                f"**{'SMOKE RUN: execution check only' if SMOKE_RUN else 'Posterior diagnostics'}**"
                f"\n\nDivergences: **{divergences}**. `ess_bulk` and `ess_tail` are "
                "effective sample sizes; `r_hat` is rank-normalized split R-hat. "
                "The summary reports equal-tail 90% credible intervals and "
                "Monte Carlo standard errors. "
                "Inspect mixing and convergence before using these results."
            ),
            mo.ui.table(summary_table.reset_index(names="parameter")),
            trace_figure,
            pair_figure,
        ]
    )
    return (chain_samples,)


@app.cell
def _(
    chain_samples,
    maximum_redshift: float,
    minimum_redshift: float,
    n_grid,
    observed,
):
    theta_samples = jnp.stack(
        [chain_samples[_name].reshape(-1) for _name in ("gamma", "kappa", "z_peak")],
        axis=-1,
    )
    log_likelihoods, densities, unit_rates = posterior_products(
        theta_samples, observed, minimum_redshift, maximum_redshift, n_grid
    )
    best_index = int(np.argmax(np.asarray(log_likelihoods)))
    if not np.isfinite(np.asarray(log_likelihoods)).all():
        raise ValueError("Posterior log likelihoods contain nonfinite values.")
    plot_grid = np.linspace(minimum_redshift, maximum_redshift, n_grid)
    mo.md(
        f"Best sampled shape (highest full-data likelihood): "
        f"gamma, kappa, z_peak = {np.asarray(theta_samples[best_index]).tolist()}"
    )
    return best_index, densities, plot_grid, unit_rates


@app.cell(hide_code=True)
def _():
    histogram_bins = mo.ui.slider(10, 200, value=50, label="Histogram bins")
    histogram_bins
    return (histogram_bins,)


@app.cell(hide_code=True)
def _(best_index, densities, histogram_bins, observed, plot_grid):
    density_figure(
        observed, plot_grid, densities, best_index, histogram_bins.value, SMOKE_RUN
    )
    return


@app.cell(hide_code=True)
def _(best_index, observation_years: float | None, observed, unit_rates):
    mo.stop(
        observation_years is None,
        mo.md(
            "Shape results only: set `observation_years` to calculate the local rate."
        ),
    )
    assert observation_years is not None
    if not np.isfinite(observation_years) or observation_years <= 0:
        raise ValueError("Observing duration must be finite and positive in years.")
    if not np.isfinite(np.asarray(unit_rates)).all() or np.any(
        np.asarray(unit_rates) <= 0
    ):
        raise ValueError("Unit total merger rates must be finite and positive.")
    local_rates = len(observed) / (observation_years * SECONDS_PER_YEAR * unit_rates)
    _low, _median, _high = np.quantile(np.asarray(local_rates), [0.05, 0.5, 0.95])
    mo.md(rf"""
    ### {"SMOKE RUN: " if SMOKE_RUN else ""}Local merger rate

    $$\widehat R_0(\theta) = \frac{{N}}{{T_{{\rm seconds}}\,\mathcal R_{{\rm unit}}(\theta)}}$$

    Best sampled rate: **{float(local_rates[best_index]):.6g}**
    $\mathrm{{Gpc}}^{{-3}}\,\mathrm{{yr}}^{{-1}}$.
    Posterior median: **{_median:.6g}**; equal-tail 90% interval:
    **[{_low:.6g}, {_high:.6g}]** in the same units.

    This interval includes **shape-induced uncertainty only**; count uncertainty
    is excluded. The count and duration are fixed. Unit total rates use
    `total_merger_rate()` on the same catalog bounds, with local rate equal to one.
    """)
    return


@app.cell(hide_code=True)
def delayed_intro():
    mo.md(r"""
    ## Time-delayed Madau-Dickinson fit

    Now the Madau-Dickinson shape describes **formation**, convolved with a
    source-frame delay distribution in lookback time. Fit `gamma`, `kappa`,
    `z_peak`, and `delay_slope` jointly using the same full catalog and cosmology.

    $$p(\tau\mid\alpha)\propto\tau^\alpha,\qquad
    \alpha=\mathrm{delay\_slope}\sim\mathrm{Uniform}(-5,0).$$

    The editable uniform prior covers slopes from -5 to 0, including the commonly
    used inverse-time law at -1
    ([BNS delay-model reference](https://link.aps.org/accepted/10.1103/PhysRevD.85.104024)).
    The bounds are a modeling choice; inspect whether the posterior reaches them.
    The fixed minimum delay is **0.02 Gyr (20 Myr)**. Formation is cut off at
    **z=20**, independently of the catalog's merger-redshift bounds. The longest
    delay is the lookback time to that formation cutoff. At each merger redshift,
    the convolution includes only delays compatible with the available cosmic time.
    These assumptions are editable below; their uncertainty is not inferred.

    The delayed rate is normalized to unit **local merger rate**, so the same
    count/duration normalization applies after fitting. The complete-catalog,
    exact-redshift and no-selection assumptions still apply. Click
    **Run delayed fit** to run all four parameters together; histogram adjustments
    reuse its posterior samples. Redshift-only inference can correlate the formation
    shape and delay slope: inspect the corner plot and convergence diagnostics.
    The editable initial values below start near the higher-support mode found for
    this catalog. Set them to `None` to check dispersed initialization. Initial values
    do not change the prior or constrain subsequent samples.
    """)
    return


@app.cell(hide_code=True)
def delayed_settings():
    from astrogwb.cosmology import lookback_time
    from astrogwb.distributions.delay import PowerLawDelayDistribution
    from astrogwb.distributions.redshift import (
        madau_dickinson_time_delayed_redshift_distribution,
    )

    # Same Uniform shape priors as the undelayed fit; editable delay-slope prior.
    delay_slope_prior = dist.Uniform(-5.0, 0.0)
    minimum_delay_gyr = 0.02
    maximum_formation_redshift = 20.0
    delay_quadrature_nodes = 48
    return (
        PowerLawDelayDistribution,
        delay_quadrature_nodes,
        delay_slope_prior,
        lookback_time,
        madau_dickinson_time_delayed_redshift_distribution,
        maximum_formation_redshift,
        minimum_delay_gyr,
    )


@app.cell(hide_code=True)
def delayed_distribution(
    PowerLawDelayDistribution,
    lookback_time,
    madau_dickinson_time_delayed_redshift_distribution,
):
    def delayed_shape_distribution(
        theta: ArrayLike,
        minimum: float,
        maximum: float,
        n_grid: int,
        minimum_delay: float,
        formation_cutoff: float,
        n_delay_nodes: int,
    ) -> RedshiftDistribution:
        """Build the delayed unit-local-rate distribution with delays measured in Gyr."""
        theta = jnp.asarray(theta)
        longest_delay = lookback_time(formation_cutoff, 67.66, 0.3096)
        return madau_dickinson_time_delayed_redshift_distribution(
            params={
                "H0": 67.66,
                "Omega_m": 0.3096,
                "local_merger_rate": 1.0,
                "gamma": theta[0],
                "kappa": theta[1],
                "z_peak": theta[2],
            },
            time_delay_distribution=PowerLawDelayDistribution(
                theta[3], minimum_delay, longest_delay
            ),
            minimum_redshift=minimum,
            maximum_redshift=maximum,
            n_grid=n_grid,
            n_delay_nodes=n_delay_nodes,
            maximum_formation_redshift=formation_cutoff,
        )

    return (delayed_shape_distribution,)


@app.cell(hide_code=True)
def delayed_model(delayed_shape_distribution):
    def delayed_redshift_model(
        observed: ArrayLike,
        minimum: float,
        maximum: float,
        n_grid: int,
        prior_bounds: ArrayLike,
        slope_prior: dist.Distribution,
        minimum_delay: float,
        formation_cutoff: float,
        n_delay_nodes: int,
    ) -> None:
        """Infer three formation-shape parameters and one power-law delay slope jointly."""
        bounds = jnp.asarray(prior_bounds)
        shape = [
            numpyro.sample(name, dist.Uniform(bounds[i, 0], bounds[i, 1]))
            for i, name in enumerate(("gamma", "kappa", "z_peak"))
        ]
        slope = numpyro.sample("delay_slope", slope_prior)
        theta = jnp.stack([*shape, slope])
        distribution = delayed_shape_distribution(
            theta,
            minimum,
            maximum,
            n_grid,
            minimum_delay,
            formation_cutoff,
            n_delay_nodes,
        )
        observed = jnp.asarray(observed)
        with numpyro.plate("catalog", observed.shape[0]):
            numpyro.sample("redshift", distribution, obs=observed)

    return (delayed_redshift_model,)


@app.cell(hide_code=True)
def delayed_fit_helper(delayed_redshift_model, lookback_time):
    def fit_delayed_catalog(
        observed: ArrayLike,
        minimum: float,
        maximum: float,
        n_grid: int,
        prior_bounds: ArrayLike,
        slope_prior: dist.Distribution,
        minimum_delay: float,
        formation_cutoff: float,
        n_delay_nodes: int,
        *,
        num_chains: int = NUM_CHAINS,
        num_warmup: int = NUM_WARMUP,
        num_samples: int = NUM_SAMPLES,
        seed: int = SEED,
        initial_values: dict[str, float] | None = None,
    ) -> MCMC:
        """Run full-data NUTS with a joint dense mass matrix for all four parameters."""
        from numpyro.infer import init_to_uniform, init_to_value

        bounds = np.asarray(prior_bounds)
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
            raise ValueError(
                "Require three finite pairs of Uniform shape-prior bounds."
            )
        if np.any(bounds[:, 0] < 0) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Require nonnegative, ordered Uniform shape-prior bounds.")
        if n_grid < 2 or n_delay_nodes < 1:
            raise ValueError("Require n_grid >= 2 and n_delay_nodes >= 1.")
        if not np.isfinite([minimum_delay, formation_cutoff]).all() or not (
            minimum_delay > 0 and formation_cutoff > maximum
        ):
            raise ValueError(
                "Require a positive delay floor and formation cutoff above support."
            )
        available_delay = float(
            lookback_time(formation_cutoff, 67.66, 0.3096)
            - lookback_time(maximum, 67.66, 0.3096)
        )
        if minimum_delay >= available_delay:
            raise ValueError(
                "Delay floor leaves zero density at the upper catalog bound."
            )
        if initial_values is not None:
            if set(initial_values) != {"gamma", "kappa", "z_peak", "delay_slope"}:
                raise ValueError(
                    "Initial values must specify all four latent parameters."
                )
            if not np.isfinite(list(initial_values.values())).all():
                raise ValueError("Initial values must be finite.")
            for i, name in enumerate(("gamma", "kappa", "z_peak")):
                if not bounds[i, 0] < initial_values[name] < bounds[i, 1]:
                    raise ValueError(
                        "Shape initial values must lie inside their priors."
                    )
        initialization = (
            init_to_uniform()
            if initial_values is None
            else init_to_value(values=initial_values)
        )
        kernel = NUTS(
            partial(delayed_redshift_model, n_grid=n_grid, n_delay_nodes=n_delay_nodes),
            target_accept_prob=TARGET_ACCEPT,
            dense_mass=DENSE_MASS,
            init_strategy=initialization,
        )
        sampler = MCMC(
            kernel,
            num_chains=num_chains,
            num_warmup=num_warmup,
            num_samples=num_samples,
            chain_method=CHAIN_METHOD,
            progress_bar=False,
        )
        sampler.run(
            jax.random.key(seed),
            observed=jnp.asarray(observed),
            minimum=minimum,
            maximum=maximum,
            prior_bounds=jnp.asarray(bounds),
            slope_prior=slope_prior,
            minimum_delay=minimum_delay,
            formation_cutoff=formation_cutoff,
        )
        return sampler

    return (fit_delayed_catalog,)


@app.cell(hide_code=True)
def delayed_products_helper(delayed_shape_distribution):
    @partial(jax.jit, static_argnames=("n_grid", "n_delay_nodes"))
    def delayed_posterior_products(
        theta: ArrayLike,
        observed: ArrayLike,
        minimum: float,
        maximum: float,
        n_grid: int,
        minimum_delay: float,
        formation_cutoff: float,
        n_delay_nodes: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Evaluate sampled delayed likelihoods, grid densities and unit total rates."""

        def evaluate(draw: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            distribution = delayed_shape_distribution(
                draw,
                minimum,
                maximum,
                n_grid,
                minimum_delay,
                formation_cutoff,
                n_delay_nodes,
            )
            return (
                jnp.sum(distribution.log_prob(observed)),
                distribution.normalized_y,
                distribution.total_merger_rate(),
            )

        return jax.lax.map(evaluate, jnp.asarray(theta))

    return (delayed_posterior_products,)


@app.cell
def delayed_initialization():
    # Editable warm start for this catalog; None uses dispersed default initialization.
    # A preliminary dispersed fit got trapped near slope -1 in a lower-support mode.
    # Local posterior optimization favored the mode below. This changes only starting
    # values, not the priors or the posterior explored by NUTS.
    delayed_initial_values: dict[str, float] | None = {
        "gamma": 2.3615,
        "kappa": 4.3018,
        "z_peak": 2.1448,
        "delay_slope": -3.6412,
    }
    return (delayed_initial_values,)


@app.cell(hide_code=True)
def delayed_run_button():
    run_delayed_fit = mo.ui.run_button(label="Run delayed fit")
    run_delayed_fit
    return (run_delayed_fit,)


@app.cell(hide_code=True)
def delayed_fit(
    delay_quadrature_nodes,
    delay_slope_prior,
    delayed_initial_values: dict[str, float] | None,
    fit_delayed_catalog,
    maximum_formation_redshift,
    maximum_redshift: float,
    minimum_delay_gyr,
    minimum_redshift: float,
    n_grid,
    observed,
    prior_bounds,
    run_delayed_fit,
):
    mo.stop(
        not run_delayed_fit.value,
        mo.md("Click **Run delayed fit** to start joint NUTS."),
    )
    delayed_sampler = fit_delayed_catalog(
        observed,
        minimum_redshift,
        maximum_redshift,
        n_grid,
        prior_bounds,
        delay_slope_prior,
        minimum_delay_gyr,
        maximum_formation_redshift,
        delay_quadrature_nodes,
        initial_values=delayed_initial_values,
    )
    return (delayed_sampler,)


@app.cell(hide_code=True)
def delayed_diagnostics(delayed_sampler):
    delayed_chain_samples = delayed_sampler.get_samples(group_by_chain=True)
    delayed_inference_data = azb.from_numpyro(delayed_sampler, log_likelihood=False)
    delayed_summary_table = azs.summary(
        delayed_inference_data,
        var_names=["gamma", "kappa", "z_peak", "delay_slope"],
        ci_prob=0.9,
        ci_kind="eti",
        round_to="none",
    )
    delayed_divergences = int(
        np.asarray(delayed_sampler.get_extra_fields()["diverging"]).sum()
    )
    delayed_trace_plots = azp.plot_trace(
        delayed_inference_data,
        var_names=["gamma", "kappa", "z_peak", "delay_slope"],
        backend="matplotlib",
        figure_kwargs={"figsize": (12, 7), "layout": "constrained"},
    )
    delayed_trace_figure = delayed_trace_plots.viz["figure"].item()
    delayed_trace_figure.suptitle(
        "SMOKE RUN: delayed traces" if SMOKE_RUN else "Time-delayed posterior traces"
    )
    delayed_pair_plots = azp.plot_pair(
        delayed_inference_data,
        var_names=["gamma", "kappa", "z_peak", "delay_slope"],
        triangle="lower",
        marginal=True,
        visuals={"divergence": True},
        backend="matplotlib",
        figure_kwargs={"figsize": (11, 11), "layout": "constrained"},
    )
    delayed_pair_figure = delayed_pair_plots.viz["figure"].item()
    delayed_pair_figure.suptitle(
        "SMOKE RUN: delayed pair plot"
        if SMOKE_RUN
        else "Time-delayed posterior pair plot"
    )
    delayed_rhat_max = float(delayed_summary_table["r_hat"].max())
    delayed_ess_min = float(delayed_summary_table[["ess_bulk", "ess_tail"]].min().min())
    delayed_converged = (
        delayed_rhat_max <= 1.01 and delayed_ess_min >= 400 and delayed_divergences == 0
    )
    delayed_convergence_message = (
        mo.md(
            "**Convergence checks passed:** R-hat <= 1.01, bulk/tail ESS >= 400, no divergences."
        )
        if delayed_converged
        else mo.callout(
            f"Convergence checks have not passed: max R-hat={delayed_rhat_max:.3f}, "
            f"min bulk/tail ESS={delayed_ess_min:.1f}, divergences={delayed_divergences}. "
            "Intervals, density bands and local rates are provisional; inspect the traces "
            "and try longer adaptation or alternative initial values before interpretation.",
            kind="warn",
        )
    )
    mo.vstack(
        [
            mo.md(
                f"**{'SMOKE RUN: execution check only' if SMOKE_RUN else 'Time-delayed diagnostics'}**"
                f"\n\nDivergences: **{delayed_divergences}**. ArviZ reports 90% equal-tail "
                "intervals, bulk/tail ESS, rank-normalized split R-hat and Monte Carlo errors. "
                "Inspect mixing and formation-shape/delay degeneracies before interpreting results."
            ),
            delayed_convergence_message,
            mo.ui.table(delayed_summary_table.reset_index(names="parameter")),
            delayed_trace_figure,
            delayed_pair_figure,
        ]
    )
    return (delayed_chain_samples,)


@app.cell(hide_code=True)
def delayed_products(
    delay_quadrature_nodes,
    delayed_chain_samples,
    delayed_posterior_products,
    maximum_formation_redshift,
    maximum_redshift: float,
    minimum_delay_gyr,
    minimum_redshift: float,
    n_grid,
    observed,
):
    delayed_theta_samples = jnp.stack(
        [
            delayed_chain_samples[_name].reshape(-1)
            for _name in ("gamma", "kappa", "z_peak", "delay_slope")
        ],
        axis=-1,
    )
    delayed_log_likelihoods, delayed_densities, delayed_unit_rates = (
        delayed_posterior_products(
            delayed_theta_samples,
            observed,
            minimum_redshift,
            maximum_redshift,
            n_grid,
            minimum_delay_gyr,
            maximum_formation_redshift,
            delay_quadrature_nodes,
        )
    )
    if not np.isfinite(np.asarray(delayed_log_likelihoods)).all():
        raise ValueError("Delayed posterior log likelihoods contain nonfinite values.")
    delayed_best_index = int(np.argmax(np.asarray(delayed_log_likelihoods)))
    delayed_plot_grid = np.linspace(minimum_redshift, maximum_redshift, n_grid)
    mo.md(
        f"Best sampled delayed fit (highest full-data likelihood): gamma, kappa, "
        f"z_peak, delay_slope = {np.asarray(delayed_theta_samples[delayed_best_index]).tolist()}"
    )
    return (
        delayed_best_index,
        delayed_densities,
        delayed_plot_grid,
        delayed_unit_rates,
    )


@app.cell(hide_code=True)
def delayed_histogram_control():
    delayed_histogram_bins = mo.ui.slider(
        10, 200, value=50, label="Delayed-fit histogram bins"
    )
    delayed_histogram_bins
    return (delayed_histogram_bins,)


@app.cell(hide_code=True)
def delayed_density(
    delayed_best_index,
    delayed_densities,
    delayed_histogram_bins,
    delayed_plot_grid,
    observed,
):
    delayed_density_figure = density_figure(
        observed,
        delayed_plot_grid,
        delayed_densities,
        delayed_best_index,
        delayed_histogram_bins.value,
        SMOKE_RUN,
    )
    delayed_density_figure.axes[0].set_title(
        "SMOKE RUN: time-delayed fit" if SMOKE_RUN else "Time-delayed redshift fit"
    )
    delayed_density_figure
    return


@app.cell(hide_code=True)
def delayed_rate(
    delayed_best_index,
    delayed_unit_rates,
    observation_years: float | None,
    observed,
):
    mo.stop(
        observation_years is None,
        mo.md(
            "Delayed shape results only: set `observation_years` for the local rate."
        ),
    )
    assert observation_years is not None
    if not np.isfinite(observation_years) or observation_years <= 0:
        raise ValueError("Observing duration must be finite and positive in years.")
    if not np.isfinite(np.asarray(delayed_unit_rates)).all() or np.any(
        np.asarray(delayed_unit_rates) <= 0
    ):
        raise ValueError("Delayed unit total rates must be finite and positive.")
    delayed_local_rates = len(observed) / (
        observation_years * SECONDS_PER_YEAR * delayed_unit_rates
    )
    _low, _median, _high = np.quantile(
        np.asarray(delayed_local_rates), [0.05, 0.5, 0.95]
    )
    mo.md(rf"""
    ### {"SMOKE RUN: " if SMOKE_RUN else ""}Time-delayed local merger rate

    Best sampled rate: **{float(delayed_local_rates[delayed_best_index]):.6g}**
    $\mathrm{{Gpc}}^{{-3}}\,\mathrm{{yr}}^{{-1}}$.
    Posterior median: **{_median:.6g}**; equal-tail 90% interval:
    **[{_low:.6g}, {_high:.6g}]** in the same units.

    This uses $N/(T_{{\rm seconds}}\mathcal R_{{\rm unit}}(\theta))$ with the delayed
    unit-local-rate distribution on the same catalog support. The interval includes
    uncertainty in all **four** inferred parameters. Count uncertainty and uncertainty
    in the fixed delay floor, formation cutoff and cosmology are excluded.
    """)
    return


if __name__ == "__main__":
    app.run()
