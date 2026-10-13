import marimo

__generated_with = "0.25.1"
app = marimo.App()

with app.setup(hide_code=True):
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


@app.cell
def _():
    # Editable catalog settings; support bounds were supplied by the catalog owner.
    catalog_path: Path | None = Path(
        "/Users/binado/work/research/phd/astrogwb/data/18321_1yrCatalogBNS.h5"
    )
    redshift_dataset = "z"  # This catalog uses "z"; standard catalogs use "redshift".
    minimum_redshift: float | None = 0.0
    maximum_redshift: float | None = 15.0
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
def _(catalog_path, maximum_redshift, minimum_redshift, redshift_dataset):
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
def _(maximum_redshift, minimum_redshift, n_grid, observed, prior_bounds, run_fit):
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
def _(chain_samples, maximum_redshift, minimum_redshift, n_grid, observed):
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


@app.cell(hide_code=True)
def _(best_index, observation_years, observed, unit_rates):
    mo.stop(
        observation_years is None,
        mo.md(
            "Shape results only: set `observation_years` to calculate the local rate."
        ),
    )
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


if __name__ == "__main__":
    app.run()
