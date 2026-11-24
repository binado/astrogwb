import marimo

__generated_with = "0.25.0"
app = marimo.App()

with app.setup(hide_code=True):
    # ruff: noqa: PLR1711
    # Marimo generates bare returns when saving output-only cells.
    import time
    from pathlib import Path

    import numpyro

    from astrogwb.runtime import configure_runtime

    # One CPU device per chain, set before any array initializes the XLA backend.
    numpyro.set_host_device_count(4)
    configure_runtime()

    import arviz_base as azb
    import arviz_plots as azp
    import arviz_stats as azs
    import h5py
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numpyro.distributions as dist
    from jax.typing import ArrayLike
    from numpyro.infer import MCMC, NUTS

    from astrogwb.constants import SECONDS_PER_YEAR
    from astrogwb.cosmology import differential_comoving_volume
    from astrogwb.distributions.redshift import MadauDickinsonRedshiftDistribution
    from astrogwb.utils import gauss_legendre_nodes_weights

    ROOT_DIR = Path(__file__).resolve().parents[1]
    PARAMETERS = ("gamma", "kappa", "z_peak")
    H0 = 67.66
    OMEGA_M = 0.3096


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Madau-Dickinson redshift fit

    Fit the Madau-Dickinson shape (`gamma`, `kappa`, `z_peak`) to every redshift
    of a complete, all-sky catalog with no selection effects, at fixed
    $H_0=67.66$ and $\Omega_m=0.3096$. The local merger rate then follows from
    the catalog size, its duration and the fitted shape.

    With the cosmology fixed, the per-event density is

    $$\log p(z_i \mid \theta) = \log\frac{\mathrm{d}V_c/\mathrm{d}z(z_i)}{1+z_i}
    + \log\psi(z_i;\theta) - \log Z(\theta),$$

    $$\psi(z) = \frac{(1+z)^\gamma}{1 + \left(\frac{1+z}{1+z_p}\right)^{\gamma+\kappa}}.$$

    The volume term does not depend on $\theta$ and is dropped, $\log\psi$ is
    closed form per event, and $Z(\theta)$ is a Gauss-Legendre quadrature in
    $u=\log(1+z)$, where $\mathrm{d}z/(1+z)=\mathrm{d}u$ cancels the time
    dilation factor. The volume at the nodes is computed once.
    """)
    return


@app.cell
def _():
    # The ET-0197A-23 description states a one-year catalog on z in [0, 15].
    catalog_path = ROOT_DIR / "data" / "18321_1yrCatalogBNS.h5"
    minimum_redshift = 0.0
    maximum_redshift = 15.0
    observation_years = 1.0
    n_nodes = 64
    # Uniform prior bounds, ordered as PARAMETERS.
    prior_bounds = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0))
    return (
        catalog_path,
        maximum_redshift,
        minimum_redshift,
        n_nodes,
        observation_years,
        prior_bounds,
    )


@app.function
def log_shape(log_one_plus_z: ArrayLike, theta: ArrayLike) -> jax.Array:
    r"""Log of the Madau-Dickinson shape :math:`\psi`, without its normalization.

    Parameters
    ----------
    log_one_plus_z
        :math:`\log(1+z)`.
    theta
        Shape parameters ordered as ``PARAMETERS``.

    Returns
    -------
    jax.Array
        :math:`\gamma u - \mathrm{softplus}((\gamma+\kappa)(u - \log(1+z_p)))`.
    """
    gamma, kappa, z_peak = jnp.asarray(theta)
    u = jnp.asarray(log_one_plus_z)
    return gamma * u - jax.nn.softplus((gamma + kappa) * (u - jnp.log1p(z_peak)))


@app.function
def volume_quadrature(
    minimum: float, maximum: float, n_nodes: int
) -> tuple[jax.Array, jax.Array]:
    r"""Gauss-Legendre nodes in :math:`u=\log(1+z)` with log volume weights.

    Parameters
    ----------
    minimum, maximum
        Catalog redshift support.
    n_nodes
        Number of quadrature nodes.

    Returns
    -------
    tuple of jax.Array
        Nodes :math:`u_j` and :math:`\log(w_j\,\mathrm{d}V_c/\mathrm{d}z(z_j))`
        in :math:`\mathrm{Mpc}^3`, so that
        :math:`Z = \sum_j \exp(\log\psi(u_j) + \log w_j V_j)`.
    """
    nodes, weights = gauss_legendre_nodes_weights(
        jnp.log1p(minimum), jnp.log1p(maximum), n_nodes
    )
    volume = differential_comoving_volume(jnp.expm1(nodes), H0, OMEGA_M)
    return nodes, jnp.log(weights * volume)


@app.function
def log_likelihood(
    theta: ArrayLike,
    log_one_plus_z: ArrayLike,
    nodes: ArrayLike,
    log_node_weights: ArrayLike,
) -> jax.Array:
    r"""Full-catalog log likelihood, up to a theta-independent constant.

    Parameters
    ----------
    theta
        Shape parameters ordered as ``PARAMETERS``.
    log_one_plus_z
        :math:`\log(1+z_i)` for every catalog event.
    nodes, log_node_weights
        Output of :func:`volume_quadrature`.

    Returns
    -------
    jax.Array
        Scalar log likelihood.
    """
    log_one_plus_z = jnp.asarray(log_one_plus_z)
    log_norm = jax.nn.logsumexp(log_shape(nodes, theta) + log_node_weights)
    return (
        jnp.sum(log_shape(log_one_plus_z, theta)) - log_one_plus_z.shape[0] * log_norm
    )


@app.function
def unit_total_rate(
    theta: ArrayLike, nodes: ArrayLike, log_node_weights: ArrayLike
) -> jax.Array:
    r"""Total merger rate per observer-frame year for :math:`\psi(0)=1`.

    Parameters
    ----------
    theta
        Shape parameters ordered as ``PARAMETERS``.
    nodes, log_node_weights
        Output of :func:`volume_quadrature`.

    Returns
    -------
    jax.Array
        Mergers per year per unit local rate in :math:`\mathrm{Gpc^{-3}\,yr^{-1}}`.
    """
    gamma, kappa, z_peak = jnp.asarray(theta)
    # psi(0) = 1 requires the factor 1 + (1 + z_p)^-(gamma + kappa).
    log_local = jax.nn.softplus(-(gamma + kappa) * jnp.log1p(z_peak))
    log_norm = jax.nn.logsumexp(log_shape(nodes, theta) + log_node_weights)
    # Mpc^3 to Gpc^3.
    return 1e-9 * jnp.exp(log_local + log_norm)


@app.function
def redshift_model(
    log_one_plus_z: ArrayLike,
    nodes: ArrayLike,
    log_node_weights: ArrayLike,
    prior_bounds: ArrayLike,
) -> None:
    r"""Analytic catalog likelihood under independent Uniform shape priors.

    Parameters
    ----------
    log_one_plus_z
        :math:`\log(1+z_i)` for every catalog event.
    nodes, log_node_weights
        Output of :func:`volume_quadrature`.
    prior_bounds
        Array of shape ``(3, 2)`` with the lower and upper Uniform bounds.
    """
    bounds = jnp.asarray(prior_bounds)
    theta = jnp.stack(
        [
            numpyro.sample(name, dist.Uniform(*bounds[i]))
            for i, name in enumerate(PARAMETERS)
        ]
    )
    numpyro.factor(
        "loglike", log_likelihood(theta, log_one_plus_z, nodes, log_node_weights)
    )


@app.cell
def _(catalog_path, maximum_redshift, minimum_redshift, n_nodes):
    with h5py.File(catalog_path, "r") as _handle:
        observed = np.asarray(_handle["z"][...], dtype=np.float64)
    log_one_plus_z = jnp.log1p(observed)
    nodes, log_node_weights = volume_quadrature(
        minimum_redshift, maximum_redshift, n_nodes
    )
    mo.md(
        f"Loaded **{observed.size}** redshifts in "
        f"[{observed.min():.4g}, {observed.max():.4g}] from `{catalog_path.name}`."
    )
    return log_node_weights, log_one_plus_z, nodes, observed


@app.cell
def _(
    log_node_weights,
    log_one_plus_z,
    maximum_redshift,
    minimum_redshift,
    nodes,
    observed,
):
    # Check against the library's grid-interpolated distribution at two thetas.
    def _library(theta):
        return MadauDickinsonRedshiftDistribution(
            params={"H0": H0, "Omega_m": OMEGA_M, **dict(zip(PARAMETERS, theta))},
            minimum_redshift=minimum_redshift,
            maximum_redshift=maximum_redshift,
        )

    _thetas = (jnp.array([2.7, 2.9, 1.9]), jnp.array([1.5, 5.0, 3.0]))
    _rate_errors = [
        float(
            unit_total_rate(_theta, nodes, log_node_weights)
            / (SECONDS_PER_YEAR * _library(_theta).total_merger_rate())
            - 1
        )
        for _theta in _thetas
    ]
    _analytic, _reference = (
        [
            float(log_likelihood(_t, log_one_plus_z, nodes, log_node_weights))
            for _t in _thetas
        ],
        [float(jnp.sum(_library(_t).log_prob(observed))) for _t in _thetas],
    )
    _gradient = jax.jit(jax.grad(log_likelihood))
    _gradient(_thetas[0], log_one_plus_z, nodes, log_node_weights).block_until_ready()
    _start = time.perf_counter()
    for _ in range(20):
        _gradient(
            _thetas[0], log_one_plus_z, nodes, log_node_weights
        ).block_until_ready()
    _milliseconds = 1e3 * (time.perf_counter() - _start) / 20
    mo.md(
        f"Relative error of the unit total rate vs the library: "
        f"{_rate_errors[0]:.2e}, {_rate_errors[1]:.2e}. "
        f"Log-likelihood difference between the two thetas, analytic minus "
        f"library: {(_analytic[0] - _analytic[1]) - (_reference[0] - _reference[1]):.3g}. "
        f"Gradient cost: **{_milliseconds:.2f} ms**."
    )
    return


@app.cell
def _(log_node_weights, log_one_plus_z, nodes, prior_bounds):
    sampler = MCMC(
        NUTS(redshift_model, target_accept_prob=0.9, dense_mass=True),
        num_chains=4,
        num_warmup=200,
        num_samples=1000,
        chain_method="parallel",
    )
    _start = time.perf_counter()
    sampler.run(
        jax.random.key(42),
        log_one_plus_z=log_one_plus_z,
        nodes=nodes,
        log_node_weights=log_node_weights,
        prior_bounds=jnp.asarray(prior_bounds),
    )
    mo.md(f"Sampled in **{time.perf_counter() - _start:.1f} s**.")
    return (sampler,)


@app.cell(hide_code=True)
def _(sampler):
    inference_data = azb.from_numpyro(sampler, log_likelihood=False)
    summary_table = azs.summary(
        inference_data,
        var_names=list(PARAMETERS),
        ci_prob=0.9,
        ci_kind="eti",
        round_to="none",
    )
    divergences = int(np.asarray(sampler.get_extra_fields()["diverging"]).sum())
    trace_figure = (
        azp.plot_trace(
            inference_data,
            var_names=list(PARAMETERS),
            backend="matplotlib",
            figure_kwargs={"figsize": (9, 7), "layout": "constrained"},
        )
        .viz["figure"]
        .item()
    )
    pair_figure = (
        azp.plot_pair(
            inference_data,
            var_names=list(PARAMETERS),
            triangle="lower",
            marginal=True,
            visuals={"divergence": True},
            backend="matplotlib",
            figure_kwargs={"figsize": (9, 9), "layout": "constrained"},
        )
        .viz["figure"]
        .item()
    )
    mo.vstack(
        [
            mo.md(
                f"### Posterior diagnostics\n\nDivergences: **{divergences}**. "
                "Intervals are equal-tail 90%."
            ),
            mo.ui.table(summary_table.reset_index(names="parameter")),
            trace_figure,
            pair_figure,
        ]
    )
    return


@app.cell
def _(
    log_node_weights,
    log_one_plus_z,
    maximum_redshift,
    minimum_redshift,
    nodes,
    sampler,
):
    _samples = sampler.get_samples()
    theta_samples = jnp.stack([_samples[_name] for _name in PARAMETERS], axis=-1)
    log_likelihoods = jax.vmap(log_likelihood, in_axes=(0, None, None, None))(
        theta_samples, log_one_plus_z, nodes, log_node_weights
    )
    unit_rates = jax.vmap(unit_total_rate, in_axes=(0, None, None))(
        theta_samples, nodes, log_node_weights
    )
    best_index = int(jnp.argmax(log_likelihoods))
    # Normalized densities on a plotting grid, sharing the quadrature normalization.
    plot_grid = np.linspace(minimum_redshift, maximum_redshift, 500)
    _log_volume = jnp.log(
        differential_comoving_volume(plot_grid, H0, OMEGA_M) / (1 + plot_grid)
    )
    densities = jax.vmap(
        lambda _theta: jnp.exp(
            log_shape(jnp.log1p(plot_grid), _theta)
            + _log_volume
            - jax.nn.logsumexp(log_shape(nodes, _theta) + log_node_weights)
        )
    )(theta_samples)
    mo.md(
        "Highest-likelihood draw: "
        + ", ".join(
            f"{_name} = {float(_value):.4g}"
            for _name, _value in zip(PARAMETERS, theta_samples[best_index], strict=True)
        )
    )
    return best_index, densities, plot_grid, unit_rates


@app.cell(hide_code=True)
def _():
    histogram_bins = mo.ui.slider(10, 200, value=50, label="Histogram bins")
    histogram_bins
    return (histogram_bins,)


@app.cell(hide_code=True)
def _(best_index, densities, histogram_bins, observed, plot_grid):
    _density = np.asarray(densities)
    _lower, _upper = np.quantile(_density, [0.05, 0.95], axis=0)
    _figure, _axis = plt.subplots()
    _axis.hist(observed, bins=histogram_bins.value, density=True, alpha=0.35)
    _axis.fill_between(plot_grid, _lower, _upper, alpha=0.25, label="90% band")
    _axis.plot(plot_grid, _density[best_index], label="Highest-likelihood draw")
    _axis.set(
        xlabel="Redshift",
        ylabel="Probability density",
        xlim=(plot_grid[0], plot_grid[-1]),
    )
    _axis.legend()
    _figure
    return


@app.cell(hide_code=True)
def _(observation_years, observed, unit_rates):
    local_rates = np.asarray(observed.size / (observation_years * unit_rates))
    _low, _median, _high = np.quantile(local_rates, [0.05, 0.5, 0.95])
    mo.md(rf"""
    ### Local merger rate

    $$R_0(\theta) = \frac{{N}}{{T\,\mathcal R_{{\rm unit}}(\theta)}}$$

    Median **{_median:.4g}**, equal-tail 90% interval
    **[{_low:.4g}, {_high:.4g}]** $\mathrm{{Gpc}}^{{-3}}\,\mathrm{{yr}}^{{-1}}$.
    The interval reflects shape uncertainty only; the count $N$ is fixed.
    """)
    return


if __name__ == "__main__":
    app.run()
