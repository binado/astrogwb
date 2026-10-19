import marimo

__generated_with = "0.25.0"
app = marimo.App()

with app.setup(hide_code=True):
    import dataclasses
    import os
    import time
    from collections.abc import Callable, Mapping, Sequence
    from functools import partial
    from pathlib import Path

    from astrogwb.paper.runtime import configure_runtime

    # One host device per chain, so the chains run in parallel. This precedes
    # waveform construction and array creation: the XLA backend is fixed once
    # it starts.
    NUM_CHAINS = 4
    _, CHAIN_METHOD = configure_runtime(num_chains=NUM_CHAINS)

    import arviz_stats
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    import numpyro.distributions as dist
    import pandas as pd
    import xarray
    from arviz_base import from_numpyro
    from matplotlib.axes import Axes
    from matplotlib.projections import register_projection
    from numpyro.distributions.transforms import biject_to
    from numpyro.handlers import condition
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.util import initialize_model
    from xarray import DataTree

    from astrogwb import __version__
    from astrogwb.inference import (
        ImportanceGaussianLikelihood,
        gwb_likelihood_model,
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
        plot_corner,
        plot_network_marginals,
        save_figures,
        use_paper_style,
    )
    from astrogwb.paper.posteriors import (
        ensure_reference_catalog,
        injection_spectrum,
        network_data,
        noise_digest,
        settings_digest,
        shot_noise_variant,
    )
    from astrogwb.populations import build_population
    from astrogwb.simulators.polarization_power import CatalogMetadata
    from astrogwb.simulators.spectra import BackgroundSpectralDensityMetadata


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # NUTS posteriors on the Madau-Dickinson parameters

    Posteriors on the shape of the merger-rate history,
    $(z_{\mathrm{peak}}, \gamma, \kappa)$ (*Inferring astrophysical
    parameters*), over all six detector networks. A grid over three shape
    parameters for six networks scales badly, so this notebook samples with
    **NUTS** instead; `cosmology_grid_posteriors.py` covers $H_0$,
    $(H_0, \Omega_m)$ and $(\Xi_0, n)$.

    The data are the same **zero-noise Poisson injection** as there: one
    Poisson-count spectrum drawn at the fiducials, with no Gaussian noise
    added, so it carries the physical shot noise of one catalog that the
    likelihood models. The model spectrum is the same **rescaled
    reference-redshift catalog**, drawn at a different seed.

    Setting `ASTROGWB_NOTEBOOK_SMOKE=1` swaps in a tiny catalog and short
    chains so the whole notebook runs in a few minutes.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    **Model.** Every site in `[priors]` is a NumPyro sample site, and the
    ones that are not sampled are pinned at their fiducial with
    `numpyro.handlers.condition`. A pinned site adds a constant prior term,
    exactly as in `grid_log_posterior`. Here $z_{\mathrm{peak}}$, $\gamma$ and
    $\kappa$ are sampled; $H_0$, $\Omega_m$, $\Xi_0$, $n$, the local merger
    rate and the mass parameters stay at their fiducials.

    **No reweighting.** None of the sampled parameters is intrinsic, so the
    reference draws are not importance-reweighted: the weights are constant
    and the relative effective sample size stays at one. The merger-rate
    history only moves the redshift quadrature weights of the rescaled
    spectrum, which is a smooth, differentiable function of the sampled
    parameters.

    **Likelihood.** One knob, `likelihood_variant`, in *Configuration*:

    - `"detector"`: the diagonal Gaussian, with no shot noise;
    - `"amplitude"`: the rank-one shot-noise covariance with $s^2$ evaluated
      at every point, which differentiates through the catalog projection
      (nested automatic differentiation through a `jax.vjp`);
    - `"fixed"`: the same covariance with $s^2$ held at the fiducial. It is
      the fallback if the gradient of `"amplitude"` is too slow: the potential
      is then cheaper and has no nested AD.

    **Sampler.** NUTS with four chains on four host devices. The likelihood is
    a model keyword argument and `jit_model_args=True`, so the catalog is a
    traced input of the compiled potential and not a constant baked into it;
    the six networks share one structure and so one compilation. Each chain
    starts at the fiducial plus Gaussian jitter in unconstrained space: that
    spreads the chains enough for $\hat R$ to mean something, and avoids prior
    draws far from the fiducial, where the spectrum barely changes and NUTS
    would stall on the flat $U(-10, 10)$ priors.

    **Acceptance.** $\hat R < 1.01$ and no divergences, for every network.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configuration
    """)


@app.cell
def _():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    FIGURES = ROOT_DIR / FIGURES_DIR / "madau_dickinson_mcmc"

    # Different seeds: the injection must not be a subset of the catalog.
    data_seed = 41
    catalog_seed = 42
    # Intrinsic draws on a scrambled Sobol net, one waveform each; the same
    # catalog as the grid notebook.
    num_samples = 2**14
    redshift_nodes = 32  # Gauss-Legendre nodes in 1/(1 + z); costs no waveforms
    catalog_chunk_size = 4096  # reference waveforms per lax.map batch
    chunk_size = 16_384  # injection waveforms per reduced chunk

    observation_time = 1.0  # years
    minimum_frequency = 2.0
    maximum_frequency = 2048.0
    fiducial_network = "ET-2L-aligned-CE-Hanford"
    network_names = [
        fiducial_network
    ]  # all six: [name for name, _ in DETECTOR_NETWORKS]

    # "detector" | "amplitude" | "fixed": the likelihood every run uses.
    likelihood_variant = "fixed"
    sampled = ("z_peak", "gamma", "kappa")
    # The chain count is fixed at the top of the notebook, before the backend
    # starts.
    num_chains, num_warmup, num_samples_mcmc = NUM_CHAINS, 2000, 2000
    target_accept_prob = 0.95
    dense_mass = True  # full mass matrix: the shape parameters are correlated
    mcmc_seed = 44
    init_jitter = 0.05  # sd of the per-chain init jitter, in unconstrained space

    SMOKE = os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") == "1"
    write_figures_default = True
    cache_name_prefix = ""
    if SMOKE:
        num_samples = 64
        redshift_nodes = 8
        catalog_chunk_size = 512
        chunk_size = 512
        num_warmup = num_samples_mcmc = 50
        write_figures_default = False
        cache_name_prefix = "smoke_"

    cache_dir = default_cache_dir() / "posteriors"
    FIDUCIALS = fiducials(root=ROOT_DIR)
    # The shared U(-10, 10) priors on gamma and kappa are restricted to positive
    # values for this notebook; every other prior is the shared one.
    positive = {"dist": "Uniform", "kwargs": {"low": 0.0, "high": 10.0}}
    PRIORS = priors(root=ROOT_DIR, gamma=positive, kappa=positive)
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
        cache_dir,
        cache_name_prefix,
        catalog_chunk_size,
        catalog_metadata,
        catalog_seed,
        chunk_size,
        data_seed,
        fiducial_network,
        frequencies,
        init_jitter,
        injection_metadata,
        likelihood_variant,
        maximum_frequency,
        mcmc_seed,
        minimum_frequency,
        network_names,
        num_chains,
        num_samples_mcmc,
        num_warmup,
        observation_time,
        population,
        redshift_nodes,
        registry,
        sampled,
        target_accept_prob,
        dense_mass,
        write_figures,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Toolbox
    """)


@app.function(hide_code=True)
def initial_params(
    key: jax.Array,
    priors: Mapping[str, dist.Distribution],
    fiducials: Mapping[str, float],
    sampled: Sequence[str],
    num_chains: int,
    jitter: float,
) -> dict[str, jax.Array]:
    """Per-chain starting points: the fiducial plus Gaussian jitter.

    The jitter is added in unconstrained space, through the bijection of each
    prior's support, so every start is inside the prior.

    Parameters
    ----------
    key
        PRNG key; one subkey per site.
    priors
        Prior of every site.
    fiducials
        Value of every site.
    sampled
        The sampled sites.
    num_chains
        Number of starting points.
    jitter
        Standard deviation of the jitter.

    Returns
    -------
    dict
        Unconstrained values by site, each of shape ``(num_chains,)``.
    """
    inits = {}
    for subkey, name in zip(jax.random.split(key, len(sampled)), sampled, strict=True):
        center = biject_to(priors[name].support).inv(jnp.asarray(fiducials[name]))
        inits[name] = center + jitter * jax.random.normal(subkey, (num_chains,))
    return inits


@app.function(hide_code=True)
def run_nuts(
    model: Callable[..., None],
    likelihood: ImportanceGaussianLikelihood,
    key: jax.Array,
    init_params: Mapping[str, jax.Array],
    *,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    target_accept_prob: float,
    dense_mass: bool,
    chain_method: str,
    progress_bar: bool,
) -> DataTree:
    """NUTS draws of ``model`` under ``likelihood``, as an ArviZ DataTree.

    The likelihood is a model keyword argument and ``jit_model_args=True``, so
    its arrays are traced inputs of the compiled potential and not constants
    baked into it. Likelihoods that share a structure share a compilation.

    Parameters
    ----------
    model
        A NumPyro model taking ``likelihood`` as a keyword argument.
    likelihood
        The likelihood passed to ``model``.
    key
        PRNG key of the run.
    init_params
        Unconstrained starting values by site, each of shape ``(num_chains,)``.
    num_warmup, num_samples, num_chains, target_accept_prob, dense_mass
        NUTS settings.
    chain_method
        NumPyro's ``chain_method``.
    progress_bar
        Whether to draw the progress bar.

    Returns
    -------
    xarray.DataTree
        With ``posterior`` and ``sample_stats`` groups.
    """
    mcmc = MCMC(
        NUTS(model, target_accept_prob=target_accept_prob, dense_mass=dense_mass),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        jit_model_args=True,
        progress_bar=progress_bar,
    )
    mcmc.run(key, likelihood=likelihood, init_params=dict(init_params))
    return from_numpyro(mcmc)


@app.function(hide_code=True)
def cached_trace(path: Path, compute: Callable[[], DataTree]) -> DataTree:
    """The trace in ``path``, computed once.

    The file records none of the settings behind it, so the caller puts a
    :func:`settings_digest` in the file name.

    Parameters
    ----------
    path
        NetCDF path.
    compute
        Called on a miss.

    Returns
    -------
    xarray.DataTree
        Loaded into memory, so the file is closed.
    """
    if path.is_file():
        return xarray.open_datatree(path, engine="h5netcdf").load()
    trace = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_netcdf(path, engine="h5netcdf")
    return trace


@app.function(hide_code=True)
def convergence_rows(
    traces: Mapping[str, DataTree], sampled: Sequence[str]
) -> list[dict[str, object]]:
    """R-hat, effective sample sizes and divergences, per network and parameter.

    Returns
    -------
    list of dict
        One row per (network, parameter), ready for :class:`pandas.DataFrame`.
    """
    label = dict(DETECTOR_NETWORKS)
    rows: list[dict[str, object]] = []
    for network, trace in traces.items():
        diagnostics = arviz_stats.summary(
            trace, var_names=list(sampled), kind="diagnostics"
        )
        divergences = int(trace["sample_stats"].dataset["diverging"].sum())
        for name in sampled:
            rows.append(
                {
                    "network": label[network],
                    "parameter": name,
                    "r_hat": float(diagnostics.loc[name, "r_hat"]),
                    "ess_bulk": float(diagnostics.loc[name, "ess_bulk"]),
                    "ess_tail": float(diagnostics.loc[name, "ess_tail"]),
                    "divergences": divergences,
                }
            )
    return rows


@app.function(hide_code=True)
def summary_rows(
    traces: Mapping[str, DataTree], sampled: Sequence[str], prob: float = 0.6827
) -> list[dict[str, object]]:
    """Median and highest-density interval of each parameter, per network.

    Returns
    -------
    list of dict
        One row per (network, parameter), ready for :class:`pandas.DataFrame`.
    """
    label = dict(DETECTOR_NETWORKS)
    rows: list[dict[str, object]] = []
    for network, trace in traces.items():
        hdi = arviz_stats.hdi(trace, prob=prob, var_names=list(sampled))
        for name in sampled:
            lower, upper = (float(v) for v in hdi[name].values)
            draws = trace["posterior"].dataset[name].values
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

    The reference catalog generates every intrinsic draw once, at the
    window's lower edge, and is read from `outputs/catalogs/` afterwards. It
    is built with `density_sites=()` and the amplitude mode, which holds the
    squared power so that every variant can be switched to afterwards. The
    injection is one Poisson spectrum, cached beside the traces.
    """)


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
    # copy made by the likelihood exists.
    catalog_data = ensure_reference_catalog(
        catalog_metadata, catalog_seed, CATALOGS_ROOT, chunk_size=catalog_chunk_size
    )
    print(
        f"reference catalog: {catalog_metadata.num_samples:,} draws, "
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
def _(
    cache_dir,
    cache_name_prefix,
    chunk_size,
    data_seed,
    frequencies,
    injection_metadata,
):
    observed, _injection_frequencies, _num_injected = injection_spectrum(
        injection_metadata,
        data_seed,
        cache_dir
        / f"{cache_name_prefix}injection_{injection_metadata.key()}_{data_seed}.npz",
        chunk_size=chunk_size,
    )
    validate_matching_frequency_grids(_injection_frequencies, frequencies)
    print(f"injection: {_num_injected:,} Poisson events")
    return (observed,)


@app.cell
def _(
    FIDUCIALS,
    catalog_metadata,
    catalog_seed,
    data_seed,
    frequencies,
    injection_metadata,
    likelihood,
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
    # Everything a cached trace depends on besides its own sampler settings.
    cache_settings = {
        "version": __version__,
        "networks": network_names,
        "noise": noise_digest(per_network),
        "seeds": (data_seed, catalog_seed),
        "band": (minimum_frequency, maximum_frequency),
        "observation_time": observation_time,
        "density_sites": (),
        "injection": injection_metadata.key(),
        "reference_catalog": catalog_metadata.key(),
        "redshift_nodes": redshift_nodes,
        # The amplitude mode's s^2: the catalog projected on each template.
        "relative_variance_mode": "projected",
    }
    # The fiducial s^2 per network; only the "fixed" variant uses it.
    fiducial_relative_variance = np.array(
        [
            float(
                likelihood.relative_variance(
                    FIDUCIALS,
                    LikelihoodNetwork(
                        jnp.asarray(data["scale"]), jnp.asarray(data["frequency_mask"])
                    ),
                )
            )
            for data in per_network.values()
        ]
    )
    labelled_networks = [(n, l) for n, l in DETECTOR_NETWORKS if n in network_names]
    return (
        cache_settings,
        fiducial_relative_variance,
        labelled_networks,
        per_network,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model and per-network likelihoods

    One model, `condition(gwb_likelihood_model, fixed)`, with every
    non-sampled site pinned at its fiducial. Each network gets the variant's
    likelihood with its own data: the observed spectrum, the Gaussian scale and
    usable-bin mask, and (for `"fixed"`) its fiducial $s^2$.
    """)


@app.cell
def _(
    FIDUCIALS,
    PRIORS,
    fiducial_relative_variance,
    likelihood,
    likelihood_variant,
    observed,
    per_network,
    sampled,
):
    variant_likelihood, _s2, variant_settings = shot_noise_variant(
        likelihood_variant, likelihood, fiducial_relative_variance
    )
    fixed = {name: value for name, value in FIDUCIALS.items() if name not in sampled}
    model = condition(partial(gwb_likelihood_model, priors=PRIORS), data=fixed)
    network_likelihoods = {
        name: dataclasses.replace(
            variant_likelihood,
            observed=jnp.asarray(observed),
            network=LikelihoodNetwork(
                data["scale"],
                data["frequency_mask"],
                None if _s2 is None else jnp.asarray(_s2[index]),
            ),
        )
        for index, (name, data) in enumerate(per_network.items())
    }
    return model, network_likelihoods, variant_settings


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Gradient check

    NUTS needs the gradient of the potential. For `"amplitude"` that is a
    nested derivative: the shot-noise variance is built from a `jax.vjp` of
    the catalog interpolation, which `jax.grad` then differentiates again. The
    cell checks the gradient at the fiducial for the fiducial network is
    finite, and times one value-and-gradient call (after warm-up), which is the
    cost of one leapfrog step.
    """)


@app.cell
def _(
    FIDUCIALS,
    PRIORS,
    fiducial_network,
    likelihood_variant,
    mcmc_seed,
    model,
    network_likelihoods,
    sampled,
):
    _lk = network_likelihoods[fiducial_network]
    _info = initialize_model(
        jax.random.key(mcmc_seed), model, model_kwargs={"likelihood": _lk}
    )
    _z = {
        name: biject_to(PRIORS[name].support).inv(jnp.asarray(FIDUCIALS[name]))
        for name in sampled
    }
    _value_and_grad = jax.jit(jax.value_and_grad(_info.potential_fn))
    _value, _grad = jax.block_until_ready(_value_and_grad(_z))
    assert np.isfinite(float(_value)), "potential is not finite at the fiducial"
    assert all(np.isfinite(float(g)) for g in _grad.values()), "gradient not finite"
    _start = time.perf_counter()
    _repeats = 10
    for _ in range(_repeats):
        jax.block_until_ready(_value_and_grad(_z))
    _seconds = (time.perf_counter() - _start) / _repeats
    mo.md(
        f"`{likelihood_variant}` at the fiducial for `{fiducial_network}`: "
        f"potential {float(_value):.3f}, gradient "
        f"{ {k: round(float(v), 3) for k, v in _grad.items()} }, "
        f"{1e3 * _seconds:.1f} ms per value-and-gradient call."
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sampling

    One NUTS run per network, each seeded from `fold_in(key(mcmc_seed), i)`.
    A trace is cached under `default_cache_dir() / "posteriors"`, keyed by a
    hash of the data, the likelihood and the sampler settings, so a repeat run
    reads it back instead of resampling.
    """)


@app.cell
def _(
    FIDUCIALS,
    PRIORS,
    cache_dir,
    cache_name_prefix,
    cache_settings,
    init_jitter,
    mcmc_seed,
    model,
    network_likelihoods,
    num_chains,
    num_samples_mcmc,
    num_warmup,
    sampled,
    target_accept_prob,
    dense_mass,
    variant_settings,
):
    _key = jax.random.key(mcmc_seed)
    _inits = initial_params(
        jax.random.fold_in(_key, len(network_likelihoods)),
        PRIORS,
        FIDUCIALS,
        sampled,
        num_chains,
        init_jitter,
    )
    _sampler = {
        "num_warmup": num_warmup,
        "num_samples": num_samples_mcmc,
        "num_chains": num_chains,
        "target_accept_prob": target_accept_prob,
        "dense_mass": dense_mass,
    }
    traces = {}
    for _index, (_name, _lk) in enumerate(network_likelihoods.items()):
        _digest = settings_digest(
            run="madau_dickinson",
            network=_name,
            sampled=sampled,
            priors={n: (float(PRIORS[n].low), float(PRIORS[n].high)) for n in sampled},
            sampler=_sampler,
            inits=(init_jitter, mcmc_seed),
            seed=mcmc_seed,
            **cache_settings,
            **variant_settings,
        )
        traces[_name] = cached_trace(
            cache_dir / f"{cache_name_prefix}madau_dickinson_{_name}_{_digest}.nc",
            partial(
                run_nuts,
                model,
                _lk,
                jax.random.fold_in(_key, _index),
                _inits,
                chain_method=CHAIN_METHOD,
                progress_bar=os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") != "1",
                **_sampler,
            ),
        )
    return (traces,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Convergence

    $\hat R$, bulk and tail effective sample sizes, and the divergence count
    of every run, against the acceptance targets: $\hat R < 1.01$ and no
    divergences.
    """)


@app.cell
def _(sampled, traces):
    convergence = pd.DataFrame(convergence_rows(traces, sampled))
    _worst = convergence["r_hat"].max()
    _divergent = convergence.loc[convergence["divergences"] > 0, "network"].unique()
    _passed = _worst < 1.01 and len(_divergent) == 0
    print(
        f"{'PASS' if _passed else 'FAIL'}: max R-hat {_worst:.4f} (target < 1.01), "
        f"{int(np.sum(convergence.drop_duplicates('network')['divergences'].to_numpy()))} "
        f"divergences" + (f" in {', '.join(_divergent)}" if len(_divergent) else "")
    )
    convergence


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Posteriors
    """)


@app.cell
def _(FIDUCIALS, PRIORS, labelled_networks, sampled, traces):
    marginals = {
        name: plot_network_marginals(
            name,
            traces,
            labelled_networks,
            FIDUCIALS[name],
            bounds=(float(PRIORS[name].low), float(PRIORS[name].high)),
        )
        for name in sampled
    }
    mo.vstack([mo.as_html(figure) for figure in marginals.values()])
    return (marginals,)


@app.cell
def _(FIDUCIALS, fiducial_network, sampled, traces):
    corner = plot_corner(
        traces[fiducial_network], {name: FIDUCIALS[name] for name in sampled}
    )
    mo.as_html(corner)
    return (corner,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Summary and output

    Median and 68% HDI of every sampled parameter, by network. Figures and the
    table are written when the switch in *Configuration* is on.
    """)


@app.cell
def _(sampled, traces):
    summary = pd.DataFrame(summary_rows(traces, sampled))
    summary
    return (summary,)


@app.cell
def _(FIGURES, corner, marginals, summary, write_figures):
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


if __name__ == "__main__":
    app.run()
