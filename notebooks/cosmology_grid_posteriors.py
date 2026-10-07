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
    from astrogwb.importance.diagnostics import relative_ess
    from astrogwb.importance.spectral import build_importance_spectrum
    from astrogwb.inference import GaussianGWBBatchedLikelihood
    from astrogwb.paper.cache import default_cache_dir
    from astrogwb.paper.catalogs import (
        ensure_catalog,
        validate_matching_frequency_grids,
    )
    from astrogwb.paper.config import (
        detector_registry,
        fiducials,
        population_metadata,
        priors,
        waveform_metadata,
    )
    from astrogwb.paper.config.detectors import DetectorRegistry
    from astrogwb.paper.config.mcmc import DEFAULT_DENSITY_SITES
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
    from astrogwb.simulators.core import batch_keys
    from astrogwb.simulators.polarization_power import CatalogMetadata
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
    spectrum drawn at the fiducials, with no Gaussian noise added on top. The
    model spectrum is a **1,000,000-source importance-reweighted proposal**
    drawn at the same fiducials, large enough that its shot noise is
    negligible. The two are drawn at different seeds, so the injection is
    not a subset of the proposal.

    Three problems, each over all six detector networks, each in its own
    section below:

    1. $H_0$ alone (*Inferring cosmological parameters*, H0).
    2. $H_0$ and $\Omega_m$ (*Inferring cosmological parameters*, H0 and Omega_m).
    3. $\Xi_0$ and $n$ (*Inferring modified propagation parameters*), with
       $H_0$ fixed.

    Setting `ASTROGWB_NOTEBOOK_SMOKE=1` swaps in a tiny proposal and coarse
    grids so the whole notebook runs in a minute or two.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    **Likelihood.** In each bin the observed spectrum is Gaussian about the
    predicted one with scale $\sigma_i = S_{\mathrm{eff},i}/\sqrt{2T\Delta f_i}$.
    The network only changes $\sigma$ (and which bins are usable), so each grid
    point is predicted once and all six networks' likelihoods are evaluated on
    that one prediction.

    **Prediction.** The proposal's polarization power is reweighted to the
    target population at each grid point,
    $\log w_i = \log p(x_i\mid\theta) - \log q(x_i) - 2\,[\log d_L(z_i\mid\theta) - \log d_i^{\mathrm{ref}}]$,
    so $\Xi_0$ and $n$ enter through $d_L$ and $H_0$, $\Omega_m$ through both
    the density and $d_L$.

    **Posterior.** The model samples every parameter in `[priors]`, so the
    prior is in the log density. Parameters that are not swept are pinned at
    their fiducial, which adds a constant. Plots normalize on the grid.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configuration
    """)


@app.cell
def _():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    FIGURES = ROOT_DIR / FIGURES_DIR / "cosmology_grid_posteriors"

    # Different seeds: the injection must not be a subset of the proposal.
    data_seed = 41
    proposal_seed = 42
    proposal_size = 1_000_000
    chunk_size = 16_384  # waveform sources per generated chunk
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

    SMOKE = os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") == "1"
    write_figures_default = True
    cache_name_prefix = ""
    if SMOKE:
        proposal_size = 2048
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

    # The injection and the proposal share the waveform (hence the frequency
    # grid) and the cosmological population; they differ in seed and count.
    waveform = waveform_metadata(root=ROOT_DIR)
    population = population_metadata(root=ROOT_DIR)
    proposal_metadata = CatalogMetadata(
        waveform=waveform,
        population=population,
        fiducials=FIDUCIALS,
        num_samples=proposal_size,
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
        chunk_size,
        data_seed,
        fiducial_network,
        grid_chunk_size,
        grid_specs,
        injection_metadata,
        maximum_frequency,
        minimum_frequency,
        network_names,
        observation_time,
        population,
        proposal_metadata,
        proposal_seed,
        registry,
        write_figures,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Toolbox
    """)


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
    log_density_fn: GaussianGWBBatchedLikelihood,
    axes: Mapping[str, tuple[float, float, int]],
    fixed: Mapping[str, float],
    observed: NDArray[np.float64],
    per_network: Mapping[str, dict[str, object]],
) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
    """Evaluate one problem's log density on its grid, for every network.

    Parameters
    ----------
    log_density_fn
        The problem's evaluator: one prediction per grid point, shared by all
        networks, which differ only in ``scale`` and ``frequency_mask``.
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
    evaluator: GaussianGWBBatchedLikelihood,
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
        The problem's :class:`GaussianGWBBatchedLikelihood`.
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
    ## Proposal and injection

    The proposal is the 1M-source catalog, generated chunk by chunk on a miss
    and read from `outputs/catalogs/` afterwards. The injection is one
    Poisson spectrum, cached beside the posteriors.
    """)


@app.cell
def _(proposal_metadata, proposal_seed):
    proposal_data, proposal_metadata_loaded = ensure_catalog(
        proposal_metadata, np.uint64(proposal_seed), CATALOGS_ROOT
    )
    frequencies = np.asarray(proposal_data["frequencies"])
    print(
        f"proposal: {proposal_data['polarization_power'].shape[1]:,} sources, "
        f"{frequencies.size} frequency bins"
    )
    return frequencies, proposal_data, proposal_metadata_loaded


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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Importance spectrum

    Built once. The population is built a single time because it hashes by
    identity. Each problem builds its own `GaussianGWBBatchedLikelihood` in
    its section, which predicts every grid point once for all six networks.
    """)


@app.cell
def _(population, proposal_data, proposal_metadata_loaded):
    target = build_population(population.model_name, **population.model_kwargs)
    spectral_density_fn, log_weights_fn = build_importance_spectrum(
        proposal_data,
        proposal_metadata_loaded,
        population=target,
        density_sites=DEFAULT_DENSITY_SITES,
    )
    return log_weights_fn, spectral_density_fn


@app.cell
def _(
    data_seed,
    frequencies,
    injection_metadata,
    maximum_frequency,
    minimum_frequency,
    network_names,
    observation_time,
    proposal_metadata_loaded,
    proposal_seed,
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
        "seeds": (data_seed, proposal_seed),
        "band": (minimum_frequency, maximum_frequency),
        "observation_time": observation_time,
        "density_sites": DEFAULT_DENSITY_SITES,
        "injection": injection_metadata.key(),
        "proposal": proposal_metadata_loaded.key(),
    }
    labelled_networks = [(n, l) for n, l in DETECTOR_NETWORKS if n in network_names]
    return cache_settings, labelled_networks, per_network


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Inferring cosmological parameters

    The two cosmological problems, with $H_0$ alone first and then $H_0$
    together with $\Omega_m$.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## H0

    The posterior on $H_0$ alone, every other parameter pinned at its
    fiducial. $H_0$ alone has one parameter, so its corner plot is its
    marginal below.
    """)


@app.cell
def _(
    FIDUCIALS,
    PRIORS,
    cache_dir,
    cache_name_prefix,
    cache_settings,
    grid_chunk_size,
    grid_specs,
    observed,
    per_network,
    spectral_density_fn,
):
    h0_posterior = grid_posterior(
        "H0",
        grid_specs["H0"],
        GaussianGWBBatchedLikelihood(
            spectral_density_fn, PRIORS, chunk_size=grid_chunk_size
        ),
        fiducials=FIDUCIALS,
        observed=observed,
        per_network=per_network,
        settings=cache_settings,
        cache_dir=cache_dir,
        prefix=cache_name_prefix,
    )
    return (h0_posterior,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Sanity: importance effective sample size.** The relative ESS of the
    weights at the edges of the $H_0$ 68% interval of the fiducial network.
    At the fiducials the target equals the proposal, so it is one there by
    construction; what matters is that it stays large away from them.
    """)


@app.cell
def _(FIDUCIALS, fiducial_network, h0_posterior, log_weights_fn):
    _grids, _log_densities = h0_posterior
    _grid = _grids["H0"]
    _density = marginal_density(_grid, _log_densities[fiducial_network], axis=0)
    _, _lo, _hi = median_and_hdi(_grid, _density)
    ess = {
        f"H0 = {value:.2f}": float(
            relative_ess(log_weights_fn({**FIDUCIALS, "H0": value}))
        )
        for value in (_lo, FIDUCIALS["H0"], _hi)
    }
    ess


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


@app.cell
def _(
    FIDUCIALS,
    PRIORS,
    cache_dir,
    cache_name_prefix,
    cache_settings,
    grid_chunk_size,
    grid_specs,
    observed,
    per_network,
    spectral_density_fn,
):
    h0_omega_m_posterior = grid_posterior(
        "H0_Omega_m",
        grid_specs["H0_Omega_m"],
        GaussianGWBBatchedLikelihood(
            spectral_density_fn, PRIORS, chunk_size=grid_chunk_size
        ),
        fiducials=FIDUCIALS,
        observed=observed,
        per_network=per_network,
        settings=cache_settings,
        cache_dir=cache_dir,
        prefix=cache_name_prefix,
    )
    return (h0_omega_m_posterior,)


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


@app.cell
def _(
    FIDUCIALS,
    PRIORS,
    cache_dir,
    cache_name_prefix,
    cache_settings,
    grid_chunk_size,
    grid_specs,
    observed,
    per_network,
    spectral_density_fn,
):
    xi_posterior = grid_posterior(
        "xi_0_xi_n",
        grid_specs["xi_0_xi_n"],
        GaussianGWBBatchedLikelihood(
            spectral_density_fn, PRIORS, chunk_size=grid_chunk_size
        ),
        fiducials=FIDUCIALS,
        observed=observed,
        per_network=per_network,
        settings=cache_settings,
        cache_dir=cache_dir,
        prefix=cache_name_prefix,
    )
    return (xi_posterior,)


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
    # Summary and output

    Median and 68% HDI of every swept parameter, by problem and network.
    Figures are written when the switch in *Configuration* is on.
    """)


@app.cell
def _(h0_omega_m_summary_rows, h0_summary_rows, xi_summary_rows):
    summary = pd.DataFrame(
        [*h0_summary_rows, *h0_omega_m_summary_rows, *xi_summary_rows]
    )
    summary


@app.cell
def _(
    FIGURES,
    h0_marginal,
    h0_omega_m_corner,
    h0_omega_m_marginal,
    write_figures,
    xi_corner,
    xi_marginal,
):
    if write_figures.value:
        save_figures(
            {
                FIGURES / "marginals_H0.pdf": h0_marginal,
                FIGURES / "marginals_H0_Omega_m.pdf": h0_omega_m_marginal,
                FIGURES / "corner_H0_Omega_m.pdf": h0_omega_m_corner,
                FIGURES / "marginals_xi_0_xi_n.pdf": xi_marginal,
                FIGURES / "corner_xi_0_xi_n.pdf": xi_corner,
            }
        )


if __name__ == "__main__":
    app.run()
