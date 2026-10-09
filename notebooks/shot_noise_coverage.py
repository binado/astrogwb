import marimo

__generated_with = "0.25.0"
app = marimo.App()

with app.setup(hide_code=True):
    import os
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
    from matplotlib.projections import register_projection
    from numpy.typing import NDArray

    from astrogwb.detector import effective_psd, gaussian_bin_scale
    from astrogwb.frequency import frequency_mask
    from astrogwb.gwb.importance import (
        build_rescaled_shot_noise,
        build_rescaled_spectrum,
        reference_catalog,
        reference_catalog_stem,
    )
    from astrogwb.inference import (
        amplitude_direction,
        amplitude_shot_noise_variance,
        rank_one_gaussian_log_likelihood,
    )
    from astrogwb.paper.cache import default_cache_dir
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
        DETECTOR_NETWORKS,
        parameter_label,
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

    # Log-likelihood of every injection at every grid point, (M, G): data is
    # (M, F), prediction and direction (G, F). One injection at a time, so
    # memory stays (G, F); compiled once for every network and variant.
    grid_log_likelihoods = jax.jit(
        lambda data, prediction, scale, direction, mask: jax.lax.map(
            lambda observed: rank_one_gaussian_log_likelihood(
                observed, prediction, scale, direction, mask
            ),
            data,
        )
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Coverage of the shot-noise likelihood for $H_0$

    `notebooks/cosmology_grid_posteriors.py` models the physical shot noise of
    its one injection as a rank-one covariance term (the amplitude mode,
    $\epsilon \sim \mathcal{N}(0, s^2)$ marginalized). This notebook checks
    that the term is calibrated: over $M$ independent Poisson injections at
    the fiducials, does the $H_0$ posterior cover the truth at the nominal
    rate, and does the detector-only likelihood not?

    Each injection is a Poisson catalog realization plus Gaussian detector
    noise, $d_f = S^{\mathrm{cat}}_f + n_f$ with $n_f \sim
    \mathcal{N}(0, \sigma_f^2)$, drawn per network. Without $n_f$ the data
    would scatter by the shot noise alone and every posterior of width
    $\ge \sigma_{\mathrm{det}}$ would over-cover. The model is the rescaled
    reference catalog of the grid notebook, at a different seed, so the bank
    is independent of the injections.

    For the amplitude direction the posterior mean of $H_0$ scatters by
    $\sqrt{1 + (\rho s)^2}$ detector widths. The detector-only posterior is
    one width wide and under-covers; with shot noise it is
    $\sqrt{1 + (\rho s)^2}$ wide. With $\rho s \approx 0.48$ the 68%
    interval covers about 63% without the term. The sharper statistic is the
    variance of $z = (\bar H_0 - H_0^{\mathrm{true}}) / \mathrm{sd}(H_0)$,
    which is one for a calibrated posterior and about $1.23$ without the
    term.

    $\mathrm{Var}(z)$ is the sum of a shot part, $(\rho s)^2$, and a detector
    part, one. With a single noise draw per injection the detector part alone
    has a sampling error of $\sqrt{2/M} \approx 0.1$ at $M = 200$, as large as
    the effect. Detector noise is free next to an injection, so each injection
    gets $R$ draws; the error is then set by the $M$ shot draws, and every
    error is a bootstrap over injections, since the $R$ draws of one injection
    share its shot noise.

    Only $H_0$ is swept. Its predictions on the grid do not depend on the
    data, so they are computed once and every injection reuses them.
    Setting `ASTROGWB_NOTEBOOK_SMOKE=1` swaps in a tiny catalog, few
    injections and fixed-count spectra.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configuration
    """)


@app.cell
def _():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    FIGURES = ROOT_DIR / FIGURES_DIR / "shot_noise_coverage"

    num_injections = 200
    injection_base_seed = 1000  # injection i draws at batch_keys(base, M)[i]
    noise_seed = 2000  # detector noise, one stream per network
    # Detector-noise draws per injection: noise is free next to an injection,
    # and one draw each leaves var(z) dominated by its own sampling error.
    noise_draws = 20
    catalog_seed = 42  # the grid notebook's reference catalog
    num_samples = 2**17
    redshift_nodes = 32
    catalog_chunk_size = 4096
    chunk_size = 16_384  # injection waveforms per reduced chunk
    grid_chunk_size = 8

    observation_time = 1.0  # years
    minimum_frequency = 2.0
    maximum_frequency = 2048.0
    fiducial_network = "ET-2L-aligned-CE-Hanford"
    network_names = [name for name, _ in DETECTOR_NETWORKS]
    h0_window, h0_size = (60.0, 76.0), 321

    SMOKE = os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") == "1"
    write_figures_default = True
    cache_name_prefix = ""
    if SMOKE:
        num_injections = 4
        noise_draws = 2
        num_samples = 64
        redshift_nodes = 8
        catalog_chunk_size = 512
        chunk_size = 512
        grid_chunk_size = 4
        h0_window, h0_size = (40.0, 100.0), 31
        write_figures_default = False
        cache_name_prefix = "smoke_"

    FIDUCIALS = fiducials(root=ROOT_DIR)
    PRIORS = priors(root=ROOT_DIR)
    registry = detector_registry(root=ROOT_DIR)
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
    )
    injection_metadata = BackgroundSpectralDensityMetadata(
        count="fixed" if SMOKE else "poisson",
        num_events=64 if SMOKE else None,
        observation_time=observation_time,
        hyperparameters={**FIDUCIALS},
        waveform=waveform,
        population=population,
    )
    cache_dir = default_cache_dir() / "coverage"

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
        fiducial_network,
        frequencies,
        grid_chunk_size,
        h0_size,
        h0_window,
        injection_base_seed,
        injection_metadata,
        maximum_frequency,
        minimum_frequency,
        network_names,
        noise_draws,
        noise_seed,
        num_injections,
        observation_time,
        population,
        redshift_nodes,
        registry,
        write_figures,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Toolbox
    """)


@app.function(hide_code=True)
def ensure_reference_catalog(
    metadata: CatalogMetadata,
    seed: int,
    directory: Path,
    *,
    chunk_size: int | None = None,
) -> PolarizationPowerData:
    """The reference catalog of ``metadata`` at ``seed``, drawn on a miss.

    The same file the grid notebook reads and writes,
    ``<directory>/reference_catalog-<key>-<seed>.h5``.

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
def ensure_injections(
    metadata: BackgroundSpectralDensityMetadata,
    base_seed: int,
    count: int,
    directory: Path,
    *,
    chunk_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """``count`` injections of ``metadata``, one cached file per injection.

    Injection ``i`` is drawn at ``batch_keys(base_seed, count)[i]`` and kept
    in ``<directory>/<key>/<base_seed>_<i>.npz``, so an interrupted run
    resumes where it stopped.

    Returns
    -------
    tuple
        Spectra ``(count, F)`` and event counts ``(count,)``.
    """
    folder = directory / metadata.key()
    folder.mkdir(parents=True, exist_ok=True)
    keys = batch_keys(base_seed, count)
    simulator = None
    spectra, events = [], []
    for index in range(count):
        path = folder / f"{base_seed}_{index}.npz"
        if not path.is_file():
            if simulator is None:
                simulator = BackgroundSpectralDensitySimulator(
                    metadata, chunk_size=chunk_size
                )
            drawn = simulator(keys[index])
            np.savez(
                path,
                spectral_density=np.asarray(drawn["spectral_density"][0]),
                n_events=int(drawn["n_events"][0]),
            )
        with np.load(path) as cached:
            spectra.append(np.asarray(cached["spectral_density"], dtype=np.float64))
            events.append(int(cached["n_events"]))
    return np.stack(spectra), np.asarray(events)


@app.function(hide_code=True)
def network_data(
    registry: DetectorRegistry,
    name: str,
    frequencies: NDArray[np.float64],
    *,
    observation_time: float,
    minimum_frequency: float,
    maximum_frequency: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """The per-bin Gaussian scale and usable-bin mask of one network.

    Unusable bins get a finite scale of one; the mask drops them.
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
    return np.asarray(jnp.where(mask, scale, 1.0)), np.asarray(mask)


@app.function(hide_code=True)
def posterior_statistics(
    grid: NDArray[np.float64],
    log_density: NDArray[np.float64],
    truth: float,
) -> dict[str, NDArray[np.float64]]:
    """PIT of the truth, mean and sd of each grid posterior.

    ``log_density`` is ``(M, G)``. The PIT is the posterior CDF at the truth,
    by the trapezoid rule; a calibrated posterior gives a uniform PIT.
    """
    density = np.exp(log_density - log_density.max(axis=-1, keepdims=True))
    density /= np.trapezoid(density, grid, axis=-1)[:, None]
    cell = np.diff(grid)
    cdf = np.concatenate(
        [
            np.zeros((density.shape[0], 1)),
            np.cumsum(0.5 * (density[:, 1:] + density[:, :-1]) * cell, axis=-1),
        ],
        axis=-1,
    )
    pit = np.array([np.interp(truth, grid, row) for row in cdf])
    mean = np.trapezoid(grid * density, grid, axis=-1)
    sd = np.sqrt(np.trapezoid((grid - mean[:, None]) ** 2 * density, grid, axis=-1))
    return {"pit": pit, "mean": mean, "sd": sd, "z": (mean - truth) / sd}


@app.function(hide_code=True)
def coverage_row(
    statistics: dict[str, NDArray[np.float64]],
    num_injections: int,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Interval coverage and the mean and variance of ``z``, with errors.

    The statistics are ``(num_injections * R,)``, injection-major: ``R``
    detector-noise draws per injection. Those draws share their injection's
    shot noise, so they are not independent, and every error is a bootstrap
    over injections, each resampled with all of its noise draws.
    """

    def summary(pit: NDArray[np.float64], z: NDArray[np.float64]) -> dict:
        axes = (-2, -1)
        values = {
            f"cover {mass:.0%}": np.mean(
                (pit > (1.0 - mass) / 2.0) & (pit < (1.0 + mass) / 2.0), axis=axes
            )
            for mass in (0.68, 0.95)
        }
        values["mean z"] = np.mean(z, axis=axes)
        values["var z"] = np.var(z, axis=axes)
        return values

    pit = statistics["pit"].reshape(num_injections, -1)
    z = statistics["z"].reshape(num_injections, -1)
    index = np.random.default_rng(seed).integers(
        0, num_injections, size=(n_bootstrap, num_injections)
    )
    point, resampled = summary(pit, z), summary(pit[index], z[index])
    row: dict[str, float] = {}
    for name, value in point.items():
        row[name] = float(value)
        row[f"{name} err"] = float(np.std(resampled[name]))
    return row


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model on the grid

    The rescaled spectrum and its shot-noise variance at every $H_0$ grid
    point, once; then $s^2$ at the fiducial per network (the adopted `fixed`
    variant) and at every grid point (`amplitude`).
    """)


@app.cell
def _(
    FIDUCIALS,
    catalog_chunk_size,
    catalog_metadata,
    catalog_seed,
    frequencies,
    grid_chunk_size,
    h0_size,
    h0_window,
    observation_time,
    population,
    redshift_nodes,
):
    catalog_data = ensure_reference_catalog(
        catalog_metadata, catalog_seed, CATALOGS_ROOT, chunk_size=catalog_chunk_size
    )
    target = build_population(population.model_name, **population.model_kwargs)
    builder_settings = {
        "population": target,
        "frequencies": frequencies,
        "num_redshift_nodes": redshift_nodes,
        "density_sites": (),
    }
    spectral_density_fn, _ = build_rescaled_spectrum(
        catalog_data, catalog_metadata, **builder_settings
    )
    variance_fn = build_rescaled_shot_noise(
        catalog_data,
        catalog_metadata,
        observation_time=observation_time,
        **builder_settings,
    )
    h0_grid = np.linspace(*h0_window, h0_size)

    @jax.jit
    def _predict(spectrum_fn, shot_noise_fn, h0):
        def point(value):
            params = {**FIDUCIALS, "H0": value}
            return spectrum_fn(params)[0], shot_noise_fn(params)

        return jax.lax.map(point, h0, batch_size=grid_chunk_size)

    grid_prediction, grid_variance = _predict(
        spectral_density_fn, variance_fn, jnp.asarray(h0_grid)
    )
    _fiducial = int(np.argmin(np.abs(h0_grid - FIDUCIALS["H0"])))
    print(f"grid: {h0_size} H0 points; fiducial node H0 = {h0_grid[_fiducial]:.3f}")
    fiducial_prediction, _ = spectral_density_fn(FIDUCIALS)
    fiducial_variance = variance_fn(FIDUCIALS)
    return (
        fiducial_prediction,
        fiducial_variance,
        grid_prediction,
        grid_variance,
        h0_grid,
    )


@app.cell
def _(
    frequencies,
    maximum_frequency,
    minimum_frequency,
    network_names,
    observation_time,
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
    return (per_network,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Injections

    $M$ Poisson spectra at the fiducials, one cached file each.
    """)


@app.cell
def _(
    cache_dir,
    cache_name_prefix,
    chunk_size,
    frequencies,
    injection_base_seed,
    injection_metadata,
    num_injections,
):
    injections, injected_events = ensure_injections(
        injection_metadata,
        injection_base_seed,
        num_injections,
        cache_dir / f"{cache_name_prefix}injections",
        chunk_size=chunk_size,
    )
    assert injections.shape[1] == frequencies.size
    print(
        f"{num_injections} injections, events {injected_events.min():,} to "
        f"{injected_events.max():,}"
    )
    return (injections,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Coverage

    Per network: $R$ detector-noise draws per injection, then the $H_0$
    posterior on the grid under each likelihood variant, with the uniform
    $H_0$ prior.
    """)


@app.cell
def _(
    FIDUCIALS,
    PRIORS,
    fiducial_prediction,
    fiducial_variance,
    grid_prediction,
    grid_variance,
    h0_grid,
    injections,
    noise_draws,
    noise_seed,
    per_network,
):
    _label = dict(DETECTOR_NETWORKS)
    _log_prior = np.asarray(PRIORS["H0"].log_prob(jnp.asarray(h0_grid)))
    _count, _bins = injections.shape
    statistics: dict[tuple[str, str], dict[str, NDArray[np.float64]]] = {}
    _rows = []
    for _k, (_name, (_scale, _mask)) in enumerate(per_network.items()):
        # (M, R, F) -> (M R, F), injection-major: R noise draws per injection.
        _noise = np.asarray(
            jax.random.normal(
                jax.random.fold_in(jax.random.key(noise_seed), _k),
                (_count, noise_draws, _bins),
            )
        )
        _data = jnp.asarray(
            (injections[:, None, :] + _noise * _scale).reshape(-1, _bins)
        )
        _s2_fixed = amplitude_shot_noise_variance(
            fiducial_prediction, fiducial_variance, _scale, _mask
        )
        _s2_grid = amplitude_shot_noise_variance(
            grid_prediction, grid_variance, _scale, _mask
        )
        _directions = {
            "detector": jnp.zeros_like(grid_prediction),
            "amplitude": amplitude_direction(grid_prediction, _s2_grid),
            "fixed": amplitude_direction(grid_prediction, _s2_fixed),
        }
        for _variant, _direction in _directions.items():
            _log_density = (
                np.asarray(
                    grid_log_likelihoods(
                        _data,
                        grid_prediction,
                        jnp.asarray(_scale),
                        _direction,
                        jnp.asarray(_mask),
                    )
                )
                + _log_prior
            )
            statistics[_name, _variant] = posterior_statistics(
                h0_grid, _log_density, FIDUCIALS["H0"]
            )
            _rows.append(
                {
                    "network": _label[_name],
                    "likelihood": _variant,
                    **coverage_row(statistics[_name, _variant], _count),
                }
            )
    coverage = pd.DataFrame(_rows)
    coverage
    return coverage, statistics


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    PIT and $z$ at the fiducial network. A calibrated posterior has a flat PIT
    histogram and $z \sim \mathcal{N}(0, 1)$; an over-confident one piles the
    PIT at both ends and widens $z$.
    """)


@app.cell
def _(fiducial_network, statistics):
    coverage_figure, (_ax_pit, _ax_z) = plt.subplots(1, 2, figsize=(9, 3.6))
    _bins = np.linspace(0.0, 1.0, 11)
    _z_grid = np.linspace(-4.0, 4.0, 201)
    for _variant, _style in (("detector", "-"), ("fixed", "--")):
        _stats = statistics[fiducial_network, _variant]
        _ax_pit.hist(
            _stats["pit"],
            bins=_bins,
            density=True,
            histtype="step",
            linestyle=_style,
            label=_variant,
        )
        _ax_z.hist(
            _stats["z"],
            bins=np.linspace(-4.0, 4.0, 17),
            density=True,
            histtype="step",
            linestyle=_style,
            label=_variant,
        )
    _ax_pit.axhline(1.0, color="0.5", lw=1)
    _ax_z.plot(
        _z_grid, np.exp(-0.5 * _z_grid**2) / np.sqrt(2 * np.pi), color="0.5", lw=1
    )
    _ax_pit.set(xlabel=f"PIT of {parameter_label('H0')}", ylabel="Density")
    _ax_z.set(xlabel=r"$z = (\bar H_0 - H_0^{\mathrm{true}}) / \mathrm{sd}$")
    _ax_pit.legend(title="likelihood", fontsize="small")
    coverage_figure.tight_layout()
    coverage_figure
    return (coverage_figure,)


@app.cell
def _(FIGURES, coverage, coverage_figure, write_figures):
    if write_figures.value:
        save_figures({FIGURES / "coverage_H0.pdf": coverage_figure})
        FIGURES.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(FIGURES / "coverage_H0.csv", index=False)


if __name__ == "__main__":
    app.run()
