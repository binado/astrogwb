import marimo

__generated_with = "0.25.0"
app = marimo.App()

with app.setup(hide_code=True):
    import os
    from collections.abc import Callable, Mapping
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
        ImportanceCatalogMetadata,
        build_importance_spectrum,
        build_rescaled_spectrum,
        importance_catalog,
        importance_catalog_stem,
        reference_catalog,
        reference_catalog_stem,
    )
    from astrogwb.inference.fisher import spectral_density_jacobian
    from astrogwb.inference.protocol import SpectralDensityFn
    from astrogwb.paper.config import (
        detector_registry,
        fiducials,
        population_metadata,
        waveform_metadata,
    )
    from astrogwb.paper.config.detectors import DetectorRegistry
    from astrogwb.paper.config.runs import CATALOGS_ROOT, FIGURES_DIR
    from astrogwb.paper.plotting import parameter_label, save_figures, use_paper_style
    from astrogwb.populations import Population, build_population
    from astrogwb.simulators.core import batch_keys, load, write
    from astrogwb.simulators.polarization_power import (
        CatalogMetadata,
        PolarizationPowerData,
    )

    # One compiled spectrum for every catalog: the spectral density function is
    # a pytree argument, so its catalog is a traced input, not a constant, and
    # catalogs of one shape share the compilation.
    evaluate_spectrum = jax.jit(lambda fn, params: fn(params)[0])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Convergence of the rescaled reference-redshift spectrum

    The grid posteriors predict the spectrum from $N$ intrinsic draws, each
    generated once at the window's lower edge $z_{\min}$ and rescaled to
    $Z$ Gauss-Legendre nodes in the scale factor $a = 1/(1+z)$,

    $$
    S(f;\Lambda) = R(\Lambda) \sum_j w_j\, p(z_j\mid\Lambda)
    \left[\frac{d^{\mathrm{ref}}}{d_L(z_j\mid\Lambda)}\right]^2 s_j^4\,
    \bar P_{\mathrm{ref}}(f s_j),
    \qquad s_j = \frac{1+z_j}{1+z_{\min}},
    $$

    with $\bar P_{\mathrm{ref}}$ the mean reference power, interpolated in
    $\ln f$. Three checks, each measured on its own:

    0. **The rescaling** on the real approximant: the rescaled spectrum
       against waveforms generated at every node, on *the same* draws.
    1. **Redshift quadrature**: a $Z = 2^i$ ladder against its top rung. $Z$
       is a builder argument, so the ladder costs no waveforms.
    2. **Monte Carlo noise** from the $N$ draws: $M$ independent catalogs at
       fixed seeds, and the scatter of their residuals about the mean.

    Errors are reported in units of the per-bin noise $\sigma_f$ of the most
    sensitive network, and as the shift of each parameter's likelihood peak
    in units of its Fisher width,

    $$
    b_a = \frac{\sum_f m_f\, g_{a,f}\, \delta S_f / \sigma_f^2}
    {\sqrt{\sum_f m_f\, g_{a,f}^2 / \sigma_f^2}},
    \qquad g_a = \partial S / \partial\theta_a .
    $$

    One catalog serves every grid point, so its Monte Carlo error is a
    coherent shift of the posterior: $b_a$ is that shift. $\xi_n$ has no
    $b$: at $\Xi_0 = 1$ the spectrum does not depend on it.

    Setting `ASTROGWB_NOTEBOOK_SMOKE=1` swaps in tiny catalogs.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configuration
    """)


@app.cell
def _():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    FIGURES = ROOT_DIR / FIGURES_DIR / "importance_convergence"
    catalog_dir = ROOT_DIR / CATALOGS_ROOT

    # Gate and redshift ladder: one fixed set of draws. The node catalog at
    # gate_nodes generates waveforms at every node; the ladder's top rung is
    # its reference.
    gate_seed = 100
    gate_samples = 256
    gate_nodes = 16
    ladder_exponents = range(2, 11)  # Z = 4 ... 1024

    # Monte Carlo: M catalogs at N_max; each N = 2**k <= N_max is a prefix.
    mc_base_seed = 300
    mc_catalogs = 16
    mc_max_samples = 2**17
    mc_min_exponent = 4  # smallest N = 16
    # None takes the smallest converged Z from the ladder.
    mc_redshift_nodes: int | None = None

    catalog_chunk_size = 4096  # waveforms per lax.map batch
    tolerance = 0.1  # |b| and max |dS| / sigma, in sigma units
    observation_time = 1.0  # years
    minimum_frequency = 2.0
    maximum_frequency = 2048.0
    yardstick_network = "ET-2L-aligned-CE-Hanford"  # the most sensitive
    peak_parameters = ("H0", "Omega_m", "xi_0")

    FIDUCIALS = fiducials(root=ROOT_DIR)
    # The grid posteriors' window edges, one parameter moved at a time.
    edges = {
        "H0": (60.0, 76.0),
        "Omega_m": (0.2856, 0.3336),
        "xi_0": (0.94, 1.06),
        "xi_n": (0.3, 3.0),
    }

    SMOKE = os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") == "1"
    write_figures_default = True
    if SMOKE:
        gate_samples = 16
        gate_nodes = 4
        ladder_exponents = range(2, 6)  # Z = 4 ... 32
        mc_catalogs = 3
        mc_max_samples = 64
        mc_min_exponent = 3
        catalog_chunk_size = 512
        edges = {}
        write_figures_default = False

    evaluation_points: dict[str, dict[str, float]] = {"fiducial": dict(FIDUCIALS)}
    for _name, _values in edges.items():
        for _value in _values:
            evaluation_points[f"{_name}={_value:g}"] = {**FIDUCIALS, _name: _value}

    # The observed grid, and the reference grid: geometric bins 1% wide from
    # the observed f_min to past the detector-frame BNS merger at z_min.
    waveform = waveform_metadata(root=ROOT_DIR)
    reference_waveform = waveform_metadata(
        root=ROOT_DIR,
        frequency_spacing="log",
        minimum_frequency=minimum_frequency,
        maximum_frequency=4000.0,
        frequency_resolution=0.02,
        turnover_frequency=None,
    )
    frequencies = np.asarray(waveform.build().frequencies)
    population = population_metadata(root=ROOT_DIR)
    target = build_population(population.model_name, **population.model_kwargs)
    registry = detector_registry(root=ROOT_DIR)

    write_figures = mo.ui.switch(value=write_figures_default, label="Write figures")
    write_figures
    use_paper_style(root=ROOT_DIR)
    # gwpy registers replacement default axes; keep matplotlib's.
    register_projection(Axes)
    return (
        FIDUCIALS,
        FIGURES,
        catalog_chunk_size,
        catalog_dir,
        evaluation_points,
        frequencies,
        gate_nodes,
        gate_samples,
        gate_seed,
        ladder_exponents,
        maximum_frequency,
        mc_base_seed,
        mc_catalogs,
        mc_max_samples,
        mc_min_exponent,
        mc_redshift_nodes,
        minimum_frequency,
        observation_time,
        peak_parameters,
        population,
        reference_waveform,
        registry,
        target,
        tolerance,
        waveform,
        write_figures,
        yardstick_network,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Toolbox
    """)


@app.function(hide_code=True)
def cached_catalog(
    path: Path,
    metadata: CatalogMetadata | ImportanceCatalogMetadata,
    seed: int,
    draw: Callable[[jax.Array], PolarizationPowerData],
) -> PolarizationPowerData:
    """The catalog at ``path``, drawn with ``draw(batch_keys(seed, 1)[0])`` on a miss.

    A hit is checked against ``metadata`` and ``seed``, as
    ``paper.catalogs.ensure_catalog`` checks a plain catalog.

    Raises
    ------
    ValueError
        If the cached file records another metadata or seed.
    """
    if path.is_file():
        data, recorded, attrs = load(path, type(metadata))
        if recorded.key() != metadata.key() or attrs.get("seed") != seed:
            raise ValueError(
                f"{path} records {recorded.key()} at seed {attrs.get('seed')}, "
                f"not the requested {metadata.key()} at seed {seed}"
            )
        return data  # ty: ignore[invalid-return-type]
    data = draw(batch_keys(seed, 1)[0])
    write(path, data, metadata, seed=seed)
    return data


@app.function(hide_code=True)
def ensure_reference_catalog(
    metadata: CatalogMetadata,
    seed: int,
    directory: Path,
    *,
    chunk_size: int | None = None,
) -> PolarizationPowerData:
    """The reference catalog of ``metadata`` at ``seed``: power ``(F_ref, N)``."""
    return cached_catalog(
        directory / f"{reference_catalog_stem(metadata, seed)}.h5",
        metadata,
        seed,
        lambda key: reference_catalog(metadata, key, chunk_size=chunk_size),
    )


@app.function(hide_code=True)
def ensure_node_catalog(
    metadata: ImportanceCatalogMetadata,
    seed: int,
    directory: Path,
    *,
    chunk_size: int | None = None,
) -> PolarizationPowerData:
    """The node catalog of ``metadata`` at ``seed``: power ``(F, Z * N)``."""
    return cached_catalog(
        directory / f"{importance_catalog_stem(metadata, seed)}.h5",
        metadata,
        seed,
        lambda key: importance_catalog(metadata, key, chunk_size=chunk_size),
    )


@app.function(hide_code=True)
def prefix_catalog(
    data: PolarizationPowerData, metadata: CatalogMetadata, n: int
) -> tuple[PolarizationPowerData, CatalogMetadata]:
    """The reference catalog of its first ``n`` draws.

    The first ``n`` of ``N`` iid draws are themselves an iid sample, so every
    smaller size is read off one catalog.
    """
    sliced: PolarizationPowerData = {
        "frequencies": data["frequencies"],
        "polarization_power": np.asarray(data["polarization_power"])[:, :n],
        "source_parameters": {
            name: np.asarray(values)[:n]
            for name, values in data["source_parameters"].items()
        },
    }
    return sliced, metadata.model_copy(update={"num_samples": n})


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

    Bins no detector pair measures have infinite effective PSD; they are
    dropped from the mask and given a finite scale of one.

    Returns
    -------
    tuple
        ``sigma`` and ``mask``, shape ``(F,)``.
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
def spectra_at(
    spectral_density_fn: SpectralDensityFn,
    points: Mapping[str, Mapping[str, float]],
) -> NDArray[np.float64]:
    """The spectrum at every evaluation point, shape ``(len(points), F)``."""
    return np.stack(
        [
            np.asarray(evaluate_spectrum(spectral_density_fn, dict(params)))
            for params in points.values()
        ]
    )


@app.function(hide_code=True)
def rescaled_fn(
    data: PolarizationPowerData,
    metadata: CatalogMetadata,
    target: Population,
    frequencies: NDArray[np.float64],
    num_redshift_nodes: int,
) -> SpectralDensityFn:
    """The rescaled spectrum of one reference catalog.

    No swept parameter is intrinsic, so no density factor enters a weight.
    """
    spectral_density_fn, _ = build_rescaled_spectrum(
        data,
        metadata,
        population=target,
        frequencies=frequencies,
        num_redshift_nodes=num_redshift_nodes,
        density_sites=(),
    )
    return spectral_density_fn


@app.function(hide_code=True)
def peak_shift(
    delta: NDArray[np.float64],
    jacobian: NDArray[np.float64],
    sigma: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Likelihood-peak shift per parameter, in units of its Fisher width.

    Parameters
    ----------
    delta
        Spectrum error, shape ``(..., F)``.
    jacobian
        ``dS/dtheta``, shape ``(F, P)``.
    sigma, mask
        The yardstick network's scale and usable bins, shape ``(F,)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(..., P)``.
    """
    weight = np.where(mask, 1.0 / sigma**2, 0.0)
    projected = np.einsum("...f,fp,f->...p", delta, jacobian, weight)
    return projected / np.sqrt(np.einsum("fp,f->p", jacobian**2, weight))


@app.function(hide_code=True)
def relative(
    delta: NDArray[np.float64],
    reference: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """``delta / reference`` on usable bins with signal, NaN elsewhere.

    The spectrum is exactly zero in band above the highest detector-frame
    merger frequency, where a relative error is undefined.
    """
    usable = mask & (reference != 0.0)
    return np.where(usable, delta / np.where(usable, reference, 1.0), np.nan)


@app.function(hide_code=True)
def in_sigma(
    delta: NDArray[np.float64], sigma: NDArray[np.float64], mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """``delta / sigma`` on usable bins and NaN elsewhere, so plots skip them."""
    return np.where(mask, delta / sigma, np.nan)


@app.function(hide_code=True)
def error_rows(
    label: str,
    value: object,
    delta: NDArray[np.float64],
    *,
    points: Mapping[str, object],
    jacobian: NDArray[np.float64],
    sigma: NDArray[np.float64],
    mask: NDArray[np.bool_],
    parameters: tuple[str, ...],
) -> list[dict[str, object]]:
    """One row per evaluation point: max ``|dS|/sigma`` and every ``b_a``."""
    max_sigma = np.nanmax(np.abs(in_sigma(delta, sigma, mask)), axis=-1)
    shift = peak_shift(delta, jacobian, sigma, mask)
    return [
        {
            label: value,
            "point": point,
            "max |dS|/sigma": float(max_sigma[p]),
            **{f"b_{name}": float(shift[p, a]) for a, name in enumerate(parameters)},
        }
        for p, point in enumerate(points)
    ]


@app.cell
def _(
    frequencies,
    maximum_frequency,
    minimum_frequency,
    observation_time,
    registry,
    yardstick_network,
):
    sigma, mask = network_data(
        registry,
        yardstick_network,
        frequencies,
        observation_time=observation_time,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    return mask, sigma


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 0. The rescaling, on the real approximant

    One reference catalog and one node catalog, at the same seed, population,
    fiducials and $N$: `importance_catalog` and `reference_catalog` draw the
    same intrinsic sources from the key, which is asserted below. The node
    catalog generates every draw at every node; the rescaled spectrum
    generates each once, at $z_{\min}$. Their difference at the same $Z$ is
    the error of the rescaling and of the $\ln f$ interpolation alone.
    """)


@app.cell
def _(
    FIDUCIALS,
    catalog_chunk_size,
    catalog_dir,
    evaluation_points,
    frequencies,
    gate_nodes,
    gate_samples,
    gate_seed,
    population,
    reference_waveform,
    target,
    waveform,
):
    gate_reference_metadata = CatalogMetadata(
        waveform=reference_waveform,
        population=population,
        fiducials=FIDUCIALS,
        num_samples=gate_samples,
    )
    gate_reference = ensure_reference_catalog(
        gate_reference_metadata,
        gate_seed,
        catalog_dir,
        chunk_size=catalog_chunk_size,
    )
    _node_metadata = ImportanceCatalogMetadata(
        waveform=waveform,
        population=population,
        fiducials=FIDUCIALS,
        num_samples=gate_samples,
        num_redshift_nodes=gate_nodes,
    )
    _nodes = ensure_node_catalog(
        _node_metadata, gate_seed, catalog_dir, chunk_size=catalog_chunk_size
    )
    assert np.array_equal(
        gate_reference["source_parameters"]["source_frame_mass_1"],
        np.asarray(_nodes["source_parameters"]["source_frame_mass_1"])[:gate_samples],
    ), "the two catalogs must share their draws"
    assert np.array_equal(np.asarray(_nodes["frequencies"]), frequencies)
    _node_fn, _ = build_importance_spectrum(
        _nodes, _node_metadata, population=target, density_sites=()
    )
    gate_node_spectra = spectra_at(_node_fn, evaluation_points)
    gate_rescaled_spectra = spectra_at(
        rescaled_fn(
            gate_reference, gate_reference_metadata, target, frequencies, gate_nodes
        ),
        evaluation_points,
    )
    print(
        f"gate: N = {gate_samples}, Z = {gate_nodes}, reference grid "
        f"{np.asarray(gate_reference['frequencies']).size} bins"
    )
    return (
        gate_node_spectra,
        gate_reference,
        gate_reference_metadata,
        gate_rescaled_spectra,
    )


@app.cell
def _(
    FIDUCIALS,
    frequencies,
    gate_reference,
    gate_reference_metadata,
    ladder_exponents,
    peak_parameters,
    target,
):
    # The score direction of b: the converged rescaled spectrum's derivative.
    _fn = rescaled_fn(
        gate_reference,
        gate_reference_metadata,
        target,
        frequencies,
        2 ** ladder_exponents[-1],
    )
    jacobian = np.asarray(spectral_density_jacobian(_fn, FIDUCIALS, peak_parameters))
    return (jacobian,)


@app.cell
def _(
    evaluation_points,
    gate_node_spectra,
    gate_nodes,
    gate_rescaled_spectra,
    jacobian,
    mask,
    peak_parameters,
    sigma,
):
    gate = pd.DataFrame(
        error_rows(
            "Z",
            gate_nodes,
            gate_rescaled_spectra - gate_node_spectra,
            points=evaluation_points,
            jacobian=jacobian,
            sigma=sigma,
            mask=mask,
            parameters=peak_parameters,
        )
    )
    gate
    return (gate,)


@app.cell
def _(frequencies, gate_node_spectra, gate_rescaled_spectra, mask, sigma):
    gate_figure, (_ax_ratio, _ax_sigma) = plt.subplots(
        2, 1, figsize=(6, 5), sharex=True
    )
    _delta = gate_rescaled_spectra[0] - gate_node_spectra[0]
    _ax_ratio.plot(frequencies, relative(_delta, gate_node_spectra[0], mask), lw=1)
    _ax_ratio.axhline(0.0, color="0.3", lw=0.6)
    _ax_ratio.set(
        ylabel=r"$(S_\mathrm{rescaled} - S_\mathrm{nodes}) / S_\mathrm{nodes}$"
    )
    _ax_sigma.plot(frequencies, np.abs(in_sigma(_delta, sigma, mask)), lw=1)
    _ax_sigma.set(
        xscale="log",
        yscale="log",
        xlabel="Frequency [Hz]",
        ylabel=r"$|S_\mathrm{rescaled} - S_\mathrm{nodes}| / \sigma_f$",
    )
    _ax_ratio.set_title("same draws, at the fiducials", fontsize="small")
    gate_figure.tight_layout()
    gate_figure
    return (gate_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Redshift quadrature: $Z = 2^i$

    The gate's reference catalog, at every rung: the draws are fixed, so the
    difference to the top rung is pure quadrature error.
    """)


@app.cell
def _(
    evaluation_points,
    frequencies,
    gate_reference,
    gate_reference_metadata,
    ladder_exponents,
    target,
):
    ladder_nodes = [2**i for i in ladder_exponents]
    ladder_spectra = {
        _nodes: spectra_at(
            rescaled_fn(
                gate_reference, gate_reference_metadata, target, frequencies, _nodes
            ),
            evaluation_points,
        )
        for _nodes in ladder_nodes
    }
    return ladder_nodes, ladder_spectra


@app.cell
def _(
    evaluation_points,
    jacobian,
    ladder_nodes,
    ladder_spectra,
    mask,
    peak_parameters,
    sigma,
    tolerance,
):
    _reference = ladder_spectra[ladder_nodes[-1]]
    quadrature = pd.DataFrame(
        [
            row
            for _nodes in ladder_nodes[:-1]
            for row in error_rows(
                "Z",
                _nodes,
                ladder_spectra[_nodes] - _reference,
                points=evaluation_points,
                jacobian=jacobian,
                sigma=sigma,
                mask=mask,
                parameters=peak_parameters,
            )
        ]
    )
    _b_columns = [f"b_{name}" for name in peak_parameters]
    _worst = (
        quadrature.assign(
            worst=np.maximum(
                quadrature["max |dS|/sigma"], quadrature[_b_columns].abs().max(axis=1)
            )
        )
        .groupby("Z")["worst"]
        .max()
    )
    _converged = _worst[_worst < tolerance]
    if _converged.empty:
        chosen_nodes = ladder_nodes[-1]
        print(
            f"no rung below Z = {chosen_nodes} meets tolerance {tolerance}; "
            "the reference itself may not be converged"
        )
    else:
        chosen_nodes = int(_converged.index.min())
    print(f"smallest converged Z: {chosen_nodes}")
    quadrature
    return chosen_nodes, quadrature


@app.cell
def _(evaluation_points, peak_parameters, quadrature, tolerance):
    quadrature_figure, (_ax_spectrum, _ax_peak) = plt.subplots(
        1, 2, figsize=(9, 3.6), sharex=True
    )
    for _point in evaluation_points:
        _rows = quadrature[quadrature["point"] == _point]
        _ax_spectrum.plot(_rows["Z"], _rows["max |dS|/sigma"], marker="o", lw=1)
    for _name in peak_parameters:
        _worst = quadrature.groupby("Z")[f"b_{_name}"].apply(lambda b: b.abs().max())
        _ax_peak.plot(
            _worst.index, _worst.values, marker="o", label=parameter_label(_name)
        )
    for _ax in (_ax_spectrum, _ax_peak):
        _ax.axhline(tolerance, color="0.5", ls="--", lw=1)
        _ax.set(xscale="log", yscale="log", xlabel="Redshift nodes $Z$")
    _ax_spectrum.set(ylabel=r"$\max_f |S_Z - S_\mathrm{ref}| / \sigma_f$")
    _ax_spectrum.set_title("every evaluation point", fontsize="small")
    _ax_peak.set(ylabel=r"worst $|b_a|$ over evaluation points")
    _ax_peak.legend(fontsize="small")
    quadrature_figure.tight_layout()
    quadrature_figure
    return (quadrature_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Monte Carlo noise: $M$ seeds, residuals about the mean

    $M$ independent reference catalogs at $N_\mathrm{max}$. Each
    $N = 2^k \le N_\mathrm{max}$ is the first $N$ draws of every catalog, so
    the whole $N$ ladder costs no extra waveforms. At each $N$ the residuals
    are taken about the mean of the $M$ estimates; their standard deviation
    uses $M - 1$ degrees of freedom, so it estimates one estimator's scatter.
    Catalogs are drawn in memory one at a time and not cached, so only one is
    resident and none is written.
    """)


@app.cell
def _(
    FIDUCIALS,
    catalog_chunk_size,
    catalog_dir,
    chosen_nodes,
    evaluation_points,
    frequencies,
    mc_base_seed,
    mc_catalogs,
    mc_max_samples,
    mc_min_exponent,
    mc_redshift_nodes,
    population,
    reference_waveform,
    target,
):
    mc_nodes = chosen_nodes if mc_redshift_nodes is None else mc_redshift_nodes
    mc_sizes = [
        2**k
        for k in range(mc_min_exponent, int(np.log2(mc_max_samples)) + 1)
        if 2**k <= mc_max_samples
    ]
    _metadata = CatalogMetadata(
        waveform=reference_waveform,
        population=population,
        fiducials=FIDUCIALS,
        num_samples=mc_max_samples,
    )
    print(
        f"Monte Carlo: {mc_catalogs} catalogs x {mc_max_samples:,} draws = "
        f"{mc_catalogs * mc_max_samples:,} waveforms, Z = {mc_nodes}"
    )
    # (M, sizes, points, F)
    _spectra = []
    for _m in range(mc_catalogs):
        # Drawn in memory, not cached: each regenerates from its seed in
        # seconds, and M of them would take M * 0.8 GB of disk.
        _data = reference_catalog(
            _metadata,
            batch_keys(mc_base_seed + _m, 1)[0],
            chunk_size=catalog_chunk_size,
        )
        if _m == 0:
            _gb = np.asarray(_data["polarization_power"]).nbytes / 1e9
            print(f"  {_gb:.2f} GB of power per catalog")
        _spectra.append(
            [
                spectra_at(
                    rescaled_fn(
                        *prefix_catalog(_data, _metadata, _n),
                        target,
                        frequencies,
                        mc_nodes,
                    ),
                    evaluation_points,
                )
                for _n in mc_sizes
            ]
        )
        # Release this catalog before the next is drawn.
        del _data
    mc_spectra = np.asarray(_spectra)
    return mc_nodes, mc_sizes, mc_spectra


@app.cell
def _(
    evaluation_points,
    jacobian,
    mask,
    mc_sizes,
    mc_spectra,
    peak_parameters,
    sigma,
):
    mc_mean = mc_spectra.mean(axis=0)  # (sizes, points, F)
    mc_residuals = mc_spectra - mc_mean
    # (M, sizes, points, P) -> std over the M estimates
    _shifts = peak_shift(mc_residuals, jacobian, sigma, mask)
    mc_shift_std = _shifts.std(axis=0, ddof=1)
    _sigma_std = in_sigma(mc_residuals.std(axis=0, ddof=1), sigma, mask)
    monte_carlo = pd.DataFrame(
        [
            {
                "N": _n,
                "point": _point,
                "max std(dS)/sigma": float(np.nanmax(_sigma_std[_s, _p])),
                **{
                    f"std b_{name}": float(mc_shift_std[_s, _p, a])
                    for a, name in enumerate(peak_parameters)
                },
            }
            for _s, _n in enumerate(mc_sizes)
            for _p, _point in enumerate(evaluation_points)
        ]
    )
    monte_carlo
    return mc_mean, mc_residuals, mc_shift_std, monte_carlo


@app.cell
def _(frequencies, mask, mc_mean, mc_residuals, mc_sizes, sigma):
    _shown = sorted({0, len(mc_sizes) // 2, len(mc_sizes) - 1})
    residual_figure, _axes = plt.subplots(
        2,
        len(_shown),
        figsize=(3.4 * len(_shown), 5.2),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for _col, _s in enumerate(_shown):
        _relative = relative(mc_residuals[:, _s, 0], mc_mean[_s, 0], mask)
        _noise = in_sigma(mc_residuals[:, _s, 0], sigma, mask)
        for _row, _values in enumerate((_relative, _noise)):
            _ax = _axes[_row, _col]
            _ax.plot(frequencies, _values.T, color="C0", lw=0.4, alpha=0.5)
            _std = np.std(_values, axis=0, ddof=1)  # NaN on masked bins
            _ax.fill_between(frequencies, -_std, _std, color="C1", alpha=0.3, lw=0)
            _ax.axhline(0.0, color="0.3", lw=0.6)
            _ax.set(xscale="log")
        _axes[0, _col].set_title(f"$N = {mc_sizes[_s]}$", fontsize="small")
        _axes[1, _col].set(xlabel="Frequency [Hz]")
    _axes[0, 0].set(ylabel=r"$(S_m - \bar S) / \bar S$")
    _axes[1, 0].set(ylabel=r"$(S_m - \bar S) / \sigma_f$")
    residual_figure.suptitle(
        "Residuals about the mean at the fiducials", fontsize="small"
    )
    residual_figure.tight_layout()
    residual_figure
    return (residual_figure,)


@app.cell
def _(mc_shift_std, mc_sizes, peak_parameters, tolerance):
    sizes = np.asarray(mc_sizes, dtype=float)
    shift_figure, _ax = plt.subplots(figsize=(5, 3.6))
    _worst = mc_shift_std.max(axis=1)  # worst over points, (sizes, P)
    for _a, _name in enumerate(peak_parameters):
        _ax.plot(sizes, _worst[:, _a], marker="o", label=parameter_label(_name))
    _anchor = _worst[-1].max()
    _ax.plot(
        sizes,
        _anchor * np.sqrt(sizes[-1] / sizes),
        color="0.5",
        lw=1,
        ls=":",
        label=r"$\propto N^{-1/2}$",
    )
    _ax.axhline(tolerance, color="0.5", ls="--", lw=1)
    _ax.set(
        xscale="log",
        yscale="log",
        xlabel="Intrinsic draws $N$",
        ylabel=r"worst std of $b_a$ [$\sigma_a$]",
    )
    _ax.legend(fontsize="small")
    shift_figure.tight_layout()
    shift_figure
    return (shift_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Recommendation

    The cheapest drawn $N$ whose worst Monte Carlo shift is below tolerance,
    and the $N$ the $1/N$ variance law extrapolates to it from $N_\mathrm{max}$,
    at the converged $Z$. The rescaling, quadrature and Monte Carlo errors are
    independent, so they are added in quadrature.
    """)


@app.cell
def _(
    chosen_nodes,
    gate,
    mc_nodes,
    mc_shift_std,
    mc_sizes,
    quadrature,
    reference_waveform,
    tolerance,
):
    _mc_worst = mc_shift_std.max(axis=(1, 2))  # (sizes,)
    # Monte Carlo variance falls as 1/N, so the size that would just meet the
    # tolerance extrapolates from the largest drawn size.
    _required = int(np.ceil(mc_sizes[-1] * (_mc_worst[-1] / tolerance) ** 2))
    _meets = [
        n for n, worst in zip(mc_sizes, _mc_worst, strict=True) if worst < tolerance
    ]
    chosen_samples = _meets[0] if _meets else mc_sizes[-1]
    if not _meets:
        print(
            f"no N <= {mc_sizes[-1]} meets tolerance {tolerance}; the 1/N "
            f"extrapolation needs N ~ {_required:,}"
        )
    _b_columns = [c for c in quadrature.columns if c.startswith("b_")]
    _rows = quadrature[quadrature["Z"] == mc_nodes]
    _quadrature = float(_rows[_b_columns].abs().max().max()) if len(_rows) else 0.0
    _rescaling = float(gate[_b_columns].abs().max().max())
    _mc = float(_mc_worst[mc_sizes.index(chosen_samples)])
    _bins = reference_waveform.build().frequencies.size
    recommendation = pd.DataFrame(
        [
            {
                "N": chosen_samples,
                "N required (1/N extrapolation)": _required,
                "Z": mc_nodes,
                "converged Z": chosen_nodes,
                "reference power [GB] at N required": _bins * _required * 8 / 1e9,
                "worst |b| rescaling": _rescaling,
                "worst |b| quadrature": _quadrature,
                "worst std b Monte Carlo": _mc,
                "combined": float(np.sqrt(_rescaling**2 + _quadrature**2 + _mc**2)),
            }
        ]
    )
    recommendation


@app.cell
def _(
    FIGURES,
    gate_figure,
    quadrature_figure,
    residual_figure,
    shift_figure,
    write_figures,
):
    if write_figures.value:
        save_figures(
            {
                FIGURES / "rescaling_gate.pdf": gate_figure,
                FIGURES / "quadrature_convergence.pdf": quadrature_figure,
                FIGURES / "monte_carlo_residuals.pdf": residual_figure,
                FIGURES / "monte_carlo_peak_shift.pdf": shift_figure,
            }
        )


if __name__ == "__main__":
    app.run()
