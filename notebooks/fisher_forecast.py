import marimo

__generated_with = "0.24.2"
app = marimo.App()

with app.setup(hide_code=True):
    from collections.abc import Mapping
    from pathlib import Path
    from typing import NamedTuple

    import corner
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    import pandas as pd
    from matplotlib.axes import Axes as MplAxes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.projections import register_projection
    from numpyro.distributions import MultivariateNormal, Normal, Uniform

    from astrogwb.paper.catalogs import load_run_catalog
    from astrogwb.paper.config import fiducials, networks, priors
    from astrogwb.paper.config.runs import ANALYSIS_PATH, CATALOGS_ROOT
    from astrogwb.paper.inference import prepare_inference_inputs
    from astrogwb.paper.plotting import (
        combo_colors,
        get_corner_kwargs,
        parameter_label,
        use_paper_style,
    )
    from astrogwb.paper.utils import load_mapping
    from astrogwb.populations import DEFAULT_DENSITY_SITES, build_population
    from astrogwb.sampling import fisher_matrix_per_bin


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Fisher forecast of the Gaussian spectrum likelihood

    Forecast $d_i \sim \mathcal{N}(S_i(\theta), \sigma_i)$ at the committed
    fiducials. $S_i(\theta)$ is the importance estimator
    `prepare_inference_inputs` builds from the injection and proposal
    catalogs. `fisher_matrix_per_bin` differentiates that spectrum once,
    through `spectral_density_jacobian`, and returns one Fisher matrix per
    frequency bin.

    The free parameters, in column order, are $H_0$, $\Omega_m$, $\Xi_0$,
    $n$, $\gamma$, $\kappa$, and $z_{\mathrm{peak}}$. `local_merger_rate`
    stays at its fiducial: it is an amplitude, degenerate with $H_0$.
    `minimum_mass` and `mass_width` stay at theirs, because a hard mass edge
    contributes no boundary term to this derivative.

    Each figure is one block. $H_0$ sits only in the cosmological block.
    Parameters outside a block stay fixed: their rows and columns are removed
    before the inversion. Low-frequency cutoffs of 2, 5, 10, and 20 Hz sum
    the bins `model_kwargs(fmin=...)` keeps. The 2 Hz cutoff is the full
    analysis band.

    Three Gaussian priors, centered at the fiducial, add $1/\sigma^2$ to a
    diagonal inside the block that contains the parameter. The posterior mean
    stays at the fiducial. $\Omega_m$ uses the production Normal scale. $n$
    (`xi_n`) and $\gamma$ take their widths from the production Uniform, in
    units of the standard-deviation slider. The slider does not rerun the
    Jacobian.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configuration

    Fiducials come from `fiducials()`, and the detector list for
    `ET-2L-aligned-CE-Hanford` from `networks()`. Observation time, the
    frequency band, the redshift grid, the population, and the catalog names
    come from `config/analysis.json`. Both roles read
    `outputs/catalogs/<name>.h5`. The notebook stops if either file is
    missing.
    """)
    return


@app.cell
def _():
    # This file lives in notebooks/, so the repository root is its grandparent.
    # `__file__` is the notebook path under `marimo edit` and when the file is
    # run as a script.
    ROOT_DIR = Path(__file__).resolve().parents[1]

    # gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes.
    # Restore matplotlib axes so plotting behaves as expected after importing
    # detector utilities.
    register_projection(MplAxes)

    jax.config.update("jax_enable_x64", True)
    use_paper_style(root=ROOT_DIR)

    FIDUCIALS = fiducials(root=ROOT_DIR)
    NETWORK = "ET-2L-aligned-CE-Hanford"
    detectors = networks(root=ROOT_DIR)[NETWORK]

    analysis = load_mapping(ROOT_DIR / ANALYSIS_PATH)["analysis"]
    observation_time = float(analysis["observation_time"])
    minimum_frequency = float(analysis["minimum_frequency"])
    maximum_frequency = float(analysis["maximum_frequency"])
    population = analysis["population"]
    population_kwargs = {
        key: int(value) if key == "n_grid" else float(value)
        for key, value in population["model_kwargs"].items()
    }
    model_name = str(population["model_name"])
    minimum_redshift = float(population_kwargs["minimum_redshift"])
    maximum_redshift = float(population_kwargs["maximum_redshift"])
    density_sites = tuple(population.get("density_sites", DEFAULT_DENSITY_SITES))
    catalog_names = analysis["catalog"]
    injection_path = ROOT_DIR / CATALOGS_ROOT / f"{catalog_names['injection']}.h5"
    proposal_path = ROOT_DIR / CATALOGS_ROOT / f"{catalog_names['proposal']}.h5"

    # Column order of the single Jacobian. H0 is only in the cosmological
    # block. The three held-fixed names stay in the parameter dict.
    COSMOLOGICAL = ("H0", "Omega_m")
    MODIFIED_PROPAGATION = ("xi_0", "xi_n")
    ASTROPHYSICAL = ("gamma", "kappa", "z_peak")
    FREE_PARAMETERS = COSMOLOGICAL + MODIFIED_PROPAGATION + ASTROPHYSICAL
    HELD_FIXED = ("local_merger_rate", "minimum_mass", "mass_width")
    classified = set(FREE_PARAMETERS) | set(HELD_FIXED)
    missing = sorted(classified - set(FIDUCIALS))
    unclassified = sorted(set(FIDUCIALS) - classified)
    if missing or unclassified:
        raise KeyError(
            "fiducial classification does not match fiducials(): "
            f"missing {missing}, unclassified {unclassified}"
        )

    CUTOFFS_HZ = (2.0, 5.0, 10.0, 20.0)
    if minimum_frequency != CUTOFFS_HZ[0]:
        raise ValueError(
            f"{CUTOFFS_HZ[0]:g} Hz is the full analysis band, but "
            f"analysis.minimum_frequency is {minimum_frequency:g} Hz"
        )
    if any(cutoff >= maximum_frequency for cutoff in CUTOFFS_HZ):
        raise ValueError(
            f"cutoffs {CUTOFFS_HZ} must lie inside the analysis band "
            f"ending at {maximum_frequency:g} Hz"
        )
    CUTOFF_COLORS = tuple(combo_colors(len(CUTOFFS_HZ)))
    N_CORNER_SAMPLES = 20_000

    print("network:", NETWORK, detectors)
    print("free:", ", ".join(FREE_PARAMETERS))
    print("fixed:", ", ".join(HELD_FIXED))
    print("injection:", injection_path)
    print("proposal:", proposal_path)
    return (
        ASTROPHYSICAL,
        COSMOLOGICAL,
        CUTOFFS_HZ,
        CUTOFF_COLORS,
        FIDUCIALS,
        FREE_PARAMETERS,
        MODIFIED_PROPAGATION,
        N_CORNER_SAMPLES,
        ROOT_DIR,
        density_sites,
        detectors,
        injection_path,
        maximum_frequency,
        maximum_redshift,
        minimum_frequency,
        minimum_redshift,
        model_name,
        observation_time,
        population_kwargs,
        proposal_path,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Importance spectrum

    `prepare_inference_inputs` restricts both catalogs to the analysis
    redshift window, builds the fiducial injection spectrum, the effective
    PSD, and the band mask, and binds the proposal catalog to the target
    population. Evaluated at the catalog fiducials, the injection draw is its
    own proposal: `config/analysis.json` names the same catalog for both
    roles. The per-bin scale and the frequency mask passed to the Fisher
    matrix are that object's `model_kwargs`. A low-frequency cutoff changes
    the mask only; the scale array stays the one fixed by the analysis band
    and the detector-coverage gaps.
    """)
    return


@app.cell
def _(
    density_sites,
    detectors,
    injection_path,
    maximum_frequency,
    maximum_redshift,
    minimum_frequency,
    minimum_redshift,
    model_name,
    observation_time,
    population_kwargs,
    proposal_path,
):
    injection = load_run_catalog(injection_path, label="injection")
    proposal = (
        injection
        if proposal_path == injection_path
        else load_run_catalog(proposal_path, label="proposal")
    )
    target = build_population(model_name, **population_kwargs)
    inputs = prepare_inference_inputs(
        injection,
        proposal,
        observation_time=observation_time,
        minimum_redshift=minimum_redshift,
        maximum_redshift=maximum_redshift,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        detectors=detectors,
        target=target,
        density_sites=density_sites,
    )
    _n_freq, _n_samples = inputs.proposal.polarization_power.shape
    print(f"proposal: n_frequency_bins={_n_freq} n_samples={_n_samples}")
    print(
        "band bins:",
        int(jnp.sum(inputs.observation.frequency_mask)),
        "of",
        _n_freq,
    )
    return (inputs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## One Jacobian

    The Jacobian is evaluated once, on the full analysis band. A cutoff below
    only selects which of those bins are summed.
    """)
    return


@app.cell
def _(CUTOFFS_HZ, FIDUCIALS, FREE_PARAMETERS, inputs):
    # One differentiation. `fisher_matrix_per_bin` calls
    # `spectral_density_jacobian`; cutoffs below only sum bins.
    _full_band = inputs.model_kwargs(fmin=CUTOFFS_HZ[0])
    per_bin_fisher = fisher_matrix_per_bin(
        inputs.spectral_density_fn,
        FIDUCIALS,
        FREE_PARAMETERS,
        scale=_full_band["scale"],
        frequency_mask=_full_band["frequency_mask"],
    )
    print(
        "per-bin Fisher",
        tuple(per_bin_fisher.shape),
        "parameters:",
        ", ".join(FREE_PARAMETERS),
    )
    return (per_bin_fisher,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Blocks and cutoffs

    For each block and each cutoff the cell adds that block's prior precision,
    prints the condition number of the result together with the $\sigma$ and
    $1/\sigma^2$ it used, then inverts. A covariance that is singular, not
    finite, or rejected by `MultivariateNormal` is reported and left out of
    the corner. Contours are about 20,000 draws at the fiducial mean.
    `plot_datapoints` is off: four sample clouds would hide the ellipses.
    """)
    return


@app.cell
def _(CUTOFFS_HZ, inputs, per_bin_fisher):
    _full_mask = np.asarray(
        inputs.model_kwargs(fmin=CUTOFFS_HZ[0])["frequency_mask"], dtype=bool
    )
    band_fishers: dict[float, tuple[int, np.ndarray]] = {}
    for _cutoff in CUTOFFS_HZ:
        _mask = np.asarray(
            inputs.model_kwargs(fmin=_cutoff)["frequency_mask"], dtype=bool
        )
        if _mask.shape != per_bin_fisher.shape[:1]:
            raise RuntimeError(
                "frequency mask and per-bin Fisher disagree on the number of bins: "
                f"{_mask.shape[0]} vs {per_bin_fisher.shape[0]}"
            )
        if not np.array_equal(_full_mask & _mask, _mask):
            raise RuntimeError(
                f"{_cutoff:g} Hz keeps bins outside the {CUTOFFS_HZ[0]:g} Hz band"
            )
        _kept = jnp.asarray(_mask)
        _total = np.asarray(
            jnp.sum(per_bin_fisher, axis=0, where=_kept[:, None, None]),
            dtype=float,
        )
        band_fishers[_cutoff] = (int(_mask.sum()), _total)
    # Likelihood information only. The standard-deviation slider adds prior
    # precision later, inside each block, and must not recompute this sum.
    return (band_fishers,)


@app.cell
def _(
    CUTOFFS_HZ,
    CUTOFF_COLORS,
    FIDUCIALS,
    FREE_PARAMETERS,
    N_CORNER_SAMPLES,
    ROOT_DIR,
):
    class BlockForecast(NamedTuple):
        n_bins: int
        condition_number: float
        covariance: np.ndarray | None
        draws: np.ndarray | None
        failure: str | None

    def _invert(block: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        try:
            covariance = np.linalg.inv(block)
        except np.linalg.LinAlgError as error:
            return None, f"inversion failed ({error})"
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        if not np.all(np.isfinite(covariance)) or np.any(eigenvalues <= 0.0):
            minimum = float(np.min(eigenvalues))
            return (
                None,
                f"covariance is not positive definite (min eigenvalue {minimum:.3e})",
            )
        return covariance, None

    def _sample(
        names: tuple[str, ...], covariance: np.ndarray, cutoff: float
    ) -> np.ndarray:
        mean = jnp.asarray([FIDUCIALS[name] for name in names], dtype=float)
        distribution = MultivariateNormal(
            loc=mean,
            covariance_matrix=jnp.asarray(covariance),
        )
        key = jax.random.fold_in(jax.random.key(0), round(cutoff))
        return np.asarray(distribution.sample(key, sample_shape=(N_CORNER_SAMPLES,)))

    def _format_priors(
        names: tuple[str, ...], prior_sigmas: Mapping[str, float]
    ) -> str:
        parts: list[str] = []
        for name in names:
            sigma = prior_sigmas.get(name)
            if sigma is None:
                continue
            precision = 1.0 / float(sigma) ** 2
            parts.append(f"{name} sigma={float(sigma):.6g}, 1/sigma**2={precision:.6g}")
        if not parts:
            return "no Gaussian prior"
        return "; ".join(parts)

    def _with_prior(
        block: np.ndarray,
        names: tuple[str, ...],
        prior_sigmas: Mapping[str, float],
    ) -> np.ndarray:
        """Add 1/sigma**2 on this block's prior diagonals.

        The prior is centered at the fiducial, so the posterior mean stays
        there. Parameters outside ``names`` are untouched: a prior is applied
        only inside the block that contains it.
        """
        updated = np.array(block, dtype=float, copy=True)
        for offset, name in enumerate(names):
            sigma = prior_sigmas.get(name)
            if sigma is None:
                continue
            updated[offset, offset] += 1.0 / float(sigma) ** 2
        return updated

    def _forecasts(
        title: str,
        names: tuple[str, ...],
        band_fishers: Mapping[float, tuple[int, np.ndarray]],
        prior_sigmas: Mapping[str, float],
    ) -> dict[float, BlockForecast]:
        index = tuple(FREE_PARAMETERS.index(name) for name in names)
        prior_note = _format_priors(names, prior_sigmas)
        forecasts: dict[float, BlockForecast] = {}
        for cutoff in CUTOFFS_HZ:
            n_bins, total = band_fishers[cutoff]
            block = _with_prior(
                np.asarray(total[np.ix_(index, index)], dtype=float),
                names,
                prior_sigmas,
            )
            condition = float(np.linalg.cond(block))
            print(
                f"{title} ({', '.join(names)}) at {cutoff:g} Hz: "
                f"{n_bins} bins, condition number {condition:.6g}; {prior_note}"
            )
            if not np.isfinite(condition):
                print(
                    "  unusable covariance (Fisher block is singular); contour skipped"
                )
                forecasts[cutoff] = BlockForecast(
                    n_bins, condition, None, None, "Fisher block is singular"
                )
                continue
            covariance, failure = _invert(block)
            draws = None
            if covariance is not None:
                try:
                    draws = _sample(names, covariance, cutoff)
                except ValueError as error:
                    covariance = None
                    failure = f"MultivariateNormal rejected the covariance ({error})"
            if failure is not None:
                print(f"  unusable covariance ({failure}); contour skipped")
            forecasts[cutoff] = BlockForecast(
                n_bins, condition, covariance, draws, failure
            )
        return forecasts

    def _corner(
        names: tuple[str, ...], forecasts: Mapping[float, BlockForecast]
    ) -> Figure | None:
        usable = {
            cutoff: result
            for cutoff, result in forecasts.items()
            if result.draws is not None
        }
        if not usable:
            return None
        stacked = np.concatenate([result.draws for result in usable.values()], axis=0)
        truth = np.asarray([FIDUCIALS[name] for name in names], dtype=float)
        low = np.minimum(stacked.min(axis=0), truth)
        high = np.maximum(stacked.max(axis=0), truth)
        width = np.maximum(high - low, 1e-12)
        pad = 0.05 * width
        span = [
            (float(lo), float(hi)) for lo, hi in zip(low - pad, high + pad, strict=True)
        ]
        labels = [parameter_label(name, root=ROOT_DIR) for name in names]
        truths = [FIDUCIALS[name] for name in names]
        colors = dict(zip(CUTOFFS_HZ, CUTOFF_COLORS, strict=True))
        corner_kwargs = get_corner_kwargs(plot_datapoints=False)
        fig: Figure | None = None
        for cutoff, result in usable.items():
            fig = corner.corner(
                result.draws,
                fig=fig,
                range=span,
                color=colors[cutoff],
                labels=labels,
                truths=truths if fig is None else None,
                hist_kwargs={"linewidth": 1.5},
                contour_kwargs={"linewidths": 1.5},
                **corner_kwargs,
            )
        if fig is None:
            return None
        handles = [
            Line2D([], [], color=colors[cutoff], lw=1.5, label=f"{cutoff:g} Hz")
            for cutoff in usable
        ]
        axes = np.asarray(fig.axes).reshape(len(names), len(names))
        axes[0, -1].legend(handles=handles, loc="center", frameon=False)
        return fig

    def _sigma_table(
        names: tuple[str, ...], forecasts: Mapping[float, BlockForecast]
    ) -> pd.DataFrame:
        columns = {
            f"{cutoff:g} Hz": (
                np.sqrt(np.diag(result.covariance))
                if result.covariance is not None
                else np.full(len(names), np.nan)
            )
            for cutoff, result in forecasts.items()
        }
        table = pd.DataFrame(
            columns,
            index=[parameter_label(name, root=ROOT_DIR) for name in names],
        )
        table.index.name = "parameter"
        return table

    def render_block(
        title: str,
        names: tuple[str, ...],
        band_fishers: Mapping[float, tuple[int, np.ndarray]],
        prior_sigmas: Mapping[str, float],
    ) -> mo.Html:
        forecasts = _forecasts(title, names, band_fishers, prior_sigmas)
        condition_rows: list[dict[str, float | str]] = []
        for cutoff, result in forecasts.items():
            row: dict[str, float | str] = {
                "cutoff": f"{cutoff:g} Hz",
                "bins": result.n_bins,
                "condition number": result.condition_number,
            }
            for name in names:
                sigma = prior_sigmas.get(name)
                if sigma is None:
                    continue
                row[f"{name} sigma"] = float(sigma)
                row[f"{name} 1/sigma**2"] = 1.0 / float(sigma) ** 2
            row["status"] = "usable" if result.failure is None else result.failure
            condition_rows.append(row)
        condition = pd.DataFrame(condition_rows).set_index("cutoff")
        pieces: list[object] = [
            mo.md(f"### {title}"),
            condition,
            mo.md(
                r"Marginal standard deviation $\sqrt{\mathrm{diag}(C)}$ of the "
                r"likelihood plus the Gaussian prior, with every parameter "
                r"outside the block held at its fiducial."
            ),
            _sigma_table(names, forecasts),
        ]
        fig = _corner(names, forecasts)
        if fig is None:
            pieces.append(mo.md("No cutoff produced a usable covariance."))
        else:
            pieces.append(fig)
        return mo.vstack(pieces)

    return (render_block,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Gaussian priors

    Each prior is a Gaussian centered at the fiducial. Its Fisher contribution
    is $1/\sigma^2$ on that parameter's diagonal, and only inside the block
    that contains the parameter. The mean of the forecast stays at the
    fiducial.

    The widths come from `priors()` (`config/priors.json`):

    - $\Omega_m$ uses the production Normal scale as $\sigma$. The slider
      does not rescale it.
    - $n$ (`xi_n`) uses $\sigma = (n_{\mathrm{fid}} - \mathrm{low}) / N_{\sigma}$,
      where $\mathrm{low}$ is the production Uniform lower edge. At the
      default, that edge sits two standard deviations below the fiducial.
    - $\gamma$ is the low-redshift power-law slope of the Madau–Dickinson
      rate, $(1+z)^{\gamma}$. Its $\sigma$ is the production Uniform width
      divided by $N_{\sigma}$, so the full production range is
      $N_{\sigma}$ standard deviations.
    """)
    return


@app.cell
def _():
    num_sigma_slider = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=2,
        debounce=True,
        show_value=True,
        label="Standard deviations spanning each constructed prior width",
    )
    num_sigma_slider
    return (num_sigma_slider,)


@app.cell
def _(num_sigma_slider):
    num_sigma = int(num_sigma_slider.value)
    return (num_sigma,)


@app.cell
def _(
    ASTROPHYSICAL,
    COSMOLOGICAL,
    FIDUCIALS,
    MODIFIED_PROPAGATION,
    ROOT_DIR,
    num_sigma,
):
    _production = priors(root=ROOT_DIR)
    _omega_m = _production["Omega_m"]
    _xi_n = _production["xi_n"]
    _gamma = _production["gamma"]
    if not isinstance(_omega_m, Normal):
        raise TypeError(
            f"Omega_m production prior must be Normal, got {type(_omega_m).__name__}"
        )
    if not isinstance(_xi_n, Uniform):
        raise TypeError(
            f"xi_n production prior must be Uniform, got {type(_xi_n).__name__}"
        )
    if not isinstance(_gamma, Uniform):
        raise TypeError(
            f"gamma production prior must be Uniform, got {type(_gamma).__name__}"
        )
    # gamma, not kappa: madau_dickinson_rate goes as (1+z)^gamma at low redshift.
    _raw_sigmas = {
        "Omega_m": float(_omega_m.scale),
        "xi_n": (float(FIDUCIALS["xi_n"]) - float(_xi_n.low)) / num_sigma,
        "gamma": (float(_gamma.high) - float(_gamma.low)) / num_sigma,
    }
    _blocks = {
        "Omega_m": COSMOLOGICAL,
        "xi_n": MODIFIED_PROPAGATION,
        "gamma": ASTROPHYSICAL,
    }
    for _name, _sigma in _raw_sigmas.items():
        if not np.isfinite(_sigma) or _sigma <= 0.0:
            raise ValueError(f"{_name} prior sigma must be positive, got {_sigma}")
        if _name not in _blocks[_name]:
            raise RuntimeError(f"{_name} is not in its Fisher block {_blocks[_name]}")
    prior_sigmas = _raw_sigmas
    for _name, _sigma in prior_sigmas.items():
        print(f"{_name}: sigma={_sigma:.6g}, 1/sigma**2={1.0 / _sigma**2:.6g}")
    return (prior_sigmas,)


@app.cell
def _(COSMOLOGICAL, band_fishers, prior_sigmas, render_block):
    cosmological = render_block(
        "Cosmological", COSMOLOGICAL, band_fishers, prior_sigmas
    )
    cosmological
    return


@app.cell
def _(MODIFIED_PROPAGATION, band_fishers, prior_sigmas, render_block):
    modified_propagation = render_block(
        "Modified propagation",
        MODIFIED_PROPAGATION,
        band_fishers,
        prior_sigmas,
    )
    modified_propagation
    return


@app.cell
def _(ASTROPHYSICAL, band_fishers, prior_sigmas, render_block):
    astrophysical = render_block(
        "Astrophysical", ASTROPHYSICAL, band_fishers, prior_sigmas
    )
    astrophysical
    return


if __name__ == "__main__":
    app.run()
