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
    from astrogwb.paper.config.mcmc import materialize_prior
    from astrogwb.paper.config.runs import ANALYSIS_PATH, CATALOGS_ROOT, assemble_run
    from astrogwb.paper.inference import prepare_inference_inputs
    from astrogwb.paper.plotting import (
        combo_colors,
        get_corner_kwargs,
        parameter_label,
        use_paper_style,
    )
    from astrogwb.paper.plotting.fisher import (
        plot_fisher_eigenmodes,
        plot_template_composition,
        plot_whitened_derivatives,
    )
    from astrogwb.paper.utils import load_mapping
    from astrogwb.populations import DEFAULT_DENSITY_SITES, build_population
    from astrogwb.sampling import (
        cumulative_template_fractions,
        derivative_cosine_matrix,
        fisher_eigenmodes,
        fisher_from_whitened_jacobian,
        fisher_svd,
        post_newtonian_templates,
        prior_sigma_along_modes,
        whitened_jacobian,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Fisher forecast of the Gaussian spectrum likelihood

    Forecast $d_i \sim \mathcal{N}(S_i(\theta), \sigma_i)$ at the committed
    fiducials. $S_i(\theta)$ is the importance estimator
    `prepare_inference_inputs` builds from the injection and proposal
    catalogs. `whitened_jacobian` differentiates that spectrum once, in noise
    units, and `fisher_from_whitened_jacobian` turns the result into one
    Fisher matrix per frequency bin.

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

    A selector picks the population. Two cases replace the baseline, and each
    frees one block while every other fiducial stays fixed:

    - **Time-delayed Madau–Dickinson** (`bns_md_time_delayed_cosmological`):
      the Madau–Dickinson law is the formation rate, and mergers follow after
      a delay $p(\tau) \propto \tau^{\alpha_\tau}$. The delay is fixed in
      Gyr, so $H_0$ reshapes the redshift law instead of only rescaling it.
      Free: $H_0$ and $\alpha_\tau$ (`delay_slope`).
    - **Gaussian masses** (`bns_md_gaussian_cosmological`): the baseline
      Madau–Dickinson redshift law, undelayed and held fixed, with both
      component masses drawn from $\mathcal{N}(\mu_m, \sigma_m^2)$ and
      ordered. Unlike the ordered-uniform masses, this law has no hard edge,
      so its derivative is complete. Free: $\mu_m$ (`mass_mean`) and
      $\sigma_m$ (`mass_sigma`). At the fiducial each derivative is a
      score-function Monte Carlo average over the catalog. For
      $32768$ sources, a bootstrap gives $\partial \ln S / \partial \mu_m
      \approx 1.07 \pm 0.13$ per $M_\odot$ (the inspiral estimate is
      $5 / 3\mu_m \approx 1.25$). $\partial \ln S / \partial \sigma_m
      \approx -0.14 \pm 0.19$ is noise: its expected value is
      $-4\sigma_m / 9\mu_m^2 \approx -0.02$, from the loss of chirp mass in
      unequal pairs. Read the $\sigma_m$ row, and the $\mu_m$–$\sigma_m$
      correlation, as limited by the catalog, not by the detectors.
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

    For the other two cases the same values come instead from a committed
    run, merged by `assemble_run` exactly as the workflow merges it:

    - `time-delay/delay-slope` adds the delayed population and its
      construction kwargs, the `delay_slope` fiducial and prior, the delayed
      injection catalog, and the $\epsilon = 0.1$ guard-mixture proposal.
    - `mass-model/gaussian-mass` adds the Gaussian-mass population, the
      `mass_mean` and `mass_sigma` fiducials and priors, and the
      Gaussian-mass catalog, which is both the injection and, as in
      `config/analysis.json`, its own proposal.

    Changing the selection reloads the catalogs and recomputes the Jacobian.
    """)
    return


@app.cell
def _():
    population_case = mo.ui.dropdown(
        options={
            "Madau–Dickinson (baseline)": "baseline",
            "Time-delayed Madau–Dickinson": "time-delay",
            "Gaussian masses": "gaussian-mass",
        },
        value="Madau–Dickinson (baseline)",
        label="Population",
    )
    population_case
    return (population_case,)


@app.cell
def _(population_case):
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

    case = str(population_case.value)
    NETWORK = "ET-2L-aligned-CE-Hanford"
    detectors = networks(root=ROOT_DIR)[NETWORK]

    # Each non-baseline case: the committed run it reads, and its one block.
    # H0 is no amplitude in the delayed case: the delay in Gyr reshapes the
    # law. The Gaussian-mass case keeps the undelayed redshift law fixed.
    _cases = {
        "time-delay": (("time-delay", "delay-slope"), ("H0", "delay_slope")),
        "gaussian-mass": (("mass-model", "gaussian-mass"), ("mass_mean", "mass_sigma")),
    }
    _titles = {"time-delay": "Time delay", "gaussian-mass": "Mass"}
    if case in _cases:
        # The committed experiment, merged as the workflow merges it, so the
        # population, catalogs, fiducials and priors cannot drift from it.
        _reference_run, _names = _cases[case]
        _run = assemble_run(*_reference_run, root=ROOT_DIR)
        analysis = _run["analysis"]
        FIDUCIALS = {name: float(value) for name, value in _run["fiducials"].items()}
        PRODUCTION_PRIORS = {
            name: materialize_prior(spec) for name, spec in _run["priors"].items()
        }
        BLOCKS = {_titles[case]: _names}
    elif case == "baseline":
        analysis = load_mapping(ROOT_DIR / ANALYSIS_PATH)["analysis"]
        FIDUCIALS = fiducials(root=ROOT_DIR)
        PRODUCTION_PRIORS = priors(root=ROOT_DIR)
        # H0 is only in the cosmological block.
        BLOCKS = {
            "Cosmological": ("H0", "Omega_m"),
            "Modified propagation": ("xi_0", "xi_n"),
            "Astrophysical": ("gamma", "kappa", "z_peak"),
        }
    else:
        raise ValueError(f"unknown population case {case!r}")
    observation_time = float(analysis["observation_time"])
    minimum_frequency = float(analysis["minimum_frequency"])
    maximum_frequency = float(analysis["maximum_frequency"])
    population = analysis["population"]
    # JSON keeps the integer kwargs (n_grid, n_delay_nodes) integers.
    population_kwargs = {
        key: value if isinstance(value, int) else float(value)
        for key, value in population["model_kwargs"].items()
    }
    model_name = str(population["model_name"])
    minimum_redshift = float(population_kwargs["minimum_redshift"])
    maximum_redshift = float(population_kwargs["maximum_redshift"])
    density_sites = tuple(population.get("density_sites", DEFAULT_DENSITY_SITES))
    catalog_names = analysis["catalog"]
    injection_path = ROOT_DIR / CATALOGS_ROOT / f"{catalog_names['injection']}.h5"
    proposal_path = ROOT_DIR / CATALOGS_ROOT / f"{catalog_names['proposal']}.h5"

    # Column order of the single Jacobian: the blocks, concatenated. Every
    # held-fixed name stays in the parameter dict at its fiducial.
    FREE_PARAMETERS = tuple(name for names in BLOCKS.values() for name in names)
    HELD_FIXED = (
        tuple(name for name in FIDUCIALS if name not in FREE_PARAMETERS)
        if case != "baseline"
        else ("local_merger_rate", "minimum_mass", "mass_width")
    )
    classified = set(FREE_PARAMETERS) | set(HELD_FIXED)
    missing = sorted(classified - set(FIDUCIALS))
    unclassified = sorted(set(FIDUCIALS) - classified)
    if missing or unclassified:
        raise KeyError(
            "fiducial classification does not match the fiducials: "
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

    print("population:", model_name)
    print("network:", NETWORK, detectors)
    print("free:", ", ".join(FREE_PARAMETERS))
    print("fixed:", ", ".join(HELD_FIXED))
    print("injection:", injection_path)
    print("proposal:", proposal_path)
    return (
        BLOCKS,
        CUTOFFS_HZ,
        CUTOFF_COLORS,
        FIDUCIALS,
        FREE_PARAMETERS,
        N_CORNER_SAMPLES,
        PRODUCTION_PRIORS,
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
        case,
        proposal_path,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Importance spectrum

    `prepare_inference_inputs` restricts both catalogs to the analysis
    redshift window, builds the fiducial injection spectrum, the effective
    PSD, and the band mask, and binds the proposal catalog to the target
    population. Without the time delay, the injection draw is its own
    proposal: `config/analysis.json` names the same catalog for both roles.
    With it, the proposal is the run's guard mixture, and the Jacobian is
    that of the importance estimator the run samples. The per-bin scale and the frequency mask passed to the Fisher
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
    # One differentiation. The whitened derivatives feed both the Fisher
    # matrices and the degeneracy figures; cutoffs below only sum bins.
    _full_band = inputs.model_kwargs(fmin=CUTOFFS_HZ[0])
    whitened = whitened_jacobian(
        inputs.spectral_density_fn,
        FIDUCIALS,
        FREE_PARAMETERS,
        scale=_full_band["scale"],
        frequency_mask=_full_band["frequency_mask"],
    )
    per_bin_fisher = fisher_from_whitened_jacobian(whitened)
    print(
        "per-bin Fisher",
        tuple(per_bin_fisher.shape),
        "parameters:",
        ", ".join(FREE_PARAMETERS),
    )
    return per_bin_fisher, whitened


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

    For the baseline, the widths come from `priors()`
    (`config/priors.json`):

    - $\Omega_m$ uses the production Normal scale as $\sigma$. The slider
      does not rescale it.
    - $n$ (`xi_n`) uses $\sigma = (n_{\mathrm{fid}} - \mathrm{low}) / N_{\sigma}$,
      where $\mathrm{low}$ is the production Uniform lower edge. At the
      default, that edge sits two standard deviations below the fiducial.
    - $\gamma$ is the low-redshift power-law slope of the Madau–Dickinson
      rate, $(1+z)^{\gamma}$. Its $\sigma$ is the production Uniform width
      divided by $N_{\sigma}$, so the full production range is
      $N_{\sigma}$ standard deviations.

    With the time delay, only $\alpha_\tau$ (`delay_slope`) carries a prior.
    Its $\sigma$ is the width of the `time-delay` experiment's Uniform divided
    by $N_{\sigma}$, as for $\gamma$. $H_0$ keeps none: its production prior
    is a wide Uniform.

    With Gaussian masses, $\mu_m$ and $\sigma_m$ each carry one, built the
    same way from the `mass-model` experiment's Uniforms.
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
def _(BLOCKS, FIDUCIALS, PRODUCTION_PRIORS, case, num_sigma):
    def _require(name: str, kind: type) -> Normal | Uniform:
        prior = PRODUCTION_PRIORS[name]
        if not isinstance(prior, kind):
            raise TypeError(
                f"{name} production prior must be {kind.__name__}, "
                f"got {type(prior).__name__}"
            )
        return prior

    def _width(name: str) -> float:
        prior = _require(name, Uniform)
        return (float(prior.high) - float(prior.low)) / num_sigma

    if case == "time-delay":
        _raw_sigmas = {"delay_slope": _width("delay_slope")}
    elif case == "gaussian-mass":
        _raw_sigmas = {name: _width(name) for name in ("mass_mean", "mass_sigma")}
    else:
        _raw_sigmas = {
            "Omega_m": float(_require("Omega_m", Normal).scale),
            "xi_n": (float(FIDUCIALS["xi_n"]) - float(_require("xi_n", Uniform).low))
            / num_sigma,
            # gamma, not kappa: madau_dickinson_rate goes as (1+z)^gamma at
            # low redshift.
            "gamma": _width("gamma"),
        }
    _free = {name for names in BLOCKS.values() for name in names}
    for _name, _sigma in _raw_sigmas.items():
        if not np.isfinite(_sigma) or _sigma <= 0.0:
            raise ValueError(f"{_name} prior sigma must be positive, got {_sigma}")
        if _name not in _free:
            raise RuntimeError(f"{_name} is in no Fisher block {tuple(BLOCKS)}")
    prior_sigmas = _raw_sigmas
    for _name, _sigma in prior_sigmas.items():
        print(f"{_name}: sigma={_sigma:.6g}, 1/sigma**2={1.0 / _sigma**2:.6g}")
    return (prior_sigmas,)


@app.cell
def _(BLOCKS, band_fishers, prior_sigmas, render_block):
    forecasts = mo.vstack(
        [
            render_block(title, names, band_fishers, prior_sigmas)
            for title, names in BLOCKS.items()
        ]
    )
    forecasts
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Degeneracies

    Two views of the same likelihood-only Fisher matrix, for each block and,
    for the baseline, for all seven free parameters together.

    **Derivative shapes.** $w_a(f) = \partial_a S / \sigma$ is each
    parameter's effect on the spectrum in noise units, and
    $F_{ab} = \sum_i w_{ia} w_{ib}$. Each curve is scaled to unit norm, so two
    curves of the same shape, up to sign, are a degeneracy: the parameters
    bend the spectrum the same way. The table gives the cosine between the
    curves, $F_{ab} / \sqrt{F_{aa} F_{bb}}$; for two parameters it is minus
    the forecast correlation. The lower panel shows the fraction of each
    parameter's information that a low-frequency cutoff at $f$ keeps.

    **Eigenmodes.** Each parameter is divided by the standard deviation of
    its production prior, the Uniform or Normal in `priors()` (or in the
    case's committed run), and the Fisher matrix is diagonalized. Without that
    rescaling the modes would depend on units: $H_0$ in km/s/Mpc would
    dominate every one. A bar is the likelihood width of one constrained
    combination; the marker is the Gaussian prior's width along it, from
    the slider above. A bar above its marker is a prior-dominated mode. The
    heat map gives each parameter's loading on each mode.
    """)
    return


@app.cell
def _(CUTOFFS_HZ):
    degeneracy_cutoff = mo.ui.dropdown(
        options={f"{cutoff:g} Hz": cutoff for cutoff in CUTOFFS_HZ},
        value=f"{CUTOFFS_HZ[0]:g} Hz",
        label="Low-frequency cutoff for the cosines and eigenmodes",
    )
    degeneracy_cutoff
    return (degeneracy_cutoff,)


@app.cell
def _(
    CUTOFFS_HZ,
    FREE_PARAMETERS,
    PRODUCTION_PRIORS,
    ROOT_DIR,
    inputs,
    whitened,
):
    _full_mask = np.asarray(
        inputs.model_kwargs(fmin=CUTOFFS_HZ[0])["frequency_mask"], dtype=bool
    )
    _band_frequencies = np.asarray(inputs.observation.frequencies)[_full_mask]
    _band_whitened = np.asarray(whitened)[_full_mask]
    # Production prior standard deviations: the units the eigenmodes use.
    _reference_scales = {
        name: float(np.sqrt(PRODUCTION_PRIORS[name].variance))
        for name in FREE_PARAMETERS
    }

    def render_degeneracies(
        title: str,
        names: tuple[str, ...],
        fisher: np.ndarray,
        prior_sigmas: Mapping[str, float],
    ) -> mo.Html:
        """Derivative shapes, cosines, and eigenmodes of one parameter set.

        ``fisher`` is the likelihood-only matrix over ``FREE_PARAMETERS`` at
        the chosen cutoff; the rows and columns outside ``names`` are held
        fixed, as in the forecast blocks.
        """
        index = [FREE_PARAMETERS.index(name) for name in names]
        labels = [parameter_label(name, root=ROOT_DIR) for name in names]
        block = np.asarray(fisher, dtype=float)[np.ix_(index, index)]
        shapes = plot_whitened_derivatives(
            _band_frequencies,
            _band_whitened[:, index],
            labels,
            colors=combo_colors(len(names)),
            cutoffs=CUTOFFS_HZ[1:],
        )
        cosines = pd.DataFrame(
            derivative_cosine_matrix(block), index=labels, columns=labels
        )
        modes = fisher_eigenmodes(
            block,
            names,
            parameter_scales=[_reference_scales[name] for name in names],
        )
        block_priors = {
            name: sigma for name, sigma in prior_sigmas.items() if name in names
        }
        prior_widths = prior_sigma_along_modes(modes, block_priors)
        eigen_figure = plot_fisher_eigenmodes(
            modes.sigmas,
            modes.directions,
            labels,
            prior_sigmas=prior_widths,
            sigma_label=r"$\sigma$ along mode [prior std]",
        )
        summary = pd.DataFrame(
            {
                "likelihood sigma": modes.sigmas,
                "prior sigma": prior_widths,
                "prior-dominated": modes.sigmas > prior_widths,
            },
            index=pd.Index(np.arange(1, len(names) + 1), name="mode"),
        )
        return mo.vstack(
            [
                mo.md(f"### {title}"),
                shapes,
                mo.md("Cosine between whitened derivative curves"),
                cosines,
                eigen_figure,
                mo.md("Widths in units of each parameter's production prior std"),
                summary,
            ]
        )

    return (render_degeneracies,)


@app.cell
def _(
    BLOCKS,
    FREE_PARAMETERS,
    band_fishers,
    degeneracy_cutoff,
    prior_sigmas,
    render_degeneracies,
):
    _targets = dict(BLOCKS)
    if len(BLOCKS) > 1:
        _targets["All free parameters"] = FREE_PARAMETERS
    _, _fisher = band_fishers[degeneracy_cutoff.value]
    degeneracies = mo.vstack(
        [
            render_degeneracies(title, names, _fisher, prior_sigmas)
            for title, names in _targets.items()
        ]
    )
    degeneracies
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Singular modes

    The same forecast, decomposed on the Jacobian instead of on the Fisher
    matrix. With every free parameter divided by its production-prior
    standard deviation, $\tilde W = W D = U \Sigma V^\top$ and
    $\tilde F = V \Sigma^2 V^\top$. Mode $k$ is the combination $V_k$,
    measured to $1/\Sigma_k$ prior standard deviations, and it changes the
    whitened spectrum along the unit template $U_k$. `local_merger_rate` stays
    fixed, so the first mode is the overall amplitude.

    - **Widths.** The table compares $1/\Sigma_k$ with the eigenmodes above.
      The SVD works at the square root of the Fisher matrix's condition
      number, so a mode that the eigen-decomposition rounds to $\infty$ can
      come back finite but weak. The last column is the Gaussian prior's
      width along the mode.
    - **Templates.** The leading $U_k$ against frequency, with the share of
      each mode's information that a cutoff at $f$ keeps.
    - **What the templates are.** Each $U_k$ is projected onto a nested
      basis: the whitened spectrum itself (a rescaling), then its relative
      1PN, 1.5PN and 2PN corrections, $S (f/f_\mathrm{ref})^{e} / \sigma$ with
      $e = 2/3, 1, 4/3$. A template spanned by the first two is the
      1PN tilt, whose population coefficient is the flux-weighted
      $\langle [M(1+z)]^{2/3} \rangle$. What the basis leaves over is shape
      from the merger or tidal part of the waveform, or noise.

    Everything here follows the cutoff dropdown above. The weakest modes sit
    near the importance estimator's Monte Carlo noise, so read their
    templates loosely.
    """)
    return


@app.cell
def _(
    CUTOFFS_HZ,
    FIDUCIALS,
    FREE_PARAMETERS,
    PRODUCTION_PRIORS,
    ROOT_DIR,
    band_fishers,
    degeneracy_cutoff,
    inputs,
    prior_sigmas,
    whitened,
):
    # Relative PN orders of an inspiral energy spectrum, as powers of f.
    _exponents = (0.0, 2.0 / 3.0, 1.0, 4.0 / 3.0)
    _basis_labels = (
        "spectrum (amplitude)",
        r"+1PN $f^{2/3}$",
        r"+1.5PN $f$",
        r"+2PN $f^{4/3}$",
    )
    _cutoff = float(degeneracy_cutoff.value)
    _band = inputs.model_kwargs(fmin=_cutoff)
    _mask = np.asarray(_band["frequency_mask"], dtype=bool)
    _frequencies = np.asarray(inputs.observation.frequencies)[_mask]
    _scales = [
        float(np.sqrt(PRODUCTION_PRIORS[name].variance)) for name in FREE_PARAMETERS
    ]
    _labels = [parameter_label(name, root=ROOT_DIR) for name in FREE_PARAMETERS]

    singular_modes = fisher_svd(
        np.asarray(whitened)[_mask], FREE_PARAMETERS, parameter_scales=_scales
    )
    _eigen = fisher_eigenmodes(
        band_fishers[_cutoff][1], FREE_PARAMETERS, parameter_scales=_scales
    )
    _mode_labels = [f"mode {k + 1}" for k in range(len(FREE_PARAMETERS))]
    _widths = pd.DataFrame(
        {
            "SVD sigma": singular_modes.sigmas,
            "eigen sigma": _eigen.sigmas,
            "prior sigma": prior_sigma_along_modes(singular_modes, prior_sigmas),
        },
        index=pd.Index(_mode_labels, name="mode"),
    )
    _loadings = pd.DataFrame(
        singular_modes.directions, index=_labels, columns=_mode_labels
    ).round(3)

    _n_shown = min(3, len(FREE_PARAMETERS))
    _templates = plot_whitened_derivatives(
        _frequencies,
        singular_modes.templates[:, :_n_shown],
        [
            f"mode {k + 1} ($\\sigma$ = {singular_modes.sigmas[k]:.2g})"
            for k in range(_n_shown)
        ],
        colors=combo_colors(_n_shown),
        cutoffs=[cutoff for cutoff in CUTOFFS_HZ if cutoff > _cutoff],
    )
    _whitened_spectrum = (
        np.asarray(inputs.spectral_density_fn(FIDUCIALS)[0])[_mask]
        / np.asarray(_band["scale"])[_mask]
    )
    _basis = post_newtonian_templates(
        _frequencies, _whitened_spectrum, _exponents, reference_frequency=10.0
    )
    template_fractions = cumulative_template_fractions(singular_modes.templates, _basis)
    _composition = plot_template_composition(
        template_fractions,
        _basis_labels,
        [
            f"{label} ($\\sigma$ = {sigma:.2g})"
            for label, sigma in zip(_mode_labels, singular_modes.sigmas, strict=True)
        ],
        colors=combo_colors(len(_basis_labels)),
    )
    _fractions = pd.DataFrame(
        template_fractions, index=_mode_labels, columns=list(_basis_labels)
    ).round(4)
    mo.vstack(
        [
            mo.md(f"### All free parameters at {_cutoff:g} Hz"),
            mo.md("Widths in prior standard deviations"),
            _widths,
            mo.md("Parameter loadings $V_k$"),
            _loadings,
            _templates,
            mo.md("Cumulative share of each template spanned by the basis"),
            _fractions,
            _composition,
        ]
    )
    return singular_modes, template_fractions


if __name__ == "__main__":
    app.run()
