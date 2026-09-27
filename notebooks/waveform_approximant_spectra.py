import marimo

__generated_with = "0.25.0"
app = marimo.App()


with app.setup(hide_code=True):
    from collections.abc import Mapping, Sequence
    from functools import partial
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.figure import Figure
    from numpyro.infer import Predictive

    from astrogwb.paper.config import fiducials, population_model, waveform_generator
    from astrogwb.paper.config.runs import FIGURES_DIR
    from astrogwb.paper.plotting import save_figures, use_paper_style
    from astrogwb.populations import IsotropicInclination
    from astrogwb.sampling import gwb_forward_model
    from astrogwb.utils import years_to_seconds


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Waveform-approximant spectral draws

    Compare stochastic-background spectra made from the *same events* with
    four Ripple frequency-domain approximants. Reusing one PRNG key for each
    NumPyro `Predictive` call makes every call replay the same event count
    and source latent variables; only `WaveformMetadata.approximant` changes.
    The solid line is the median of the retained draws and the shaded region
    is the central percentile band.

    Higher-mode waveforms depend on inclination, so the source model is
    wrapped with `IsotropicInclination`: each event draws $\iota$ from
    the isotropic law ($\cos\iota$ uniform on $[-1, 1]$). Returning
    `inclination` also disables the analytic $2/5$ face-on-to-isotropic
    rescaling, which is only valid for quadrupole waveforms. The shared PRNG
    key then replays the same orientations for every approximant.

    The lower panel shows fractional residuals
    $(S_h^A - S_h^{\mathrm{ref}}) / S_h^{\mathrm{ref}}$ against the reference
    approximant chosen below. Bins where the reference is exactly zero are
    undefined and are masked rather than divided.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Notebook configuration
    """)
    return


@app.cell
def _():
    # This file lives in notebooks/, so the repository root is its grandparent.
    # `__file__` is the notebook path under `marimo edit` and when the file is
    # run as a script.
    ROOT_DIR = Path(__file__).resolve().parents[1]

    # Configure precision before constructing a JAX array or querying a device.
    jax.config.update("jax_enable_x64", True)
    use_paper_style(root=ROOT_DIR)

    #: Where this notebook's figure goes: under the one output root the
    #: workflow, the figure scripts and `config/plotting.json` all agree on.
    OUTPUT_PATH = ROOT_DIR / FIGURES_DIR / "waveform_approximant_spectra.pdf"

    # The hyperparameters every approximant is drawn at, from
    # config/fiducials.json. `root=` because the accessors resolve paths
    # against the working directory.
    FIDUCIALS = fiducials(root=ROOT_DIR)

    # Population overrides, kept fixed and off the controls: every approximant
    # must see the same events on the same redshift support for the comparison
    # to mean anything. A coarser cosmology grid than config/population.json's
    # 4096, and a floor at z = 0.3 rather than the file's 0.35.
    minimum_redshift = 0.3
    maximum_redshift = 20.0
    n_grid = 256

    # batch_size chunks the waveform generation; n_max_sigma sizes the static
    # event plate a Poisson tail above the mean count.
    batch_size = 128
    n_max_sigma = 5.0

    # Ripple calls its registered tidal model `IMRPhenomXAS_NRTidalv3`
    # (lower-case `v`), while plot text uses the conventional
    # `IMRPhenomXAS_NRTidalV3` spelling. DISPLAY_LABELS carries the display
    # form; the "(reference)" tag is appended in the plotting cell, to
    # whichever approximant the reference control selects.
    APPROXIMANTS = (
        "TaylorF2",
        "IMRPhenomXAS",
        "IMRPhenomHM",
        "IMRPhenomXAS_NRTidalv3",
    )
    DISPLAY_LABELS = {
        "TaylorF2": "TaylorF2",
        "IMRPhenomXAS": "IMRPhenomXAS",
        "IMRPhenomHM": "IMRPhenomHM",
        "IMRPhenomXAS_NRTidalv3": "IMRPhenomXAS_NRTidalV3",
    }
    COLORS = {
        "TaylorF2": "#0072B2",
        "IMRPhenomXAS": "#E69F00",
        "IMRPhenomHM": "#009E73",
        "IMRPhenomXAS_NRTidalv3": "#D55E00",
    }
    return (
        APPROXIMANTS,
        COLORS,
        DISPLAY_LABELS,
        FIDUCIALS,
        OUTPUT_PATH,
        ROOT_DIR,
        batch_size,
        maximum_redshift,
        minimum_redshift,
        n_grid,
        n_max_sigma,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Interactive configuration options

    The reference approximant and the percentile band only restyle the figure:
    the paired draws are unchanged, so both are cheap to move. The observation
    time, the retained-draw count, and the seed re-run the forward model --
    the first rescales the Poisson event count, the second trades a smoother
    median and band for a longer run, and the third replays a different,
    still perfectly paired, set of draws, which is the cheapest check that
    the comparison is not an artifact of one seed.

    The population, the hyperparameters, and the waveform grid carry no
    controls on purpose: the comparison only means anything when every
    approximant sees the same events on the same bins, so those stay pinned
    to the shared configuration below.
    """)
    return


@app.cell
def _(APPROXIMANTS, DISPLAY_LABELS):
    # The dropdown shows the conventional spelling; its value is the name
    # Ripple registered.
    reference_choice = mo.ui.dropdown(
        options={DISPLAY_LABELS[name]: name for name in APPROXIMANTS},
        value=DISPLAY_LABELS["IMRPhenomXAS_NRTidalv3"],
        label="Reference approximant",
    )
    # The value is the central band level; the value cell below turns it into
    # a (low, median, high) percentile triple.
    band_choice = mo.ui.dropdown(
        options={"50 %": 50.0, "80 %": 80.0, "90 %": 90.0, "95 %": 95.0},
        value="80 %",
        label="Central percentile band",
    )
    # Scales the Poisson mean, so the event count and the background level
    # grow together with the observing run.
    observation_time_slider = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=1,
        debounce=True,
        show_value=True,
        label="Observation time (years)",
    )
    # More draws stabilize the median and the band, at a linear cost in
    # waveform passes.
    draw_count_slider = mo.ui.slider(
        start=2,
        stop=12,
        step=1,
        value=4,
        debounce=True,
        show_value=True,
        label="Retained draws",
    )
    # A different seed replays a different, still matched, set of draws.
    seed_number = mo.ui.number(
        start=0,
        step=1,
        value=20250314,
        debounce=True,
        label="Seed",
    )
    write_figures_switch = mo.ui.switch(
        value=True,
        label="Write figure",
    )
    mo.vstack(
        [
            reference_choice,
            band_choice,
            observation_time_slider,
            draw_count_slider,
            seed_number,
            write_figures_switch,
        ]
    )
    return (
        band_choice,
        draw_count_slider,
        observation_time_slider,
        reference_choice,
        seed_number,
        write_figures_switch,
    )


@app.cell
def _(band_choice, reference_choice):
    reference_approximant = reference_choice.value
    _band_level = float(band_choice.value)
    # (low, median, high) percentiles of the central band.
    band_percentiles = (
        50.0 - _band_level / 2.0,
        50.0,
        50.0 + _band_level / 2.0,
    )
    return (band_percentiles, reference_approximant)


@app.cell
def _(draw_count_slider, observation_time_slider, seed_number):
    observation_time = float(observation_time_slider.value)
    draw_count = int(draw_count_slider.value)
    seed = int(seed_number.value)
    return (draw_count, observation_time, seed)


@app.cell
def _(write_figures_switch):
    write_figures = write_figures_switch.value
    return (write_figures,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Shared simulation configuration

    There is one configuration for the population, the hyperparameters, the
    waveform grid, batching, and the capacity-tail rule; the observing
    duration, the number of retained draws, and the seed are the controls
    above. The four generators receive the same grid settings; their
    approximant is their only differing metadata field. Fiducials, population,
    and waveform settings come from the shared `astrogwb.paper.config`
    accessors, so this comparison cannot drift from the catalog configuration.
    Ripple calls its registered tidal model `IMRPhenomXAS_NRTidalv3`
    (lower-case `v`), while plot text uses the conventional
    `IMRPhenomXAS_NRTidalV3` spelling.
    """)
    return


@app.cell
def _(APPROXIMANTS, ROOT_DIR, maximum_redshift, minimum_redshift, n_grid):
    population = population_model(
        root=ROOT_DIR,
        minimum_redshift=minimum_redshift,
        maximum_redshift=maximum_redshift,
        n_grid=n_grid,
    )
    source_model = IsotropicInclination(population.source_model)
    merger_rate_fn = population.merger_rate_fn
    if merger_rate_fn is None:
        raise ValueError("configured population cannot simulate event counts")

    generators = {
        approximant: waveform_generator(root=ROOT_DIR, approximant=approximant)
        for approximant in APPROXIMANTS
    }
    return (generators, merger_rate_fn, source_model)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Draw matched spectra

    `Predictive` assigns keys deterministically by sample-site name. Calling
    the same model with the same fixed key therefore reproduces `n_events`,
    masses, redshifts, spins, tidal deformabilities, and inclinations exactly
    for every approximant. Splitting the key in the loop would instead
    produce unrelated catalogs and would confound waveform differences with
    Monte Carlo variation. Non-tidal approximants deliberately do not consume
    the shared tidal latent variables.
    """)
    return


@app.cell
def _(
    APPROXIMANTS,
    FIDUCIALS,
    batch_size,
    draw_count,
    generators,
    merger_rate_fn,
    n_max_sigma,
    observation_time,
    seed,
    source_model,
):
    _rate = float(jnp.asarray(merger_rate_fn(FIDUCIALS)))
    _mean_count = _rate * years_to_seconds(observation_time)
    _max_events = max(int(np.ceil(_mean_count + n_max_sigma * np.sqrt(_mean_count))), 1)
    _shared_key = jax.random.key(seed)

    spectral_draws: dict[str, np.ndarray] = {}
    _event_counts: dict[str, np.ndarray] = {}
    for _approximant, _generator in generators.items():
        _predictive = Predictive(
            partial(
                gwb_forward_model,
                source_model=source_model,
                merger_rate_fn=merger_rate_fn,
                generator=_generator,
                observation_time=observation_time,
                batch_size=batch_size,
                max_events=_max_events,
            ),
            num_samples=draw_count,
            return_sites=("spectral_density", "n_events"),
        )
        _result = _predictive(_shared_key, FIDUCIALS)
        spectral_draws[_approximant] = np.asarray(_result["spectral_density"])
        _event_counts[_approximant] = np.asarray(_result["n_events"])

    # The identical counts are a cheap explicit check that the stochastic
    # traces stayed paired. The fixed-key construction also pairs every named
    # source site.
    _reference_counts = _event_counts[APPROXIMANTS[0]]
    for _approximant, _counts in _event_counts.items():
        np.testing.assert_array_equal(
            _counts,
            _reference_counts,
            err_msg=f"unpaired event counts for {_approximant}",
        )

    {"Events per draw": _event_counts[APPROXIMANTS[0]].tolist()}
    return (spectral_draws,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Validate the common frequency grid

    A pointwise comparison is meaningful only if every generator returns
    exactly the same bins. Fail before plotting if shape or values differ.
    """)
    return


@app.cell
def _(APPROXIMANTS, draw_count, generators, spectral_draws):
    frequencies = np.asarray(generators[APPROXIMANTS[0]].frequencies)
    for _approximant, _generator in generators.items():
        _candidate = np.asarray(_generator.frequencies)
        np.testing.assert_array_equal(
            _candidate,
            frequencies,
            err_msg=f"frequency grid differs for {_approximant}",
        )
        if spectral_draws[_approximant].shape != (draw_count, frequencies.size):
            raise ValueError(
                f"unexpected spectral shape for {_approximant}: "
                f"{spectral_draws[_approximant].shape}"
            )

    {"Frequency bins": int(frequencies.size)}
    return (frequencies,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Median spectra and fractional residuals

    Both panels summarize all retained draws: lines are draw-wise medians and
    bands span the central interval chosen above. Residuals are computed for
    each paired draw against the selected reference before percentiles are
    taken, preserving event-level comparability. A reference bin equal to
    zero is represented by `NaN` and omitted safely.
    """)
    return


@app.cell(hide_code=True)
def _():
    def plot_median_spectra_and_residuals(
        frequencies: np.ndarray,
        spectral_draws: Mapping[str, np.ndarray],
        approximants: Sequence[str],
        *,
        reference_approximant: str,
        percentiles: tuple[float, float, float],
        labels: Mapping[str, str],
        colors: Mapping[str, str],
    ) -> Figure:
        """Median spectra over fractional residuals to a chosen reference.

        ``percentiles`` is the ``(low, median, high)`` triple of the central
        band. Residuals are computed per paired draw against
        ``reference_approximant`` before percentiles are taken, preserving
        event-level comparability; a reference bin equal to zero is masked
        rather than divided.
        """
        low_pct, _, high_pct = percentiles
        reference_draws = spectral_draws[reference_approximant]

        fig, (spectrum_ax, residual_ax) = plt.subplots(
            2,
            1,
            figsize=(7.0, 6.5),
            sharex=True,
            gridspec_kw={"height_ratios": (2.1, 1.0), "hspace": 0.08},
        )
        for approximant in approximants:
            draws = spectral_draws[approximant]
            median, low, high = np.percentile(draws, (50.0, low_pct, high_pct), axis=0)
            color = colors[approximant]
            label = labels[approximant]
            spectrum_ax.loglog(frequencies, median, color=color, label=label)
            spectrum_ax.fill_between(frequencies, low, high, color=color, alpha=0.18)

            residuals = np.full_like(draws, np.nan)
            np.divide(
                draws - reference_draws,
                reference_draws,
                out=residuals,
                where=reference_draws != 0.0,
            )
            residual_median = np.nanmedian(residuals, axis=0)
            residual_low, residual_high = np.nanpercentile(
                residuals, (low_pct, high_pct), axis=0
            )
            residual_ax.semilogx(frequencies, residual_median, color=color, label=label)
            residual_ax.fill_between(
                frequencies,
                residual_low,
                residual_high,
                color=color,
                alpha=0.18,
            )

        spectrum_ax.set_ylabel(r"$S_h(f)\ [\mathrm{Hz}^{-1}]$")
        spectrum_ax.legend(loc="best")
        spectrum_ax.grid(alpha=0.25)
        residual_ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        residual_ax.set_xlabel(r"Frequency $f\ [Hz]$")
        residual_ax.set_ylabel("Fractional\nresidual")
        residual_ax.grid(alpha=0.25)
        fig.align_ylabels()
        return fig

    return (plot_median_spectra_and_residuals,)


@app.cell
def _(
    APPROXIMANTS,
    COLORS,
    DISPLAY_LABELS,
    OUTPUT_PATH,
    ROOT_DIR,
    band_percentiles,
    frequencies,
    plot_median_spectra_and_residuals,
    reference_approximant,
    spectral_draws,
    write_figures,
):
    _labels = {
        approximant: DISPLAY_LABELS[approximant]
        + (" (reference)" if approximant == reference_approximant else "")
        for approximant in APPROXIMANTS
    }
    _fig = plot_median_spectra_and_residuals(
        frequencies,
        spectral_draws,
        APPROXIMANTS,
        reference_approximant=reference_approximant,
        percentiles=band_percentiles,
        labels=_labels,
        colors=COLORS,
    )
    if write_figures:
        save_figures({OUTPUT_PATH: _fig}, root=ROOT_DIR)
    _fig
    return


if __name__ == "__main__":
    app.run()
