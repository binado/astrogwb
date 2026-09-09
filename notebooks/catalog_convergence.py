# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: astrogwb (3.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Catalog convergence: Monte-Carlo size and frequency resolution
#
# A catalog-based stochastic background carries two discretization errors, and
# they are different kinds of thing:
#
# 1. **Number of sources $N$.** `spectral_density` is literally a sample mean,
#    `factor * R_tot * (P @ w) / N`, so the catalog spectrum approaches the
#    analytic population spectrum as a Monte-Carlo estimator: error
#    $\propto N^{-1/2}$, with a variance set by how heavy-tailed the per-source
#    power is.
# 2. **Frequency resolution $\Delta f$.** `spectral_snr_squared` is a Riemann
#    sum $2T\,\Delta f \sum_i S_{h,i}^2/S_{{\rm eff},i}^2$, so it approaches an
#    integral as $\Delta f \to 0$. This one has a trap:
#    `apply_frequency_mask`'s docstring warns that `df` must be passed
#    explicitly. Subsample a grid `::k` and forget $\Delta f \to k\,\Delta f$
#    and the SNR falls by $\sqrt{k}$ — which looks exactly like
#    non-convergence, and is not.
#
# `packages/astrogwb/tests/test_frequency_resolution.py` and
# `test_mock_population.py` own the tolerances. This notebook owns the picture:
# a tolerance on a mean ratio cannot show the *shape* of a residual across
# frequency, nor whether an error is falling like $N^{-1/2}$ or has hit a bias
# floor.
#
# **This notebook needs no external data**; it names a registered population
# model and caches the catalog it builds to `notebooks/convergence_catalog.h5`
# (gitignored).

# %% [markdown]
# ## Imports

# %%
import os
import warnings
from itertools import pairwise
from pathlib import Path
from typing import Any

# lal warns about SWIG stdout redirection on import, but only under IPython --
# so it fires in Jupyter and under `jupytext --execute`, not in a terminal. It
# is pulled in transitively by astrogwb.detector, so the filter goes first.
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection

from astrogwb.catalog import Catalog, PopulationMetadata
from astrogwb.constants import ISCO_ALPHA, SECONDS_PER_YEAR
from astrogwb.detector import effective_psd, gaussian_bin_scale, load_sensitivity_map
from astrogwb.distributions.rates import madau_dickinson_rate
from astrogwb.frequency import apply_frequency_mask, frequency_mask
from astrogwb.gwb import (
    analytic_spectral_density_from_mass_moments,
    omega_gw_from_spectral_density,
    spectral_density,
    spectral_snr,
    uniform_prior_mass_moments,
)
from astrogwb.importance.estimator import SpectralDensityImportanceEstimator
from astrogwb.populations import (
    TOTAL_MERGER_RATE_SITE,
    BNSMadauDickinson,
    BNSMadauDickinsonModifiedPropagation,
)
from astrogwb.waveform import AnalyticInspiralGenerator

# gwpy, pulled in by gwmock-signal behind astrogwb.detector, replaces
# matplotlib's registered rectilinear axes with its own subclass on import.
register_projection(MplAxes)
plt.rcParams.update({"figure.dpi": 120, "figure.constrained_layout.use": True})

# Last, and after every astrogwb import: x64 must be enabled before the first
# array is created, and no core module builds one at import time.
jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Notebook configuration
#
# **Two grids, one population.** The notebook's two halves want opposite things
# from a frequency grid, and no single choice serves both, so the same sources
# are reduced onto two of them:
#
# | section | band | $\Delta f$ | why |
# |---|---|---|---|
# | $\Omega_{\rm gw}$, Monte-Carlo convergence | [2, 2048] Hz | 1 Hz | a pointwise ratio of two spectra, integrating nothing |
# | SNR, $\Delta f$, likelihood | [2, 256] Hz | 0.125 Hz | must resolve the 7 Hz SNR peak |
#
# Both catalogs are drawn from the same `POPULATION_SEED` at the same
# `NUM_SOURCES`, so they describe the *same sources*; only the grid differs.
#
# **Why the SNR band is `[2, 256]` Hz and `FINE_DF` is 0.125 Hz.** With the ET
# effective PSD, the SNR integrand $S_h^2/S_{\rm eff}^2$ is a *narrow peak near
# 7 Hz*: about 90% of $\rho^2$ accumulates between 5 and 11 Hz, and 99.9% below
# 150 Hz. A reference grid must resolve that peak or it is not a reference at
# all — at $\Delta f = 1$ Hz the peak is sampled about four times, and the
# "converged" anchor is itself several parts in $10^3$ off. At
# $\Delta f = 0.125$ Hz the SNR is stable to $10^{-5}$ under a further halving,
# which is what makes the residuals below meaningful. Extending the band past
# 256 Hz would only add bins that contribute nothing *to the SNR*.
#
# **Why the $\Omega_{\rm gw}$ band is `[2, 2048]` Hz and `OMEGA_DF` is 1 Hz.**
# $\Omega_{\rm gw} \propto f^3 S_h \propto f^{2/3}$ *rises* across the whole SNR
# band and turns over only where the population starts running out of inspiral:
# the analytic spectrum peaks near 410 Hz and falls to zero at the ISCO cutoff
# of the lightest, nearest binary in the population, above 1.6 kHz. A comparison
# stopped at 256 Hz sees neither. Nothing in that section is integrated, so the
# coarse grid costs nothing and the wider band is nearly free — 2047 bins by
# 1024 sources is about 17 MB.
#
# `FINE_DF` is a negative power of two on purpose: `k * FINE_DF` is then exact
# in binary, so the subsampled grids match a directly-built coarse grid to the
# last bit rather than to a tolerance.

# %%
SMOKE = os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") == "1"

NUM_SOURCES = 256 if SMOKE else 1024

#: Reference frequency resolution, and the band the SNR and likelihood
#: sections run on.
FINE_DF = 0.125
F_MIN = 2.0
SNR_F_MAX = 256.0

#: The Omega_gw comparison grid. Wide enough to contain the ~410 Hz spectral
#: peak and the ~1.7 kHz analytic cutoff; df is coarse because that section
#: compares two spectra pointwise and integrates nothing.
OMEGA_DF = 1.0
OMEGA_F_MAX = 2048.0

#: Log-spaced band edges, exact powers of four. The lowest three bands sit
#: entirely below the ~125 Hz common-support edge; [128, 512) straddles it and
#: contains the spectral peak.
BAND_EDGES: tuple[float, ...] = (2.0, 8.0, 32.0, 128.0, 512.0, 2048.0)
NUM_BANDS = len(BAND_EDGES) - 1

POPULATION_SEED = 41

#: Grid-coarsening factors. `k` means "keep every k-th bin", giving
#: `df = k * FINE_DF`, from 0.125 Hz up to 8 Hz.
SUBSAMPLE_FACTORS: tuple[int, ...] = (
    (1, 4, 16, 64) if SMOKE else (1, 2, 4, 8, 16, 32, 64)
)

#: Catalog sizes the Monte-Carlo convergence is measured at, and how many
#: independent resamples are drawn at each.
CATALOG_SIZES: tuple[int, ...] = (
    (32, 64, 128, 256) if SMOKE else (64, 128, 256, 512, 1024)
)
NUM_REALIZATIONS = 4 if SMOKE else 16
RNG_SEED = 20260901

DETECTORS: tuple[str, ...] = ("E1", "E2", "E3")

#: Observation time the likelihood scan is evaluated at, in years.
OBSERVATION_TIME = 1.0

#: H0 values the log-likelihood is scanned over.
H0_SCAN = np.linspace(60.0, 76.0, 17 if SMOKE else 33)

#: The notebook may be executed from the repository root
#: (`just test-notebooks`) or from its own directory (Jupyter).
NOTEBOOK_DIR = Path("notebooks") if Path("notebooks").is_dir() else Path()
SNR_CATALOG_PATH = NOTEBOOK_DIR / (
    "convergence_catalog_smoke.h5" if SMOKE else "convergence_catalog.h5"
)
OMEGA_CATALOG_PATH = NOTEBOOK_DIR / (
    "convergence_omega_catalog_smoke.h5" if SMOKE else "convergence_omega_catalog.h5"
)

# %% [markdown]
# ## The source population
#
# One declaration, used twice: `Population.sample` samples from it, and the
# importance weights below evaluate the *same* model's density at the stored
# samples. That is what makes this catalog exactly its own proposal
# ($\log w \equiv 0$ at the fiducials) rather than approximately so.
#
# The component-mass bounds are load-bearing here: `uniform_prior_mass_moments`
# below is given the *same* bounds, and the analytic spectrum it feeds is only
# the right oracle for this catalog if they agree. So is the zero `inclination`
# column the population declares, which pairs with
# `average_mode="analytic_inclination"`.

# %%
FIDUCIALS: dict[str, float] = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "xi_0": 1.0,
    "xi_n": 1.91,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
}

#: The generating population takes no propagation parameters: modified
#: propagation is target-side only, and at `xi_0 = 1` the two agree exactly.
POPULATION_PARAMS: dict[str, float] = {
    name: value for name, value in FIDUCIALS.items() if name not in {"xi_0", "xi_n"}
}

Z_MIN = 0.3
Z_MAX = 20.0
N_GRID = 256

MINIMUM_COMPONENT_MASS = 1.0
MAXIMUM_COMPONENT_MASS = 2.5

#: The registered model, and the settings bound into it. Both travel into the
#: generated file, so a cached catalog says which population produced it.
POPULATION_MODEL = "bns_md_cosmological"
POPULATION_MODEL_KWARGS: dict[str, float | int] = {
    "z_min": Z_MIN,
    "z_max": Z_MAX,
    "n_grid": 4096,
}


def population_model_fn():
    """The generating population, with its construction settings bound."""
    return BNSMadauDickinson(**POPULATION_MODEL_KWARGS)


def target_model_fn():
    """The target population: the same sources under modified propagation."""
    return BNSMadauDickinsonModifiedPropagation(z_min=Z_MIN, z_max=Z_MAX, n_grid=N_GRID)


def make_redshift_grid() -> jax.Array:
    """The redshift grid every cosmology integral in this notebook runs on."""
    return jnp.linspace(Z_MIN, Z_MAX, N_GRID)


# %% [markdown]
# ## Building or loading the catalogs
#
# Two catalogs, drawn from one population. `build_catalog` re-runs
# `Population.sample` at the same seed for each, so the two files hold the *same*
# sources reduced onto different frequency grids — the wide 1 Hz grid the
# $\Omega_{\rm gw}$ comparison needs, and the fine 0.125 Hz grid everything from
# the SNR section on runs against.
#
# Within the fine grid, every coarser grid below is obtained by subsampling, so
# those all describe the same sources too and the only thing that varies is
# $\Delta f$.
#
# The luminosity distance is the population's own `numpyro.deterministic`,
# computed in the same batched pass every later density evaluation takes, which
# is what makes the catalog exactly its own importance proposal
# ($\log w \equiv 0$) rather than approximately so.
#
# A cached file is reused only when its recorded population and waveform grid
# still describe the configuration cell. That guard matters more here than in
# `mcmc_example_models.py`: `FINE_DF` *is* the subject, so silently reusing a
# catalog built at a different resolution would invalidate every result below
# while looking perfectly healthy. The `grid` attribute records which of the two
# a file holds.
#
# Each cell below ends by displaying its dataset: the rendering carries the
# shape and the grid, the attributes the rest of the provenance -- population,
# seed, source count, package version, and the total merger rate, which
# `unpack` derives and stamps because the cache does not store it.


# %%
def build_catalog(*, df: float, f_max: float, grid: str) -> Catalog:
    """Draw the population and reduce it onto the `[F_MIN, f_max]` grid.

    `grid` is a label carried into the population provenance, and from there
    into the file's attributes, so the two cache files below are
    self-describing rather than distinguished by filename.

    The derived columns -- distances, detector-frame masses -- come from the
    population itself, in one batched pass, so they are bit-identical to what
    every later density evaluation recomputes from the stored samples.
    """
    population_metadata = PopulationMetadata(
        name=POPULATION_MODEL,
        seed=POPULATION_SEED,
        num_samples=NUM_SOURCES,
        source_type="bns",
        provenance={
            "notebook": "catalog_convergence",
            "grid": grid,
            "termination_alpha": ISCO_ALPHA,
        },
    )
    parameters = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in population_model_fn()
        .sample(
            jax.random.PRNGKey(POPULATION_SEED),
            POPULATION_PARAMS,
            num_samples=NUM_SOURCES,
        )
        .items()
    }

    return Catalog.from_generator(
        parameters,
        generator=AnalyticInspiralGenerator(
            alpha=ISCO_ALPHA,
            approximant="AnalyticInspiral",
            minimum_frequency=F_MIN,
            maximum_frequency=f_max,
            reference_frequency=F_MIN,
            sampling_frequency=2.0 * f_max,
            df=df,
        ),
        population_metadata=population_metadata,
        model_name=POPULATION_MODEL,
        model_kwargs=POPULATION_MODEL_KWARGS,
        population_params=POPULATION_PARAMS,
        density_sites=("redshift",),
    )


def catalog_matches_configuration(catalog: Catalog, *, df: float, f_max: float) -> bool:
    """Does a cached catalog still describe the configuration cell?

    The population half of the question no longer needs asking: the file
    records its own model, settings and hyperparameters, and `Catalog.load`
    refuses a file whose columns no longer match them. What is left is the
    waveform grid and the draw size, which the population record does not
    cover.
    """
    waveform = catalog.waveform_metadata
    return (
        waveform.df == df
        and waveform.minimum_frequency == F_MIN
        and waveform.maximum_frequency == f_max
        and catalog.population_metadata.num_samples == NUM_SOURCES
        and catalog.population_metadata.seed == POPULATION_SEED
        and catalog.population_model_name == POPULATION_MODEL
        and dict(catalog.population_params) == POPULATION_PARAMS
        and dict(catalog.population_model_kwargs) == POPULATION_MODEL_KWARGS
    )


def load_or_build_catalog(*, df: float, f_max: float, grid: str, path: Path) -> Catalog:
    """Return the cached catalog if it is still current, else rebuild it.

    A file written by an older astrogwb is *rejected* by `Catalog.load` rather
    than merely failing the configuration check below, so the read is guarded:
    a stale cache is a rebuild, not a crash.
    """
    if path.is_file():
        try:
            cached = Catalog.load(path)
        except (OSError, KeyError, ValueError) as error:
            print(f"{path} is not a current astrogwb catalog ({error}); rebuilding")
        else:
            if catalog_matches_configuration(cached, df=df, f_max=f_max):
                print(f"Loaded {path}")
                return cached
            print(f"{path} does not match this notebook's configuration; rebuilding")
    catalog = build_catalog(df=df, f_max=f_max, grid=grid)
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.save(path)
    print(f"Built and wrote {path}")
    return catalog


def describe(catalog: Catalog) -> pd.Series:
    """A one-glance summary of what a catalog file holds."""
    waveform = catalog.waveform_metadata
    return pd.Series(
        {
            "population": catalog.population_model_name,
            "seed": catalog.population_metadata.seed,
            "num_sources": catalog.population_metadata.num_samples,
            "grid": catalog.population_metadata.provenance.get("grid", ""),
            "num_frequencies": waveform.frequencies.size,
            "df_hz": waveform.df,
            "f_min_hz": waveform.minimum_frequency,
            "f_max_hz": waveform.maximum_frequency,
            "total_merger_rate_per_s": float(catalog_merger_rate(catalog)),
        }
    )


def catalog_merger_rate(catalog: Catalog) -> jax.Array:
    """The observer-frame rate this catalog's own population implies."""
    model = catalog.get_population_model()
    params = catalog.population_params
    values = catalog.source_parameters
    _, trace = model.evaluate(params, values)
    return trace[TOTAL_MERGER_RATE_SITE]["value"]


def unpack(
    catalog: Catalog,
) -> tuple[np.ndarray, np.ndarray, dict[str, jax.Array], jax.Array]:
    """The four things every section wants out of a catalog."""
    frequencies = np.asarray(catalog.waveform_metadata.frequencies)
    power = np.asarray(catalog.polarization_power)
    catalog_samples = {
        name: jnp.asarray(values) for name, values in catalog.source_parameters.items()
    }
    return frequencies, power, catalog_samples, catalog_merger_rate(catalog)


# The wide, coarse grid: the Omega_gw comparison and the Monte-Carlo
# convergence below both run on this one.
wide_catalog = load_or_build_catalog(
    df=OMEGA_DF, f_max=OMEGA_F_MAX, grid="omega", path=OMEGA_CATALOG_PATH
)
wide_frequencies, wide_power, wide_samples, wide_merger_rate = unpack(wide_catalog)

# The rate is derived from the file's own population rather than persisted:
# it is a property of the population and the redshift window, so a stored copy
# would be stale the moment either moved.
describe(wide_catalog)

# %%
# The narrow, fine grid: everything from the SNR section on.
catalog = load_or_build_catalog(
    df=FINE_DF, f_max=SNR_F_MAX, grid="snr", path=SNR_CATALOG_PATH
)
fine_frequencies, fine_power, samples, total_merger_rate = unpack(catalog)

describe(catalog)

# %% [markdown]
# ## Analytic vs sample-mean $\Omega_{\rm gw}$
#
# `analytic_spectral_density_from_mass_moments` integrates the *population*:
# the exact same Newtonian-inspiral physics, over the same uniform ordered
# mass prior and the same Madau-Dickinson rate, with no sampling anywhere. It
# is the thing the catalog contraction is a Monte-Carlo estimate of, so their
# ratio is the estimator's error and nothing else.
#
# Both are pushed through `omega_gw_from_spectral_density`, which is a
# multiplication by $4\pi^2 f^3/(3H_0^2)$ — the *same* factor for both, so the
# residual is identical in $S_h$ and in $\Omega_{\rm gw}$. Plotting
# $\Omega_{\rm gw}$ anyway is the point: it is the quantity the literature
# quotes, and it is the one function of the pair that no test on this branch
# otherwise exercises.
#
# **One convention.** The source-frame merger rate passed to the analytic
# spectrum deliberately omits the $1/(1+z)$ time dilation:
# `analytic_spectral_density_from_mass_moments` carries it inside its
# $(1+z)^{4/3}$ factor, whereas the population's redshift density applies it
# inside $p(z) \propto \psi(z)/(1+z)\,dV_c/dz$. Dividing in both places double-counts it
# — exactly the class of error these two independent paths are crossed to
# catch.


# %%
def source_frame_merger_rate(redshift: jax.Array, hyperparameters) -> jax.Array:
    """Absolute source-frame merger-rate density in Gpc^-3 yr^-1."""
    return hyperparameters["local_merger_rate"] * madau_dickinson_rate(
        redshift,
        hyperparameters["gamma"],
        hyperparameters["kappa"],
        hyperparameters["z_peak"],
    )


mass_moments = uniform_prior_mass_moments(
    jnp.asarray(wide_frequencies),
    minimum_redshift=Z_MIN,
    maximum_redshift=Z_MAX,
    minimum_component_mass=MINIMUM_COMPONENT_MASS,
    maximum_component_mass=MAXIMUM_COMPONENT_MASS,
    # The same truncation the catalog's polarization power was built with.
    alpha=ISCO_ALPHA,
)
analytic_spectrum = np.asarray(
    analytic_spectral_density_from_mass_moments(
        jnp.asarray(wide_frequencies),
        FIDUCIALS,
        source_frame_merger_rate,
        mass_moments,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
    )
)


def contract(power: np.ndarray, rate: jax.Array) -> np.ndarray:
    """Unweighted catalog contraction at the fiducials, for `power`'s sources."""
    num = power.shape[1]
    return np.asarray(
        spectral_density(
            jnp.asarray(power),
            jnp.ones(num),
            rate,
            average_mode="analytic_inclination",
        )
    )


def to_omega(spectrum: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Convert a strain spectral density on `frequencies` to Omega_gw."""
    return np.asarray(
        omega_gw_from_spectral_density(
            jnp.asarray(spectrum),
            jnp.asarray(frequencies),
            hubble_constant=FIDUCIALS["H0"],
        )
    )


catalog_spectrum = contract(wide_power, wide_merger_rate)
omega_catalog = to_omega(catalog_spectrum, wide_frequencies)
omega_analytic = to_omega(analytic_spectrum, wide_frequencies)

# Bins where *every* source still emits. Above the smallest sampled cutoff a
# finite catalog loses sources one at a time while the analytic population
# spectrum, which integrates over the whole mass-redshift plane, does not --
# so the residual there measures the catalog's discreteness, not its Monte-
# Carlo error. Summary statistics are quoted on the common support; the plot
# shows the full band so the transition is visible rather than cropped away.
common_support = np.all(wide_power > 0.0, axis=1)
# Both sides need guarding now that the band runs past the analytic cutoff:
# above ~1.7 kHz the analytic spectrum is zero too, and the ratio is undefined
# rather than merely uninteresting.
valid = (catalog_spectrum > 0.0) & (analytic_spectrum > 0.0)
residual = np.full_like(omega_catalog, np.nan)
residual[valid] = omega_catalog[valid] / omega_analytic[valid] - 1.0
support_edge = float(wide_frequencies[common_support][-1])
omega_peak = float(wide_frequencies[int(np.argmax(omega_analytic))])
analytic_edge = float(wide_frequencies[analytic_spectrum > 0.0][-1])

print(f"common support: {int(common_support.sum())} bins, up to {support_edge:g} Hz")
print(f"analytic Omega_gw peaks at {omega_peak:g} Hz, ends at {analytic_edge:g} Hz")
probes = (support_edge, 256.0, omega_peak, 1024.0)
emitting_table = pd.DataFrame(
    {
        f"sources still emitting (of {NUM_SOURCES})": [
            int(
                np.sum(
                    wide_power[int(np.argmin(np.abs(wide_frequencies - probe)))] > 0.0
                )
            )
            for probe in probes
        ],
    },
    index=pd.Index(probes, name="probe [Hz]"),
)
print(
    "relative residual on the common support: "
    f"mean {np.mean(residual[common_support]):+.4f}, "
    f"spread {np.ptp(residual[common_support]):.2e}"
)

emitting_table.style.format_index("{:.0f}")

# %%
fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True, height_ratios=(2, 1))
axes[0].loglog(
    wide_frequencies, omega_analytic, lw=1.4, color="k", label="analytic population"
)
axes[0].loglog(
    wide_frequencies[valid],
    omega_catalog[valid],
    lw=1.2,
    color="tab:blue",
    label=f"catalog, $N = {NUM_SOURCES}$",
)
axes[0].set_ylabel(r"$\Omega_{\rm gw}(f)$")
axes[0].set_title(r"Catalog contraction against the analytic $\Omega_{\rm gw}$")
# Both curves fall off a cliff at the cutoff; without a floor the decades of
# empty axis below it squash the part worth looking at into a sliver.
axes[0].set_ylim(1.0e-4 * omega_analytic.max(), 2.0 * omega_analytic.max())
axes[0].legend(loc="lower left")

axes[1].axhline(0.0, color="k", lw=0.8)
axes[1].plot(wide_frequencies, residual, lw=1.2, color="tab:blue")
axes[1].set_xscale("log")
axes[1].set_xlabel("frequency [Hz]")
axes[1].set_ylabel("relative residual")
for ax in axes:
    ax.axvspan(wide_frequencies[0], support_edge, color="0.9", zorder=0)
    ax.axvline(omega_peak, color="tab:red", lw=0.9, ls="--")
axes[0].annotate(
    f"analytic peak, {omega_peak:g} Hz",
    xy=(omega_peak, 0.06),
    xycoords=("data", "axes fraction"),
    xytext=(-4, 0),
    textcoords="offset points",
    ha="right",
    fontsize=8,
    color="tab:red",
)
axes[1].annotate(
    "shaded: every source still emits",
    xy=(0.03, 0.86),
    xycoords="axes fraction",
    fontsize=8,
)
plt.show()

# %% [markdown]
# **Two regimes, and the plot now shows both.**
#
# Across the shaded region the residual is **flat**, and that is not a
# coincidence. Below the smallest sampled cutoff every source contributes at
# every bin, and both spectra are exactly $\propto f^{-7/3}$ there, so their
# ratio cannot depend on frequency: the whole Monte-Carlo error collapses to a
# single normalization offset. That is the regime the SNR lives in — the
# integrand peaks near 7 Hz, two decades below the support edge — and it is why
# a 1024-source catalog is an excellent estimator of the quantity the rest of
# this notebook measures.
#
# Above the edge it is a poor one. The catalog runs out of sources one ISCO
# cutoff at a time while the analytic curve, which integrates the whole
# mass-redshift plane, does not; the residual acquires structure, turns
# systematically negative, and at the spectral peak itself the catalog is
# missing a third of its emitters. Both curves die by the cutoff of the
# lightest, nearest binary the population can contain.
#
# So "the residual at catalog size $N$" is one number *below* the support edge
# and a curve above it. The next section measures both, band by band.
# %% [markdown]
# ## Monte-Carlo convergence in catalog size
#
# **How the smaller catalogs are drawn.** Seeded random column subsets of the
# single $N = 1024$ catalog, **with replacement**, several realizations per
# size — not separate `num_sources=N` builds, and not subsets drawn *without*
# replacement.
#
# Both alternatives are actively misleading:
#
# - `build_catalog(num_sources=N)` slices a deterministic *prefix* of the same
#   draw, so the $N$ and $2N$ catalogs share their first $N$ sources and their
#   residuals are strongly correlated. Any slope fitted through them is
#   meaningless.
# - Subsets drawn without replacement from a 1024-source parent carry the
#   finite-population correction $(1 - n/N)$, which forces the variance to
#   *zero* at $n = N$. Measured on this catalog that fakes an 11$\times$ drop
#   in residual across a 16$\times$ range in $n$, where $N^{-1/2}$ predicts
#   4$\times$ — a beautifully clean, entirely artificial slope.
#
# Sampling with replacement estimates what a *fresh* catalog of size $n$ would
# have looked like, which is the question. The error bars are wide because
# per-source power is heavy-tailed in mass and distance; that is the honest
# amount of information in 1024 sources.
#
# **Resolved by frequency band.** The previous section reduced the residual to
# a single rms over the common support, which was defensible only because the
# residual is flat there. It is not flat above the support edge, so the error
# is measured in each of the five bands of `BAND_EDGES` separately. Two
# statistics come out of each realization:
#
# - the **rms** relative residual, which is what converges, and
# - the **signed mean** relative residual, which is what *correlates*. An rms
#   is positive by construction, so a correlation computed from it would report
#   agreement between bands that share only a magnitude.
#
# The bands are fixed constants rather than derived from the catalog's own
# support on purpose: the support edge *moves with $N$* — a bigger catalog
# samples further into the tail of light, nearby binaries and so keeps emitting
# to higher frequency — and bands that moved with it would not be comparable
# across the sizes being compared.

# %%
rng = np.random.default_rng(RNG_SEED)

# Bands are half-open, [lo, hi), and restricted to bins the analytic spectrum
# actually reaches: above ~1.7 kHz there is nothing to take a ratio against.
positive_analytic = analytic_spectrum > 0.0
band_masks = [
    (wide_frequencies >= low) & (wide_frequencies < high) & positive_analytic
    for low, high in pairwise(BAND_EDGES)
]
band_labels = [f"[{low:g}, {high:g}) Hz" for low, high in pairwise(BAND_EDGES)]

band_rms: dict[int, np.ndarray] = {}
band_mean: dict[int, np.ndarray] = {}
largest_curves = np.full((NUM_REALIZATIONS, wide_frequencies.size), np.nan)

for size in CATALOG_SIZES:
    rms = np.empty((NUM_REALIZATIONS, NUM_BANDS))
    signed = np.empty((NUM_REALIZATIONS, NUM_BANDS))
    for realization in range(NUM_REALIZATIONS):
        columns = rng.integers(0, NUM_SOURCES, size=size)
        # The rate is a property of the population and the redshift window,
        # not of which sources were drawn, so resampling columns changes the
        # contraction and nothing else.
        subset_omega = to_omega(
            contract(wide_power[:, columns], wide_merger_rate), wide_frequencies
        )
        # Where the subset has no source left emitting the ratio is exactly -1.
        # That looks like a numerical artifact and is not: it is the
        # discreteness error itself, and it is the whole content of the top
        # band, so it is kept rather than masked out.
        curve = np.full_like(subset_omega, np.nan)
        curve[positive_analytic] = (
            subset_omega[positive_analytic] / omega_analytic[positive_analytic] - 1.0
        )
        for band, mask in enumerate(band_masks):
            rms[realization, band] = np.sqrt(np.mean(curve[mask] ** 2))
            signed[realization, band] = np.mean(curve[mask])
        if size == CATALOG_SIZES[-1]:
            largest_curves[realization] = curve
    band_rms[size] = rms
    band_mean[size] = signed

pd.DataFrame(
    [band_rms[size].mean(axis=0) for size in CATALOG_SIZES],
    index=pd.Index(CATALOG_SIZES, name="N"),
    columns=band_labels,
).style.format("{:.5f}").set_caption(
    f"rms relative residual, averaged over {NUM_REALIZATIONS} bootstrap realizations"
)

# %%
sizes = np.array(CATALOG_SIZES, dtype=float)
colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.85, NUM_BANDS))

band_curves = {
    band: (
        np.array([band_rms[s][:, band].mean() for s in CATALOG_SIZES]),
        np.array(
            [
                band_rms[s][:, band].std(ddof=1) / np.sqrt(NUM_REALIZATIONS)
                for s in CATALOG_SIZES
            ]
        ),
    )
    for band in range(NUM_BANDS)
}

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

for band, (label, color) in enumerate(zip(band_labels, colors, strict=True)):
    means, errors = band_curves[band]
    axes[0].errorbar(
        sizes, means, yerr=errors, fmt="o-", lw=1.3, capsize=3, color=color, label=label
    )
axes[0].plot(
    sizes,
    band_curves[0][0][0] * np.sqrt(sizes[0] / sizes),
    ls="--",
    color="k",
    lw=1.1,
    label=r"$\propto N^{-1/2}$, anchored on the lowest band",
)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
# Open a decade under the lowest point so the six-entry legend has somewhere to
# sit that is not on top of the curves.
floor = min(means.min() for means, _ in band_curves.values())
axes[0].set_ylim(bottom=0.25 * floor)
axes[0].set_xlabel("catalog size $N$")
axes[0].set_ylabel("rms relative residual")
axes[0].set_title("Monte-Carlo convergence, by band")
axes[0].legend(fontsize=7, loc="lower left")

axes[1].axhline(0.0, color="k", lw=0.8)
axes[1].axvspan(wide_frequencies[0], support_edge, color="0.9", zorder=0)
for realization in range(NUM_REALIZATIONS):
    axes[1].plot(
        wide_frequencies,
        largest_curves[realization],
        lw=0.7,
        alpha=0.5,
        color="tab:blue",
    )
for edge in BAND_EDGES[1:-1]:
    axes[1].axvline(edge, color="0.5", lw=0.8, ls=":")
axes[1].set_xscale("log")
axes[1].set_xlim(wide_frequencies[0], BAND_EDGES[-1])
axes[1].set_xlabel("frequency [Hz]")
axes[1].set_ylabel("signed relative residual")
axes[1].set_title(
    f"{NUM_REALIZATIONS} realizations at $N = {CATALOG_SIZES[-1]}$; "
    "dotted lines are band edges"
)
plt.show()

pd.DataFrame(
    {
        "fitted slope": [
            np.polyfit(np.log(sizes), np.log(band_curves[band][0]), 1)[0]
            for band in range(NUM_BANDS)
        ],
    },
    index=pd.Index(band_labels, name="band"),
).style.format("{:+.3f}").set_caption("(Monte-Carlo prediction: -0.500)")

# %%
correlation = np.corrcoef(band_mean[CATALOG_SIZES[-1]], rowvar=False)
pd.DataFrame(
    correlation,
    index=pd.Index(band_labels, name="band"),
    columns=band_labels,
).style.format("{:.3f}").set_caption(
    f"Pearson correlation of the signed mean residual across {NUM_REALIZATIONS} "
    f"realizations at N = {CATALOG_SIZES[-1]}"
)

# %% [markdown]
# **The lowest three bands are one number, not three.** Their correlation is
# essentially 1: a realization that runs 3% high at 5 Hz runs 3% high at 100 Hz
# too, because below the support edge every source contributes to every bin and
# the residual is a pure normalization offset. The right panel shows the same
# thing directly — inside the shaded region the realizations are flat, parallel
# lines, each one a different constant. That is *why* a single rms over the
# common support was an adequate summary before, and it is the cleanest
# available demonstration of it.
#
# Above the edge the lines fan out and the correlation with the low bands
# falls away — to about 0.6 for `[512, 2048)`. It does not fall to zero, because
# that band still inherits the overall normalization offset the bulk of the
# catalog sets; the missing 0.4 is the part of its error that depends on *which*
# of the eight-odd sources still emitting at 1 kHz a given bootstrap draw
# happened to pick up, and that is independent of how the bulk came out.
#
# The convergence panel splits the same way. The lowest three bands lie exactly
# on top of one another there, so one line stands for all three; the coincidence
# is the result, not a plotting fault. The low bands sit on the
# $N^{-1/2}$ guide, which is what a sample mean does. The top band sits far
# above it and barely moves with $N$: its error is not Monte-Carlo noise on a
# well-sampled mean but the catalog running out of sources, and no amount of
# $N^{-1/2}$ fixes a residual that is pinned near $-1$ because the estimator
# has nothing left to average.
#
# The point at $N = 1024$ in the low bands sits above the residual the actual
# 1024-source catalog achieved a few cells up, and it should: a bootstrap
# resample of size $N$ from an $N$-source parent contains only about 63%
# distinct sources, so it is a *different* catalog of the same size. What the
# curve estimates is the error of a typical size-$N$ catalog, not the error of
# this particular one — which is a single draw from that distribution, and
# happens to have landed on the low side.

# %% [markdown]
# ## Frequency-resolution convergence
#
# **First, the premise.** The catalog grid is `f_min + df * arange(n)`, so
# `frequencies[::k]` of a `df`-spaced catalog is *exactly* the grid a
# `k * df`-spaced catalog would have been built on — same start, same spacing,
# same values to the last bit (which is why `FINE_DF` is a power of two).
# Subsampling is therefore a legitimate stand-in for regenerating at a coarser
# resolution, and it holds the sources fixed so $\Delta f$ is the only thing
# that varies.
#
# **Then, the trap.** Every call site must be told the new width. `df` is not
# measured off the grid anywhere in `astrogwb` — masks may drop interior bins,
# so the mean spacing of what survives is not the bin width. Subsample `::k`,
# leave `df` at its old value, and `spectral_snr` falls by exactly $\sqrt{k}$:
# a clean, plausible, entirely spurious "convergence".

# %%
sensitivities = load_sensitivity_map(DETECTORS)


def analysis_at(factor: int) -> dict[str, Any]:
    """Re-derive the whole analysis on the grid coarsened by `factor`."""
    frequencies = jnp.asarray(fine_frequencies[::factor])
    power = jnp.asarray(fine_power[::factor])
    # The line the trap is about: df tracks the subsampling.
    df = factor * FINE_DF

    psd = jnp.asarray(effective_psd(np.asarray(frequencies), DETECTORS, sensitivities))
    mask = frequency_mask(frequencies, fmin=F_MIN, fmax=SNR_F_MAX) & jnp.isfinite(psd)
    frequencies, power, psd = apply_frequency_mask(mask, frequencies, power, psd)
    spectrum = spectral_density(
        power,
        jnp.ones(NUM_SOURCES),
        total_merger_rate,
        average_mode="analytic_inclination",
    )
    return {
        "factor": factor,
        "df": df,
        "num_bins": int(frequencies.shape[0]),
        "frequencies": frequencies,
        "power": power,
        "effective_psd": psd,
        "spectrum": spectrum,
        "snr": float(
            spectral_snr(spectrum, psd, OBSERVATION_TIME * SECONDS_PER_YEAR, df)
        ),
    }


runs = {factor: analysis_at(factor) for factor in SUBSAMPLE_FACTORS}

# The premise, checked rather than asserted.
for factor in SUBSAMPLE_FACTORS:
    direct_bins = int(np.floor((SNR_F_MAX - F_MIN) / (factor * FINE_DF))) + 1
    direct = F_MIN + factor * FINE_DF * np.arange(direct_bins, dtype=np.float64)
    assert np.array_equal(direct, fine_frequencies[::factor]), factor
print(
    f"subsampled grids match directly-built ones exactly for k in {SUBSAMPLE_FACTORS}"
)

reference = runs[SUBSAMPLE_FACTORS[0]]
snr_residual = {
    factor: run["snr"] / reference["snr"] - 1.0 for factor, run in runs.items()
}

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

integrand = np.asarray(reference["spectrum"] / reference["effective_psd"]) ** 2
reference_frequencies = np.asarray(reference["frequencies"])
axes[0].semilogx(reference_frequencies, integrand / integrand.max(), lw=1.3)
axes[0].set_xlabel("frequency [Hz]")
axes[0].set_ylabel(r"$S_h^2/S_{\rm eff}^2$, normalized")
axes[0].set_title(f"The SNR integrand at $\\Delta f$ = {FINE_DF} Hz")
peak = reference_frequencies[int(np.argmax(integrand))]
axes[0].axvline(peak, color="tab:red", lw=0.9, ls="--")
axes[0].annotate(
    f"peak at {peak:g} Hz",
    xy=(peak, 0.55),
    xytext=(6, 0),
    textcoords="offset points",
    fontsize=8,
    color="tab:red",
)

coarse = [f for f in SUBSAMPLE_FACTORS if f != SUBSAMPLE_FACTORS[0]]
widths = np.array([runs[f]["df"] for f in coarse])
magnitudes = np.array([abs(snr_residual[f]) for f in coarse])
axes[1].loglog(widths, magnitudes, "o-", lw=1.3, label="measured")
# The first-order guide is anchored in the middle of the range, not at the
# finest point: that point sits where the residual changes sign and is nearly
# zero by accident, so anchoring there would put the reference four decades
# below the data and demonstrate nothing.
anchor = len(coarse) // 2
axes[1].loglog(
    widths,
    magnitudes[anchor] * widths / widths[anchor],
    ls="--",
    color="k",
    lw=1.1,
    label=r"$\mathcal{O}(\Delta f)$ guide",
)
axes[1].set_xlabel(r"$\Delta f$ [Hz]")
axes[1].set_ylabel("|relative SNR residual|")
axes[1].set_title("SNR convergence under refinement")
axes[1].legend()
plt.show()

pd.DataFrame(
    {
        "df [Hz]": [runs[factor]["df"] for factor in SUBSAMPLE_FACTORS],
        "bins": [runs[factor]["num_bins"] for factor in SUBSAMPLE_FACTORS],
        "SNR": [runs[factor]["snr"] for factor in SUBSAMPLE_FACTORS],
        "residual": [snr_residual[factor] for factor in SUBSAMPLE_FACTORS],
    },
    index=pd.Index(SUBSAMPLE_FACTORS, name="k"),
).style.format({"df [Hz]": "{:.3f}", "SNR": "{:.4f}", "residual": "{:+.4e}"})

# %% [markdown]
# The residual falls monotonically under refinement, by more than four orders
# of magnitude across this range — but it does **not** follow a single power
# law, and the first-order guide fits only near $\Delta f \approx 1$ Hz. The
# left panel says why. The integrand is a peak a few hertz wide, so this range
# straddles a crossover: below about 1 Hz the grid resolves the peak and the
# error is ordinary discretization error, while at 4 and 8 Hz the grid steps
# over the peak instead. Undersampling is not a refinement regime, and no
# slope describes it.
#
# (The sign flips once, at $\Delta f = 0.25$ Hz, where the residual is
# $7\times10^{-6}$ and consistent with zero. Its position on a log-magnitude
# axis carries no information.)
#
# The practical reading: for this band and this network, $\Delta f \lesssim 1$
# Hz is a converged SNR, and $\Delta f = 8$ Hz — the resolution
# `mcmc_example_models.py` runs at for speed — costs about 13%.

# %% [markdown]
# ## Likelihood convergence
#
# The quantity that converges is the log-likelihood **ratio**, not
# $\log\mathcal{L}$ itself.
#
# The Gaussian bin scale is
# $\sigma_i = S_{{\rm eff},i}/\sqrt{2T\Delta f}$, so coarsening by $k$ grows
# every $\sigma_i$ by $\sqrt{k}$ while dropping the bin count by $k$. The
# normalization term $-\tfrac{1}{2}\sum_i \log(2\pi\sigma_i^2)$ therefore
# scales with the number of bins and **does not converge to anything** — it is
# a property of the discretization, not of the data. Only differences at fixed
# resolution, where it cancels, are meaningful. **Do not "fix" that drift** by
# normalizing per bin: it is not a defect, and removing it would rescale the
# ratio that does converge.
#
# There is no picture of the drift, because there is nothing to see. The data
# here *is* the template at the fiducial, so the $\chi^2$ term is identically
# zero and $\log\mathcal{L}$ **is** the normalization — two curves that
# coincide by algebra and cannot disagree. The numbers are printed with the
# closed-form check below instead.

# %%
# The estimator caches the proposal density and the reference distances once,
# from the catalog's own recorded population, so the H0 scan below pays for the
# target evaluation only. The reference distance is the stored distance column
# -- the one the stored power was generated at -- never a freshly interpolated
# cosmology table.
scan_estimator = SpectralDensityImportanceEstimator.from_catalog(
    catalog,
    model=target_model_fn(),
    average_mode="analytic_inclination",
)


def log_likelihood(run: dict[str, Any], hubble_constant: float) -> float:
    """Gaussian log-density of the injection under the H0-shifted template."""
    noise_scale = gaussian_bin_scale(run["effective_psd"], OBSERVATION_TIME, run["df"])
    params = {**FIDUCIALS, "H0": hubble_constant}
    _, extras = scan_estimator(params)
    model = spectral_density(
        run["power"],
        jnp.exp(scan_estimator.log_weights(params)),
        jnp.asarray(extras["total_merger_rate"]),
        average_mode="analytic_inclination",
    )
    return float(jnp.sum(dist.Normal(model, noise_scale).log_prob(run["spectrum"])))


scans = {
    factor: np.array([log_likelihood(run, h0) for h0 in H0_SCAN])
    for factor, run in runs.items()
}
at_fiducial = {
    factor: log_likelihood(run, FIDUCIALS["H0"]) for factor, run in runs.items()
}
delta = {factor: scans[factor] - at_fiducial[factor] for factor in SUBSAMPLE_FACTORS}

normalization = {
    factor: float(
        -0.5
        * jnp.sum(
            jnp.log(
                2.0
                * jnp.pi
                * gaussian_bin_scale(run["effective_psd"], OBSERVATION_TIME, run["df"])
                ** 2
            )
        )
    )
    for factor, run in runs.items()
}

fig, ax = plt.subplots(figsize=(7.0, 4.0))
for factor in SUBSAMPLE_FACTORS:
    ax.plot(
        H0_SCAN, delta[factor], lw=1.2, label=rf"$\Delta f$ = {runs[factor]['df']:g} Hz"
    )
ax.axvline(FIDUCIALS["H0"], color="k", lw=0.8)
ax.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
ax.set_ylabel(
    r"$\Delta\log\mathcal{L} = \log\mathcal{L}(H_0) - \log\mathcal{L}(H_0^{\rm fid})$"
)
ax.set_title("Converges: the log-likelihood ratio")
ax.legend(fontsize=7)
plt.show()

# %% [markdown]
# The curves lie on top of one another as $\Delta f \to 0$.
#
# The identity $\log\mathcal{L}(H_0^{\rm fid}) = -\tfrac{1}{2}\sum_i
# \log(2\pi\sigma_i^2)$ also gives the ratio's residual in closed form: with
# $S_h(H_0) = A\,S_h(H_0^{\rm fid})$ and $A = H_0^{\rm fid}/H_0$,
#
# $$\Delta\log\mathcal{L}(H_0) = -\tfrac{1}{2}(1 - A)^2\rho^2 ,$$
#
# so the relative residual of $\Delta\log\mathcal{L}$ must be **exactly** the
# relative residual of $\rho^2$, at every $H_0$ alike. That is a strong
# statement about the code, so it is checked rather than asserted:

# %%
scan_mask = np.abs(H0_SCAN - FIDUCIALS["H0"]) > 1.0
ratios = {
    factor: delta[factor][scan_mask] / delta[SUBSAMPLE_FACTORS[0]][scan_mask] - 1.0
    for factor in SUBSAMPLE_FACTORS
}
pd.DataFrame(
    {
        "df [Hz]": [runs[factor]["df"] for factor in SUBSAMPLE_FACTORS],
        "from Delta logL": [ratios[factor].mean() for factor in SUBSAMPLE_FACTORS],
        "from SNR^2": [
            (1.0 + snr_residual[factor]) ** 2 - 1.0 for factor in SUBSAMPLE_FACTORS
        ],
        "spread over H0": [np.ptp(ratios[factor]) for factor in SUBSAMPLE_FACTORS],
        "logL(H0_fid)": [at_fiducial[factor] for factor in SUBSAMPLE_FACTORS],
        "normalization": [normalization[factor] for factor in SUBSAMPLE_FACTORS],
    },
    index=pd.Index(SUBSAMPLE_FACTORS, name="k"),
).style.format(
    {
        "df [Hz]": "{:.3f}",
        "from Delta logL": "{:.6e}",
        "from SNR^2": "{:.6e}",
        "spread over H0": "{:.2e}",
        "logL(H0_fid)": "{:.6e}",
        "normalization": "{:.6e}",
    }
)

# %% [markdown]
# The last two columns are equal to every printed digit -- the $\chi^2$ term is
# identically zero here -- and both track the bin count rather than converging.
# That is the discretization, not the data.

# %% [markdown]
# ## Summary

# %%
reference_width = FIDUCIALS["H0"] / reference["snr"]
pd.DataFrame(
    {
        "df [Hz]": [runs[factor]["df"] for factor in SUBSAMPLE_FACTORS],
        "bins": [runs[factor]["num_bins"] for factor in SUBSAMPLE_FACTORS],
        "SNR": [runs[factor]["snr"] for factor in SUBSAMPLE_FACTORS],
        "SNR resid": [snr_residual[factor] for factor in SUBSAMPLE_FACTORS],
        "sigma_H0": [
            FIDUCIALS["H0"] / runs[factor]["snr"] for factor in SUBSAMPLE_FACTORS
        ],
        "width resid": [
            FIDUCIALS["H0"] / runs[factor]["snr"] / reference_width - 1.0
            for factor in SUBSAMPLE_FACTORS
        ],
    },
    index=pd.Index(SUBSAMPLE_FACTORS, name="k"),
).style.format(
    {
        "df [Hz]": "{:.3f}",
        "SNR": "{:.4f}",
        "SNR resid": "{:+.3e}",
        "sigma_H0": "{:.4f}",
        "width resid": "{:+.3e}",
    }
)

# %% [markdown]
# $\sigma_{H_0} = H_0^{\rm fid}/\rho$ is the Fisher width the noiseless linear
# model predicts, so an under-resolved grid does not merely mis-state the SNR:
# it reports a posterior that is too wide by the same factor.
