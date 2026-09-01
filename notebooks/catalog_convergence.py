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
# **This notebook needs no external data**; it carries its own population graph
# inline and caches the catalog it builds to `notebooks/convergence_catalog.h5`
# (gitignored).

# %% [markdown]
# ## Imports

# %%
import importlib.metadata
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import xarray as xr
from astrogwb.constants import ISCO_ALPHA, SECONDS_PER_YEAR
from astrogwb.detector import effective_psd, gaussian_bin_scale, load_sensitivity_map
from astrogwb.frequency import apply_frequency_mask, frequency_mask
from astrogwb.gwb import (
    analytic_spectral_density_from_mass_moments,
    omega_gw_from_spectral_density,
    spectral_density,
    spectral_snr,
    uniform_prior_mass_moments,
)
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
    madau_dickinson_rate,
    make_merger_rate_and_log_weights_fn,
)
from astrogwb.waveform import (
    inspiral_polarization_power,
    load_catalog,
    make_catalog,
    save_catalog,
)
from gwmock_pop import GraphSimulator
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection

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
# **Why the band is `[2, 256]` Hz and `FINE_DF` is 0.125 Hz.** With the ET
# effective PSD, the SNR integrand $S_h^2/S_{\rm eff}^2$ is a *narrow peak near
# 7 Hz*: about 90% of $\rho^2$ accumulates between 5 and 11 Hz, and 99.9% below
# 150 Hz. A reference grid must resolve that peak or it is not a reference at
# all — at $\Delta f = 1$ Hz the peak is sampled about four times, and the
# "converged" anchor is itself several parts in $10^3$ off. At
# $\Delta f = 0.125$ Hz the SNR is stable to $10^{-5}$ under a further halving,
# which is what makes the residuals below meaningful. Extending the band past
# 256 Hz would only add bins that contribute nothing.
#
# `FINE_DF` is a negative power of two on purpose: `k * FINE_DF` is then exact
# in binary, so the subsampled grids match a directly-built coarse grid to the
# last bit rather than to a tolerance.

# %%
SMOKE = os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") == "1"

NUM_SOURCES = 256 if SMOKE else 1024

#: Reference frequency resolution, and the analysis band.
FINE_DF = 0.125
F_MIN = 2.0
F_MAX = 256.0

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
CATALOG_PATH = NOTEBOOK_DIR / (
    "convergence_catalog_smoke.h5" if SMOKE else "convergence_catalog.h5"
)

# %% [markdown]
# ## The source population
#
# The graph below is an inline mirror of
# `packages/astrogwb/tests/fixtures/mock_bns_population.yaml`, the frozen
# population the core test suite draws its committed fixture from. **Keep the
# two in step**: nothing enforces their agreement, and that is deliberate.
#
# The component-mass bounds are load-bearing here in a way they are not in
# `mcmc_example_models.py`: `uniform_prior_mass_moments` below is given the
# *same* bounds, and the analytic spectrum it feeds is only the right oracle
# for this catalog if they agree. So is the zero `inclination` column, which
# pairs with `average_mode="analytic_inclination"`.

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

Z_MIN = 0.3
Z_MAX = 20.0
N_GRID = 256

MINIMUM_COMPONENT_MASS = 1.0
MAXIMUM_COMPONENT_MASS = 2.5

POPULATION_GRAPH: dict[str, Any] = {
    "luminosity_distance": {
        "transform": {
            "function": "redshift_to_luminosity_distance",
            "arguments": {
                "redshift": "@redshift",
                "hubble_constant": FIDUCIALS["H0"],
                "omega_m": FIDUCIALS["Omega_m"],
                "max_redshift": Z_MAX,
            },
        }
    },
    "mass_pair": {
        "intermediate": True,
        "sampler": {
            "function": "joint_uniform_mass_pair",
            "arguments": {
                "m1_min": MINIMUM_COMPONENT_MASS,
                "m1_max": MAXIMUM_COMPONENT_MASS,
                "m2_min": MINIMUM_COMPONENT_MASS,
                "m2_max": MAXIMUM_COMPONENT_MASS,
                "ordered": True,
            },
        },
    },
    "source_frame_mass_1": {
        "transform": {
            "function": "take_row",
            "arguments": {"matrix": "@mass_pair", "index": 0},
        }
    },
    "source_frame_mass_2": {
        "transform": {
            "function": "take_row",
            "arguments": {"matrix": "@mass_pair", "index": 1},
        }
    },
    "spin_1z": {
        "sampler": {
            "function": "uniform",
            "arguments": {"minimum": -0.05, "maximum": 0.05},
        }
    },
    "spin_2z": {
        "sampler": {
            "function": "uniform",
            "arguments": {"minimum": -0.05, "maximum": 0.05},
        }
    },
    "lambda_1": {
        "sampler": {
            "function": "uniform",
            "arguments": {"minimum": 0.0, "maximum": 2000.0},
        }
    },
    "lambda_2": {
        "sampler": {
            "function": "uniform",
            "arguments": {"minimum": 0.0, "maximum": 2000.0},
        }
    },
    "inclination": {
        "transform": {
            "function": "constant_like",
            "arguments": {"reference": "@redshift", "value": 0.0},
        }
    },
    "coa_phase": {
        "transform": {
            "function": "constant_like",
            "arguments": {"reference": "@redshift", "value": 0.0},
        }
    },
    "coa_time": {
        "transform": {
            "function": "constant_like",
            "arguments": {"reference": "@redshift", "value": 0.0},
        }
    },
    "redshift": {
        "sampler": {
            "function": "madau_dickinson_redshift",
            "arguments": {
                "z_min": Z_MIN,
                "z_max": Z_MAX,
                "gamma": FIDUCIALS["gamma"],
                "kappa": FIDUCIALS["kappa"],
                "z_peak": FIDUCIALS["z_peak"],
                "hubble_constant": FIDUCIALS["H0"],
                "omega_m": FIDUCIALS["Omega_m"],
                "n_grid": 4096,
            },
        }
    },
}

CATALOG_PARAMETERS = (
    "redshift",
    "source_frame_mass_1",
    "source_frame_mass_2",
    "inclination",
)


def make_redshift_grid() -> jax.Array:
    """The redshift grid every cosmology integral in this notebook runs on."""
    return jnp.linspace(Z_MIN, Z_MAX, N_GRID)


# %% [markdown]
# ## Building or loading the catalog
#
# One catalog, at the finest resolution: every coarser grid below is obtained
# from it by subsampling, so all of them describe the *same* sources and the
# only thing that varies is $\Delta f$.
#
# `build_catalog` discards the luminosity distance the graph produced and
# recomputes it from `compute_merger_rate_distance_and_logprob` at the
# fiducials, which is what makes the catalog exactly its own importance
# proposal ($\log w \equiv 0$).
#
# The cached file is reused only when its stored attributes still describe the
# configuration cell. That guard matters more here than in
# `mcmc_example_models.py`: `FINE_DF` *is* the subject, so silently reusing a
# catalog built at a different resolution would invalidate every result below
# while looking perfectly healthy.


# %%
def build_catalog() -> xr.Dataset:
    """Draw the population and reduce it to a `waveform_catalog` Dataset."""
    simulator = GraphSimulator(
        POPULATION_GRAPH, source_type="bns", seed=POPULATION_SEED
    )
    drawn = dict(simulator.simulate(NUM_SOURCES))
    parameters = {
        name: np.asarray(drawn[name], dtype=np.float64) for name in CATALOG_PARAMETERS
    }
    _, luminosity_distance, _ = compute_merger_rate_distance_and_logprob(
        FIDUCIALS,
        {"redshift": jnp.asarray(parameters["redshift"])},
        redshift_grid=make_redshift_grid(),
    )
    parameters["luminosity_distance"] = np.asarray(luminosity_distance)

    num_bins = int(np.floor((F_MAX - F_MIN) / FINE_DF)) + 1
    frequencies = F_MIN + FINE_DF * np.arange(num_bins, dtype=np.float64)
    power = np.asarray(
        inspiral_polarization_power(frequencies, parameters, alpha=ISCO_ALPHA)
    ).T
    return make_catalog(
        frequencies=frequencies,
        polarization_power=power,
        source_parameters=parameters,
        approximant="AnalyticInspiral",
        minimum_frequency=F_MIN,
        maximum_frequency=F_MAX,
        reference_frequency=F_MIN,
        sampling_frequency=2.0 * F_MAX,
        df=FINE_DF,
        extra_attrs={
            "notebook": "catalog_convergence",
            "population": "madau-dickinson",
            "population_seed": POPULATION_SEED,
            "num_sources": NUM_SOURCES,
            "termination_alpha": ISCO_ALPHA,
            "gwmock_pop_version": importlib.metadata.version("gwmock-pop"),
            **{f"fiducial_{name}": value for name, value in FIDUCIALS.items()},
        },
    )


def catalog_matches_configuration(catalog: xr.Dataset) -> bool:
    """Does a cached catalog still describe the configuration cell?

    Attributes survive the netCDF round trip as numpy scalars, so both sides
    are cast before comparing.
    """
    attrs = catalog.attrs
    return (
        float(attrs["df"]) == FINE_DF
        and float(attrs["minimum_frequency"]) == F_MIN
        and float(attrs["maximum_frequency"]) == F_MAX
        and int(attrs.get("num_sources", -1)) == NUM_SOURCES
        and int(attrs.get("population_seed", -1)) == POPULATION_SEED
        and all(
            float(attrs.get(f"fiducial_{name}", float("nan"))) == value
            for name, value in FIDUCIALS.items()
        )
    )


def load_or_build_catalog() -> xr.Dataset:
    """Return the cached catalog if it is still current, else rebuild it."""
    if CATALOG_PATH.is_file():
        cached = load_catalog(CATALOG_PATH)
        if catalog_matches_configuration(cached):
            print(f"Loaded {CATALOG_PATH}")
            return cached
        print(
            f"{CATALOG_PATH} does not match this notebook's configuration; rebuilding"
        )
    catalog = build_catalog()
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_catalog(CATALOG_PATH, catalog)
    print(f"Built and wrote {CATALOG_PATH}")
    return catalog


catalog = load_or_build_catalog()

fine_frequencies = np.asarray(catalog.frequency.values)
fine_power = np.asarray(catalog.polarization_power.values)
samples = {
    str(name): jnp.asarray(catalog.source_parameters.sel(parameter=name).values)
    for name in catalog.parameter.values
}
total_merger_rate, _, proposal_logprob = compute_merger_rate_distance_and_logprob(
    FIDUCIALS, samples, redshift_grid=make_redshift_grid()
)

print(f"  shape        {dict(catalog.sizes)}")
print(
    f"  grid         {fine_frequencies[0]}-{fine_frequencies[-1]} Hz, df = {FINE_DF} Hz"
)
print(f"  sources      {NUM_SOURCES}, seed {POPULATION_SEED}")
print(f"  gwmock-pop   {catalog.attrs['gwmock_pop_version']}")
print(f"  total merger rate {float(total_merger_rate):.6e} /s")

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
# $(1+z)^{4/3}$ factor, whereas `compute_merger_rate_distance_and_logprob`
# applies it inside its own density. Dividing in both places double-counts it
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
    jnp.asarray(fine_frequencies),
    minimum_redshift=Z_MIN,
    maximum_redshift=Z_MAX,
    minimum_component_mass=MINIMUM_COMPONENT_MASS,
    maximum_component_mass=MAXIMUM_COMPONENT_MASS,
    # The same truncation the catalog's polarization power was built with.
    alpha=ISCO_ALPHA,
)
analytic_spectrum = np.asarray(
    analytic_spectral_density_from_mass_moments(
        jnp.asarray(fine_frequencies),
        FIDUCIALS,
        source_frame_merger_rate,
        mass_moments,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
    )
)


def contract(power: np.ndarray, catalog_samples: dict[str, jax.Array]) -> np.ndarray:
    """Unweighted catalog contraction at the fiducials, for `power`'s sources."""
    num = power.shape[1]
    rate, _, _ = compute_merger_rate_distance_and_logprob(
        FIDUCIALS, catalog_samples, redshift_grid=make_redshift_grid()
    )
    return np.asarray(
        spectral_density(
            jnp.asarray(power),
            jnp.ones(num),
            rate,
            average_mode="analytic_inclination",
        )
    )


def to_omega(spectrum: np.ndarray) -> np.ndarray:
    """Convert a strain spectral density to the dimensionless Omega_gw."""
    return np.asarray(
        omega_gw_from_spectral_density(
            jnp.asarray(spectrum),
            jnp.asarray(fine_frequencies),
            hubble_constant=FIDUCIALS["H0"],
        )
    )


catalog_spectrum = contract(fine_power, samples)
omega_catalog = to_omega(catalog_spectrum)
omega_analytic = to_omega(analytic_spectrum)

# Bins where *every* source still emits. Above the smallest sampled cutoff a
# finite catalog loses sources one at a time while the analytic population
# spectrum, which integrates over the whole mass-redshift plane, does not --
# so the residual there measures the catalog's discreteness, not its Monte-
# Carlo error. Summary statistics are quoted on the common support; the plot
# shows the full band so the transition is visible rather than cropped away.
common_support = np.all(fine_power > 0.0, axis=1)
nonzero = catalog_spectrum > 0.0
residual = np.full_like(omega_catalog, np.nan)
residual[nonzero] = omega_catalog[nonzero] / omega_analytic[nonzero] - 1.0
support_edge = float(fine_frequencies[common_support][-1])

print(f"common support: {int(common_support.sum())} bins, up to {support_edge:g} Hz")
print(
    "relative residual on the common support: "
    f"mean {np.mean(residual[common_support]):+.4f}, "
    f"spread {np.ptp(residual[common_support]):.2e}"
)

fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True, height_ratios=(2, 1))
axes[0].loglog(
    fine_frequencies, omega_analytic, lw=1.4, color="k", label="analytic population"
)
axes[0].loglog(
    fine_frequencies[nonzero],
    omega_catalog[nonzero],
    lw=1.2,
    color="tab:blue",
    label=f"catalog, $N = {NUM_SOURCES}$",
)
axes[0].set_ylabel(r"$\Omega_{\rm gw}(f)$")
axes[0].set_title(r"Catalog contraction against the analytic $\Omega_{\rm gw}$")
axes[0].legend()

axes[1].axhline(0.0, color="k", lw=0.8)
axes[1].plot(fine_frequencies, residual, lw=1.2, color="tab:blue")
axes[1].set_xscale("log")
axes[1].set_xlabel("frequency [Hz]")
axes[1].set_ylabel("relative residual")
for ax in axes:
    ax.axvspan(
        fine_frequencies[0],
        support_edge,
        color="0.9",
        zorder=0,
        label=None,
    )
axes[1].annotate(
    "shaded: every source still emits",
    xy=(0.03, 0.86),
    xycoords="axes fraction",
    fontsize=8,
)
plt.show()

# %% [markdown]
# The residual is **flat** across the shaded region, and that is not a
# coincidence. Below the smallest sampled cutoff every source contributes at
# every bin, and both spectra are exactly $\propto f^{-7/3}$ there, so their
# ratio cannot depend on frequency: the whole Monte-Carlo error collapses to a
# single normalization offset. All the frequency *structure* lives above the
# shaded edge, where the catalog runs out of sources one cutoff at a time and
# the analytic curve does not.
#
# That is worth knowing before reading the next section: "the residual at
# catalog size $N$" is one number, not a curve.

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

# %%
rng = np.random.default_rng(RNG_SEED)
support_analytic = omega_analytic[common_support]

size_residuals: dict[int, np.ndarray] = {}
for size in CATALOG_SIZES:
    draws = np.empty(NUM_REALIZATIONS)
    for realization in range(NUM_REALIZATIONS):
        columns = rng.integers(0, NUM_SOURCES, size=size)
        subset_omega = to_omega(
            contract(
                fine_power[:, columns],
                {name: values[columns] for name, values in samples.items()},
            )
        )
        draws[realization] = np.sqrt(
            np.mean((subset_omega[common_support] / support_analytic - 1.0) ** 2)
        )
    size_residuals[size] = draws
    print(
        f"N = {size:5d}   rms relative residual = "
        f"{draws.mean():.5f} +/- {draws.std(ddof=1) / np.sqrt(NUM_REALIZATIONS):.5f}"
    )

sizes = np.array(CATALOG_SIZES, dtype=float)
means = np.array([size_residuals[s].mean() for s in CATALOG_SIZES])
errors = np.array(
    [size_residuals[s].std(ddof=1) / np.sqrt(NUM_REALIZATIONS) for s in CATALOG_SIZES]
)

fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.errorbar(sizes, means, yerr=errors, fmt="o-", lw=1.3, capsize=3, label="measured")
ax.plot(
    sizes,
    means[0] * np.sqrt(sizes[0] / sizes),
    ls="--",
    color="k",
    lw=1.1,
    label=r"$\propto N^{-1/2}$, anchored at the first point",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("catalog size $N$")
ax.set_ylabel("rms relative residual on the common support")
ax.set_title("Monte-Carlo convergence of the catalog spectrum")
ax.legend()
plt.show()

slope = np.polyfit(np.log(sizes), np.log(means), 1)[0]
print(f"fitted log-log slope: {slope:+.3f}   (Monte-Carlo prediction: -0.500)")

# %% [markdown]
# The point at $N = 1024$ sits well above the residual the actual 1024-source
# catalog achieved a few cells up, and it should: a bootstrap resample of size
# $N$ from an $N$-source parent contains only about 63% distinct sources, so it
# is a *different* catalog of the same size. What the curve estimates is the
# error of a typical size-$N$ catalog, not the error of this particular one --
# which is a single draw from that distribution, and happens to have landed on
# the low side.

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
    mask = frequency_mask(frequencies, fmin=F_MIN, fmax=F_MAX) & jnp.isfinite(psd)
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
    direct_bins = int(np.floor((F_MAX - F_MIN) / (factor * FINE_DF))) + 1
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

for factor in SUBSAMPLE_FACTORS:
    run = runs[factor]
    print(
        f"k = {factor:3d}   df = {run['df']:6.3f} Hz   bins = {run['num_bins']:5d}   "
        f"SNR = {run['snr']:9.4f}   residual = {snr_residual[factor]:+.4e}"
    )

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
# resolution, where it cancels, are meaningful.
#
# The second panel exists so that nobody later "fixes" that drift.

# %%
weights_fn = make_merger_rate_and_log_weights_fn(
    fiducials=FIDUCIALS,
    redshift_grid=make_redshift_grid(),
    proposal_logprob=proposal_logprob,
)


def log_likelihood(run: dict[str, Any], hubble_constant: float) -> float:
    """Gaussian log-density of the injection under the H0-shifted template."""
    noise_scale = gaussian_bin_scale(run["effective_psd"], OBSERVATION_TIME, run["df"])
    rate, log_weights = weights_fn({**FIDUCIALS, "H0": hubble_constant}, samples)
    model = spectral_density(
        run["power"],
        jnp.exp(log_weights),
        rate,
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

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
for factor in SUBSAMPLE_FACTORS:
    axes[0].plot(
        H0_SCAN, delta[factor], lw=1.2, label=rf"$\Delta f$ = {runs[factor]['df']:g} Hz"
    )
axes[0].axvline(FIDUCIALS["H0"], color="k", lw=0.8)
axes[0].set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
axes[0].set_ylabel(
    r"$\Delta\log\mathcal{L} = \log\mathcal{L}(H_0) - \log\mathcal{L}(H_0^{\rm fid})$"
)
axes[0].set_title("Converges: the ratio")
axes[0].legend(fontsize=7)

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
widths_all = np.array([runs[f]["df"] for f in SUBSAMPLE_FACTORS])
axes[1].loglog(
    widths_all,
    [abs(at_fiducial[f]) for f in SUBSAMPLE_FACTORS],
    "o-",
    lw=1.3,
    label=r"$|\log\mathcal{L}(H_0^{\rm fid})|$",
)
axes[1].loglog(
    widths_all,
    [abs(normalization[f]) for f in SUBSAMPLE_FACTORS],
    "s--",
    lw=1.1,
    label=r"$|-\frac{1}{2}\sum_i\log(2\pi\sigma_i^2)|$",
)
axes[1].set_xlabel(r"$\Delta f$ [Hz]")
axes[1].set_ylabel("absolute log-density")
axes[1].set_title("Does not converge: the normalization")
axes[1].legend(fontsize=8)
plt.show()

# %% [markdown]
# The left panel's curves lie on top of one another as $\Delta f \to 0$; the
# right panel's fall like the bin count, straight through every resolution,
# with $\log\mathcal{L}$ and the normalization term indistinguishable.
#
# They are indistinguishable because the data here *is* the template at the
# fiducial, so the $\chi^2$ term is identically zero and $\log\mathcal{L}$ is
# nothing but the normalization. That also gives the ratio's residual in closed
# form: with $S_h(H_0) = A\,S_h(H_0^{\rm fid})$ and $A = H_0^{\rm fid}/H_0$,
#
# $$\Delta\log\mathcal{L}(H_0) = -\tfrac{1}{2}(1 - A)^2\rho^2 ,$$
#
# so the relative residual of $\Delta\log\mathcal{L}$ must be **exactly** the
# relative residual of $\rho^2$, at every $H_0$ alike. That is a strong
# statement about the code, so it is checked rather than asserted:

# %%
scan_mask = np.abs(H0_SCAN - FIDUCIALS["H0"]) > 1.0
print(
    f"{'df [Hz]':>9} {'from Delta logL':>18} {'from SNR^2':>14} {'spread over H0':>16}"
)
for factor in SUBSAMPLE_FACTORS:
    ratio = delta[factor][scan_mask] / delta[SUBSAMPLE_FACTORS[0]][scan_mask] - 1.0
    predicted = (1.0 + snr_residual[factor]) ** 2 - 1.0
    print(
        f"{runs[factor]['df']:9.3f} {ratio.mean():18.6e} {predicted:14.6e} "
        f"{np.ptp(ratio):16.2e}"
    )

# %% [markdown]
# ## Summary

# %%
header = (
    f"{'k':>4} {'df [Hz]':>9} {'bins':>6} {'SNR':>10} {'SNR resid':>12} "
    f"{'sigma_H0':>10} {'width resid':>12}"
)
print(header)
print("-" * len(header))
for factor in SUBSAMPLE_FACTORS:
    run = runs[factor]
    width = FIDUCIALS["H0"] / run["snr"]
    reference_width = FIDUCIALS["H0"] / reference["snr"]
    print(
        f"{factor:4d} {run['df']:9.3f} {run['num_bins']:6d} {run['snr']:10.4f} "
        f"{snr_residual[factor]:+12.3e} {width:10.4f} "
        f"{width / reference_width - 1.0:+12.3e}"
    )
print()
print(
    "sigma_H0 = H0_fid / rho is the Fisher width the noiseless linear model "
    "predicts, so an under-resolved grid does not merely mis-state the SNR: it "
    "reports a posterior that is too wide by the same factor."
)
