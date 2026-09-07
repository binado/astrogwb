# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: astrogwb (3.12.9)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Catalog shot noise in the GWB spectral density
#
# `notebooks/cosmological_parameters_grid.py` originally drew its proposal from an
# independently seeded catalog against an independently seeded injection, and its
# $H_0$ posteriors quietly missed the fiducial line. The cause was not a
# weight-formula bug -- the raw importance weights stay within a hair of uniform
# whenever the proposal is drawn at the same fiducial cosmology as the target,
# regardless of catalog size or seed (`astrogwb.importance.diagnostics.relative_ess`
# alone is ≈1 throughout this notebook too) -- but Monte Carlo shot noise:
# polarization power scales as $1/d_L^2$ and the redshift density vanishes toward
# $z=0$, so the spectral-density power *sum* is dominated by whichever handful of
# samples land nearest the analysis window's `minimum_redshift` edge, no matter how
# uniform the weights multiplying them are. `astrogwb.importance.diagnostics`
# gained `power_weighted_relative_ess` for exactly this notebook: it folds
# `polarization_power` into the Kish effective-sample-size calculation instead of
# looking at the weights alone, and what it measures below is not subtle -- at the
# standard `minimum_redshift=0.3` cutoff every catalog here, self-matched or
# independent, retains only about 35-40% of its nominal sample count
# (`n_eff` in the thousands out of tens of thousands, essentially independent of
# $N$), and that fraction collapses by more than 25x, to roughly 1%
# (`n_eff` under 1000 even at $N=65536$), as the cutoff tightens toward $z=0.03$.
# Two independently seeded catalogs at the *same* size and cutoff still disagree at
# the percent level in $H_0$ purely from this. That notebook was fixed by reusing
# its injection catalog as its own proposal (the same trick already used for the
# IMRPhenom-vs-itself "systematics baseline" run), which removes the resulting
# importance-weight mismatch rather than showing the underlying noise source.
#
# This notebook makes the mechanism itself the subject: three figures showing how
# the recovered $H_0$ posterior degrades as (1) the proposal catalog shrinks,
# holding the redshift cutoff fixed, (1b) that same size sweep's relative bias
# against the fiducial, and (2) the redshift cutoff moves toward $z=0$, holding
# catalog size fixed. It uses the **default detector network only**
# (`DEFAULT_NETWORK`, `ET-2L-aligned-CE-Hanford`) to keep the figures to a small,
# readable set of curves.
#
# Because this is shot noise and not a systematic, a single realization's shift
# does **not** shrink monotonically with catalog size. Every proposal catalog here
# is a genuinely independent draw at its own seed (`PROPOSAL_SEEDS`), not a nested
# prefix of one shared stream, which if anything strengthens the point: a
# "worse" outcome at a larger $N$ is not an artifact of subsetting one draw, it is
# what independent shot noise actually looks like. That is expected, not a bug to
# chase.
#
# **No input files.** The injection and every proposal catalog are generated
# in-process from one Madau-Dickinson population graph (`POPULATION_GRAPH`
# below) and reduced to polarization power with a single shared `RippleGenerator`
# (`TaylorF2`, `f_max = 2048` Hz -- narrower than production's 4096 Hz, which
# halves the two resident frequency-by-sample arrays; see the memory-budget note
# below). The injection is drawn at `INJECTION_SEED`/`INJECTION_SIZE`; each
# size-sweep point is an independent draw at its own seed. **TaylorF2 changes the
# numbers, not the mechanism**: shot noise is a property of the Monte Carlo sum,
# not the approximant, so every quoted number below is measured from this
# notebook's own output and is not comparable to a previously committed run.
#
# **Outputs (when `SAVE_OUTPUTS`):** `figures/H0-catalog-size-sweep.pdf` + `.csv` +
# `.tex`, `figures/H0-relative-bias-vs-size.pdf` (no separate `.csv`/`.tex`; rides on
# `SIZE_SHIFT_TABLE`'s export), `figures/H0-redshift-cutoff-sweep.pdf` + `.csv` +
# `.tex`, and raw grids under `grids/catalog_shot_noise.npz` plus a JSON metadata
# sidecar.
#
# This notebook needs the repository root as its working directory.

# %% [markdown]
# ## Imports and JAX configuration

# %%
import json
import os
import time
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection

from astrogwb.catalog import Catalog, PopulationMetadata, simulate_population
from astrogwb.catalog.io import catalog_to_dataset
from astrogwb.importance.diagnostics import power_weighted_relative_ess
from astrogwb.importance.population import importance_log_weights
from astrogwb.paper.config.catalogs import (
    MadauDickinsonProposal,
    MixtureProposal,
    ProposalComponent,
    resolve_proposal,
)
from astrogwb.paper.config.constants import (
    DEFAULT_NETWORK,
    FIDUCIALS,
    NETWORK_DETECTORS,
    PARAMETER_LABELS,
)
from astrogwb.paper.config.mcmc import AnalysisGrid
from astrogwb.paper.inference import prepare_inference_inputs
from astrogwb.paper.plotting import (
    CATEGORY,
    DETECTOR_NETWORKS,
    MERGER_RATE_LEGEND,
    TRUTH,
    Network,
    combo_colors,
    use_paper_style,
)
from astrogwb.paper.snr import compute_network_snrs
from astrogwb.sampling import LogDensityFn, gwb_spectral_density_model
from astrogwb.waveform import RippleGenerator

# gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes. Restore
# matplotlib axes so plotting behaves as expected after importing detector utilities.
register_projection(MplAxes)
# %config InlineBackend.figure_format = 'retina'
use_paper_style()

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Notebook configuration

# %%
SMOKE = os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") == "1"

INJECTION_SEED: int = 41
INJECTION_SIZE: int = 2048 if SMOKE else 65536

# Independent draws, one seed per size -- not nested prefixes of one stream (see
# the markdown above). PROPOSAL_SEEDS[INJECTION_SIZE] additionally serves the
# z_min sweep below, since evaluate_h0_posteriors generates it once and loops
# over every requested minimum_redshift internally.
PROPOSAL_SEEDS: dict[int, int] = (
    {2048: 42, 512: 43}
    if SMOKE
    else {65536: 42, 32768: 43, 16384: 44, 8192: 45, 4096: 46}
)

# Held against the largest proposal draw: only minimum_redshift moves.
ZMIN_SWEEP_VALUES: tuple[float, ...] = (0.3, 0.1, 0.03)

# Frequency band and redshift grid, mirroring config/analysis/base/model.toml
# except f_max, narrowed from 4096 to 2048 Hz -- see the memory-budget note
# below. minimum_redshift here is the size sweep's (and the reference curve's)
# fixed cutoff; evaluate_h0_posteriors overrides it for each z_min sweep point.
ANALYSIS_GRID = AnalysisGrid(
    observation_time=1.0,
    f_min=2.0,
    f_max=2048.0,
    minimum_redshift=0.3,
    maximum_redshift=20.0,
    n_grid=256,
)

# TaylorF2 waveform generation, shared by the injection and every proposal draw
# so their frequency grids are bit-identical by construction --
# validate_matching_frequency_grids checks this with np.array_equal.
# sampling_frequency drives generation cost; f_max (via ANALYSIS_GRID) drives
# peak memory.
WAVEFORM_APPROXIMANT: str = "TaylorF2"
SAMPLING_FREQUENCY: float = 8192.0
REFERENCE_FREQUENCY: float = 20.0
FREQUENCY_RESOLUTION: float = 1.0
GENERATOR_CHUNK_SIZE: int = 2048

# Inlined to match config/analysis/base/parameters.toml, as in
# cosmological_parameters_grid.py. Every fiducial carries a prior:
# gwb_spectral_density_model samples every key, and LogDensityFn's `fixed=` pins
# the ones a given sweep is not gridding.
PRIORS: dict[str, dist.Distribution] = {
    "H0": dist.Uniform(20.0, 140.0),
    "Omega_m": dist.Normal(0.3096, 0.006),
    "xi_0": dist.Uniform(0.5, 5.0),
    "xi_n": dist.Uniform(0.3, 3.0),
    "gamma": dist.Uniform(-10.0, 10.0),
    "kappa": dist.Uniform(-10.0, 10.0),
    "z_peak": dist.Uniform(0.0, 2.5),
    "local_merger_rate": dist.Normal(770.0, 7.7),
}

# Widened from cosmological_parameters_grid.py's 5.0: the z_min=0.03 sweep point
# shifts the MAP by roughly 8 sigma at the Fisher-predicted scale, and 15 sigma of
# half-width leaves margin to see the posterior shape around that excursion.
COVERAGE_SIGMAS: float = 15.0
NPOINTS_1D: int = 32 if SMOKE else 256
CHUNK_SIZE: int = 64  # LogDensityFn batch_size; bounds peak memory

SAVE_OUTPUTS: bool = True
# `grids/` and `figures/` are repository-root artifacts (see CLAUDE.md), never
# notebook-local ones. Interactive use (Jupyter Lab/VS Code) launches with cwd
# already at the repository root, but `jupytext --execute` launches with cwd at
# this file's own directory -- the same quirk `catalog_convergence.py`'s
# `NOTEBOOK_DIR` works around. `config/` exists only at the repository root, so
# its presence tells the two cases apart.
_REPO_ROOT = Path() if Path("config").is_dir() else Path("..")
GRID_DIR = _REPO_ROOT / "grids"
FIGURE_DIR = _REPO_ROOT / "figures"

# %% [markdown]
# ## Helper functions
#
# `safe_exponentiate`, `fisher_window`, and `uniform_grid` are copied verbatim from
# `cosmological_parameters_grid.py`, which defines them locally rather than exporting
# them from `astrogwb.paper`; duplicating a few short functions matches this repo's
# existing pattern of self-contained notebooks over a new shared module.


# %%
def safe_exponentiate(log_values: jax.Array) -> np.ndarray:
    """Stably exponentiate a log-density array, off the JAX device.

    Subtracts the max before `exp` so the largest exponent is 0 and overflow
    cannot occur; the overall normalization is irrelevant for a density that
    is about to be renormalized by `np.trapezoid` anyway.
    """
    values = np.asarray(log_values, dtype=np.float64)
    return np.exp(values - values.max())


def fisher_window(
    center: float, sigma: float, *, sigmas: float, support: tuple[float, float]
) -> tuple[float, float]:
    """A Fisher-predicted window, centred on `center`, clipped to `support`."""
    low = max(center - sigmas * sigma, support[0])
    high = min(center + sigmas * sigma, support[1])
    return low, high


def uniform_grid(low: float, high: float, npoints: int) -> jax.Array:
    """A uniform grid."""
    return jnp.linspace(low, high, npoints)


def posterior_summary(
    grid: jax.Array, log_density: jax.Array
) -> tuple[float, float, float]:
    """MAP, mean, and standard deviation of a normalized 1D posterior grid.

    The MAP (`grid[argmax(log_density)]`) is what the shift tables below
    report -- the shift is framed as a shift in the posterior *peak*. Mean and
    standard deviation come along too, since a 15-sigma-wide posterior needs
    more than an assumed-symmetric interval to summarize its width.
    """
    grid_np = np.asarray(grid, dtype=np.float64)
    log_density_np = np.asarray(log_density, dtype=np.float64)
    h0_map = float(grid_np[np.argmax(log_density_np)])

    density = safe_exponentiate(log_density_np)
    density /= np.trapezoid(density, grid_np)
    mean = float(np.trapezoid(density * grid_np, grid_np))
    variance = float(np.trapezoid(density * (grid_np - mean) ** 2, grid_np))
    return h0_map, mean, float(np.sqrt(variance))


def shift_table_latex(table: pd.DataFrame, *, caption: str, label: str) -> str:
    """Format a shift table as a publication LaTeX tabular."""
    latex_table = table.rename(
        columns={
            "label": "Catalog",
            "h0_map": r"$H_0^{\rm MAP}$",
            "shift": r"$H_0^{\rm MAP} - H_0^{\rm fid}$",
            "sigma": r"$\sigma_{H_0}$",
            "shift_sigma": r"shift$/\sigma_{H_0}$",
            "rel_bias": r"$(H_0^{\rm MAP} - H_0^{\rm fid})/H_0^{\rm fid}$",
            "rel_sigma": r"$\sigma_{H_0}/H_0^{\rm fid}$",
            "n_eff": r"$N_{\rm eff}$",
            "rel_ess": r"$N_{\rm eff}/N$",
        }
    )
    return latex_table.to_latex(
        index=False,
        escape=False,
        float_format="%.3g",
        caption=caption,
        label=label,
    )


def write_shift_table(
    table: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
    *,
    caption: str,
    label: str,
) -> str:
    """Write the machine-readable and publication-formatted shift tables."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    latex = shift_table_latex(table, caption=caption, label=label)
    tex_path.write_text(latex, encoding="utf-8")
    return latex


# %% [markdown]
# ## Source population and waveform generator
#
# `POPULATION_GRAPH` is an inline mirror of
# `config/populations/madau-dickinson.yaml`. **Keep the two in step**: nothing
# enforces their agreement, and `GraphSimulator` draws RNG keys in topological
# order, broken by declaration order, so reordering these blocks silently
# changes every generated catalog.
#
# **`z_min = 0.0` here**, not the `0.3` `catalog_convergence.py` inlines: the
# z_min sweep reaches down to `0.03`, and `resolve_proposal` requires
# `mixture.z_min <= minimum_redshift`, so a graph drawn from `z_min = 0.3` would
# reject the two smallest cutoff sweep points.
#
# `GENERATOR` is the single `RippleGenerator` instance shared by the injection
# and every proposal draw below. That is what makes their frequency grids
# bit-identical by construction rather than by coincidence --
# `prepare_inference_inputs` checks exact equality with `np.array_equal`. Its
# `frequencies` property raises until the first call (Ripple sizes its own FFT
# segment), so nothing here inspects it before generating.
#
# `generate_catalog_dataset` mirrors `scripts/generate_catalog.py`'s generation
# path minus the config-layer loading: gwmock only provides the EM-frame ->
# source-frame mass conversion, so the inverse `(1 + z)` redshift to
# detector-frame masses is applied here, inline, exactly as production does.
# The graph's own `luminosity_distance` is kept as-is (unlike
# `catalog_convergence.py`, which recomputes it at the fiducials) because
# production catalogs store the graph's EM distance and this notebook is
# diagnosing production.

# %%
POPULATION_GRAPH: dict[str, Any] = {
    "luminosity_distance": {
        "transform": {
            "function": "redshift_to_luminosity_distance",
            "arguments": {
                "redshift": "@redshift",
                "hubble_constant": FIDUCIALS["H0"],
                "omega_m": FIDUCIALS["Omega_m"],
                "max_redshift": 20.0,
            },
        }
    },
    "mass_pair": {
        "intermediate": True,
        "sampler": {
            "function": "joint_uniform_mass_pair",
            "arguments": {
                "m1_min": 1.0,
                "m1_max": 2.5,
                "m2_min": 1.0,
                "m2_max": 2.5,
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
                "z_min": 0.0,
                "z_max": 20.0,
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

GENERATOR = RippleGenerator(
    approximant=WAVEFORM_APPROXIMANT,
    sampling_frequency=SAMPLING_FREQUENCY,
    minimum_frequency=ANALYSIS_GRID.f_min,
    maximum_frequency=ANALYSIS_GRID.f_max,
    reference_frequency=REFERENCE_FREQUENCY,
    frequency_resolution=FREQUENCY_RESOLUTION,
    chunk_size=GENERATOR_CHUNK_SIZE,
)

# The single Madau-Dickinson component every draw above follows, built from the
# same graph dict rather than read back off a file -- so the density and the
# draw cannot disagree.
PROPOSAL_MIXTURE = MixtureProposal(
    components=(
        ProposalComponent(
            weight=1.0,
            density=MadauDickinsonProposal.from_gwmock_dict(
                POPULATION_GRAPH["redshift"]["sampler"]["arguments"]
            ),
        ),
    )
)


def generate_catalog_dataset(*, seed: int, num_samples: int) -> xr.Dataset:
    """Draw one population and reduce it to polarization power via `GENERATOR`."""
    metadata = PopulationMetadata(
        name="madau-dickinson",
        seed=seed,
        num_samples=num_samples,
        source_type="bns",
        provenance={"notebook": "catalog_shot_noise"},
    )
    population = simulate_population(POPULATION_GRAPH, metadata=metadata)
    one_plus_z = 1.0 + population["redshift"]
    samples = {
        **population,
        "detector_frame_mass_1": population["source_frame_mass_1"] * one_plus_z,
        "detector_frame_mass_2": population["source_frame_mass_2"] * one_plus_z,
    }
    return catalog_to_dataset(
        Catalog.from_generator(
            samples, generator=GENERATOR, population_metadata=metadata
        )
    )


# %% [markdown]
# ## Building one sweep point's H0 posteriors
#
# The one substantive new helper. It generates one proposal catalog, then loops
# `minimum_redshift` *inside*: reloading a fresh proposal per z_min value would
# regenerate the same 64k catalog three times over, once per z_min sweep point,
# for no reason -- generating once and evaluating three redshift windows against
# it is what lets the largest size-sweep point double as the z_min sweep's whole
# input.
#
# **Reusing the injection as its own proposal needs no sentinel.** When
# `(seed, num_samples) == (INJECTION_SEED, INJECTION_SIZE)` the draw would be
# bitwise identical to `injection_catalog` anyway, so the function returns that
# object directly instead of regenerating it -- which is what makes the
# self-matched reference's importance weights exactly zero-mismatch rather than
# zero to rounding.
#
# **Memory budget.** At `f_max = 2048`, `df = 1.0 Hz`, the grid is $F = 2047$
# bins, so one 64k catalog's `polarization_power` is
# `2047 x 65536 x 8 B ~ 1.07 GB`. The invariant this function's return type
# holds is **at most two such arrays resident at once** -- the injection (alive
# for the whole notebook) and the proposal currently under evaluation, which
# becomes unreachable the moment this function returns a `PosteriorPoint`
# holding only small 1D arrays. Measured directly (per-sweep-point RSS tracing):
# the baseline between sweep points stays flat at 2.3-3.3 GB across all five
# sizes -- confirming no proposal leaks past its own point -- but the *transient*
# peak *during* the largest ($N=65536$) point's `LogDensityFn` evaluation reaches
# roughly 7-8 GB, well above a naive `1.07 (injection) + 1.07 (proposal numpy) +
# 1.07 (proposal device) ~ 3.2 GB` estimate: chunked evaluation
# (`CHUNK_SIZE` batches of the $H_0$ grid) keeps more than one band-masked
# device copy of the proposal power alive at once while XLA's allocator holds
# buffers rather than freeing them immediately. Smaller catalogs peak
# proportionally lower. Plan for a peak in the single-digit GB range at
# $N=65536$, not the ~3 GB the raw array arithmetic alone would suggest.
#
# **`LogDensityFn` recompiles at every sweep point.** The catalog's sample count
# `N` is baked into the traced array shapes, so a new proposal size forces a new
# trace; this is a consequence of the design, not an oversight to fix.


# %%
class PosteriorPoint(NamedTuple):
    """One `(proposal, minimum_redshift)` evaluation's H0 posterior and diagnostics."""

    h0_grid: jax.Array
    log_posterior: jax.Array
    relative_ess: float
    """N_eff / N of the Monte-Carlo power sum at the fiducials (Kish ESS of
    w_i * P_i, not of the importance weights alone -- see
    power_weighted_relative_ess)."""
    n_kept: int
    """Proposal samples surviving the minimum_redshift truncation."""


def evaluate_h0_posteriors(
    injection_catalog: xr.Dataset,
    *,
    seed: int,
    num_samples: int,
    minimum_redshifts: Sequence[float],
    network: Network,
    snr: float,
) -> dict[float, PosteriorPoint]:
    """H0 log-posterior grids for one proposal catalog, at every requested z_min."""
    proposal_catalog = (
        injection_catalog
        if (seed, num_samples) == (INJECTION_SEED, INJECTION_SIZE)
        else generate_catalog_dataset(seed=seed, num_samples=num_samples)
    )

    results: dict[float, PosteriorPoint] = {}
    for minimum_redshift in minimum_redshifts:
        grid = replace(ANALYSIS_GRID, minimum_redshift=minimum_redshift)
        proposal_config = resolve_proposal(
            PROPOSAL_MIXTURE,
            minimum_redshift=grid.minimum_redshift,
            maximum_redshift=grid.maximum_redshift,
            label=f"s{seed}-n{num_samples}",
        )

        inputs = prepare_inference_inputs(
            injection_catalog,
            proposal_catalog,
            fiducials=FIDUCIALS,
            proposal_config=proposal_config,
            grid=grid,
            detectors=network.detectors,
        )
        # importance_relative_ess (the estimator's own extras key) measures only
        # the raw weight ratios, which are ~uniform whenever the proposal was
        # drawn at the same fiducial cosmology as the target -- true for every
        # catalog here, so it stays pinned near 1.0 regardless of catalog size
        # or seed and says nothing about the mechanism this notebook is about.
        # power_weighted_relative_ess folds in polarization_power, evaluated at
        # the lowest surviving in-band frequency: the dominant 1/d_L(z)^2
        # blowup is common to every frequency bin for a given sample, so any
        # single bin gives a representative reading of the Monte-Carlo power
        # sum's actual effective sample size.
        catalog = inputs.estimator.catalog
        target = inputs.estimator.population_fn(FIDUCIALS).compute_population_terms(
            catalog.source_parameters
        )
        log_weights = importance_log_weights(
            target,
            proposal_log_prob=catalog.proposal_log_prob,
            log_reference_distance=catalog.log_reference_distance,
        )
        relative_ess = float(
            power_weighted_relative_ess(log_weights, catalog.polarization_power[0])
        )
        n_kept = int(inputs.proposal.sizes["sample"])

        model = partial(
            gwb_spectral_density_model,
            spectral_density_fn=inputs.estimator,
            priors=PRIORS,
        )
        log_density_fn = LogDensityFn(model, chunk_size=CHUNK_SIZE)

        h0_window = fisher_window(
            FIDUCIALS["H0"],
            FIDUCIALS["H0"] / snr,
            sigmas=COVERAGE_SIGMAS,
            support=(float(PRIORS["H0"].low), float(PRIORS["H0"].high)),
        )
        h0_grid = uniform_grid(*h0_window, NPOINTS_1D)
        fixed = {name: FIDUCIALS[name] for name in PRIORS if name != "H0"}
        logpost = jax.block_until_ready(
            log_density_fn({"H0": h0_grid}, fixed=fixed, **inputs.masked_model_kwargs())
        )
        results[minimum_redshift] = PosteriorPoint(
            h0_grid=h0_grid,
            log_posterior=logpost,
            relative_ess=relative_ess,
            n_kept=n_kept,
        )
    return results


# %% [markdown]
# ## Generating the injection catalog
#
# Generated once, at `INJECTION_SEED`/`INJECTION_SIZE`, and reused as
# `injection_catalog` for every call to `evaluate_h0_posteriors` below -- only
# the proposal side is regenerated per sweep point. It is the one array kept
# alive for the whole notebook (see the memory-budget note above).

# %%
injection_catalog = generate_catalog_dataset(
    seed=INJECTION_SEED, num_samples=INJECTION_SIZE
)

NETWORK: Network = Network(
    DEFAULT_NETWORK,
    dict(DETECTOR_NETWORKS)[DEFAULT_NETWORK],
    NETWORK_DETECTORS[DEFAULT_NETWORK],
)
print(f"Default network: {NETWORK.label} ({', '.join(NETWORK.detectors)})")

# %% [markdown]
# ## Matched-filter SNR for the default network
#
# Sizes every sweep point's $H_0$ grid via the Fisher prediction $\sigma_{H_0}
# \approx H_0/\mathrm{SNR}$, computed once at `ANALYSIS_GRID`'s fixed
# `minimum_redshift=0.3` and reused unchanged across every sweep point below --
# including the z_min sweep, whose own SNR does shift slightly with the cutoff, but
# only the grid's *width* depends on it, and `COVERAGE_SIGMAS=15` already leaves
# ample margin against that small mismatch.
#
# `compute_network_snrs` takes the in-memory `injection_catalog` directly --
# `astrogwb.paper.snr` widened its first parameter to accept either a path or an
# already-loaded dataset for exactly this notebook, since there is no file to
# point it at.

# %%
SNR_TABLE = compute_network_snrs(
    injection_catalog, [NETWORK], FIDUCIALS, grid=ANALYSIS_GRID
)
snr = float(SNR_TABLE["snr"].iloc[0])
print(f"SNR ({NETWORK.label}): {snr:.1f}")
SNR_TABLE

# %% [markdown]
# ## Zero-mismatch reference curve
#
# `evaluate_h0_posteriors` called with `seed=INJECTION_SEED,
# num_samples=INJECTION_SIZE`: that combination reuses `injection_catalog` as
# its own proposal (see above), so the importance weights carry zero catalog
# mismatch and the posterior sits close to the fiducial line, up to the
# injection's own finite-N noise.
#
# Its `relative_ess` (the power-weighted diagnostic, see `PosteriorPoint`) comes
# out **statistically indistinguishable** from the independently drawn
# `N=INJECTION_SIZE` proposal in the sweep below (~0.38 either way): this
# diagnostic reflects each catalog's own intrinsic Monte-Carlo noise, present
# whether or not the proposal matches the injection, not the importance-weight
# mismatch between the two. What differs is the recovered $H_0$ shift itself --
# the self-matched curve's MAP sits almost exactly on the fiducial (zero weight
# mismatch by construction), while the independently drawn catalog's does not,
# because it combines two *separate* finite-$N$ noisy realizations. Evaluated
# only at `ANALYSIS_GRID`'s fixed `minimum_redshift`: the z_min sweep below
# already shows how `relative_ess` moves with the cutoff, via the independently
# drawn catalogs, so re-evaluating this self-matched curve at every z_min would
# not add anything past that.

# %%
REFERENCE_POINT = evaluate_h0_posteriors(
    injection_catalog,
    seed=INJECTION_SEED,
    num_samples=INJECTION_SIZE,
    minimum_redshifts=(ANALYSIS_GRID.minimum_redshift,),
    network=NETWORK,
    snr=snr,
)[ANALYSIS_GRID.minimum_redshift]
print(f"self-matched relative_ess: {REFERENCE_POINT.relative_ess:.6f}")

# %% [markdown]
# ## Figure 1 -- H0 posterior vs. proposal catalog size
#
# ≙ `H0-catalog-size-sweep.pdf`. `minimum_redshift` is held fixed at `0.3`; only
# the proposal catalog's sample count and seed vary. **Shot noise does not
# shrink monotonically in a single realization** -- the curves below are not
# expected to nest tightest to widest with $N$, and a "worse" outcome at a
# larger $N$ is not a bug.
#
# The loop below also fills `ZMIN_SWEEP_RESULTS`: the largest sweep point
# (`N=INJECTION_SIZE`) is evaluated at every `ZMIN_SWEEP_VALUES` entry in one
# call, since `evaluate_h0_posteriors` generates that catalog once and reuses it
# across `minimum_redshift` -- Figure 3 below reads its result directly rather
# than generating a seventh catalog.

# %%
_size_colors = combo_colors(len(PROPOSAL_SEEDS))

SIZE_SWEEP_RESULTS: dict[int, PosteriorPoint] = {}
ZMIN_SWEEP_RESULTS: dict[float, PosteriorPoint] = {}
for n_samples, seed in PROPOSAL_SEEDS.items():
    minimum_redshifts = (
        ZMIN_SWEEP_VALUES
        if n_samples == INJECTION_SIZE
        else (ANALYSIS_GRID.minimum_redshift,)
    )
    start = time.perf_counter()
    points = evaluate_h0_posteriors(
        injection_catalog,
        seed=seed,
        num_samples=n_samples,
        minimum_redshifts=minimum_redshifts,
        network=NETWORK,
        snr=snr,
    )
    print(f"N={n_samples} (s{seed}): {time.perf_counter() - start:.2f}s")
    SIZE_SWEEP_RESULTS[n_samples] = points[ANALYSIS_GRID.minimum_redshift]
    if n_samples == INJECTION_SIZE:
        ZMIN_SWEEP_RESULTS = points

fig_size_sweep, ax = plt.subplots()
for (n_samples, point), color in zip(
    SIZE_SWEEP_RESULTS.items(), _size_colors, strict=True
):
    grid_np = np.asarray(point.h0_grid)
    density = safe_exponentiate(point.log_posterior)
    density /= np.trapezoid(density, grid_np)
    ax.plot(
        grid_np,
        density,
        label=f"N={n_samples} (s{PROPOSAL_SEEDS[n_samples]})",
        color=color,
    )

_reference_grid_np = np.asarray(REFERENCE_POINT.h0_grid)
_reference_density = safe_exponentiate(REFERENCE_POINT.log_posterior)
_reference_density /= np.trapezoid(_reference_density, _reference_grid_np)
ax.plot(
    _reference_grid_np,
    _reference_density,
    label=f"self-matched (N={INJECTION_SIZE})",
    color=str(TRUTH["color"]),
    linestyle="--",
    linewidth=TRUTH["linewidth"],
)
ax.axvline(FIDUCIALS["H0"], **TRUTH)
ax.set(xlabel=PARAMETER_LABELS["H0"], ylabel="Posterior density")
ax.legend(**MERGER_RATE_LEGEND)
fig_size_sweep.tight_layout()
fig_size_sweep

# %% [markdown]
# `SIZE_SHIFT_TABLE`: MAP shift from the fiducial, in units of the grid posterior's
# own standard deviation, plus the measured effective sample size (`n_eff`,
# `rel_ess`), for each sweep point and the self-matched reference.

# %%
_size_rows: list[dict[str, object]] = []
for n_samples, point in SIZE_SWEEP_RESULTS.items():
    h0_map, _, sigma = posterior_summary(point.h0_grid, point.log_posterior)
    _size_rows.append(
        {
            "label": f"N={n_samples} (s{PROPOSAL_SEEDS[n_samples]})",
            "h0_map": h0_map,
            "shift": h0_map - FIDUCIALS["H0"],
            "sigma": sigma,
            "shift_sigma": (h0_map - FIDUCIALS["H0"]) / sigma,
            "rel_ess": point.relative_ess,
            "n_eff": point.relative_ess * point.n_kept,
        }
    )
_reference_h0_map, _, _reference_sigma = posterior_summary(
    REFERENCE_POINT.h0_grid, REFERENCE_POINT.log_posterior
)
_size_rows.append(
    {
        "label": f"self-matched (N={INJECTION_SIZE})",
        "h0_map": _reference_h0_map,
        "shift": _reference_h0_map - FIDUCIALS["H0"],
        "sigma": _reference_sigma,
        "shift_sigma": (_reference_h0_map - FIDUCIALS["H0"]) / _reference_sigma,
        "rel_ess": REFERENCE_POINT.relative_ess,
        "n_eff": REFERENCE_POINT.relative_ess * REFERENCE_POINT.n_kept,
    }
)
SIZE_SHIFT_TABLE = pd.DataFrame(_size_rows).assign(
    rel_bias=lambda df: df["shift"] / FIDUCIALS["H0"],
    rel_sigma=lambda df: df["sigma"] / FIDUCIALS["H0"],
)
SIZE_SHIFT_TABLE

# %% [markdown]
# ## Figure 2 -- Relative bias in H0 vs. catalog size
#
# ≙ `H0-relative-bias-vs-size.pdf`. The same `SIZE_SHIFT_TABLE` values plotted
# against `N` instead of overlaid as posterior curves: relative bias
# `(H0_MAP - H0_fid)/H0_fid`, error bars at `sigma_H0/H0_fid`. As in Figure 1,
# five points from one independent realization each is not enough to fit a
# shot-noise scaling law -- this is a visual comparison against the
# self-matched floor, not a fitted trend.

# %%
_size_ns = np.asarray(list(PROPOSAL_SEEDS), dtype=np.float64)
_size_sweep_rows = SIZE_SHIFT_TABLE.iloc[: len(_size_ns)]
_reference_row = SIZE_SHIFT_TABLE.iloc[-1]

fig_size_relative_bias, ax = plt.subplots()
ax.errorbar(
    _size_ns,
    _size_sweep_rows["rel_bias"],
    yerr=_size_sweep_rows["rel_sigma"],
    fmt="o-",
    lw=1.3,
    capsize=3,
    color=CATEGORY["cosmology"],
    label="size sweep",
)
ax.errorbar(
    [INJECTION_SIZE],
    [_reference_row["rel_bias"]],
    yerr=[_reference_row["rel_sigma"]],
    fmt="D",
    capsize=3,
    color=str(TRUTH["color"]),
    label=f"self-matched (N={INJECTION_SIZE})",
)
ax.axhline(0.0, **TRUTH)
ax.set_xscale("log")
ax.set_xlabel("catalog size $N$")
ax.set_ylabel(r"relative bias in $H_0$")
ax.legend(**MERGER_RATE_LEGEND)
fig_size_relative_bias.tight_layout()
fig_size_relative_bias

# %% [markdown]
# ## Figure 3 -- H0 posterior vs. redshift cutoff
#
# ≙ `H0-redshift-cutoff-sweep.pdf`. The proposal catalog is held fixed at
# `N=INJECTION_SIZE` (seed `PROPOSAL_SEEDS[INJECTION_SIZE]`); only
# `minimum_redshift` varies. No new catalog generation or SNR computation is
# needed here: `ZMIN_SWEEP_RESULTS` was already filled by the Figure 1 loop
# above, and `fisher_window` reuses the `snr` computed earlier.

# %%
_zmin_colors = combo_colors(len(ZMIN_SWEEP_VALUES))

fig_zmin_sweep, ax = plt.subplots()
for (minimum_redshift, point), color in zip(
    ZMIN_SWEEP_RESULTS.items(), _zmin_colors, strict=True
):
    grid_np = np.asarray(point.h0_grid)
    density = safe_exponentiate(point.log_posterior)
    density /= np.trapezoid(density, grid_np)
    ax.plot(grid_np, density, label=f"z_min={minimum_redshift}", color=color)

ax.plot(
    _reference_grid_np,
    _reference_density,
    label=f"self-matched (z_min={ANALYSIS_GRID.minimum_redshift})",
    color=str(TRUTH["color"]),
    linestyle="--",
    linewidth=TRUTH["linewidth"],
)
ax.axvline(FIDUCIALS["H0"], **TRUTH)
ax.set(xlabel=PARAMETER_LABELS["H0"], ylabel="Posterior density")
ax.legend(**MERGER_RATE_LEGEND)
fig_zmin_sweep.tight_layout()
fig_zmin_sweep

# %% [markdown]
# `ZMIN_SHIFT_TABLE`: same treatment as `SIZE_SHIFT_TABLE`, one row per swept
# `z_min` plus the self-matched reference.

# %%
_zmin_rows: list[dict[str, object]] = []
for minimum_redshift, point in ZMIN_SWEEP_RESULTS.items():
    h0_map, _, sigma = posterior_summary(point.h0_grid, point.log_posterior)
    _zmin_rows.append(
        {
            "label": f"z_min={minimum_redshift}",
            "h0_map": h0_map,
            "shift": h0_map - FIDUCIALS["H0"],
            "sigma": sigma,
            "shift_sigma": (h0_map - FIDUCIALS["H0"]) / sigma,
            "rel_ess": point.relative_ess,
            "n_eff": point.relative_ess * point.n_kept,
        }
    )
_zmin_rows.append(
    {
        "label": f"self-matched (z_min={ANALYSIS_GRID.minimum_redshift})",
        "h0_map": _reference_h0_map,
        "shift": _reference_h0_map - FIDUCIALS["H0"],
        "sigma": _reference_sigma,
        "shift_sigma": (_reference_h0_map - FIDUCIALS["H0"]) / _reference_sigma,
        "rel_ess": REFERENCE_POINT.relative_ess,
        "n_eff": REFERENCE_POINT.relative_ess * REFERENCE_POINT.n_kept,
    }
)
ZMIN_SHIFT_TABLE = pd.DataFrame(_zmin_rows)
ZMIN_SHIFT_TABLE

# %% [markdown]
# ## Saving the grids and figures

# %%
if SAVE_OUTPUTS:
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        GRID_DIR / "catalog_shot_noise.npz",
        reference_grid=_reference_grid_np,
        reference_logpost=np.asarray(REFERENCE_POINT.log_posterior),
        **{
            f"size_grid_{n_samples}": np.asarray(point.h0_grid)
            for n_samples, point in SIZE_SWEEP_RESULTS.items()
        },
        **{
            f"size_logpost_{n_samples}": np.asarray(point.log_posterior)
            for n_samples, point in SIZE_SWEEP_RESULTS.items()
        },
        **{
            f"zmin_grid_{minimum_redshift}": np.asarray(point.h0_grid)
            for minimum_redshift, point in ZMIN_SWEEP_RESULTS.items()
        },
        **{
            f"zmin_logpost_{minimum_redshift}": np.asarray(point.log_posterior)
            for minimum_redshift, point in ZMIN_SWEEP_RESULTS.items()
        },
    )
    (GRID_DIR / "catalog_shot_noise.json").write_text(
        json.dumps(
            {
                "fiducials": FIDUCIALS,
                "coverage_sigmas": COVERAGE_SIGMAS,
                "npoints_1d": NPOINTS_1D,
                "network": NETWORK.name,
                "snr": snr,
                "approximant": WAVEFORM_APPROXIMANT,
                "f_max": ANALYSIS_GRID.f_max,
                "injection_seed": INJECTION_SEED,
                "injection_size": INJECTION_SIZE,
                "proposal_seeds": PROPOSAL_SEEDS,
                "zmin_sweep_values": list(ZMIN_SWEEP_VALUES),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fig_size_sweep.savefig(
        FIGURE_DIR / "H0-catalog-size-sweep.pdf", bbox_inches="tight"
    )
    fig_size_relative_bias.savefig(
        FIGURE_DIR / "H0-relative-bias-vs-size.pdf", bbox_inches="tight"
    )
    fig_zmin_sweep.savefig(
        FIGURE_DIR / "H0-redshift-cutoff-sweep.pdf", bbox_inches="tight"
    )

    write_shift_table(
        SIZE_SHIFT_TABLE,
        FIGURE_DIR / "H0-catalog-size-sweep.csv",
        FIGURE_DIR / "H0-catalog-size-sweep.tex",
        caption=(
            "Recovered $H_0$ MAP shift from the fiducial value, in units of the "
            "grid posterior's own standard deviation, as the proposal catalog "
            rf"size grows at fixed $z_{{\rm min}}={ANALYSIS_GRID.minimum_redshift}$, "
            "each point an independent draw at its own seed."
        ),
        label="tab:catalog_shot_noise_size_sweep",
    )
    write_shift_table(
        ZMIN_SHIFT_TABLE,
        FIGURE_DIR / "H0-redshift-cutoff-sweep.csv",
        FIGURE_DIR / "H0-redshift-cutoff-sweep.tex",
        caption=(
            "Recovered $H_0$ MAP shift from the fiducial value, in units of the "
            "grid posterior's own standard deviation, as the redshift cutoff "
            rf"moves toward $z=0$ at fixed $N={INJECTION_SIZE}$."
        ),
        label="tab:catalog_shot_noise_zmin_sweep",
    )
    print("saved grids to", GRID_DIR, "and figures to", FIGURE_DIR)
