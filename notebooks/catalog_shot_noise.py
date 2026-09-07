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
# independently seeded catalog (`s42`) against an `s41` injection, and its $H_0$
# posteriors quietly missed the fiducial line. The cause was not a weight-formula
# bug -- the importance weights themselves are exact -- but Monte Carlo shot noise:
# polarization power scales as $1/d_L^2$ and the redshift density vanishes toward
# $z=0$, so the spectral-density power sum is dominated by whichever handful of
# samples land nearest the analysis window's `minimum_redshift` edge, an effective
# sample size of order 10-50 out of tens of thousands of draws. Two independently
# seeded catalogs disagree at the percent level purely from this. That notebook was
# fixed by reusing its injection catalog as its own proposal (the same trick already
# used for the IMRPhenom-vs-itself "systematics baseline" run), which removes the
# noise source rather than showing it.
#
# This notebook makes the mechanism itself the subject: three figures showing how the
# recovered $H_0$ posterior degrades as (1) the proposal catalog shrinks, holding the
# redshift cutoff fixed, (1b) that same size sweep's relative bias against the
# fiducial, and (2) the redshift cutoff moves toward $z=0$, holding catalog size
# fixed. It uses the **default detector network only** (`DEFAULT_NETWORK`,
# `ET-2L-aligned-CE-Hanford`) to keep the figures to a small, readable set of curves.
#
# Because this is shot noise and not a systematic, a single realization's shift does
# **not** shrink monotonically with catalog size -- see the size-sweep figure below,
# where $N=16384$ sits farther from the fiducial than $N=32768$. That is expected,
# not a bug to chase.
#
# **Inputs:** `outputs/catalogs/md-imrphenom-s41-n32768.h5` (injection, and reused as
# its own proposal for the zero-mismatch reference curve),
# `outputs/catalogs/md-imrphenom-s42-n{8192,16384,32768}.h5` (size sweep -- the
# smaller files are exact prefixes of the larger one, same seed), and
# `outputs/catalogs/md-imrphenom-s42-n32768.h5` again at three `minimum_redshift`
# values (0.3, 0.1, 0.03) for the cutoff sweep. All already exist; no new catalog
# generation needed.
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
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection

from astrogwb.paper.catalogs import load_run_catalog
from astrogwb.paper.config.catalogs import (
    CatalogProvenance,
    check_fiducials_match,
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

# gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes. Restore
# matplotlib axes so plotting behaves as expected after importing detector utilities.
register_projection(MplAxes)
# %config InlineBackend.figure_format = 'retina'
use_paper_style()

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Notebook configuration

# %%
INJECTION_CATALOG_PATH = Path("outputs/catalogs/md-imrphenom-s41-n32768.h5")

# Same seed (s42) at three sizes: the smaller files are exact prefixes of
# md-imrphenom-s42-n32768.h5, so this sweep varies only the effective sample size,
# not the draw itself.
SIZE_SWEEP_PROPOSAL_PATHS: dict[int, Path] = {
    8192: Path("outputs/catalogs/md-imrphenom-s42-n8192.h5"),
    16384: Path("outputs/catalogs/md-imrphenom-s42-n16384.h5"),
    32768: Path("outputs/catalogs/md-imrphenom-s42-n32768.h5"),
}

# Held against the largest s42 catalog: only minimum_redshift moves.
ZMIN_SWEEP_VALUES: tuple[float, ...] = (0.3, 0.1, 0.03)
ZMIN_SWEEP_PROPOSAL_PATH = SIZE_SWEEP_PROPOSAL_PATHS[32768]

# Frequency band and redshift grid, mirroring config/analysis/base/model.toml.
# minimum_redshift here is the size sweep's (and the reference curve's) fixed
# cutoff; evaluate_h0_posterior overrides it for each z_min sweep point.
ANALYSIS_GRID = AnalysisGrid(
    observation_time=1.0,
    f_min=2.0,
    f_max=4096.0,
    minimum_redshift=0.3,
    maximum_redshift=20.0,
    n_grid=256,
)

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
NPOINTS_1D: int = 256
CHUNK_SIZE: int = 64  # LogDensityFn batch_size; bounds peak memory

SAVE_OUTPUTS: bool = True
GRID_DIR = Path("grids")
FIGURE_DIR = Path("figures")

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
# ## Building one sweep point's H0 posterior
#
# The one substantive new helper. Each sweep point re-derives its own proposal
# density and analysis inputs from scratch -- unlike
# `cosmological_parameters_grid.py`'s per-network evaluator, there is no shared
# `LogDensityFn` to reuse across a sweep's own points, since either the proposal
# file or `minimum_redshift` changes at every call. This replaces re-inlining the
# same ~15-line block seven times (three size points, three z_min points, one
# self-matched reference).


# %%
def evaluate_h0_posterior(
    injection_catalog: xr.Dataset,
    proposal_path: Path,
    *,
    minimum_redshift: float,
    network: Network,
    snr: float,
) -> tuple[jax.Array, jax.Array]:
    """H0 log-posterior grid for one proposal catalog and redshift cutoff."""
    proposal_catalog = load_run_catalog(proposal_path, label="proposal")
    grid = replace(ANALYSIS_GRID, minimum_redshift=minimum_redshift)

    provenance = CatalogProvenance.from_file(proposal_path)
    check_fiducials_match(provenance, FIDUCIALS, label=str(proposal_path))
    proposal_config = resolve_proposal(
        provenance.redshift_proposal,
        minimum_redshift=grid.minimum_redshift,
        maximum_redshift=grid.maximum_redshift,
        label=str(proposal_path),
    )

    inputs = prepare_inference_inputs(
        injection_catalog,
        proposal_catalog,
        fiducials=FIDUCIALS,
        proposal_config=proposal_config,
        grid=grid,
        detectors=network.detectors,
    )
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
    return h0_grid, logpost


# %% [markdown]
# ## Loading the injection catalog
#
# Loaded once and reused as `injection_catalog` for every call to
# `evaluate_h0_posterior` below -- only the proposal side changes between sweep
# points.

# %%
injection_catalog = load_run_catalog(INJECTION_CATALOG_PATH, label="injection")

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

# %%
SNR_TABLE = compute_network_snrs(
    INJECTION_CATALOG_PATH, [NETWORK], FIDUCIALS, grid=ANALYSIS_GRID
)
snr = float(SNR_TABLE["snr"].iloc[0])
print(f"SNR ({NETWORK.label}): {snr:.1f}")
SNR_TABLE

# %% [markdown]
# ## Zero-mismatch reference curve
#
# `evaluate_h0_posterior` called with `proposal_path = INJECTION_CATALOG_PATH`: the
# same file plays both roles, so the importance weights carry zero catalog mismatch
# and the posterior sits on the fiducial line up to numerical noise. Computed once
# here and reused as a fixed dashed reference line in **both** figures below.
# Re-evaluating it at each swept `z_min` would show nothing new: with the same file
# on both sides, the residual is at the bit level regardless of `minimum_redshift`.

# %%
REFERENCE_GRID, REFERENCE_LOGPOST = evaluate_h0_posterior(
    injection_catalog,
    INJECTION_CATALOG_PATH,
    minimum_redshift=ANALYSIS_GRID.minimum_redshift,
    network=NETWORK,
    snr=snr,
)

# %% [markdown]
# ## Figure 1 -- H0 posterior vs. proposal catalog size
#
# ≙ `H0-catalog-size-sweep.pdf`. `minimum_redshift` is held fixed at `0.3`; only the
# proposal catalog's sample count varies. **Shot noise does not shrink monotonically
# in a single realization** -- the curves below are not expected to nest tightest to
# widest with $N$, and a "worse" outcome at a larger $N$ is not a bug.

# %%
_size_colors = combo_colors(len(SIZE_SWEEP_PROPOSAL_PATHS))

SIZE_SWEEP_RESULTS: dict[int, tuple[jax.Array, jax.Array]] = {}
for n_samples, proposal_path in SIZE_SWEEP_PROPOSAL_PATHS.items():
    start = time.perf_counter()
    SIZE_SWEEP_RESULTS[n_samples] = evaluate_h0_posterior(
        injection_catalog,
        proposal_path,
        minimum_redshift=ANALYSIS_GRID.minimum_redshift,
        network=NETWORK,
        snr=snr,
    )
    print(f"N={n_samples}: {time.perf_counter() - start:.2f}s")

fig_size_sweep, ax = plt.subplots()
for (n_samples, (grid, logpost)), color in zip(
    SIZE_SWEEP_RESULTS.items(), _size_colors, strict=True
):
    grid_np = np.asarray(grid)
    density = safe_exponentiate(logpost)
    density /= np.trapezoid(density, grid_np)
    ax.plot(grid_np, density, label=f"N={n_samples}", color=color)

_reference_grid_np = np.asarray(REFERENCE_GRID)
_reference_density = safe_exponentiate(REFERENCE_LOGPOST)
_reference_density /= np.trapezoid(_reference_density, _reference_grid_np)
ax.plot(
    _reference_grid_np,
    _reference_density,
    label="self-matched (N=32768)",
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
# own standard deviation, for each sweep point plus the self-matched reference.

# %%
_size_rows: list[dict[str, object]] = []
for n_samples, (grid, logpost) in SIZE_SWEEP_RESULTS.items():
    h0_map, _, sigma = posterior_summary(grid, logpost)
    _size_rows.append(
        {
            "label": f"N={n_samples}",
            "h0_map": h0_map,
            "shift": h0_map - FIDUCIALS["H0"],
            "sigma": sigma,
            "shift_sigma": (h0_map - FIDUCIALS["H0"]) / sigma,
        }
    )
_reference_h0_map, _, _reference_sigma = posterior_summary(
    REFERENCE_GRID, REFERENCE_LOGPOST
)
_size_rows.append(
    {
        "label": "self-matched (N=32768)",
        "h0_map": _reference_h0_map,
        "shift": _reference_h0_map - FIDUCIALS["H0"],
        "sigma": _reference_sigma,
        "shift_sigma": (_reference_h0_map - FIDUCIALS["H0"]) / _reference_sigma,
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
# three points from one realization each is not enough to fit a shot-noise
# scaling law -- this is a visual comparison against the self-matched floor at
# N=32768, not a fitted trend.

# %%
_size_ns = np.asarray(list(SIZE_SWEEP_PROPOSAL_PATHS), dtype=np.float64)
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
    [32768],
    [_reference_row["rel_bias"]],
    yerr=[_reference_row["rel_sigma"]],
    fmt="D",
    capsize=3,
    color=str(TRUTH["color"]),
    label="self-matched (N=32768)",
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
# `md-imrphenom-s42-n32768.h5`; only `minimum_redshift` varies. No new SNR
# computation is needed -- `fisher_window` reuses the `snr` computed above, and only
# `minimum_redshift` changes per grid.

# %%
_zmin_colors = combo_colors(len(ZMIN_SWEEP_VALUES))

ZMIN_SWEEP_RESULTS: dict[float, tuple[jax.Array, jax.Array]] = {}
for minimum_redshift in ZMIN_SWEEP_VALUES:
    start = time.perf_counter()
    ZMIN_SWEEP_RESULTS[minimum_redshift] = evaluate_h0_posterior(
        injection_catalog,
        ZMIN_SWEEP_PROPOSAL_PATH,
        minimum_redshift=minimum_redshift,
        network=NETWORK,
        snr=snr,
    )
    print(f"z_min={minimum_redshift}: {time.perf_counter() - start:.2f}s")

fig_zmin_sweep, ax = plt.subplots()
for (minimum_redshift, (grid, logpost)), color in zip(
    ZMIN_SWEEP_RESULTS.items(), _zmin_colors, strict=True
):
    grid_np = np.asarray(grid)
    density = safe_exponentiate(logpost)
    density /= np.trapezoid(density, grid_np)
    ax.plot(grid_np, density, label=f"z_min={minimum_redshift}", color=color)

ax.plot(
    _reference_grid_np,
    _reference_density,
    label="self-matched (z_min=0.3)",
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
for minimum_redshift, (grid, logpost) in ZMIN_SWEEP_RESULTS.items():
    h0_map, _, sigma = posterior_summary(grid, logpost)
    _zmin_rows.append(
        {
            "label": f"z_min={minimum_redshift}",
            "h0_map": h0_map,
            "shift": h0_map - FIDUCIALS["H0"],
            "sigma": sigma,
            "shift_sigma": (h0_map - FIDUCIALS["H0"]) / sigma,
        }
    )
_zmin_rows.append(
    {
        "label": "self-matched (z_min=0.3)",
        "h0_map": _reference_h0_map,
        "shift": _reference_h0_map - FIDUCIALS["H0"],
        "sigma": _reference_sigma,
        "shift_sigma": (_reference_h0_map - FIDUCIALS["H0"]) / _reference_sigma,
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
        reference_logpost=np.asarray(REFERENCE_LOGPOST),
        **{
            f"size_grid_{n_samples}": np.asarray(grid)
            for n_samples, (grid, _) in SIZE_SWEEP_RESULTS.items()
        },
        **{
            f"size_logpost_{n_samples}": np.asarray(logpost)
            for n_samples, (_, logpost) in SIZE_SWEEP_RESULTS.items()
        },
        **{
            f"zmin_grid_{minimum_redshift}": np.asarray(grid)
            for minimum_redshift, (grid, _) in ZMIN_SWEEP_RESULTS.items()
        },
        **{
            f"zmin_logpost_{minimum_redshift}": np.asarray(logpost)
            for minimum_redshift, (_, logpost) in ZMIN_SWEEP_RESULTS.items()
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
                "size_sweep_n": list(SIZE_SWEEP_PROPOSAL_PATHS),
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
            r"size grows at fixed $z_{\rm min}=0.3$."
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
            "moves toward $z=0$ at fixed $N=32768$."
        ),
        label="tab:catalog_shot_noise_zmin_sweep",
    )
    print("saved grids to", GRID_DIR, "and figures to", FIGURE_DIR)
