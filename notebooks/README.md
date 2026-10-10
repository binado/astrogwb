# Notebooks

Workflows in this directory are stored as plain `.py` files. Most are [Jupytext](https://jupytext.readthedocs.io/) **py:percent** notebooks — a way to represent Jupyter notebooks as Python source instead of `.ipynb` JSON. That keeps diffs readable and lets normal Python tooling (Ruff, `ty`) work on notebook code. The `.py` is the source of truth; `*.ipynb` is gitignored.

[`fiducial_spectrum.py`](fiducial_spectrum.py) and [`spectrum_snrs.py`](spectrum_snrs.py) are [marimo](https://docs.marimo.io/) notebooks: plain `.py` files whose cells form a reactive graph.

## Two kinds, one directory

They used to live in two trees, one per workspace package. The packages merged;
the distinction did not, and it is worth knowing which kind you are opening.

**Self-contained** — demonstrations of the core `astrogwb` library. They name a
registered population model and build their own catalog through
`astrogwb.populations` and `astrogwb.catalog`, so they run against a clean
checkout with `astrogwb[io]` installed and nothing else. They read no file
outside themselves and import nothing from `tests/core`.

- **`catalog_convergence.py`** — how the catalog contraction approaches the
  analytic spectrum as the number of sources grows ($\propto N^{-1/2}$), and
  how the SNR and the log-likelihood *ratio* converge as the frequency
  resolution $\Delta f$ is refined. The companion to
  `tests/core/test_frequency_resolution.py`, which owns the
  tolerances; the notebook owns the picture.
- **`waveform_approximant_spectra.py`** — compare TaylorF2, IMRPhenomXAS,
  IMRPhenomHM, and IMRPhenomXAS_NRTidalV3 spectra on identical stochastic
  event draws, including percentile bands and fractional residuals to the
  tidal reference. Higher-mode spectra use matched isotropic inclination draws
  from the recorded population, without applying the quadrupole analytic average.
  It requires the `notebook` extra and the `jupyter` tooling group. It writes
  `outputs/figures/waveform_approximant_spectra.pdf`; there is no data cache,
  so every execution deterministically regenerates the draws from its fixed
  seed. Execute it from the repository root with:

  ```bash
  uv run --extra notebook --group jupyter jupytext --to notebook --execute \
      notebooks/waveform_approximant_spectra.py
  ```

  To convert without executing, omit `--execute`; the resulting sibling
  `.ipynb` is gitignored.

**Paper analyses** — these drive real waveform catalogs through the
`astrogwb.paper` configuration layer and are not self-contained by design. They
need built catalogs under `outputs/catalogs/`, the `notebook` extra, and the
repository root as the working directory.

- **`mcmc.py`** — importance-weighted NUTS inference (NumPyro port of ASGWB.jl)
- **`mcmc_plotting.py`** — load saved chains and produce diagnostic and corner
  plots
- **`logposterior_grid.py`** — evaluate the log-posterior on a parameter grid
- **`fiducial_spectrum.py`** — marimo notebook. One seeded forward-model draw
  of the fiducial $S_h$ / $\Omega_{\mathrm{GW}}$, network $S_{\mathrm{eff}}$,
  $\sigma$, and per-network SNR. No catalog file.
- **`spectrum_snrs.py`** — **deprecated**: it depends on the removed
  amplitude-marginalized model and fails at import; to be resolved in a
  follow-up. Marimo notebook for the appendix's shot-noise
  argument. A physics-first walkthrough of how catalog shot noise scales with
  the number of injections $N$ and the minimum redshift $z_{\min}$, judged
  against the expected $\sigma(H_0)$. It fits every template to a common
  reference spectrum, measures the MAP offset in units of a fixed
  $\sigma_{\mathrm{ref}}$ with bootstrap errors, and ends with a pass/fail
  verdict against a tolerance on $\mathrm{sd}(r)$. Uses the checked spectrum
  cache and writes three paper figures (A1-A3) and eleven supporting
  figures. Smoke test:
  `ASTROGWB_NOTEBOOK_SMOKE=1 uv run --extra notebook --group dev python notebooks/spectrum_snrs.py`.
- **`cosmology_grid_posteriors.py`** — marimo notebook. Grid posteriors for
  $H_0$, $(H_0, \Omega_m)$ and $(\Xi_0, n)$ across the six detector networks,
  from a zero-noise Poisson injection, modeled with the rescaled
  reference-redshift spectrum (`astrogwb.gwb.importance`: $2^{14}$ intrinsic
  draws on a scrambled Sobol net, one waveform each at the window's lower edge, rescaled to 32 redshift
  nodes; a different seed). The log densities are cached under
  `default_cache_dir() / "posteriors"`, keyed by a hash of their settings, so
  the figures iterate without recomputing. Each grid point is predicted once
  and shared by all six networks (`GaussianGWBBatchedLikelihood`); the numbers
  equal the per-network evaluation, so existing cache files stay valid. The
  injection is one catalog realization, so its physical shot noise is modeled
  as a rank-one covariance term from the same reference catalog
  (`build_rescaled_shot_noise`); each problem is evaluated with every
  likelihood variant (`detector`, `amplitude`, `fixed`, `per_frequency`), the
  figures show `likelihood_variant`, and a comparison section reports the
  widening, the fixed vs per-point $s^2$ verdict, and the per-frequency
  leakage check. Writes per-problem marginals and corner plots, and an $H_0$
  overlay of the variants. Smoke test:
  `ASTROGWB_NOTEBOOK_SMOKE=1 uv run --extra notebook --group dev python notebooks/cosmology_grid_posteriors.py`.
- **`importance_convergence.py`** — marimo notebook. Chooses the grid
  posteriors' $(N, Z)$: redshift-quadrature convergence on a $Z = 2^i$
  ladder over one fixed set of draws (for the spectrum and, separately, for
  its shot-noise amplitude scatter $s$), and Monte
  Carlo noise as the scatter of $M$ fixed-seed catalogs about their mean, with
  every $N = 2^k$ read off as a prefix; the same ladder for $M$ scrambled
  Sobol catalogs (`CatalogMetadata(sampling="sobol")`), compared at equal $N$
  (`sampling_comparison.pdf`). Errors are in units of the most
  sensitive network's per-bin noise and of each parameter's Fisher width.
  Smoke test:
  `ASTROGWB_NOTEBOOK_SMOKE=1 uv run --extra notebook --group dev python notebooks/importance_convergence.py`.
- **`mean_spectrum_snr_table.py`** — marimo notebook. Mean-spectrum SNR per
  network and low-frequency cut, as LaTeX tables, from a 200-draw Poisson
  ensemble (`IMRPhenomXAS`, seed 41), plus the shot noise of one draw about
  the mean ($\rho\,\mathrm{sd}(\epsilon)$ and the correlation across bins).
  The ensemble is cached one draw per file under
  `default_cache_dir() / "ensembles"` (`spectra_ensemble`) and shared with
  `shot_noise_coverage.py`. Smoke test:
  `ASTROGWB_NOTEBOOK_SMOKE=1 uv run --extra notebook --group dev python notebooks/mean_spectrum_snr_table.py`.
- **`shot_noise_coverage.py`** — marimo notebook. Calibration of the
  shot-noise likelihood for $H_0$: the Poisson ensemble of
  `mean_spectrum_snr_table.py` as injections (read from the shared cache),
  plus Gaussian detector noise per network, scored on an $H_0$ grid against an
  independent `IMRPhenomXAS` reference catalog with the `detector`,
  `amplitude` and `fixed` likelihoods. Compares the model's $\rho s$ with the
  injections' realized scatter, and reports 68%/95% coverage and the mean and
  variance of $z = (\bar H_0 - H_0^{\mathrm{true}})/\mathrm{sd}$ with
  bootstrap errors, and a PIT/$z$ figure at the fiducial network; writes
  `coverage_H0.pdf`, `coverage_H0.csv` and
  `shot_noise_model_vs_injections.csv`. Smoke test:
  `ASTROGWB_NOTEBOOK_SMOKE=1 uv run --extra notebook --group dev python notebooks/shot_noise_coverage.py`.

`fiducial_spectrum.py` and `spectrum_snrs.py` read the shared fiducials, detector registry, and default draw's waveform
and population through `astrogwb.paper.config`.
Their analysis window, draw seed, and plotting choices live in editable
configuration cells. Editing the shared analysis table does not update those
local controls; mirror the change there by hand.

`fiducial_spectrum.py` uses one seeded Poisson draw and the ordered network
legend from `astrogwb.paper.plotting.DETECTOR_NETWORKS`. `spectrum_snrs.py` uses
fixed-count ensembles for one selected network. Neither requires a waveform
catalog under `outputs/catalogs/`; their spectra are served or generated through
`simulate`.

Open either notebook from the repository root:

```bash
uv run --extra notebook --group jupyter marimo edit notebooks/fiducial_spectrum.py
# Or open the SNR distribution sweeps:
uv run --extra notebook --group jupyter marimo edit notebooks/spectrum_snrs.py
```

For the shared scientific values on their own, without standing in for a
particular run, read them from the package rather than retyping them:

```python
from astrogwb.paper.config import detector_registry, fiducials, networks, priors

fid = fiducials()
detectors = networks()["ET-2L-aligned-CE-Hanford"]
registry = detector_registry()
geometry, sensitivities = registry.build_network("ET-2L-aligned-CE-Hanford")
higher_h0 = fiducials(H0=70.0)  # keyword overrides, merged over the file
```

These are the `[fiducials]` and `[priors]` tables of `config/defaults.toml`, and
the `[networks]` and `[detectors]` tables of `config/detectors.toml` — the shared
layers the workflow merges into every run, so
a notebook cannot drift from what the runs sample. `priors()` returns live
NumPyro distributions. Each accessor also takes keyword overrides merged over
the file, so varying one value does not mean retyping the table; a network name
is hyphenated, so override one by unpacking a mapping
(`networks(**{"ET-2L-aligned": ("S1", "R1", "C1")})`). Each accessor returns
fresh settings from one cached merge of the shared layers. To see file edits
in a long-lived kernel, call `astrogwb.paper.config._shared.cache_clear()`.

## Running them

The self-contained notebook converts and executes through `just`:

```bash
just test-notebooks
```

That recipe installs the `simulation` and `io` extras and the `jupyter` group,
and nothing else — which is what keeps the notebook's self-containment honest.
The notebook names the population it draws from and the parameters it draws at,
then passes `sample_sources`'s output through
`PolarizationPowerCatalog.from_generator(..., generator=AnalyticInspiralGenerator(...))`.
The
luminosity distance is the population's own `numpyro.deterministic`, computed
in the same batched pass every later density evaluation takes, which is what
makes the catalog exactly its own importance proposal.

`ASTROGWB_NOTEBOOK_SMOKE=1` shrinks the catalog and the convergence sweeps. It
changes only how long the notebook runs, never which
cells execute — there is one code path in both modes:

```bash
ASTROGWB_NOTEBOOK_SMOKE=1 just test-notebooks
```

## Opening in Jupyter

The percent notebooks open in the classic notebook UI after a conversion to `.ipynb`. The two marimo notebooks open in marimo, as above.

To convert a percent notebook:

```bash
uvx jupytext --to ipynb notebooks/catalog_convergence.py
```

Jupytext writes a sibling `.ipynb` you can open in JupyterLab. Install the
environment first — the `notebook` extra is the library stack the notebooks
import, the `jupyter` group is the tooling that runs them:

```bash
uv sync --extra notebook --group jupyter
```

## The catalog cache

The self-contained notebook writes the catalog it builds to a
`notebooks/*.h5` file
(gitignored) and reuses it on the next run. The file is a cache, not an input:
delete it and the notebook rebuilds from the population model it names.
Persistence is not part of the core dependency set: `astrogwb` builds an
array-native `PolarizationPowerCatalog` in memory, and its `.save` / `.load`
reach HDF5 behind the `io` extra, so the notebook needs `astrogwb[io]` and
nothing else.

The reuse is guarded on more than the file existing, but the population half of
that guard is no longer the notebook's job: a catalog records its own model,
construction settings and hyperparameters, and
`PolarizationPowerCatalog.load` refuses a file whose columns no longer match
them. What the notebook still checks is the
waveform grid and the draw size, which the population record does not cover.
Without that, editing `CATALOG_DF` and re-running would silently analyse the
old frequency grid, which in `catalog_convergence.py` would invalidate the
entire result while looking perfectly healthy.
