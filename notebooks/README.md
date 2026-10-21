# Notebooks

Workflows in this directory are plain `.py` files edited in [marimo](https://docs.marimo.io/): each one is a notebook whose cells form a reactive graph. That keeps diffs readable and lets normal Python tooling (Ruff, `ty`) work on notebook code. The `.py` is the source of truth; `*.ipynb` is gitignored.

Every notebook here drives the `astrogwb.paper` configuration layer, so they all need the `notebook` extra for their imports and the `notebook` dependency group for the marimo editor. Run them from the repository root.

## Notebooks

- **`fiducial_spectrum.py`** — marimo notebook. Interactive explorer of one
  seeded forward-model draw: characteristic amplitudes $h_c$ / $h_n$ (or
  $S_h$ / $\Omega_{\mathrm{GW}}$), network $S_{\mathrm{eff}}$, $\sigma$, and
  cumulative per-network SNR, with sliders for approximant, resolution and
  observation time. No catalog file, and no paper SNR table: a single draw is
  not the quoted number (see `ensemble_spectrum.py`).
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
  figures.
- **`cosmology_grid_posteriors.py`** — marimo notebook. Grid posteriors for
  $H_0$, $(H_0, \Omega_m)$ and $(\Xi_0, n)$ across the six detector networks,
  from a zero-noise Poisson injection as each network records it
  (`network_spectra_ensemble`: one catalog, six spectra through the per-source
  antenna patterns), modeled with the rescaled
  reference-redshift spectrum (`astrogwb.gwb.importance`: $2^{14}$ intrinsic
  draws on a scrambled Sobol net, one waveform each at the window's lower edge, rescaled to 32 redshift
  nodes; a different seed). The log densities are cached under
  `default_cache_dir() / "posteriors"`, keyed by a hash of their settings, so
  the figures iterate without recomputing. Each grid point is predicted once
  and shared by all six networks (`grid_log_posterior`); the numbers
  equal the per-network evaluation, so existing cache files stay valid. The
  injection is one catalog realization, so its physical shot noise is modeled
  as a rank-one covariance term from the same reference catalog, with each
  network's response (`Network.response`, its moments held per network)
  (`ImportanceGaussianLikelihood`'s `shot_noise` mode); each problem is evaluated with every
  likelihood variant (`detector`, `amplitude`, `fixed`, `per_frequency`), the
  figures show `likelihood_variant`, and a comparison section reports the
  widening, the fixed vs per-point $s^2$ verdict, and the per-frequency
  leakage check. Writes per-problem marginals and corner plots, and an $H_0$
  overlay of the variants.
- **`madau_dickinson_grid.py`** — marimo notebook. Posteriors on the
  Madau-Dickinson shape $(z_{\mathrm{peak}}, \gamma, \kappa)$ for the six
  detector networks (the paper's astrophysical-parameter section), from the
  same network-recorded Poisson injection and rescaled reference catalog as
  `cosmology_grid_posteriors.py`, with the response-aware `modes` likelihood,
  on a **ridge-aligned grid**. NUTS is not
  used: the posterior is too thin and curved for it to sample reliably. The data pin the
  spectrum amplitude to about $10^{-4}$ to $10^{-3}$ of the prior range, so the
  posterior is a thin curved sheet that a uniform grid steps over. The grid runs
  over $(\gamma, \kappa)$ and solves $z_{\mathrm{peak}}$ at every point: a
  coarse scan and Newton iterations locate the spike, and the trapezoid rule
  integrates a window around it. One knob, `likelihood_variant` (`detector`,
  `amplitude`, `fixed`). A 200x120 grid takes a few seconds per network. Reports, per
  network, the posterior mass in columns where Newton did not converge or the
  ridge sits on the support edge, and the quantile shift under a 2x refinement;
  writes marginals, a corner plot, `summary.csv` / `summary.tex` and
  `checks.csv`.
- **`importance_convergence.py`** — marimo notebook. Chooses the grid
  posteriors' $(N, Z)$: redshift-quadrature convergence on a $Z = 2^i$
  ladder over one fixed set of draws (for the spectrum and, separately, for
  its shot-noise amplitude scatter $s$), and Monte
  Carlo noise as the scatter of $M$ fixed-seed catalogs about their mean, with
  every $N = 2^k$ read off as a prefix; the same ladder for $M$ scrambled
  Sobol catalogs (`CatalogMetadata(sampling="sobol")`), compared at equal $N$
  (`sampling_comparison.pdf`). Errors are in units of the most
  sensitive network's per-bin noise and of each parameter's Fisher width.
- **`ensemble_spectrum.py`** — marimo notebook. The paper's spectrum
  figure and SNR tables, from a 1000-draw Poisson ensemble (`IMRPhenomXAS_NRTidalv3`,
  seed 41): per-network SNR and low-frequency cut as the mean over draws of
  each draw's SNR with its 16-84% range (LaTeX tables, ET-COBA and ET-D PSDs),
  the characteristic amplitudes $h_c$ (median) against $h_n$ and
  the $d\mathrm{SNR}^2/d\ln f$ curves, as two figures (`characteristic_amplitudes.pdf`,
  `snr_density.pdf`),
  and the shot noise of one draw about the mean ($\rho\,\mathrm{sd}(\epsilon)$
  and the correlation across bins). It also scans the window's lower edge
  $z_{\min}$: the amplitude uncertainty
  $\sigma_{\mathrm{tot}}/A = (\rho^{-2} + s^2)^{1/2}$ per network
  (`minimum_redshift_scan.pdf`), from one reference catalog generated at the
  redshift where $d_L = 1$ Mpc and rescaled to each window, checked against the
  draws' $\rho\,\mathrm{sd}(\epsilon)$. The ensemble is cached one draw per file
  under `default_cache_dir() / "ensembles"` (`spectra_ensemble`) and shared
  with `shot_noise_coverage.py`.
- **`shot_noise_coverage.py`** — marimo notebook. Calibration of the
  shot-noise likelihood for $H_0$: the catalogs of the Poisson ensemble of
  `ensemble_spectrum.py`, each recorded by every network
  (`network_spectra_ensemble`, cached one file per network and draw), plus
  Gaussian detector noise per network, scored on an $H_0$ grid against an
  independent `IMRPhenomXAS_NRTidalv3` reference catalog with the `detector`,
  `amplitude`, `fixed` and `amplitude_sky` (no response) likelihoods. Compares the model's $\rho s$ with the
  injections' realized scatter, and reports 68%/95% coverage and the mean and
  variance of $z = (\bar H_0 - H_0^{\mathrm{true}})/\mathrm{sd}$ with
  bootstrap errors, and a PIT/$z$ figure at the fiducial network; writes
  `coverage_H0.pdf`, `coverage_H0.csv` and
  `shot_noise_model_vs_injections.csv`.

Open a notebook from the repository root:

```bash
uv sync --extra notebook --group notebook
uv run --extra notebook --group notebook marimo edit notebooks/fiducial_spectrum.py
# Or through the just recipe, which runs the same editor:
just run-notebook notebooks/spectrum_snrs.py
```

The shared fiducials, detector registry, and default draw's waveform and
population come from `astrogwb.paper.config`. Each notebook's analysis window,
draw seed, and plotting choices live in its own editable configuration cell.
Editing the shared analysis table does not update those local controls; mirror
the change there by hand.

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
layers the workflow merges into every run, so a notebook cannot drift from what
the runs sample. `priors()` returns live NumPyro distributions. Each accessor
also takes keyword overrides merged over the file, so varying one value does
not mean retyping the table; a network name is hyphenated, so override one by
unpacking a mapping (`networks(**{"ET-2L-aligned": ("S1", "R1", "C1")})`). Each
accessor returns fresh settings from one cached merge of the shared layers. To
see file edits in a long-lived kernel, call
`astrogwb.paper.config._shared.cache_clear()`.

## Smoke runs

`ASTROGWB_NOTEBOOK_SMOKE=1` shrinks the catalogs and the convergence sweeps. It
changes only how long a notebook runs, never which cells execute — there is one
code path in both modes. Run any notebook as a script to check it end to end:

```bash
ASTROGWB_NOTEBOOK_SMOKE=1 uv run --extra notebook --group dev python notebooks/fiducial_spectrum.py
```

`marimo check --strict notebooks/<name>.py` lints a notebook's cell graph without running it.
