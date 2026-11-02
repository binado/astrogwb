# Notebooks

Workflows in this directory are plain `.py` files edited in [marimo](https://docs.marimo.io/): each one is a notebook whose cells form a reactive graph. That keeps diffs readable and lets normal Python tooling (Ruff, `ty`) work on notebook code. The `.py` is the source of truth; `*.ipynb` is gitignored.

Every notebook here drives the `astrogwb.paper` configuration layer, so they all need the `notebook` extra for their imports and the `notebook` dependency group for the marimo editor. Run them from the repository root.

## Inputs

The notebooks read simulation data and never generate it. Each configuration
cell names its inputs, all under `outputs/`:

| Input | Read by |
| --- | --- |
| `ensembles/fiducial/` | `cosmology_grid_posteriors`, `madau_dickinson_grid`, `shot_noise_coverage`, `ensemble_spectrum` |
| `catalogs/reference-fiducial.h5` | `cosmology_grid_posteriors`, `madau_dickinson_grid`, `shot_noise_coverage`, `ensemble_spectrum` |
| `catalogs/reference-ladder.h5` | `importance_convergence` |

`Snakefile` writes them (`docs/coverage-draws-on-slurm.md`).
A missing input is a `FileNotFoundError`, never a simulation.

## Notebooks

- **`fiducial_spectrum.py`** — marimo notebook. Interactive explorer of one
  seeded forward-model draw: characteristic amplitudes $h_c$ / $h_n$ (or
  $S_h$ / $\Omega_{\mathrm{GW}}$), network $S_{\mathrm{eff}}$, $\sigma$, and
  cumulative per-network SNR, with sliders for approximant, resolution and
  observation time. No catalog file, and no paper SNR table: a single draw is
  not the quoted number (see `ensemble_spectrum.py`).
- **`cosmology_grid_posteriors.py`** — marimo notebook. Grid posteriors for
  $H_0$, $(H_0, \Omega_m)$ and $(\Xi_0, n)$ across the six detector networks,
  from a zero-noise Poisson injection as each network records it
  (`network_spectra_ensemble` over the fiducial ensemble: one catalog, six
  spectra through the per-source antenna patterns), modeled with the rescaled
  reference-redshift spectrum (`astrogwb.importance.spectrum`: $2^{14}$ intrinsic
  draws on a scrambled Sobol net, one waveform each at reference redshift $10^{-4}$, rescaled to 32 redshift
  nodes; catalog seed 42, the injection being seed 41). The log densities are computed on every run. Each
  grid point is predicted once and shared by all six networks
  (`grid_log_posterior`); the numbers equal the per-network evaluation. The
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
  each draw's SNR with its 16-84% range (LaTeX table),
  the characteristic amplitudes $h_c$ (median) against $h_n$ and
  the $d\mathrm{SNR}^2/d\ln f$ curves, as two figures (`characteristic_amplitudes.pdf`,
  `snr_density.pdf`),
  and the shot noise of one draw about the mean ($\rho\,\mathrm{sd}(\epsilon)$
  and the correlation across bins). It also scans the window's lower edge
  $z_{\min}$: the amplitude uncertainty
  $\sigma_{\mathrm{tot}}/A = (\rho^{-2} + s^2)^{1/2}$ per network
  (`minimum_redshift_scan.pdf`), from one reference catalog generated at the
  reference redshift $10^{-4}$ and rescaled to each window, checked against the
  draws' $\rho\,\mathrm{sd}(\epsilon)$. The ensemble is the fiducial ensemble,
  shared with `shot_noise_coverage.py`.
- **`shot_noise_coverage.py`** — marimo notebook. Calibration of the
  shot-noise likelihood for $H_0$: the catalogs of the Poisson ensemble of
  `ensemble_spectrum.py`, each recorded by every network
  (`network_spectra_ensemble`), plus
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
just run-notebook notebooks/ensemble_spectrum.py
```

The shared fiducials, detector registry, and default draw's waveform and
population come from `astrogwb.config`. Each notebook's analysis window,
draw seed, and plotting choices live in its own editable configuration cell.
Editing the shared analysis table does not update those local controls; mirror
the change there by hand.

For the shared scientific values on their own, without standing in for a
particular run, read them from the package rather than retyping them:

```python
from astrogwb.config import load_config

config = load_config()
fid = config.fiducials
registry = config.registry
detectors = registry.build_network("ET-2L-aligned-CE-Hanford")
priors = {name: spec.build() for name, spec in config.priors.items()}
higher_h0 = {**fid, "H0": 70.0}
```

`load_config()` merges the top-level TOML files in `config/` with packaged
detector defaults. Pass a different config directory or ordered override files
when needed. Each call returns independent settings and reads the current files.
Prior specifications are built into NumPyro distributions where needed.

## Checking notebooks

`marimo check --strict notebooks/<name>.py` lints a notebook's cell graph without running it.
