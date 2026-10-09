# Paper figures

Experiment figures are part of the same DAG as their chains. The split for
presentation is that *order and structure* are code, *values* are data.

Order and structure stay hard-coded in the script that draws each figure: the
run IDs it compares, and the sequence they appear in. Changing that is a code
change, reviewed alongside the plot it affects. The six detector networks
compared by more than one figure are the one shared piece, and they live in
`astrogwb.paper.plotting.DETECTOR_NETWORKS` as ordered
`(run name, LaTeX label)` pairs -- a network's label stays there because
nothing reads it without also needing the order it sits in.

Values live in [`config/plotting.toml`](../config/plotting.toml): the LaTeX
label for each *parameter*, plus `figure_dpi` and `figure_format`. Scripts
reach them through `astrogwb.paper.plotting.parameter_label`,
`figure_dpi` and `figure_format`. Parameter labels used to be declared
separately in three scripts; `figure_dpi` was a literal `300` in four of them
*and* in `paper.mplstyle`. `use_paper_style()` applies the dpi and format as
rcParams, so `paper.mplstyle` no longer declares either and no figure script
takes a `--figure-dpi` flag.

Input and output paths are both named literally in
[`Snakefile`](../Snakefile), and every output is a valid Snakemake target.
Shared scientific values -- fiducials, priors, detector networks, frequency
bounds, the target population and its redshift grid -- arrive on argv as
repeated `--config` layer files, the same list the rule declares as `input:`, so a figure reports exactly
what was sampled and a layer edit retriggers the figure. A script that needs
one of the top-level tables outside a run context can also call
`astrogwb.paper.config.fiducials()` / `priors()` / `networks()` directly; both
paths read the same files.

Detector *lists* are never hard-coded next to a label. The rule passes
`--network-run <experiment>/<run>` once per network, in legend order, and
`astrogwb.paper.config.runs.resolve_networks` merges each run's own config
layers, reads the `analysis.network` that run names, and resolves it through
the `[networks]` table those same layers carry. The detectors a figure reports
an SNR for are therefore always the ones its chain was sampled with.

That indirection is deliberate. Every network run happens to be named after the
network it uses, so looking the legend name up in the shared `[networks]` table
directly would give the same answer today -- but that is a property of the
current tree, not a derivation. Going through the run means a run that changed
its `network` moves the figure with it, instead of the figure quietly
reporting one network's SNRs beside another network's chain.

`resolve_networks` matches the `--network-run` list against the legend
*positionally* and rejects a mismatch. That check matters more than it looks:
declaration order drives chain order, legend order, and colour assignment, so a
swapped pair would render a perfectly good figure with the wrong labels on the
wrong curves rather than failing.

The workflow imports that same tuple and expands its chain paths from it, so
chain order and legend order are one list rather than two that have to be kept
in step:

```python
from astrogwb.paper.plotting import DETECTOR_NETWORK_RUNS

chains=expand("outputs/chains/cosmological-parameters/{run}.nc",
              run=DETECTOR_NETWORK_RUNS),
```

## Experiment figures

The detector-network, merger-rate, and Omega-m analyses form one paper section.
The `plot_cosmological_parameters` rule consumes all eight chains and produces their
five figures and two CSV/LaTeX table pairs in one script invocation. All
artifacts live under `outputs/figures/cosmological-parameters/`.

Preview or build the section (from the repository root):

```bash
snakemake --snakefile Snakefile \
  --allowed-rules run_mcmc plot_cosmological_parameters \
  --profile profiles/local --cores 8 --dry-run plot_cosmological_parameters
snakemake --snakefile Snakefile \
  --allowed-rules run_mcmc plot_cosmological_parameters \
  --profile profiles/slurm plot_cosmological_parameters
```

Every artifact remains a valid Snakemake target, but because the rule has
multiple outputs, requesting one builds the complete section:

```bash
snakemake --snakefile Snakefile \
  --allowed-rules run_mcmc plot_cosmological_parameters \
  --profile profiles/local --cores 8 \
  outputs/figures/cosmological-parameters/H0-by-detector.pdf
```

With a SLURM profile, sampling runs remotely and figure rules run locally on the
submit host after their chains finish. The submit host must remain attached,
share the output filesystem, and provide plotting dependencies.

Use `run_experiment_cosmological_parameters` to sample the constituent
experiment without running post-processing.

The modified-propagation section works the same way:
`plot_modified_propagation` builds its propagation figures and tables from the
chains of `run_experiment_modified_propagation`.

## Standalone figures

The importance-weight grids are the standalone figure rules in the unified
workflow. `importance_weights_grid` reads its proposal *density* from the
catalog file it is handed, the same way `scripts/run_mcmc.py` does -- so the
weights it plots divide by the same denominator the chains did.

The fiducial injection spectrum, network $S_{\mathrm{eff}}$, $\sigma$ overlay,
and per-network SNR figures live in
[`notebooks/fiducial_spectrum.py`](../notebooks/fiducial_spectrum.py) rather
than the DAG. That notebook resolves no run config: it reads the shared
fiducials and detector networks from the shared `[fiducials]` and `[networks]`
tables through `astrogwb.paper.config`, and the ordered network
legend from `astrogwb.paper.plotting.DETECTOR_NETWORKS`, but keeps its analysis
window (`observation_time`, the frequency band, and the redshift bounds) and
local plotting choices as literals of its own. Those mirror
the shared `[analysis]` table, so editing that table does not change these figures --
update the notebook's configuration cell too.

```bash
snakemake --snakefile Snakefile --cores 1 \
  --allowed-rules importance_weights_grid \
  outputs/figures/standalone/importance_weights_grid_H0_Omega_m.pdf \
  outputs/figures/standalone/importance_weights_grid_Xi0_n.pdf
```

Or build any one of them directly:

```bash
snakemake --snakefile Snakefile --cores 1 \
  --allowed-rules importance_weights_grid \
  outputs/figures/standalone/importance_weights_grid_H0_Omega_m.pdf
```

Standalone workflow figure products are written under `outputs/figures/`.

## Spectrum-realization shot-noise analysis

`notebooks/spectrum_snrs.py` is a marimo notebook that justifies two choices of
the appendix, the number of injections `N` (128k-256k recommended) and the
minimum redshift `z_min`, by one criterion: **catalog shot noise stays below
the expected `sigma(H0)`**. Open it from the repository root:

```bash
uv run --extra notebook --group jupyter marimo edit notebooks/spectrum_snrs.py
```

The notebook is a physics-first walkthrough: a markdown cell explains each
step, and the code lives in named top-level functions collected in a toolbox
section. Edit its configuration cell to choose the network, observation time,
frequency band, draw count, seed, batch size, cache directory, sweep values,
`tolerance` and `n_bootstrap`. Scientific fiducials, waveform settings,
population settings, and the detector registry come from the shared
configuration accessors, as in `fiducial_spectrum.py`. There are no CLI
configuration flags or Snakemake rules for this analysis.

### Ensembles

Every ensemble is a fixed-count set of independent draws (100 by default) at
the fiducial hyperparameters, served from the checked spectrum cache or
generated on a miss. Other defaults are seed 41, batch size 1024, one observing
year, the 2-2048 Hz band, and network `ET-2L-aligned-CE-Hanford`.

- The **count sweep** uses `N = 2^12 ... 2^18` at `z_min = 0.35`.
- The **grid** uses `N` in `{2^14, 2^16, 2^17, 2^18}` for `z_min` in
  `{0.05, 0.15, 0.35}`. It reuses the count sweep's ensembles at the baseline
  cutoff.
- The **Poisson** ensemble draws its count from the population's rate.

Cache keys depend on count, cutoff, seed, and draw count, not on the detector
or analysis band. `cache_only = True` requires existing spectra and fails
explicitly on a missing one.

### Offset statistic

Each Monte Carlo template is fitted to a common reference spectrum with the
amplitude-marginalized model's sufficient statistics (`amplitude_mle`,
`template_optimal_snr`). With `S_h ∝ 1/H0` and the shared uniform prior on H0,
the MAP is `H0_fid / amplitude_mle`, clipped to the prior's support; boundary
MAPs are flagged in the displayed tables.

The residual is `r = (H0_MAP - H0_fid) / sigma_ref` with the **fixed** width
`sigma_ref = H0_fid / rho_ref`, where `rho_ref` is the SNR of the reference
spectrum. Dividing by each draw's own width instead would mix numerator and
denominator and skew the tails, so that variant is kept only as a cross-check
column (`sd_per_draw`) and in the Fisher-mixture figure. Because shot noise and
detector noise add in quadrature, `sd(r)` is the fractional inflation of
`sigma(H0)` up to `sqrt(1 + sd^2) - 1`; the tolerance `sd(r) <= 0.1` is about
0.5%. Errors on `mean`, `sd`, `rms`, `q95(|r|)` and `P(|r| > 1)` are seeded
bootstrap errors from resampling draws.

The reference is `data_reference = "largest_mean"` by default, the mean spectrum
over all draws at the largest `N` (a mean of spectra, not of SNRs), or
`"poisson"`, one independently seeded Poisson draw (seed 42). The default
shares draws with the largest ensemble, so it measures convergence relative to
that ensemble and cannot reveal an error common to every draw. In the grid each
`z_min` uses its own reference and `sigma_ref`, and its `N_max` point shares
draws with that reference (about 1% of one template's scatter).

### Figures

With `write_figures = True`, the notebook writes fourteen figures under
`outputs/figures/spectrum_snrs/`. The three **paper figures** are:

- `paper_shot_noise_vs_detector.pdf` (A1): shot-noise standard deviation of the
  spectrum for the paper's source counts against the per-bin detector
  uncertainty `S_eff / sqrt(2 T Delta_f)`, with bin widths from the full grid
  before band selection.
- `paper_offset_scaling.pdf` (A2): left, kernel densities of the fixed-width
  residuals with the tolerance band; right, `sd(r)` against `N` with bootstrap
  errors, an `N^(-1/2)` guide through the smallest `N`, the tolerance line and
  the Poisson ensemble as a separate marker.
- `paper_offset_vs_min_redshift.pdf` (A3): `sd(r)` (solid) and `q95(|r|)`
  (dashed) against `N`, one curve per `z_min`, with the guide and tolerance.

The eleven **supporting figures** are `spectrum_mean.pdf`,
`spectrum_relative_variance.pdf`, `num_events_spectrum_sensitivity.pdf`,
`minimum_redshift_spectrum_sensitivity.pdf`,
`num_events_frequency_correlation.pdf`,
`minimum_redshift_frequency_correlation.pdf`, `num_events_snr_distribution.pdf`,
`num_events_sigma_H0_distribution.pdf`, `num_events_H0_fisher_distribution.pdf`,
`minimum_redshift_snr_distribution.pdf` and
`minimum_redshift_sigma_H0_distribution.pdf`. The minimum-redshift supporting
figures use the recommended `N = 2^17`. Overlays use ECDFs if any case is
constant and KDEs otherwise; KDEs extend beyond the observed extrema.

The notebook ends with a verdict table (`sd(r)` and `q95(|r|)` at the two
largest grid counts for each `z_min`, with pass/fail against the tolerance) and
a conclusion filled in from it. Set `write_figures = False` for display only.
Figure styling uses the shared paper style, including its LaTeX requirement.
No CSV or provenance JSON is exported.

Draws contain source fluctuations only, with the population's inclination
convention retained, and no detector-noise realizations. Execute the smoke
test, which checks that the whole notebook runs, with:

```bash
uv run --extra notebook --group dev marimo check --strict notebooks/spectrum_snrs.py
ASTROGWB_NOTEBOOK_SMOKE=1 uv run --extra notebook --group dev python notebooks/spectrum_snrs.py
```

Smoke mode uses three draws, counts `[8, 16, 32]`, a 64-source fixed-count
stand-in for the Poisson option, and no figure saving. A full run
(`N = 2^18`, 100 draws, three cutoffs) is expensive the first time, until the
spectrum cache is populated.

## Grid posteriors

`notebooks/cosmology_grid_posteriors.py` is a marimo notebook that evaluates
the posteriors of `H0`, `(H0, Omega_m)` and `(xi_0, xi_n)` on a grid for all six
networks, with no sampler. The data are a zero-noise Poisson injection
(`data_seed = 41`); the model spectrum is the rescaled reference-redshift
spectrum of `astrogwb.gwb.importance` (`catalog_seed = 42`): `2^19` intrinsic
draws, each generated once at the window's lower edge by `reference_catalog`,
and `build_rescaled_spectrum` rescaling their mean power to 32 Gauss-Legendre
nodes in the scale factor `1/(1 + z)` at each grid point, with no intrinsic
reweighting. The log densities are cached in `default_cache_dir() /
"posteriors"`, in a file
whose name carries a hash of the seeds, grids, band, observing time, version
and catalog keys, so a stale cache is never served. The first run generates the
reference catalog (`outputs/catalogs/reference_catalog-<key>-42.h5`, chunked
by `reference_catalog(..., chunk_size=...)`).
Figures go to `outputs/figures/cosmology_grid_posteriors/` when the
`Write figures` switch is on:

```bash
uv run --extra notebook --group dev marimo check --strict notebooks/cosmology_grid_posteriors.py
ASTROGWB_NOTEBOOK_SMOKE=1 uv run --extra notebook --group dev python notebooks/cosmology_grid_posteriors.py
```

## Importance convergence

`notebooks/importance_convergence.py` measures the two errors of the
rescaled reference-redshift spectrum separately, to choose the grid posteriors'
`(N, Z)`. The rescaling itself holds for the (2,2)-mode aligned-spin waveforms
used; `tests/core/test_gwb_importance.py` checks it against waveforms generated
at every node. The redshift quadrature runs a `Z = 2^i` ladder (4 to 1024 nodes) on one
reference catalog -- the node count is a builder argument, so the ladder costs
no waveforms. The Monte Carlo noise is the scatter of `M`
catalogs at fixed seeds about their mean; each `N = 2^k` is the first `N`
draws of every catalog, so the `N` ladder costs no extra waveforms. Errors are
reported in units of the most sensitive network's per-bin `sigma` and as each
parameter's likelihood-peak shift in Fisher widths, at the fiducials and the
grid windows' edges. A recommendation cell reports the cheapest pair below
tolerance and the `N` the `1/N` variance law extrapolates to. Catalogs are
cached in `outputs/catalogs/`; figures go to
`outputs/figures/importance_convergence/` when `Write figures` is on:

```bash
uv run --extra notebook --group dev marimo check --strict notebooks/importance_convergence.py
ASTROGWB_NOTEBOOK_SMOKE=1 uv run --extra notebook --group dev python notebooks/importance_convergence.py
```

## Scripts

Figure entry points are plain Python scripts under `scripts/`. Each reads its
own fiducials and analysis settings from an assembled run config, whose path the
library owns, and hard-codes its own labels and run order. Snakemake passes only
what it owns: the chain and catalog paths it built, and the output paths it
declared.

The assembled config is a declared input of each rule, so editing a fiducial or
a detector list rebuilds the figure; editing a label is a code change and
rebuilds it the same way. Config parsing stays free of JAX, so `--help` and
config errors stay cheap.
