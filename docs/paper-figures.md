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
  --allowed-rules validate run_mcmc plot_cosmological_parameters \
  --profile profiles/local --cores 8 --dry-run plot_cosmological_parameters
snakemake --snakefile Snakefile \
  --allowed-rules validate run_mcmc plot_cosmological_parameters \
  --profile profiles/slurm plot_cosmological_parameters
```

Every artifact remains a valid Snakemake target, but because the rule has
multiple outputs, requesting one builds the complete section:

```bash
snakemake --snakefile Snakefile \
  --allowed-rules validate run_mcmc plot_cosmological_parameters \
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

## Spectrum-realization SNR analysis

`notebooks/spectrum_snrs.py` is a marimo notebook with two local, fixed-count
spectrum sweeps for one detector network. Open it from the repository root:

```bash
uv run --extra notebook --group jupyter marimo edit notebooks/spectrum_snrs.py
```

Edit its configuration cell to choose the network, observation time, frequency
band, draw count, seed, batch size, cache directory, and sweep values. Scientific
fiducials, waveform settings, population settings, and the detector registry
come from the shared configuration accessors, as in `fiducial_spectrum.py`.
There are no CLI configuration flags or Snakemake rules for this analysis.

The source-count section compares `num_events = [16384, 32768, 65536]` at
`minimum_redshift = 0.35`. The minimum-redshift section compares
`[0.05, 0.15, 0.35]` at `num_events = 16384`. Each case has 1000 independent
realizations at fixed fiducial hyperparameters by default. Other defaults are
seed 41, batch size 1024, one observing year, the 2–2048 Hz analysis band, and
network `ET-2L-aligned-CE-Hanford`. The waveform and its grid are read from the
shared default draw. Editing plotting cells reuses the calculated case results.

Both sections display an SNR distribution overlay and an overlay of
amplitude-only Fisher widths rather than a full multiparameter Fisher
calculation. The count sweep evaluates widths at fitted MAPs, as described
below; the redshift sweep uses `sigma(H0) = H0_fid / SNR`. Case labels and colors
follow the configured list order. If any case is constant (including a single draw), all
curves in that overlay use ECDFs; otherwise they use KDEs.
KDEs extend beyond the observed extrema and do not reflect density at those
sample-dependent edges; tiny smoke ensembles verify execution and layout,
not the shape of the physical distribution. Compact summary
tables are displayed in the notebook, including sample SD, quantiles, relative
scatter, and the mean spectrum's SNR. `mean(SNR)` and `SNR(mean spectrum)` remain
separate because SNR is nonlinear.

The source-count section also overlays an amplitude-only Fisher approximation
to the inferred H0 distribution. Each catalog draw is a Monte Carlo template
fitted to one common reference spectrum. By default, `data_reference = "largest_mean"`
averages the full spectra from all 1000 draws at the largest configured count
(`max(num_events)`, 65536 by default), then uses that same spectrum for every N.
It averages spectra, not SNRs, and reuses that ensemble's checked artifact.
Set `data_reference = "poisson"` to use one independently seeded physical data
realization at the same fiducials, redshift bounds, waveform, and observing time.
The Poisson seed is 42; template seed is 41. The amplitude-marginalized model supplies
`amplitude_mle = (data|template)/(template|template)` and `template_optimal_snr`
using the full spectra and its Gaussian noise weights. With `S_h ∝ 1/H0` and
the shared uniform prior on H0, the MAP is `H0_fid / amplitude_mle`, clipped
to that prior's support. Boundary MAPs are flagged in the displayed summary.
The local Fisher width is evaluated at the MAP,
`H0_MAP**2 / (H0_fid * template_optimal_snr)`. The plotted density is the
equally weighted mixture of these per-draw Gaussians, evaluated directly without
additional posterior sampling or KDE. It includes both template-induced MAP
scatter and conditional Fisher uncertainty. If draws were observations fitted
with a fixed mean template, the amplitude fit would instead use that fixed
template and the varying draws as data. Fisher Gaussians at prior boundaries
do not describe the truncated posterior.

It also overlays the per-template normalized MAP residuals,
`(H0_MAP - H0_fid) / sigma(H0)`, against a standard normal reference and marks
the interval `[-1, 1]`. A displayed table reports mean, sample SD, RMS, and
the fraction of offsets exceeding one Fisher sigma. These draws contain no
detector noise, so a unit-width Gaussian is a comparison scale, not a required
sampling distribution. A width greater than one indicates template-induced MAP
scatter larger than the Fisher uncertainty. Every N uses the same reference;
the ensembles are not separately recentered. The default reference shares
draws with the largest template ensemble and measures convergence relative to
that ensemble. It cannot detect systematic error shared by all draws. Averaging
1000 draws makes random reference error much smaller than individual-template
scatter. With the Poisson option, a mean residual can also reflect fluctuations
in the physical data realization; separating those effects requires repeated
independent data realizations.

Fixed-count results measure **finite-catalog estimator scatter**. Increasing
`num_events` tests convergence about the same rate-normalized spectrum;
changing the minimum redshift changes the population and can change both the
spectrum and its scatter. The draws include source fluctuations and exclude
detector-noise realizations. The shared population's inclination convention
is retained. Nearby sources can strongly affect tails, which require enough
realizations to characterize reliably.

The notebook uses `simulate` and the existing content-addressed spectrum cache
at `default_cache_dir() / "spectra"`, shared across worktrees. Changing count,
cutoff, seed, or draw count changes the key; detector settings and the analysis
band do not. Identical baseline settings in the two sections share one
artifact. The default largest-ensemble mean requires no additional artifact;
the optional Poisson data spectrum is a sixth cached artifact.
Set `cache_only = True` to require existing spectra; a missing or
mismatched artifact fails explicitly.

With `write_figures = True`, the only analysis files written are six overlays
under `outputs/figures/spectrum_snrs/`:

- `num_events_snr_distribution.pdf`
- `num_events_sigma_H0_distribution.pdf`
- `num_events_H0_fisher_distribution.pdf`
- `num_events_H0_normalized_residual_distribution.pdf`
- `minimum_redshift_snr_distribution.pdf`
- `minimum_redshift_sigma_H0_distribution.pdf`

Set `write_figures = False` for display only. Figure styling and export settings
use the shared paper style, including its LaTeX requirement. No CSV or
provenance JSON is exported; each case retains its metadata in notebook memory
and the checked spectrum artifact carries the generating record.

`just test-spectrum-snrs-notebook` checks the marimo graph and runs both sweeps
with `ASTROGWB_NOTEBOOK_SMOKE=1`: three realizations, source counts `[8, 16, 32]`,
a redshift-sweep count of 8, and figure saving disabled. Both reference options
are tested; the Poisson option substitutes a separate 64-source fixed-count
data draw for the physical realization. The execution
tests use a temporary spectrum cache and check that repeat execution needs no generation.

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
