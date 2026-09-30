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

`scripts/analyze_spectrum_snrs.py` analyzes one spectrum ensemble and one named
detector network per invocation. Run it from the repository root with the
`notebook` extra for the plotting dependencies. The shared paper style requires
the same LaTeX installation as the other paper figures.

For fixed-count spectra at shared fiducials, generate on a cache miss and then
write SNR tables and a density plot:

```bash
uv run --extra notebook python scripts/analyze_spectrum_snrs.py \
    --spectra-config config/defaults.toml \
    --spectra-config config/waveforms.toml \
    --spectra-config config/populations.toml \
    --spectra-config config/simulations/spectrum/fixed.toml \
    --detector-config config/detectors.toml \
    --network ET-2L-aligned-CE-Hanford
```

The script validates an in-script `SNRConfig` containing `SpectraMetadata`, the
resolved `DetectorRegistry`, the selected network and the frequency band.
Repeat `--spectra-config` for the shared scientific layers and one case's
simulation layers; repeat `--detector-config` for detector registry layers.
The two groups merge independently and references resolve within their own
group. `load_detector_config` supplies the packaged geometry and sensitivity
definitions, then merges the supplied `[detectors]` and `[networks]` tables
in order and validates a `DetectorRegistry`. Other resolved top-level tables
in detector layers are rejected. `--network` defaults to
`ET-2L-aligned-CE-Hanford`.

The analysis band comes from `[analysis].minimum_frequency` and
`maximum_frequency` in the spectrum layers; the observing time comes from the
spectrum metadata and is converted from years to seconds. Each SNR uses the
full frequency axis and its derived bin widths before applying the band mask.

The same checked cache mechanism as the generator serves
`<cache-dir>/<spectrum-key>.h5`, or generates and atomically saves a miss.
Both spectrum scripts default to the user cache directory's `astrogwb/spectra`
subdirectory, shared across worktrees. `platformdirs` honors `XDG_CACHE_HOME`
on Linux and macOS; without an override, the default is
`~/.cache/astrogwb/spectra` on Linux and `~/Library/Caches/astrogwb/spectra`
on macOS. `--cache-dir` changes the SNR script's cache directory;
`--output-dir` selects the generation script's directory. These paths are
local preferences and do not enter the spectrum metadata or cache key.
`--batch-size` defaults to 128
and only controls waveform memory on a miss. Generation can also be done first
with `scripts/simulate_spectra.py` using those spectrum layers as repeated
`--config` flags. `--cache-only` requires a hit and raises an error naming the
missing key/path. A metadata mismatch always fails, without regeneration.

Compare another network using the same saved spectra:

```bash
uv run --extra notebook python scripts/analyze_spectrum_snrs.py \
    --spectra-config config/defaults.toml \
    --spectra-config config/waveforms.toml \
    --spectra-config config/populations.toml \
    --spectra-config config/simulations/spectrum/fixed.toml \
    --detector-config config/detectors.toml \
    --network ET-triangular --cache-only
```

Network selection, geometry/PSD overrides, and the analysis band stay outside
the spectrum generation key. A spectrum-generation setting, including seed,
source count, draw count or generating population, changes that key. Specify
different cases in separate invocations; repeated `--spectra-config` flags
merge one case and do not define a sweep.

Outputs default to `outputs/snr/<spectrum-key>/<network>/`; `--output-dir`
overrides this. Repeating an analysis replaces its outputs, so use distinct
directories to retain band or detector-override comparisons:

- `snr_draws.csv`: spectrum key, network, distribution label, zero-based draw
  index, event count and per-realization SNR.
- `snr_summary.csv`: detector membership, draw count, mean, median, unbiased
  sample SD (`ddof=1`), 5th/95th percentiles, relative scatter (`SD/mean`) and
  the mean spectrum's SNR. Single-draw SD and zero-mean relative scatter are
  undefined and written as empty CSV cells.
- `snr_distribution.pdf`: one SNR density plot via `arviz_plots.plot_dist`
  with a KDE, using the configured paper figure format and resolution (`pdf`
  by default). A single draw or constant ensemble uses an ECDF instead.
- `<param>_fisher_scatter.pdf`: one extra figure per parameter passed to
  `--plot-param-fisher-scatter` (repeatable; accepts several names), the
  density of that fixed hyperparameter's fiducial value divided by the
  per-draw SNR — the Fisher scatter a single network realization implies for
  the parameter. The x-axis labels it as `$\sigma_{H_0}$`-style LaTeX from
  `config/plotting.toml`.
- `provenance.json`: the validated configuration, the command-line flags as
  invoked (`sys.argv[1:]`), and the analysis software version.

`mean(SNR)` and `SNR(mean spectrum)` are separate statistics: SNR is nonlinear.
Fixed-count results measure **finite-catalog estimator scatter**; Poisson
results represent **finite-observation realizations**. Both include source
fluctuations and exclude detector-noise realizations. The source inclination
convention is determined by the recorded population and software version. The
current default BNS models sample isotropic inclinations, retaining orientation
fluctuations. Setting the recorded population's `sample_inclination = false`
selects analytic quadrupole averaging instead.

Only fixed-hyperparameter ensembles are accepted. The supplied Poisson example
samples `local_merger_rate` and is rejected: add a later simulation layer
setting `[spectra.hyperparameters].local_merger_rate` to
`"${fiducials.local_merger_rate}"` to study source fluctuations at a fixed rate.
Pass `--plot-param-fisher-scatter H0 xi_0` to add the Fisher-scatter figures;
the fiducial values come from the spectrum metadata, so every requested name
must be a fixed hyperparameter.
There are no bootstrap intervals or sweep/convergence diagnostics in this
first version, and no new workflow rule or generation cache format.

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
