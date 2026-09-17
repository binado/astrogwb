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

Values live in [`config/plotting.json`](../config/plotting.json): the LaTeX
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
network it uses, so looking the legend name up in `config/networks.json`
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
fiducials and detector networks from `config/fiducials.json` and
`config/networks.json` through `astrogwb.paper.config`, and the ordered network
legend from `astrogwb.paper.plotting.DETECTOR_NETWORKS`, but keeps its analysis
grid and local plotting choices as literals of its own. Those mirror
`config/analysis/base/model.toml`, so editing that file does not change these
figures -- update the notebook's configuration cell too.

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

All new figure products are written under `outputs/figures/`.

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
