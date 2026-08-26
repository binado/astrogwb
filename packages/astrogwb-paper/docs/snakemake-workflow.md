# Snakemake workflows

The paper application has one top-level [`Snakefile`](../Snakefile). It contains
the bank rules that generate waveform banks, and the experiment rules that
assemble configs, sample chains, and build figures.

`snakemake` is invoked directly; preview with `--dry-run` (Snakemake executes
for real unless it is passed).

All commands run with `packages/astrogwb-paper/` as their working directory.
Source inputs live under `config/`; generated artifacts live under `outputs/`.

The Snakefile computes its own inputs by globbing that tree -- `discover_banks()`
over `config/banks/*.toml` and `discover_runs()` over
`config/analysis/runs/*/` -- and imports exactly one config function,
`assemble_run`, because a run's bank names are known only after the three-layer
merge. No registry file translates a name into a path.

## Bank workflow

Bank configs are committed in [`config/banks/`](../config/banks/), one TOML per
bank. Build all banks before running experiments:

```bash
snakemake --snakefile Snakefile --cores 1 \
  --allowed-rules waveform_bank banks --dry-run banks
snakemake --snakefile Snakefile --cores 1 \
  --allowed-rules waveform_bank banks banks
```

Or build individual banks:

```bash
snakemake --snakefile Snakefile --cores 1 \
  --allowed-rules waveform_bank --dry-run \
  outputs/banks/md-imrphenom-s41.h5 \
  outputs/banks/md-imrphenom-s42.h5
```

One rule does the whole thing: it reads the bank config, simulates that bank's
population graph in-process, and generates waveforms for those rows. The
population is not a workflow node -- it was a `temp()` output with exactly one
consumer, and two banks share one graph at different seeds. All durable banks
live under `outputs/banks/`. The `--allowed-rules` filter keeps bank generation
explicit. MCMC commands omit these rules, so a missing bank stops MCMC with a
`MissingInputException`.

See [bank generation](bank-generation.md) for what a bank records about itself.

## Experiment workflow

[`config/analysis/`](../config/analysis/) holds the shared base, one `_base.toml`
per experiment, and one TOML per run. The local `assemble_config` rule merges
those three layers for **one run** and writes:

```text
outputs/configs/<experiment>/<run>.json
```

The generic `run_mcmc` rule then writes:

```text
outputs/chains/<experiment>/<run>.nc
```

Both rules are per-run wildcards on `{experiment}/{run}`, so a config edit
retriggers exactly its own chain. That is what removed the stale-input wrapper
the old workflow needed on `run_mcmc`'s config input: one rule used to emit all
26 configs at once, so any edit invalidated every one of them -- and the wrapper
meant config changes never retriggered sampling at all. The `configs` target
builds all 26 configs without sampling anything.

The chain is protected. Its assembled config under `outputs/configs/` is the
record of the resolved scientific configuration.

The experiments are:

| Experiment | Runs | Figures |
| --- | ---: | --- |
| `cosmological-parameters` | 6 detector runs plus `H0-Omega_m` and `H0-merger-rate` | input to `plot_cosmological_parameters` |
| `modified-propagation` | 6 detector runs plus `Xi_0` and `Xi_0-H0` | corners, marginal comparison, and tables |
| `astrophysical-parameters` | `Madau-Dickinson` and `z_peak` | chains only |
| `variable-catalog-size` | 8192, 16384, and 32768 proposal samples | chains only |
| `variable-proposal-guard` | 1e-1, 1e-2, and 1e-3 proposal guard fractions | chains only |
| `waveform-approximant` | `IMRPhenom` and `TaylorF2` proposals | chains only |

Run one experiment's chains:

```bash
snakemake --snakefile Snakefile \
  --allowed-rules assemble_config run_mcmc run_experiment_cosmological_parameters \
  --profile profiles/local --cores 8 --dry-run run_experiment_cosmological_parameters
snakemake --snakefile Snakefile \
  --allowed-rules assemble_config run_mcmc run_experiment_cosmological_parameters \
  --profile profiles/slurm run_experiment_cosmological_parameters
```

Build the paper's complete cosmological-parameter section:

```bash
snakemake --snakefile Snakefile \
  --allowed-rules assemble_config run_mcmc plot_cosmological_parameters \
  --profile profiles/slurm plot_cosmological_parameters
```

This single local post-processing rule consumes all eight chains from
`cosmological-parameters`, then writes five figures and two CSV/LaTeX table
pairs. Requesting any one cosmological-parameter output path schedules the full
section.

Multiple chain targets may be supplied. Each experiment exposes a
`run_experiment_<name>` target (`run_experiment_cosmological_parameters`,
`run_experiment_modified_propagation`, ...) that builds its chains only;
figures are opt-in via the plot rules above. The `experiments` target
builds every experiment's chains.

When passing CLI `--config` overrides to a CPU profile (`local`, `slurm-cpu`),
repeat `jax_platforms=cpu` in the same `--config` group: Snakemake replaces
the profile's whole `config:` list rather than merging.

## SLURM chains and local figures

The committed `slurm` and `slurm-cpu` profiles use the SLURM executor for
`run_mcmc`. Config assembly and all plotting/table rules are declared with
Snakemake's `localrules`, so they execute on the submit host.

For the `plot_cosmological_parameters` target, Snakemake:

1. assembles and validates the eight configs it needs, one local job each;
2. submits the eight missing chains to SLURM;
3. waits for the chain outputs;
4. executes the one figure-and-table rule locally.

Keep the Snakemake controller alive for the whole run. The submit host must
share the output filesystem with the compute nodes and have the `plotting`
dependency group installed. Both SLURM profiles set a two-core local-rule
budget.

Profile summary:

| Profile | Executor | MCMC resources | JAX backend |
| --- | --- | --- | --- |
| `local` | local | 4 threads per run | CPU |
| `slurm` | SLURM GPU | 1 GPU, 4 CPUs, 8 GB, 12 h | CUDA |
| `slurm-cpu` | SLURM CPU | 4 CPUs, 16 GB, 24 h | CPU |

Runtime platform settings belong to profiles, not the analysis config tree, and
do not enter the scientific config hash.

## Standalone figures

Figures without an MCMC experiment remain explicit local rules in the unified
Snakefile -- there is no aggregate target, so request the rules or their
outputs directly:

```bash
snakemake --snakefile Snakefile --cores 1 amplitude_toy \
  --allowed-rules amplitude_toy fiducial_spectrum importance_weights_grid \
  fiducial_spectrum importance_weights_grid
```

These cover the amplitude toy model, fiducial spectrum, effective
detector PSDs, and importance-weight grids. Input and output paths are both
named literally in each rule. Presentation -- labels, run order, plot limits --
is hard-coded in the scripts in [`scripts/`](../scripts/) rather than passed on
argv or loaded from a config the workflow has to parse first.

## Re-running protected results

Config changes now *do* invalidate a chain -- one run's config is one rule's
output, so only the affected chain is retriggered. Because chains are
`protected()`, Snakemake refuses to overwrite one without being told to: make
the chain writable and use Snakemake's force controls. Always dry-run first.

The old `out/`, `chains/`, and `figures/` trees are not migrated automatically;
the refactored workflow writes only beneath `outputs/`.
