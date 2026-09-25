# Snakemake workflows

The paper application has one top-level [`Snakefile`](../Snakefile). It contains
the catalog rules that generate waveform catalogs, and the experiment rules that
assemble configs, sample chains, and build figures.

`snakemake` is invoked directly; preview with `--dry-run` (Snakemake executes
for real unless it is passed).

All commands run with the repository root as their working directory.
Source inputs live under `config/`; generated artifacts live under `outputs/`.

The Snakefile computes its own inputs by globbing that tree --
`discover_catalog_names()` over `config/catalogs/*.json` and `discover_runs()`
over `config/runs/*/` -- and imports only the path and merge helpers from
`astrogwb.paper.config.runs`, because a run's catalog names are known only
after its layers are merged. No registry file translates a name into a path.

## Catalog workflow

Catalog configs are committed in [`config/catalogs/`](../config/catalogs/):
one `<name>.json` per catalog over the shared `config/waveform.json`,
`config/population.json` and `config/fiducials.json` layers. Build all
catalogs before running experiments:

```bash
snakemake --snakefile Snakefile --cores 1 \
  --allowed-rules waveform_catalog catalogs --dry-run catalogs
snakemake --snakefile Snakefile --cores 1 \
  --allowed-rules waveform_catalog catalogs catalogs
```

Or build individual catalogs:

```bash
snakemake --snakefile Snakefile --cores 1 \
  --allowed-rules waveform_catalog --dry-run \
  outputs/catalogs/md-imrphenom-s41-n32768.h5 \
  outputs/catalogs/md-imrphenom-s42-n16384.h5
```

One rule does the whole thing: it merges the catalog's config layers with `jq`,
draws the registered population they name in-process, and generates waveforms
for those rows. The population is not a workflow node and no longer a file
either -- the layer list *is* the dependency edge, so editing
`config/waveform.json`, `config/population.json` or `config/fiducials.json`
invalidates every catalog. `jq` is therefore a workflow dependency, alongside
`uv`. All durable
catalogs live under `outputs/catalogs/`. The `--allowed-rules` filter
keeps catalog generation explicit. MCMC commands omit these rules, so a missing
catalog stops MCMC with a
`MissingInputException`.

See [catalog generation](catalog-generation.md) for what a catalog records about
itself.

## Experiment workflow

[`config/runs/`](../config/runs/) holds one `_base.json` per experiment and one
JSON file per run, over the five shared `config/*.json` layers. `run_mcmc`
declares those layers as its own `input:` and folds them with `jq` into one
`--<block>` flag per shared block, then writes:

```text
outputs/chains/<experiment>/<run>.nc      the chain (protected)
outputs/chains/<experiment>/<run>.json    the config it was sampled with
```

There is no intermediate assembled config, and deleting that rule cost nothing:
its `input:` was already exactly these three files, so it only turned files the
chain already depended on into a JSON copy of themselves. Re-run granularity is
unchanged -- the rule is a per-run wildcard on `{experiment}/{run}`, so editing
a run's TOML retriggers exactly its own chain and editing a `base/` file
retriggers all 27.

That granularity is what removed the stale-input wrapper the old workflow
needed: one rule used to emit all 27 configs at once, so any edit invalidated
every one of them -- and the wrapper meant config changes never retriggered
sampling at all.

The `validate` rule replaces the old `configs` target. It merges, validates,
and catalog-checks all 27 runs without building anything, so a config typo fails
before any catalog is built.

The experiments are:

| Experiment | Runs | Figures |
| --- | ---: | --- |
| `cosmological-parameters` | 6 detector runs plus `H0-Omega_m` and `H0-merger-rate` | input to `plot_cosmological_parameters` |
| `modified-propagation` | 6 detector runs plus `Xi_0` and `Xi_0-H0` | corners, marginal comparison, and tables |
| `astrophysical-parameters` | `madau-dickinson` and `redshift-peak` | chains only |
| `variable-catalog-size` | 8192, 16384, and 32768 proposal samples | chains only |
| `variable-proposal-guard` | 1e-1, 1e-2, and 1e-3 proposal guard fractions | chains only |
| `waveform-approximant` | `IMRPhenom` and `TaylorF2` proposals | chains only |

Run one experiment's chains:

```bash
snakemake --snakefile Snakefile \
  --allowed-rules validate run_mcmc run_experiment_cosmological_parameters \
  --profile profiles/local --cores 8 --dry-run run_experiment_cosmological_parameters
snakemake --snakefile Snakefile \
  --allowed-rules validate run_mcmc run_experiment_cosmological_parameters \
  --profile profiles/slurm run_experiment_cosmological_parameters
```

Build the paper's complete cosmological-parameter section:

```bash
snakemake --snakefile Snakefile \
  --allowed-rules validate run_mcmc plot_cosmological_parameters \
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
snakemake --snakefile Snakefile --cores 1 \
  --allowed-rules importance_weights_grid \
  importance_weights_grid
```

This covers the importance-weight grids. The fiducial spectrum, network PSDs,
and SNR figures are the paper notebook
[`notebooks/fiducial_spectrum.py`](../notebooks/fiducial_spectrum.py), not a
workflow rule. The grid rule's input and output paths are named literally in
the Snakefile. Presentation -- labels, run order, and plot limits -- is
hard-coded in its script rather than passed on argv or loaded from a config the
workflow has to parse first.

## Re-running protected results

Config changes now *do* invalidate a chain -- one run's config is one rule's
output, so only the affected chain is retriggered. Because chains are
`protected()`, Snakemake refuses to overwrite one without being told to: make
the chain writable and use Snakemake's force controls. Always dry-run first.

The old `out/`, `chains/`, and `figures/` trees are not migrated automatically;
the refactored workflow writes only beneath `outputs/`.
