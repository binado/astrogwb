# Snakemake workflows

The paper application has one top-level [`Snakefile`](../Snakefile). It
contains both the catalog rules that generate populations and waveform
catalogs, and the experiment rules that assemble configs, sample chains, and
build figures.

`snakemake` is invoked directly; preview with `--dry-run` (Snakemake executes
for real unless it is passed).

All commands run with `packages/astrogwb-paper/` as their working directory.
Source inputs live under `inputs/`; generated artifacts live under `outputs/`.

## Catalog workflow

Catalog recipes are committed in [`inputs/catalogs.yaml`](../inputs/catalogs.yaml).
Build a catalog explicitly before running an experiment:

```bash
snakemake --snakefile Snakefile --cores 1 \
  --allowed-rules population_config population waveform_catalog \
  --dry-run \
  outputs/catalogs/injection-bns-n32768-eps=0-df1.h5 \
  outputs/catalogs/bns-n16384-eps=0.1-df1.h5
snakemake --snakefile Snakefile --cores 1 \
  --allowed-rules population_config population waveform_catalog \
  outputs/catalogs/injection-bns-n32768-eps=0-df1.h5 \
  outputs/catalogs/bns-n16384-eps=0.1-df1.h5
```

The DAG deep-merges the two population graph variants, draws a fresh temporary
population for each catalog, and generates waveforms for those rows. All
durable catalogs live directly under `outputs/catalogs/`. The
`--allowed-rules` filter keeps catalog generation explicit. MCMC commands omit
these rules, so a missing input stops MCMC with a `MissingInputException`.

## Experiment workflow

[`inputs/experiments.yaml`](../inputs/experiments.yaml) contains the shared base, a
`networks` block naming each detector network once, and all four experiment
groups. Runs alias a network rather than restating its detector list, while
experiment and run mappings override inherited values. The
local `assemble_config` rule validates all 21 runs in one job and writes:

```text
outputs/configs/<experiment>/<run>.json
```

The generic `run_mcmc` rule then writes:

```text
outputs/chains/<experiment>/<run>.nc
```

The chain is protected. Its assembled config under `outputs/configs/` is the
record of the resolved scientific configuration.

The curated inventory is:

| Experiment | Runs | Figures |
| --- | ---: | --- |
| `cosmological-parameters` | 6 detector runs plus `H0-Omega_m` and `H0-merger-rate` | input to `plot_cosmological_parameters` |
| `modified-propagation` | 6 detector runs plus `Xi_0` and `Xi_0-H0` | corners, marginal comparison, and tables |
| `astrophysical-parameters` | `Madau-Dickinson` and `z_peak` | chains only |
| `variable-proposal-size` | 8192, 16384, and 32768 proposal catalogs | chains only |
| `variable-proposal-guard` | 1e-1, 1e-2, and 1e-3 proposal guard fractions | chains only |

Run one experiment's chains:

```bash
snakemake --snakefile Snakefile \
  --allowed-rules assemble_config run_mcmc cosmological_parameters_chains \
  --profile profiles/local --cores 8 --dry-run cosmological_parameters_chains
snakemake --snakefile Snakefile \
  --allowed-rules assemble_config run_mcmc cosmological_parameters_chains \
  --profile profiles/slurm cosmological_parameters_chains
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

Multiple chain targets may be supplied. Each experiment also has a complete
target named after it (`cosmological_parameters`, `modified_propagation`, ...),
which builds its figures where it has a figure rule and its chains otherwise.
The `experiments` target builds all four.

When passing CLI `--config` overrides to a CPU profile (`local`, `slurm-cpu`),
repeat `jax_platforms=cpu` in the same `--config` group: Snakemake replaces
the profile's whole `config:` list rather than merging.

## SLURM chains and local figures

The committed `slurm` and `slurm-cpu` profiles use the SLURM executor for
`run_mcmc`. Config assembly and all plotting/table rules are declared with
Snakemake's `localrules`, so they execute on the submit host.

For the `plot_cosmological_parameters` target, Snakemake:

1. assembles and validates all 21 configs in one local job;
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

Runtime platform settings belong to profiles, not the experiment inventory, and
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

Config mtimes do not invalidate expensive chains; the assembled config under
`outputs/configs/` records their actual inputs. To intentionally resample, make
the protected chain writable and use Snakemake's force controls. Always dry-run
first.

The old `out/`, `chains/`, and `figures/` trees are not migrated automatically;
the refactored workflow writes only beneath `outputs/`.
