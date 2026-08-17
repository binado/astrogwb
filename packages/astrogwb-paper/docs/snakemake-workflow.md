# Snakemake workflows

The paper application has two Snakefiles:

- [`workflow/catalog.smk`](../workflow/catalog.smk) generates populations and
  waveform catalogs from committed recipes.
- [`workflow/mcmc.smk`](../workflow/mcmc.smk) assembles explicit experiment
  configs, samples chains, and builds experiment and standalone figures.

`snakemake` is invoked directly; preview with `--dry-run` (Snakemake executes
for real unless it is passed).

All commands run with `packages/astrogwb-paper/` as their working directory.
Source inputs live under `inputs/` and `experiments/`; generated artifacts live
under `outputs/`.

## Catalog workflow

Catalog recipes are committed under [`inputs/catalogs/`](../inputs/catalogs).
Build a catalog explicitly before running an experiment:

```bash
snakemake --snakefile workflow/catalog.smk --cores 1 \
  --dry-run outputs/catalogs/bns-n16384-df1.h5
snakemake --snakefile workflow/catalog.smk --cores 1 \
  outputs/catalogs/bns-n16384-df1.h5
```

The DAG first creates `outputs/populations/<catalog>.h5`, then creates
`outputs/catalogs/<catalog>.h5`. The experiment workflow deliberately does not
include these rules. A missing catalog therefore stops MCMC with a
`MissingInputException`.

## Experiment workflow

Each committed `experiments/<experiment>.toml` describes one experiment. The
local `assemble_config` rule merges `[runs.<run>]` with
[`inputs/mcmc.base.toml`](../inputs/mcmc.base.toml), validates the result, and
writes:

```text
outputs/configs/<experiment>/<run>.json
```

The generic `run_mcmc` rule then writes:

```text
outputs/chains/<experiment>/<run>.nc
outputs/chains/<experiment>/<run>.json
```

The chain is protected, and the sidecar records catalog and config hashes plus
the resolved scientific configuration.

The curated inventory is:

| Experiment | Runs | Figures |
| --- | ---: | --- |
| `H0-all-detectors` | 6 detector networks | posterior comparison and constraint table |
| `modified-propagation-all-detectors` | 6 detector runs plus `Xi_0` and `Xi_0-H0` | corners, marginal comparison, and tables |
| `H0-merger-rate` | fixed and sampled merger rate | density comparison, corner, and table |
| `H0-omega-m` | one amplitude-marginalized run | corner and relative-ESS corner |
| `astrophysical-parameters` | one Madau-Dickinson run | chains only |
| `star-formation-peak` | one `z_peak` run | chains only |
| `variable-injection-size` | 8192, 16384, and 32768 injections | chains only |

Run one complete experiment:

```bash
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/local --cores 8 --dry-run H0_all_detectors
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/slurm H0_all_detectors
```

The default target includes experiment figures where they exist. Request only
the chains with the `_chains` target:

```bash
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/slurm H0_all_detectors_chains
```

Multiple experiment targets may be supplied. The `experiments` target builds
every experiment.

When passing CLI `--config` overrides to a CPU profile (`local`, `slurm-cpu`),
repeat `jax_platforms=cpu` in the same `--config` group: Snakemake replaces
the profile's whole `config:` list rather than merging.

## SLURM chains and local figures

The committed `slurm` and `slurm-cpu` profiles use the SLURM executor for
`run_mcmc`. Config assembly and all plotting/table rules are declared with
Snakemake's `localrules`, so they execute on the submit host.

For a complete experiment target, Snakemake:

1. assembles and validates configs locally;
2. submits missing chains to SLURM;
3. waits for chain and sidecar outputs;
4. executes dependent figures locally.

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

Runtime platform settings belong to profiles, not experiment TOMLs, and do not
enter the scientific config hash.

## Standalone figures

Figures without an MCMC experiment remain explicit local rules in the unified
Snakefile:

```bash
snakemake --snakefile workflow/mcmc.smk --cores 1 --dry-run standalone_figures
snakemake --snakefile workflow/mcmc.smk --cores 1 standalone_figures
```

The aggregate includes the amplitude toy model, fiducial spectrum, effective
detector PSDs, and importance-weight grids. Output paths, detector networks,
and labels live under
[`inputs/figures/`](../inputs/figures); Snakemake passes those values to the
scripts in [`scripts/`](../scripts/).

## Re-running protected results

Config mtimes do not invalidate expensive chains; sidecar content hashes record
their actual inputs. To intentionally resample, make the protected chain and
sidecar writable and use Snakemake's force controls. Always dry-run first.

The old `out/`, `chains/`, and `figures/` trees are not migrated automatically;
the refactored workflow writes only beneath `outputs/`.
