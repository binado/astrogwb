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
| `H0-all-detectors` | 6 detector networks | input to `plot_cosmological_parameters` |
| `modified-propagation-all-detectors` | 6 detector runs plus `Xi_0` and `Xi_0-H0` | corners, marginal comparison, and tables |
| `H0-merger-rate` | fixed and sampled merger rate | input to `plot_cosmological_parameters` |
| `H0-omega-m` | one amplitude-marginalized run | input to `plot_cosmological_parameters` |
| `astrophysical-parameters` | one Madau-Dickinson run | chains only |
| `star-formation-peak` | one `z_peak` run | chains only |
| `variable-injection-size` | 8192, 16384, and 32768 injections | chains only |

Run one experiment's chains:

```bash
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/local --cores 8 --dry-run H0_all_detectors_chains
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/slurm H0_all_detectors_chains
```

Build the paper's complete cosmological-parameter section:

```bash
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/slurm plot_cosmological_parameters
```

This single local post-processing rule consumes all nine chains from
`H0-all-detectors`, `H0-merger-rate`, and `H0-omega-m`, then writes their five
figures and two CSV/LaTeX table pairs. These three experiments expose only
their `_chains` targets; their former complete targets were removed. Requesting
any one cosmological-parameter output path schedules the full section.

Multiple chain targets may be supplied. The `experiments` target builds every
experiment and the combined cosmological-parameter products.

When passing CLI `--config` overrides to a CPU profile (`local`, `slurm-cpu`),
repeat `jax_platforms=cpu` in the same `--config` group: Snakemake replaces
the profile's whole `config:` list rather than merging.

## SLURM chains and local figures

The committed `slurm` and `slurm-cpu` profiles use the SLURM executor for
`run_mcmc`. Config assembly and all plotting/table rules are declared with
Snakemake's `localrules`, so they execute on the submit host.

For the `plot_cosmological_parameters` target, Snakemake:

1. assembles and validates configs locally;
2. submits the nine missing chains to SLURM;
3. waits for chain and sidecar outputs;
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

Runtime platform settings belong to profiles, not experiment TOMLs, and do not
enter the scientific config hash.

## Standalone figures

Figures without an MCMC experiment remain explicit local rules in the unified
Snakefile -- there is no aggregate target, so request the rules or their
outputs directly:

```bash
snakemake --snakefile workflow/mcmc.smk --cores 1 amplitude_toy \
  fiducial_spectrum importance_weights_grid
```

These cover the amplitude toy model, fiducial spectrum, effective
detector PSDs, and importance-weight grids. Input and output paths are both
named literally in each rule. Presentation -- labels, run order, plot limits --
is hard-coded in the scripts in [`scripts/`](../scripts/) rather than passed on
argv or loaded from a config the workflow has to parse first.

## Re-running protected results

Config mtimes do not invalidate expensive chains; sidecar content hashes record
their actual inputs. To intentionally resample, make the protected chain and
sidecar writable and use Snakemake's force controls. Always dry-run first.

The old `out/`, `chains/`, and `figures/` trees are not migrated automatically;
the refactored workflow writes only beneath `outputs/`.
