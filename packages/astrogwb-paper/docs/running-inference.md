# Running inference

## Ad-hoc runs

`astrogwb-run-mcmc` accepts an assembled run config plus the bank files its
catalogs draw from. The project ships no standalone example config; assemble one
first:

```bash
uv run astrogwb-assemble-config \
  --experiment cosmological-parameters \
  --run ET-2L-aligned-CE-Hanford \
  --output outputs/configs/cosmological-parameters/ET-2L-aligned-CE-Hanford.json

uv run astrogwb-run-mcmc \
  --config outputs/configs/cosmological-parameters/ET-2L-aligned-CE-Hanford.json \
  --bank md-imrphenom-s41=outputs/banks/md-imrphenom-s41.h5 \
  --bank md-imrphenom-s42=outputs/banks/md-imrphenom-s42.h5
```

Repeat `--bank NAME=PATH` once per distinct bank the config's
`[catalog.injection]` and `[catalog.proposal]` tables name. `--all` assembles
every run in one process instead.

Any config under `outputs/configs/<experiment>/<run>.json` works for direct
runner and profiling work.

## The configuration tree

[`config/analysis/`](../config/analysis/) is the sole MCMC configuration source.
A run config is three layers merged in order:

```text
config/analysis/base/*.toml                     settings every run shares
config/analysis/runs/<experiment>/_base.toml    the experiment override
config/analysis/runs/<experiment>/<run>.toml    the run override
  -> outputs/configs/<experiment>/<run>.json
  -> outputs/chains/<experiment>/<run>.nc
```

Filenames are the mapping. There is no inventory file: `discover_runs()` globs
the tree, and a new run is a new TOML. `_base.toml` is required in every
experiment directory rather than optional -- a conditional Snakemake input
complicates the DAG for no gain.

The four base files partition disjoint concerns:

| File | Owns |
| --- | --- |
| `base/sampling.toml` | the sampling RNG seed and NUTS defaults |
| `base/model.toml` | observing time, frequency band, cosmology grid |
| `base/parameters.toml` | fiducial values and every parameter's prior |
| `base/catalogs.toml` | the injection catalog and the default proposal catalog |

`base/model.toml` deliberately declares no `analysis.detectors`: a run without a
detector list must fail rather than silently inherit someone else's network.

Nested mappings merge and lists replace, except that each overridden
`[priors.<param>]` table replaces the inherited one *wholesale*. That is
load-bearing, not incidental: key-merging a normal prior onto a uniform one
would leave stale `low` / `high` behind.

The six experiments and their 26 runs:

| Experiment | Runs |
| --- | ---: |
| `cosmological-parameters` | six detector networks, `H0-Omega_m`, and `H0-merger-rate` |
| `astrophysical-parameters` | `Madau-Dickinson` and `z_peak` |
| `modified-propagation` | six detector networks, `Xi_0`, and `Xi_0-H0` |
| `variable-catalog-size` | `n8192`, `n16384`, and `n32768` |
| `variable-proposal-guard` | `eps1e-1`, `eps1e-2`, and `eps1e-3` |
| `waveform-approximant` | `IMRPhenom` and `TaylorF2` |

`assemble_config` is one local Snakemake job **per run**, so editing a run's
TOML retriggers exactly that run's config and chain. Editing a `base/` file
retriggers all of them, which is correct.

## Catalogs and the proposal density

Every run states its own two catalogs inline. There is no named-composition
registry: a catalog is a bank name (or an MD bank plus a uniform bank) and the
mixture parameters, composed in memory at run time.

Every run shares one injection catalog -- it is the "observed" data -- so it
lives in `base/catalogs.toml` and no run overrides it. Only
`variable-catalog-size`, `variable-proposal-guard`, `astrophysical-parameters`,
and `waveform-approximant` override the proposal catalog.

The importance-sampling *proposal density* is **not** in the config. It is
derived at run time from the proposal bank's own provenance attributes (see
[bank generation](bank-generation.md)), restricted to the run's analysis
redshift window, and checked against the run's `[fiducials]`. Resolution happens
at run time rather than assemble time so that `astrogwb-assemble-config` stays
cheap: a config typo fails without any bank having to exist.

The resolved density is stamped into the saved chain's posterior attributes, so
the `.nc` remains the self-describing record of what was sampled.

## Curated experiment runs

Run one experiment's chains through Snakemake from `packages/astrogwb-paper/`:

```bash
snakemake --snakefile Snakefile \
  --allowed-rules assemble_config run_mcmc run_experiment_cosmological_parameters \
  --profile profiles/local --cores 8 run_experiment_cosmological_parameters
snakemake --snakefile Snakefile \
  --allowed-rules assemble_config run_mcmc run_experiment_modified_propagation \
  --profile profiles/slurm run_experiment_modified_propagation
```

Build all cosmological chains, figures, and tables with
`plot_cosmological_parameters`.

## Outputs

Every labelled run has a deterministic `.nc` path under
`outputs/chains/<experiment>/`. The assembled config at
`outputs/configs/<experiment>/<run>.json` is the record of the settings that
produced it; the run itself writes no provenance sidecar.

Ad-hoc unlabelled runs retain the timestamped
`mcmc-<params>-det=<detectors>-seed<n>-<timestamp>` convention.

## Bank prerequisite

Experiments consume existing banks and never generate them implicitly. Build the
required banks first:

```bash
snakemake --snakefile Snakefile --allowed-rules waveform_bank banks \
  --profile profiles/local --cores 8 banks
```

If a required bank is absent, the MCMC workflow fails with a
`MissingInputException` instead of silently scheduling hours of waveform
generation.
