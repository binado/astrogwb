# Running inference

## Ad-hoc runs

`scripts/run_mcmc.py` takes a run's config layers plus the bank files its
catalogs draw from. One `--config` per layer, in merge order -- the same list
the workflow declares as the rule's `input:` and passes straight back on argv:

```bash
uv run --extra paper python scripts/run_mcmc.py \
  --config config/analysis/base/catalogs.toml \
  --config config/analysis/base/model.toml \
  --config config/analysis/base/parameters.toml \
  --config config/analysis/base/sampling.toml \
  --config config/analysis/runs/cosmological-parameters/_base.toml \
  --config config/analysis/runs/cosmological-parameters/ET-2L-aligned-CE-Hanford.toml \
  --bank md-imrphenom-s41=outputs/banks/md-imrphenom-s41.h5 \
  --bank md-imrphenom-s42=outputs/banks/md-imrphenom-s42.h5
```

Repeat `--bank NAME=PATH` once per distinct bank the merged config's
`[catalog.injection]` and `[catalog.proposal]` tables name.

Order is yours to get right: nothing owns it any more, and a wrong-but-valid
order produces a valid-but-wrong run. The resolved order is logged at INFO
before the merge and stamped into the chain's `config_layers` attribute, so a
finished chain says which files produced it.

The stack is open-ended, which is the practical gain over a fixed assembled
artifact: append one more `--config` to override anything for a single
invocation -- a shorter chain, a smaller catalog -- without editing a committed
layer or writing a throwaway config.

`scripts/profile_model.py` takes the same flags and runs as:

```bash
uv run --extra paper python scripts/profile_model.py --help
```

## The configuration tree

[`config/analysis/`](../config/analysis/) is the sole MCMC configuration source.
A run config is three layers merged in order:

```text
config/analysis/base/*.toml                     settings every run shares
config/analysis/runs/<experiment>/_base.toml    the experiment override
config/analysis/runs/<experiment>/<run>.toml    the run override
  -> outputs/chains/<experiment>/<run>.nc       the chain
  -> outputs/chains/<experiment>/<run>.json     the config it was sampled with
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

`run_mcmc` declares a run's three layers as its own inputs, so editing a run's
TOML retriggers exactly that chain. Editing a `base/` file retriggers all 26,
which is correct.

`snakemake validate` merges and bank-checks every run without building
anything. Run it before a campaign: it fails on the first invalid run *before
any bank is built*, and a bank is a GPU job.

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
at run time rather than at config time so that `snakemake validate` stays
cheap: a config typo fails without any bank having to exist.

The resolved density is stamped into the saved chain's posterior attributes, so
the `.nc` remains the self-describing record of what was sampled.

## Curated experiment runs

Run one experiment's chains through Snakemake from the repository root:

```bash
snakemake --snakefile Snakefile \
  --allowed-rules validate run_mcmc run_experiment_cosmological_parameters \
  --profile profiles/local --cores 8 run_experiment_cosmological_parameters
snakemake --snakefile Snakefile \
  --allowed-rules validate run_mcmc run_experiment_modified_propagation \
  --profile profiles/slurm run_experiment_modified_propagation
```

Build all cosmological chains, figures, and tables with
`plot_cosmological_parameters`.

## Outputs

Every labelled run has a deterministic `.nc` path under
`outputs/chains/<experiment>/`, and `run_mcmc` writes the record of its
settings beside it as `<run>.json` -- the defaults-filled config, so two runs
that reach the same settings by different overrides produce identical files.
The ordered layer paths that produced it go into the chain's `config_layers`
attribute, which is the one thing a merged config cannot carry.

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
