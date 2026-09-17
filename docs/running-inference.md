# Running inference

## Ad-hoc runs

`scripts/run_mcmc.py` takes a run's config layers plus the two catalog files
it samples against. One `--config` per layer, in merge order -- the same list
the workflow declares as the rule's `input:` and passes straight back on argv:

```bash
uv run --extra paper python scripts/run_mcmc.py \
  --config config/analysis/base/catalogs.toml \
  --config config/analysis/base/model.toml \
  --config config/fiducials.json \
  --config config/priors.json \
  --config config/networks.json \
  --config config/analysis/base/sampling.toml \
  --config config/analysis/runs/cosmological-parameters/_base.toml \
  --config config/analysis/runs/cosmological-parameters/ET-2L-aligned-CE-Hanford.toml \
  --injection-catalog outputs/catalogs/md-imrphenom-s41-n32768.h5 \
  --proposal-catalog outputs/catalogs/md-imrphenom-s42-n16384.h5
```

The two roles are fixed, so the files are named flags rather than a
name-to-path mapping. They are the catalogs the merged config's `[catalog]`
block names.

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
A run config is four layers merged in order:

```text
config/{fiducials,priors,networks}.json         the shared scientific values
config/analysis/base/*.toml                     the remaining shared settings
config/analysis/runs/<experiment>/_base.toml    the experiment override
config/analysis/runs/<experiment>/<run>.toml    the run override
  -> outputs/chains/<experiment>/<run>.nc       the chain
  -> outputs/chains/<experiment>/<run>.json     the config it was sampled with
```

Filenames are the mapping. There is no inventory file: `discover_runs()` globs
the tree, and a new run is a new TOML. `_base.toml` is required in every
experiment directory rather than optional -- a conditional Snakemake input
complicates the DAG for no gain.

Layer 0 is JSON, and top-level, because it is read by more than the workflow:
the notebooks and figure scripts consume the same files through
`astrogwb.paper.config`, and `jq` reads them without importing the package.
Each is a single-key object, so nothing special-cases them in the merge.

| File | Owns |
| --- | --- |
| `fiducials.json` | the fiducial value of every parameter |
| `priors.json` | the prior on every parameter |
| `networks.json` | each detector network, by name |
| `base/sampling.toml` | the sampling RNG seed and NUTS defaults |
| `base/model.toml` | observing time, frequency band, target population |
| `base/catalogs.toml` | the injection catalog and the default proposal catalog |

Fiducials are **not** the injection: what was injected is recorded in the
injection catalog file, which is where the observed spectrum's rate and density
come from. They are where NUTS initializes each sampled parameter, what the
non-sampled sites are conditioned at, and the reference point an
amplitude-marginalized run forms its ratio against. Nothing cross-checks them
against a catalog, because nothing needs to. Every fiducial carries a prior;
`RunConfig` retains the complete table and `analysis.sampled_params` selects
the NUTS latents, leaving the rest to NumPyro effect handlers.

A prior is `{"dist": "<numpyro.distributions class name>", "kwargs": {...}}`.
The class is looked up on `numpyro.distributions` by name, so adding a
distribution needs no code change. There is deliberately no positional `args`
form: the serializer can only emit kwargs, so a second spelling would make the
config `run_mcmc` writes next to each chain fail to round-trip.

A run names a network -- `[analysis] network = "ET-2L-aligned-CE-Hanford"` --
and `networks.json` resolves it to a detector list. `base/model.toml`
deliberately declares no `analysis.network`: a run without one must fail rather
than silently inherit someone else's. A run may not write out `detectors`
alongside a `network`; to try a network that is not committed, add it in an
overlay layer and name it:

```json
{"networks": {"scratch": ["S1", "R1"]}, "analysis": {"network": "scratch"}}
```

The chain's own config records both the resolved `detectors` and the `network`
name they came from, so an archived chain stays checkable against a later edit
to `networks.json`.

Nested mappings merge and lists replace, except that each overridden
`[priors.<param>]` table replaces the inherited one *wholesale*. That is
load-bearing, not incidental: key-merging a normal prior onto a uniform one
would leave stale `low` / `high` behind.

The six experiments and their 26 runs:

| Experiment | Runs |
| --- | ---: |
| `cosmological-parameters` | six detector networks, `H0-Omega_m`, and `H0-merger-rate` |
| `astrophysical-parameters` | `madau-dickinson` and `redshift-peak` |
| `modified-propagation` | six detector networks, `Xi_0`, and `Xi_0-H0` |
| `variable-catalog-size` | `n8192`, `n16384`, and `n32768` |
| `variable-proposal-guard` | `eps1e-1`, `eps1e-2`, and `eps1e-3` |
| `waveform-approximant` | `IMRPhenom` and `TaylorF2` |

`run_mcmc` declares a run's four layers as its own inputs, so editing a run's
TOML retriggers exactly that chain. Editing a `base/` file or one of the three
top-level JSONs retriggers all 26, which is correct.

`snakemake validate` merges and catalog-checks every run without building
anything. Run it before a campaign: it fails on the first invalid run *before
any catalog is built*, and a catalog is a GPU job.

## Catalogs and the proposal density

Every run names two catalogs, one per role:

```toml
[catalog]
injection = "md-imrphenom-s41-n32768"
proposal  = "md-imrphenom-s42-n16384"
```

That is the whole block, and injection versus proposal is two filenames and
nothing else. How a catalog was drawn -- its population model, that model's
construction settings, and the hyperparameters -- lives in
`config/catalogs/<name>.json` and its shared layers and, once the file exists,
in the file itself. Never in the run config.

Every run shares one injection catalog -- it is the "observed" data -- so it
lives in `base/catalogs.toml` and no run overrides it. Only
`variable-catalog-size`, `variable-proposal-guard`, `astrophysical-parameters`,
and `waveform-approximant` override the proposal catalog.

The importance-sampling *proposal density* is **not** in the config, and it is
not derived from the config either. It is the proposal catalog's *own* recorded
population, evaluated at the parameters it was drawn at (see
[catalog generation](catalog-generation.md)), narrowed to the run's analysis
redshift window by `PolarizationPowerCatalog.restrict_redshift` -- samples
and recorded density
together. There is nothing left to cross-check against the run's `[fiducials]`,
which is why the old exact-float-equality gate over five hard-coded parameter
names is gone.

The window is narrower than what was generated, which is why the per-sample
density cannot be baked into the file. Nothing about it is resolved at config
time, so `snakemake validate` stays cheap: a config typo, or an unregistered
population name, fails without any catalog having to exist.

The `[analysis.population]` block is the *target* population the sampled
hyperparameters describe:

```toml
[analysis.population]
model_name = "bns_md_modified_propagation"

[analysis.population.model_kwargs]
minimum_redshift = 0.3
maximum_redshift = 20.0
n_grid = 256
```

`model_name` is the default, and every committed run uses it. It reduces
exactly to the plain cosmological population at `xi_0 = 1`, which is how a run
that does not sample the propagation parameters gets the standard law without
naming a second model.

`model_kwargs` is the one statement of the redshift window and grid: the same
three numbers build the target callables and define the grid the spectral
integral runs on, so `AnalysisConfig.grid` reads them back rather than a second
block restating them. A third key, `density_sites`, selects the source-density
factors importance weighting includes; it defaults to redshift and the ordered
mass pair, and no committed run overrides it. It lives here rather than on a
catalog because no draw depends on it — the samples are the same whichever of
their densities a later weight counts.

The proposal catalog's recorded population, narrowed to the analysis window, is
stamped into the saved chain's posterior attributes, so the `.nc` remains the
self-describing record of what was sampled.

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

## Catalog prerequisite

Experiments consume existing catalogs and never generate them implicitly. Build
the required catalogs first:

```bash
snakemake --snakefile Snakefile --allowed-rules waveform_catalog catalogs \
  --profile profiles/local --cores 8 catalogs
```

If a required catalog is absent, the MCMC workflow fails with a
`MissingInputException` instead of silently scheduling hours of waveform
generation.
