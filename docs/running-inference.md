# Running inference

## Ad-hoc runs

`scripts/run_mcmc.py` takes the run's config layers, one `--config` flag per
file in merge order, plus the two catalog files it samples against. The layers
are the same list the workflow declares as the rule's `input:`:

```bash
LAYERS="config/defaults.toml \
  config/detectors.toml \
  config/runs/cosmological-parameters/_base.toml \
  config/runs/cosmological-parameters/ET-2L-aligned-CE-Hanford.toml"

uv run --extra paper python scripts/run_mcmc.py \
  $(for layer in $LAYERS; do printf -- '--config %s ' "$layer"; done) \
  --injection-catalog outputs/catalogs/<injection key>.h5 \
  --proposal-catalog outputs/catalogs/<proposal key>.h5
```

The merge is `astrogwb.paper.config.runs.merge_config_layers`, which hands the
layers to `knf` (pyknf): a deep merge left to right, arrays and scalars
replacing, except that each `priors.<param>` table replaces the inherited one
wholesale (`PRIOR_SHALLOW = "priors.*"`), so a Normal prior never inherits a
Uniform's `low` / `high`. The `knf` CLI is the same engine, so this prints
exactly the config the run will validate:

```bash
uv run --extra paper knf $LAYERS --shallow 'priors.*'
```

Merge order is yours to get right, and a wrong-but-valid order fails silently,
so `run_mcmc` logs it; the workflow builds it from `run_config_paths`. To
override something for a single invocation, add a layer to `LAYERS` -- the
stack is open-ended, which is the practical gain over a fixed assembled
artifact.

The two catalog roles are fixed, so the files are named flags. `just catalogs`
prints each run's two keys. `run_mcmc` resolves the request each role of its
config asks for and refuses a file that does not record exactly that request,
so a file handed to the wrong role, or one built before the config changed,
fails before JAX starts.

`scripts/profile_model.py` still takes a run's layer *paths* as repeated
`--config` flags -- it merges them in process, as the figure scripts do -- and
runs as:

```bash
uv run --extra paper python scripts/profile_model.py --help
```

## The configuration tree

[`config/`](../config/) is the sole MCMC configuration source. A run config is
four layers merged in order:

```text
config/defaults.toml                                       shared scientific values
config/detectors.toml                                      shared detector settings
config/runs/<experiment>/_base.toml                        the experiment override
config/runs/<experiment>/<run>.toml                        the run override
  -> outputs/chains/<experiment>/<run>.nc                  the chain
  -> outputs/chains/<experiment>/<run>.json                the config it was sampled with
```

Filenames are the mapping. There is no inventory file: `discover_runs()` globs
the tree, and a new run is a new TOML file. `_base.toml` is required in every
experiment directory rather than optional -- a conditional Snakemake input
complicates the DAG for no gain. What each committed run is *for* is written
as a comment at the top of its own file; [`config/runs/README.md`](../config/runs/README.md)
indexes the experiments and the catalogs they share.

`config/defaults.toml` declares the scientific defaults; `config/detectors.toml`
declares `[networks]` and an empty `[detectors]` table for optional overrides.
Each block is commented with what it owns. Several are read by more than the workflow: the
notebooks and figure scripts consume `fiducials`, `priors` and `networks`
through `astrogwb.paper.config`.

| Block | Owns |
| --- | --- |
| `[analysis]` | observing time, frequency band, target population, and the two catalogs |
| `[fiducials]` | the fiducial value of every parameter |
| `[networks]` | each detector network, by name (in `detectors.toml`) |
| `[detectors]` | optional geometry, PSD, and label overrides (in `detectors.toml`) |
| `[priors]` | the prior on every parameter |
| `[sampler]` | the sampling RNG seed and NUTS defaults |
| `[waveform]` | the waveform settings every catalog of a run inherits |
| `[population]` | the population a run's catalogs are drawn from, unless a role overrides it |

Fiducials are also the hyperparameters a run's catalogs are drawn at, so the
injection is drawn at exactly the values NUTS initializes at. What was
injected is still read back off the injection catalog file, which is where the
observed spectrum's rate and density come from. Fiducials are where NUTS
initializes each sampled parameter, what the
non-sampled sites are conditioned at, and the reference point an
amplitude-marginalized run forms its ratio against. Every fiducial carries a prior;
`RunConfig` retains the complete table and `analysis.sampled_params` selects
the NUTS latents, leaving the rest to NumPyro effect handlers.

A prior is `{ dist = "<numpyro.distributions class name>", kwargs = { ... } }`.
The class is looked up on `numpyro.distributions` by name, so adding a
distribution needs no code change. There is deliberately no positional `args`
form: the serializer can only emit kwargs, so a second spelling would make the
config `run_mcmc` writes next to each chain fail to round-trip.

A run names a network -- `[analysis] network = "ET-2L-aligned-CE-Hanford"` --
and the `[networks]` table resolves it to a detector list. `config/defaults.toml`
deliberately declares no `analysis.network`: a run without one must fail rather
than silently inherit someone else's. A run may not write out `detectors`
alongside a `network`; to try a network that is not committed, add it in an
overlay layer and name it:

```toml
[networks]
scratch = ["S1", "R1"]

[analysis]
network = "scratch"
```

The chain's own config records both the resolved `detectors` and the `network`
name they came from. It also records the complete `detector_registry`, including
geometry, PSD references, labels, and network membership. Reloading that JSON
uses the resolved settings without merging newer packaged defaults.

Detector definitions default to the packaged `geometry.toml` and
`sensitivity.toml`. Both use the same `[detectors.<name>]` structure as
`config/detectors.toml`, and pyknf merges them before the four run layers.
`DetectorRegistry` validates the resulting complete definitions. A shared,
experiment, or run layer can override individual fields; later layers win:

```toml
[detectors.E1]
psd_reference = "data/noise/e1_psd.txt"
label = "ET channel 1"

[detectors.E1.geometry]
xarm_azimuth_rad = 0.3141592653589793  # Original: 72.0 deg counter-clockwise from East
```

Geometry uses gwmock's `CustomDetector` fields: `latitude_rad`,
`longitude_rad`, `xarm_azimuth_rad`, `yarm_azimuth_rad`, `xarm_tilt_rad`,
`yarm_tilt_rad`, and `elevation_m`. Angles are radians; azimuths run clockwise
from North, and tilts measure altitude above the local horizon. Elevation is
in metres. Known detectors accept partial geometry or PSD-only overrides. New
names require latitude, longitude, elevation, both azimuths, and `psd_reference`;
both tilts default to zero. Unknown fields, incomplete definitions, and
undefined network members fail validation. Labels default to detector names and
affect presentation only.
The optional `duty_factor` is reference metadata; it does not rescale a PSD or
the observation time. Sensitivities for gwmock presets with upstream geometry
live separately in the packaged `presets.toml` and remain available through
the core sensitivity loaders.

`psd_reference` keeps its existing resolution order: gwmock-noise preset,
packaged noise-curve file, local file, then HTTP(S) URL. Local paths stay relative
to the caller's working directory, including paths in override files. The
workflow declares the packaged detector tables and selected external local
PSDs as chain and figure inputs.
Detector changes leave population draws, waveform catalogs, catalog keys, and
the package version unchanged.

Notebooks can call `detector_registry()` and then
`geometry, sensitivities = registry.build_network(name)`. A run uses
`config.detector_registry`; network-comparison figures carry each run's own
registry. Config loading and validation leave the JAX backend uninitialized;
runtime detector objects are built lazily.

Nested mappings merge and lists replace, except that each overridden
`[priors.<param>]` table replaces the inherited one *wholesale*. That is
load-bearing, not incidental: key-merging a normal prior onto a uniform one
would leave stale `low` / `high` behind.

The seven experiments and their 27 runs:

| Experiment | Runs |
| --- | ---: |
| `cosmological-parameters` | six detector networks, `H0-Omega_m`, and `H0-merger-rate` |
| `astrophysical-parameters` | `madau-dickinson` and `redshift-peak` |
| `modified-propagation` | six detector networks, `Xi_0`, and `Xi_0-H0` |
| `variable-catalog-size` | `n8192`, `n16384`, and `n32768` |
| `variable-proposal-guard` | `eps1e-1`, `eps1e-2`, and `eps1e-3` |
| `waveform-approximant` | `IMRPhenom` and `TaylorF2` |
| `time-delay` | `delay-slope` |

`run_mcmc` declares a run's layers as its own inputs, so editing a run's file
retriggers exactly that chain, editing an experiment's `_base.toml` retriggers
that experiment, and editing either shared layer (`config/defaults.toml` or
`config/detectors.toml`) retriggers all 27.

`snakemake validate` merges and catalog-checks every run without building
anything. Run it before a campaign: it fails on the first invalid run *before
any catalog is built*, and a catalog is a GPU job.

## Catalogs and the proposal density

Every run declares its two catalogs, one per role, as partial specs over its
own `[waveform]`, `[population]` and `[fiducials]` blocks:

```toml
[analysis.catalog]
injection = { seed = 41, num_samples = 32768 }
proposal = { seed = 41, num_samples = 32768 }
```

That is `config/defaults.toml`'s default, and a role overrides only what
differs -- the seed and size, a population, an approximant. Each role resolves
to a `CatalogMetadata`, whose key names the file; see
[catalog generation](catalog-generation.md).

Only `time-delay` overrides the injection, because its target population is not
the one the default injection was drawn from. Only `variable-catalog-size`,
`variable-proposal-guard`, `astrophysical-parameters`, `waveform-approximant`
and `time-delay` override the proposal.

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
