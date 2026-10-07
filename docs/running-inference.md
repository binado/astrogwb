# Running inference

## Ad-hoc runs

`scripts/run_mcmc.py` takes the run's config layers, one `--config` flag per
file in merge order. The layers are the same list the workflow declares as the
rule's `input:`:

```bash
LAYERS="config/defaults.toml \
  config/waveforms.toml \
  config/populations.toml \
  config/detectors.toml \
  config/runs/cosmological-parameters/_base.toml \
  config/runs/cosmological-parameters/ET-2L-aligned-CE-Hanford.toml"

uv run --extra paper python scripts/run_mcmc.py \
  $(for layer in $LAYERS; do printf -- '--config %s ' "$layer"; done)
```

The merge is `astrogwb.paper.config.runs.merge_config_layers`, which hands the
layers to `knf` (pyknf): a deep merge left to right, arrays and scalars
replacing, except that each `priors.<param>` table replaces the inherited one
wholesale (`PRIOR_SHALLOW = "priors.*"`), so a Normal prior never inherits a
Uniform's `low` / `high`. After the merge every `"${a.b}"` reference is
resolved (see [references](#references)). The `knf` CLI is the same engine, so
this prints exactly the config the run will validate:

```bash
uv run --extra paper knf $LAYERS --shallow 'priors.*' --interpolate --merge-key extends
```

Merge order is yours to get right, and a wrong-but-valid order fails silently,
so `run_mcmc` logs it; the workflow builds it from `run_config_paths`. To
override something for a single invocation, add a layer to `LAYERS` -- the
stack is open-ended, which is the practical gain over a fixed assembled
artifact.

The catalogs are fetched the way a notebook fetches them:
`astrogwb.simulators.polarization_power.polarization_power` looks each role's
request (its metadata and `[analysis.seeds]` seed) up in `--catalog-dir`
(default `outputs/catalogs`), checks a hit against the request, and generates a
miss. Hits are served before JAX starts, so a file filed under the wrong name
fails cheaply; a miss is generated only after the runtime is
configured, because drawing a catalog initializes the XLA backend.
`--cached-only` makes a miss an error instead -- the workflow passes it, since
`rule waveform_catalog` builds the catalogs upstream. `just catalogs` prints
each run's two keys.

`scripts/profile_model.py` still takes a run's layer *paths* as repeated
`--config` flags -- it merges them in process, as the figure scripts do -- and
runs as:

```bash
uv run --extra paper python scripts/profile_model.py --help
```

## The configuration tree

[`config/`](../config/) is the sole MCMC configuration source. A run config is
six layers merged in order:

```text
config/defaults.toml                                       shared scientific values
config/waveforms.toml                                      named waveforms
config/populations.toml                                    named populations
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

`config/defaults.toml` declares the scientific defaults; `config/waveforms.toml`
and `config/populations.toml` declare the named waveforms and populations that
catalogs and targets refer to; `config/detectors.toml` declares `[networks]`
and an empty `[detectors]` table for optional overrides. Each block is
commented with what it owns. Several are read by more than the workflow: the
notebooks and figure scripts consume `fiducials`, `priors` and `networks`
through `astrogwb.paper.config`.

| Block | Owns |
| --- | --- |
| `[analysis]` | observing time, frequency band, target population, density sites, and the two catalogs |
| `[catalog]` | the default draw both roles refer to, shaped like `CatalogMetadata` |
| `[fiducials]` | the fiducial value of every parameter |
| `[priors]` | the prior on every parameter |
| `[sampler]` | the sampling RNG seed and NUTS defaults |
| `[waveforms.<name>]` | a named `WaveformMetadata` (in `waveforms.toml`) |
| `[populations.<name>]` | a named `PopulationMetadata` (in `populations.toml`) |
| `[networks]` | each detector network, by name (in `detectors.toml`) |
| `[detectors]` | optional geometry, PSD, and label overrides (in `detectors.toml`) |

`[catalog]`, `[waveforms]` and `[populations]` exist only to be referenced:
once the merge resolves every reference, what they said lives in the tables
that named them, so `RunConfig` drops them and the saved config holds only
resolved records.

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
`config/detectors.toml`, and pyknf merges them before the six run layers.
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

`psd_reference` keeps its existing resolution order: gwmock-noise preset,
packaged noise-curve file, local file, then HTTP(S) URL. Local paths stay relative
to the caller's working directory, including paths in override files. The
workflow declares the packaged detector tables as inputs of every chain and of
`importance_weights_grid`. A local or remote PSD is loaded when the run executes.
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

### References

After the merge, every string that is exactly `"${a.b}"` is replaced by the
merged value at `a.b` -- a table, a number, whatever it is, with its type
kept. That is how one table reuses another rather than restating it:
`[catalog]` names its waveform as `"${waveforms.default}"`, and each catalog
role in `[analysis]` names the fields of `[catalog]`. Four rules follow from
how `knf` resolves them:

- **References resolve against the final merge.** A run that overrides
  `[fiducials]` reaches every catalog drawn at `"${fiducials}"`.
- **A reference to a table merges as that table.** A layer that sets a key
  *under* a reference overrides that field and keeps the rest, so
  `[analysis.proposal.waveform] approximant = "TaylorF2"` gives the default
  waveform with a different approximant, and a run can override `num_samples`
  or a role's seed (`[analysis.seeds]`) alone. To change a named variant for
  every role that names it in one run, override it at its source --
  `[populations.guard] model_kwargs.uniform_mixing_fraction = 0.01` -- and every
  role that names it follows. `--shallow` forces
  replacement instead; `priors.*` is the one place we use it.
- **A reference can be reached through.** `"${catalog.population.model_kwargs.n_grid}"`
  resolves even though `[catalog].population` is itself
  `"${populations.cosmological}"`.
- **`extends` inherits a table.** `extends = "${a.b}"` starts a table from
  `a.b` and lets its own fields win, and the key is removed from the result.
  `[populations.guard.model_kwargs]` is the cosmological population's
  `model_kwargs` plus a mixing fraction. A table takes one base.

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
that experiment, and editing any of the four shared layers retriggers all 27.

A dry run (`snakemake --dry-run experiments`) merges every run and resolves its
catalog requests without building anything. Run it before a campaign: a
malformed config fails there, *before any catalog is built*, and a catalog is a
GPU job.

## Catalogs and the proposal density

Every run declares its two catalogs, `[analysis.injection]` and
`[analysis.proposal]`, each a `CatalogMetadata` once its references resolve.
Both default to the shared draw, `[catalog]`:

```toml
[catalog]
waveform = "${waveforms.default}"
fiducials = "${fiducials}"
num_samples = 32768
population = "${populations.cosmological}"
```

and a run overrides only what differs -- a size, a seed (`[analysis.seeds]`), a named population,
a named waveform:

```toml
[analysis.proposal]
population = "${populations.guard}"
num_samples = 16384
```

Each role's key names its file; see [catalog generation](catalog-generation.md).

Only `time-delay` overrides the injection, because its target population is not
the one the default injection was drawn from. Only `variable-catalog-size`,
`variable-proposal-guard`, `astrophysical-parameters`, `waveform-approximant`
and `time-delay` override the proposal.

The importance-sampling *proposal density* is **not** in the config, and it is
not derived from the config either. It is the proposal catalog's *own* recorded
population, evaluated at the parameters it was drawn at (see
[catalog generation](catalog-generation.md)), narrowed to the run's analysis
redshift window by `restrict_redshift` -- samples
and recorded density
together. There is nothing left to cross-check against the run's `[fiducials]`,
which is why the old exact-float-equality gate over five hard-coded parameter
names is gone.

The window is narrower than what was generated, which is why the per-sample
density cannot be baked into the file. Nothing about it is resolved at config
time, so the dry run stays cheap: a config typo fails without any catalog
having to exist.

`analysis.population` is the *target* population the sampled hyperparameters
describe, a `PopulationMetadata` like any catalog's, though a target is evaluated rather
than drawn from:

```toml
[analysis]
population = "${populations.target}"
```

`[populations.target]` is `bns_coba`, which every run but `time-delay` uses.
Modified propagation reduces exactly to the standard law at `xi_0 = 1`, which is
how a run that does not sample the propagation parameters gets the standard law
without naming a second model.

Its `model_kwargs` are the one statement of the analysis redshift window and
grid: the same three numbers build the target callables and define the grid
the spectral integral runs on. `analysis.density_sites` selects the
source-density factors importance weighting includes; it defaults to redshift
and the ordered mass pair, and no committed run overrides it. It lives on the
analysis rather than on a catalog because no draw depends on it -- the samples
are the same whichever of their densities a later weight counts.

The proposal catalog's recorded population, narrowed to the analysis window, is
stamped into the saved chain's posterior attributes, so the `.nc` remains the
self-describing record of what was sampled.

## Curated experiment runs

Run one experiment's chains through Snakemake from the repository root:

```bash
snakemake --snakefile Snakefile \
  --allowed-rules run_mcmc run_experiment_cosmological_parameters \
  --profile profiles/local --cores 8 run_experiment_cosmological_parameters
snakemake --snakefile Snakefile \
  --allowed-rules run_mcmc run_experiment_modified_propagation \
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
