# Catalog generation

A **catalog** is one persisted waveform draw: a population drawn from a
registered NumPyro model, with its frequency-domain polarization power reduced
and written to `outputs/catalogs/polarization_power-<key>-<digest>.h5`. Catalogs are the only expensive
artifact in the workflow and the only generated input a run consumes.

There is no separate "bank" any more. Catalogs used to be split in two: banks
were persisted single-component draws, and a catalog was a cheap in-memory
prefix or mixture over them, composed at run time. Storing only polarization
power made every catalog cheap enough to persist, so the composition step and
its plumbing went away.

## Catalogs are content-addressed

A run declares *what* each of its two catalogs draws, and at which seed; the
file is named by a hash of each:

```text
config/runs/<experiment>/<run>.toml [analysis.<role>] + [analysis.seeds]
    -> (CatalogMetadata, seed)
    -> outputs/catalogs/polarization_power-<metadata.key()>-<digest of seed>.h5
```

There is no catalog config tree and no name for a catalog. Two runs that ask
for the same draw resolve to the same file name and share one file; a run that
changes anything about its draw -- a seed, a kwarg, a fiducial -- gets a new
one. `just catalogs` maps the names back to what they draw, at which seed, and
which runs use them.

### What a run declares

Each role, `[analysis.injection]` and `[analysis.proposal]`, is a complete
`CatalogMetadata` once the merge resolves its `${...}` references (see
[references](running-inference.md#references)). Both default to the shared
draw, `[catalog]` in `config/defaults.toml`, field by field, and a run
overrides only what differs:

```toml
[analysis.proposal]
population = "${populations.guard}"
num_samples = 16384
```

The fields a role resolves to:

1. `waveform` — a named `[waveforms.<name>]` from `config/waveforms.toml`,
   `default` unless the role names another. Only
   `waveform-approximant/TaylorF2` does (`"${waveforms.TaylorF2}"`). The stored
   band matches `[analysis]`'s `minimum_frequency` and `maximum_frequency`: the
   catalog grid *is* the array every model is evaluated on, and a run's band
   selects bins on it with a mask rather than compressing it.
   The grid is `frequency_spacing` over that band: `"loglinear"` (the default) is
   uniform at `frequency_resolution` up to `turnover_frequency` and geometric
   above it, with the bin width continuous at the turn, which is ~400 bins for
   2--2048 Hz at 1 Hz and 100 Hz against 2047 for `"linear"`. `"log"` is
   geometric throughout, `frequency_resolution` wide at `minimum_frequency`.
   `turnover_frequency` belongs to `"loglinear"` alone: required there and
   rejected elsewhere, so one grid has one record and one key. The turn falls on
   the last uniform point not above `turnover_frequency`. For a `"linear"` Ripple
   waveform `sampling_frequency` is the backend's Nyquist, not the stored grid;
   the other spacings are built directly to `maximum_frequency` and do not use
   it. `[waveforms.linear]` is the default on the uniform grid, for comparing the
   two.
   `approximant="AnalyticInspiral"` selects the closed-form inspiral, and is
   the only approximant accepting the optional `alpha` key (the inspiral
   termination constant, defaulting to the Schwarzschild ISCO value); naming it
   alongside a Ripple approximant is rejected.
   `astrogwb.paper.config.waveform_generator()` builds the default draw's
   generator for a notebook.
2. `population` — a `PopulationMetadata`: `model_name`, a key in the
   `astrogwb.populations` registry, `model_kwargs`, the construction settings
   bound into it. The named populations live in `config/populations.toml`; a
   run that changes a named population overrides it at its source
   (`[populations.guard.model_kwargs] uniform_mixing_fraction = 0.01`). The
   analysis target is `analysis.population`, a separate record. There is no
   seed in it: a seed picks a realization of the density, so it is an input
   beside the metadata, stated per role in `[analysis.seeds]` (default 41 for
   both; `config/defaults.toml`). A run that wants another realization of a
   role sets `[analysis.seeds] proposal = 62`.
3. `fiducials` — the hyperparameters the draw is made at: `"${fiducials}"`, the
   run's own merged table, so the injection is drawn at exactly the values the
   run initializes at. `time-delay` sets `delay_slope = -1` once, as a run
   fiducial, and its injection inherits it.
4. `num_samples` — the draw's size.

Every named population states the redshift window by reference to
`[populations.cosmological]`, so a guarded proposal keeps the shared window and
grid and adds the one setting it takes. `RunConfig` validates the roles as
`CatalogMetadata` and the workflow keys the same merged tables, and a test pins
the two agreeing for every run.

Drawing at the run's fiducials has one visible cost: a run-level fiducial
override also changes the proposal's key, even for a population that never
reads it. `time-delay`'s proposal is therefore its own copy of the eps = 0.1
guard catalog. The draws are identical; only the recorded fiducials differ.

### The key, and what invalidates it

`CatalogMetadata` (in `astrogwb.simulators.polarization_power`) is the waveform
settings, the population record, the fiducials, the sample count, and the
`astrogwb` version. Its `key()` is the first 16 hex digits of a SHA-256 over
its canonical JSON; the file name adds the same 16 digits of a SHA-256 over the
seed input. Anything in the record or the seed invalidates the file by
renaming it, so the workflow's catalog rule declares no config inputs at all.

The record used to be called `CatalogRequest` and nested the waveform and
population under a `metadata` field. `key()` still hashes that nested shape,
so flattening it renamed no file.

The version is in the key so that *code* changes invalidate catalogs too -- but
only if it is bumped. **Bump `version` in `pyproject.toml` whenever a change
alters what a population draw or a waveform generator produces.** Every key
changes with it, so the next `snakemake catalogs` regenerates everything.
`just catalogs --orphans` lists the files no run asks for any more.

## The population is a registered model

A catalog names a population by its key in the `astrogwb.populations`
registry, and supplies the construction kwargs it takes -- the guarded proposal
above, for example. The redshift window and grid resolution are referenced
from `[populations.cosmological]` and the hyperparameters from the run's
`[fiducials]`; `model_kwargs` is one mapping, passed whole to the factory. The
population declares its density factors and source outputs.

A role inherits every field it does not name, whether or not the population it
names reads all of it: the guard inherits `[fiducials]` whole,
`local_merger_rate` included, even though `bns_md_uniform_mixture` declares no
merger rate. That is right for a guard mixture: it is a sampling density, and
nothing reads a rate off a proposal (see below).

**A registry key, not an import path.** Registry keys change only on purpose;
module paths move as collateral whenever a module is reorganized, so a
persisted `module:function` string is a reference that silently rots. An
unknown key fails pre-flight, in `snakemake validate`, listing what is
registered — before a GPU job is queued.

A registered population is a *factory*: it takes the construction kwargs and
returns the source model and the merger rate together, each a
`functools.partial` with those kwargs bound. Hyperparameters and source
arrays remain arguments.

```python
from astrogwb.populations import DEFAULT_DENSITY_SITES, build_population
from astrogwb.populations.evaluation import evaluate_sources, sample_sources

source_model, merger_rate_fn = build_population(
    "bns_md_cosmological", minimum_redshift=0.0, maximum_redshift=20.0, n_grid=4096
)

sources = sample_sources(source_model, key, params, num_samples=1024)
log_prob, outputs = evaluate_sources(
    source_model, params, sources, density_sites=DEFAULT_DENSITY_SITES
)  # log_prob: shape (1024,); outputs["luminosity_distance"]: shape (1024,)
total_merger_rate = merger_rate_fn(params)  # shape ()
```

One name, not two. The source model and its merger rate are both
normalizations of the same redshift law, and composing them freely is how every
guarded-proposal catalog came to record the plain Madau-Dickinson rate -- a
number that is not the normalization of the density its samples were drawn
from. A population that has no physical rate returns `None` for it, so a
proposal catalog used as an injection fails by name rather than scaling an
observed spectrum by the wrong factor.

The factory's signature *is* the construction-settings schema. A key the named
population does not take raises `TypeError` naming the population and what it
accepts, rather than being silently filtered on its way to one of two
separately built callables.

- The source model's returned mapping defines the stored columns, including
  spins, detector-frame masses, and `luminosity_distance`. Its sample sites are
  exactly the inputs needed to replay it; a missing one raises `KeyError`.
- `density_sites` selects the density factors included in importance
  weighting. It is an argument, not a stored field: no sample depends on it, so
  the analysis that reweights a draw states it — `DEFAULT_DENSITY_SITES`,
  redshift and the ordered mass pair, for every committed run — and one value
  is used for both sides of every weight. Omitting factors does not
  marginalize variables.
- `evaluate_sources` runs the model once, under a `sources` plate, with every
  column conditioned in, and returns the selected log density together with the
  model's recomputed outputs.

`sample_sources` uses `Predictive` followed by one batched replay through
`evaluate_sources`. Conditioning affects sample sites only, so stored
deterministic values never override the model's recomputation. That replay is
the same code path density evaluation takes, which preserves exactly zero
self-reweighting errors. Both functions isolate their NumPyro effects from
enclosing inference models with `handlers.block`.

A bound partial hashes by identity: build it once per run and close a
JIT-compiled function over it, while hyperparameters and source arrays are
traced. To compile sampling, keep `num_samples` static.

### Mass models

Component masses are an ordered pair: `source_frame_mass_1` is the larger one.
Two mass laws share the rest of the BNS Madau-Dickinson declaration:

- **Ordered uniforms** (`bns_md_cosmological`, `bns_md_modified_propagation`,
  `bns_md_uniform_mixture`). Parameters `minimum_mass` and `mass_width`; the
  fiducial support is `[1.0, 2.5]` solar masses, with constant joint density
  `2 / width**2` on the ordered triangle. That triangle is compact, so a NUTS
  step that moves the edges can send catalog samples outside the support and
  drop their importance weights to zero.
- **Ordered Gaussians** (`bns_md_gaussian_cosmological`,
  `bns_md_gaussian_modified_propagation`, `bns_md_gaussian_uniform_mixture`).
  Both components are i.i.d. `Normal(mass_mean, mass_sigma)`, then ordered.
  The joint density `2 N(m1) N(m2)` lives on the half-plane `m1 >= m2`, with
  no compact mass support, so moving `(mass_mean, mass_sigma)` never zeros a
  weight. Galactic BNS masses motivate the shape (a Gaussian around
  `1.33 Msun` with width `~0.09 Msun`). Default catalogs and run configs still
  use the uniform triangle.

### Guard mixtures are one density, not two draws

`bns_md_uniform_mixture` blends a fraction ε of uniform-in-redshift draws into
the Madau-Dickinson density with `numpyro.distributions.MixtureGeneral`. The
same mixture that draws the redshifts evaluates their log density, so the
recorded guard fraction can never be something other than what was drawn. It
replaced a pair of gwmock graphs differing only in their redshift block, a
weighted `MixtureSimulator`, and a hand-written `logaddexp` mixture density in
the analysis layer.

A guard mixture declares **no merger rate**: it is a sampling density, not a
physical population, and the Madau-Dickinson total rate is the normalization of
the Madau-Dickinson redshift density, not of a mixture of it with a uniform
component. Nothing reads a rate off a proposal -- importance weighting takes
the *target's* -- so this costs nothing, and a catalog drawn from a guard
mixture now raises if used as an injection or named as an analysis target,
where before it would have returned a finite rate wrong by the guard
fraction.

### Prefix stability across sizes

The three seed-42 `variable-catalog-size` proposals are generated independently and their
*source parameters* are still exact nested draws, which is what makes
`variable-catalog-size` a clean series rather than three unrelated runs.
`Predictive` allocates its per-draw keys with `jax.random.split`, which is
prefix-stable — a property of the installed JAX rather than an API promise, so
`tests/core/test_populations.py` checks it directly.

Prefix stability is a property of the population draw, not of the waveforms.
Generation is exactly reproducible at a fixed sample count, but the same source
generated in a batch of 8192 and a batch of 32768 gets polarization power
agreeing only to ~1e-14 relative: XLA picks different reduction orders at
different batch sizes and floating-point addition is not associative. The
differences land on the deep tail of the spectrum — values some three orders of
magnitude below the array peak — so they are numerically irrelevant, but the
catalogs are not byte-identical to one another.

## Generate a catalog

Through the workflow, from the repository root:

```bash
# every catalog any run asks for
snakemake --snakefile Snakefile --cores 1 \
  --allowed-rules waveform_catalog catalogs catalogs

# one catalog, by file stem (`just catalogs` lists them)
snakemake --snakefile Snakefile --cores 1 --allowed-rules waveform_catalog \
  outputs/catalogs/polarization_power-<key>-<digest>.h5
```

`rule waveform_catalog` hands `scripts/generate_catalog.py` the resolved
`CatalogMetadata` as JSON, the `--seed`, and the output path. The script refuses
a path that is not `polarization_power.path(...)` of that metadata and seed, so
one draw cannot be filed under another's address, then calls the cached node,
which writes atomically so an interrupted job never leaves a partial file.

The `--allowed-rules` filter keeps catalog generation explicit. MCMC commands
omit these rules, so a missing catalog stops the run with a
`MissingInputException` rather than silently scheduling waveform generation.

From Python the same node sits behind the same cache:

```python
import numpy as np

from astrogwb.paper.catalogs import run_catalog
from astrogwb.simulators.polarization_power import polarization_power

# a committed run's catalog: resolved from its config, generated on a miss
data, metadata = run_catalog("variable-proposal-guard", "eps1e-2", "proposal")

# or any CatalogMetadata at any seed, against any cache directory
data = polarization_power(
    {"seed": np.uint64(41)}, metadata, cache_dir="outputs/catalogs"
)
```

A hit is read and checked against the metadata it was asked for; a miss is
generated and saved under the name `polarization_power.path` gives.

## Catalogs record the density that drew them

Each cached file stores its complete `CatalogMetadata` in a root HDF5
attribute, written once at generation time:

```python
handle.attrs["metadata"] = metadata.model_dump_json()
```

The JSON record nests `waveform` and `population`, and includes `fiducials`,
`num_samples`, and `version`. The population records its registered
`model_name` and construction `model_kwargs`. Reading the file back, a caller
validates the attribute with `CatalogMetadata.model_validate_json()`; Pydantic
handles the serialization and validation without a field-by-field HDF5 codec.

What is *not* stored is a callable: `metadata.population.build()`
looks the name up in the registry and binds the recorded settings, returning
both callables at once (they hash by identity, so two getters would force a
recompile on every call).

That is enough to reconstruct the exact map from hyperparameters to source
density. With the version, it is the file's whole `CatalogMetadata` --
the `metadata` half of the `(data, metadata)` pair -- which is what `run_mcmc` checks each
catalog against before sampling.

The density factors are deliberately *not* part of the record. They change no
sample: the stored columns and power are the same whichever of their densities
a later weight counts, so the choice belongs to the analysis rather than to the
file. What still matters is that one value covers both sides of a ratio — a
proposal density computed with the mass factors excluded, reweighted against a
target that includes them, gives silently wrong weights with no shape error
anywhere — which is why `build_importance_spectrum` takes it once and threads
that one value into both callables it returns.

### The file format

Every cached node writes the same layout, through
`astrogwb.simulators.core.cache`:

| Where | What |
| --- | --- |
| root attribute `metadata` | the metadata's `model_dump_json()` |
| root attributes `node`, `version`, `created`, `host` | which function, which `astrogwb`, when and where |
| group `inputs/` | the node's inputs (the seed) |
| group `outputs/` | the node's outputs: a nested dict maps to subgroups, an array to a dataset |

For a catalog, `outputs/` holds `frequencies` `(F,)`, `polarization_power`
`(F, N)` and a `source_parameters/` group of `(N,)` columns. There are no
format names or version files: the path says which node, metadata and inputs a
file answers, and `read(path)` returns `(inputs, outputs, metadata_json)` to a
caller that was handed a file by path.
The arrays are used as read (the writer is the only source of files, so they
are not re-validated), and `metadata.population.build()` rebuilds the recorded population from the registry: an
unknown name raises `KeyError` listing what is registered, and a construction
setting the population does not take raises `TypeError`. Bin widths are not
stored; they are derived from `frequencies`.

## The cache

Catalogs and spectra share one cache, `astrogwb.simulators.core.cached`, which
wraps a simulator function `fn(inputs, metadata, **settings)`:

| | node | metadata | inputs | outputs wrap as |
| --- | --- | --- | --- | --- |
| catalogs | `polarization_power` | `CatalogMetadata` | `{"seed": uint64 scalar}` | `(data, metadata)` as is |
| spectra | `spectra` | `SpectraMetadata` | `{"seeds": uint64 array}` | `SpectralDensityCatalog.from_arrays` |

A node's result lives at
`<cache_dir>/<fn.__name__>-<metadata.key()>-<digest(inputs)>.h5`, where
`digest` hashes the inputs' paths, dtypes, shapes and bytes. A miss runs the
body and writes atomically; a hit is read and its recorded metadata compared
with the request. `generate=False` serves hits only and raises
`FileNotFoundError` on a miss; `run_mcmc` fetches its catalogs this way under
`--cached-only`, so a chain job never generates one. Settings such as
`chunk_size` are passed to the body and kept out of the path. Cached nodes need
concrete inputs, so a traced input raises a clear `TypeError`; call the
undecorated body (`node.__wrapped__`) inside a transformation instead.

Seeds are inputs because they pick a realization rather than describe a
distribution: `split_seed(seed, n)` derives `n` prefix-stable child seeds
(`SeedSequence(seed, spawn_key=(i,))`) without JAX, and
`astrogwb.simulators._keys.seed_key` is the one place a 64-bit seed becomes a
JAX key.

The version is in the key so that code changes invalidate the cache -- but
only if it is bumped. Bump `version` in `pyproject.toml` whenever a change
alters what a population draw or a waveform generator produces. Version
**0.4.0** moved the seed out of the metadata and changed the file layout, so
every file from earlier versions is unreachable; remove the old
`outputs/catalogs/` and `outputs/spectra/` files and regenerate. The spectra
simulator's later split into a population draw and a packed reduction changes
the spectra record, hence its keys, but no catalog draw, so it did not bump.

The waveform metadata records `frequency_spacing`, `frequency_resolution` and
`turnover_frequency` -- what was *requested* of the generating backend -- while
the bin widths used in every integral are derived from the `frequency` dataset
itself (`astrogwb.frequency.bin_widths`, used directly on
`data["frequencies"]`); the backend chooses the actual grid, so
the two can differ. The grid need only be strictly increasing: each bin's width
is half the distance between its neighbours, which is the grid spacing on a
uniform grid and grows with frequency above the turn of a `"loglinear"` one.

Adding the grid fields re-keyed every catalog and spectra file, and the version
was bumped with them (0.2.0), so files from earlier versions are never served;
regenerate them.

## Inclination convention

All shipped BNS population models sample isotropic inclination by default:
`cos(iota)` is uniform on `[-1, 1]`, and the returned `inclination` sample site
is in radians. Polarization-power catalogs store this column, and density
reconstruction conditions on the recorded values even though inclination is
excluded from the default importance-weight factors. The common,
hyperparameter-independent inclination law cancels in those weights.

Both fixed-count and Poisson spectrum realizations use the same recorded source
model. Waveform power already includes each sampled inclination, so contraction
uses an inclination factor of one. Spectrum files store contracted draws and
population metadata, rather than the individual event columns. Sampling
inclination restores orientation fluctuations in both count modes; fixed-count
estimator scatter and Poisson observation scatter remain different experiments.

For explicit analytic quadrupole averaging, set `sample_inclination = false`
in the population's `model_kwargs`. For example, a layer can override the
shared population and every role that inherits its settings:

```toml
[populations.cosmological.model_kwargs]
sample_inclination = false
```

Or construct it directly with
`build_population("bns_md_cosmological", sample_inclination=False, **kwargs)`.
This omits the inclination sample site and column. Waveform generators then use
face-on power, and contraction applies the analytic `2/5` factor. It preserves
the quadrupole ensemble mean but removes orientation fluctuations; it is not a
universal orientation average for higher-mode waveforms. The waveform comparison
notebook uses the sampled default, without an additional `IsotropicInclination`
wrapper. That handler remains available for custom models that omit inclination.

The choice is a boolean construction setting, recorded in provenance and cache
identity. It is not an option on the waveform or artifact generator. Numeric
substitutes such as `0` and `1` are rejected for this setting.

Version **0.3.0** changes the default population draw. Both catalog and spectrum
artifacts therefore receive new keys; old face-on artifacts are not reused as
sampled-inclination draws. From the repository root, regenerate workflow catalogs
and chains with:

```bash
uv run --group workflow snakemake --snakefile Snakefile --cores 1 validate
uv run --group workflow snakemake --snakefile Snakefile --cores 1 experiments
```

Regenerate spectra with the existing CLI, choosing either the `fixed` or
`poisson` simulation layer:

```bash
uv run --extra paper python scripts/simulate_spectra.py \
    --config config/defaults.toml --config config/waveforms.toml \
    --config config/populations.toml --config config/detectors.toml \
    --config config/simulations/spectrum/fixed.toml
```

A follow-up notebook will compare sampled and analytically averaged ensemble
means, variance and frequency covariance at fixed hyperparameters in both count
modes. Detector noise, shot-noise likelihoods and SNR studies are separate work.

## The population node

A spectrum is a population draw reduced through a waveform, and the first half
is a node of its own: `astrogwb.simulators.population.population`.

- **metadata** -- a `PopulationDrawMetadata`: the population, each
  hyperparameter's fixed value or prior, `observation_time`, `count`,
  `num_events` and the version. No waveform, so two waveforms' spectra records
  share one population key (`SpectraMetadata.sources.key()`).
- **inputs** -- `{"seeds": ...}`, as for spectra.
- **outputs** -- one draw per call: `count`, `total_merger_rate` and
  `hyperparameters/<name>` are 0-d and each `source_parameters/<name>` column
  is `(count,)` (`PopulationData`).

Draw once and reduce through several waveforms -- the same events, so the
differences are the waveform's alone:

```python
population = PopulationSimulator(metadata.sources)
simulator_a, simulator_b = SpectraSimulator(metadata_a), SpectraSimulator(metadata_b)
parts_a, parts_b = [], []
for key in batch_keys(seed, n):
    draw = population(key)
    parts_a.append(simulator_a.reduce(draw))
    parts_b.append(simulator_b.reduce(draw))
spectra_a, spectra_b = stack_spectra(parts_a), stack_spectra(parts_b)
```

Persisting a population pays when it is reused like this; a simulation loop
calls `SpectraSimulator(...)(key)` per draw and keeps nothing.

## The spectral-density node

The sibling artifact is a `SpectralDensityCatalog`. It holds the forward
model's *contraction* rather than the power it contracts, so a run that only
needs predicted spectra never materializes `(F, N)` waveforms.

It is produced by the cached node `astrogwb.simulators.spectra.spectra`:

- **metadata** -- a `SpectraMetadata`: the waveform, the population, each
  hyperparameter's fixed value *or* prior, `observation_time`, `count`,
  `num_events`, and the `astrogwb` version. It extends the waveform-free
  `PopulationDrawMetadata` (see [the population node](#the-population-node)); its
  `.sources` is that part. No seed and no draw count.
- **inputs** -- `{"seeds": ...}`, a 1-d `uint64` array with no duplicates, one
  seed per draw, usually `split_seed(seed, num_draws)`. Each seed is one draw
  (hyperparameters and sources alike), so a draw depends on its own seed alone,
  not on its batchmates, and the same seed gives the same spectrum in any call.
- **settings** -- `chunk_size` chunks the waveform reduction;
  `source_chunk_size` (default `chunk_size`) sets the size of the pieces
  Poisson-count sources are drawn in. Neither consumes randomness, so neither is in the
  path. Memory is one draw's sources.

```python
from astrogwb.paper.cache import default_cache_dir
from astrogwb.paper.config import fiducials, population_metadata, waveform_metadata
from astrogwb.simulators.core import split_seed
from astrogwb.simulators.spectra import SpectraMetadata, SpectralDensityCatalog, spectra

metadata = SpectraMetadata(
    waveform=waveform_metadata(),
    population=population_metadata(),
    hyperparameters={
        **fiducials(),
        "local_merger_rate": {"dist": "Normal", "kwargs": {"loc": 770.0, "scale": 7.7}},
    },
    observation_time=1.0,
    count="poisson",
)
outputs = spectra(
    {"seeds": split_seed(41, 64)},
    metadata,
    cache_dir=default_cache_dir() / "spectra",
    chunk_size=1024,
)
catalog = SpectralDensityCatalog.from_arrays(outputs, metadata)
```

A hyperparameter is a number to fix it for every draw, or a
`{"dist", "kwargs"}` spec -- the format of the shared `[priors]` table, validated by
`astrogwb.distributions.config.DistributionConfig` -- to draw it independently once per draw. Priors
are data, so an edited bound re-keys the draws without a version bump. Each
seed is split into a hyperparameter key, a count key and a source key, and event
`i` of a draw is drawn from the source key folded with `i`. Counts are exact --
there is no padded capacity -- and the first `n` events of a draw do not depend
on how many were drawn, so `chunk_size` changes cost, never the draws.

A draw's events are reduced chunk by chunk (`ChunkedPowerSum`): chunks of
`chunk_size` sources run through the waveform and are added into one `(F,)`
sum, with a mask on the tail of the last chunk, so one compilation serves any
count and a prior that spreads the merger rate several-fold costs no more
waveforms than the counts need. A simulator call is one draw; `stack_spectra`
joins a loop over `batch_keys` into the draw-first layout below.
`SpectraSimulator` owns the built population, generator and jitted stages and is
memoized per record, so a loop that calls `spectra` again does not recompile.

`count="poisson"` (the default) uses `poisson_counts_forward_model`: it draws
`N ~ Poisson(R * T)` and forms `S_h = A_inc * sum(P_i) / T`, with `T` in
seconds and the observer-frame rate `R` in mergers per second, from the
realized sources of that exact count. `num_events` must be omitted.

`count="fixed"` draws exactly the positive integer `num_events` sources and
forms `S_h = A_inc * R * sum(P_i) / num_events`. `n_events` is deterministic and
equals that count in every row. Both modes require a population with a physical
merger rate. They share the chunked reduction -- the count mode only changes the
per-draw count and the normalization factor -- and the inclination convention:
`A_inc` averages face-on power over isotropic inclinations when the source
model omits inclination, and is one when inclination is supplied.

`num_events` is the source count within each fixed realization; the length of
`seeds` is the number of independent realizations. `observation_time` remains positive
in both modes and is recorded in the key. It controls Poisson counts and
cancels from fixed-count normalization.

The complete normalized metadata is hashed. `n_max_sigma` is gone from the
record, so earlier spectrum cache addresses are unreachable; rerun spectrum
generation under the new keys. There is no key migration. Polarization-power
catalog draws and waveform algorithms are unchanged, so this does not bump the
package version or invalidate polarization-power catalog caches.

`scripts/simulate_spectra.py` is the same node from a shell. It takes
config layers like `run_mcmc` does -- the four shared `config/*.toml` layers,
then `config/simulations/spectrum/<name>.toml` -- and validates the merged
`[spectra]` table as the `SpectraMetadata`. In that table a hyperparameter is a
`"${fiducials.X}"` reference (fixed) or a `"${priors.X}"` one (sampled). The
sibling `[draws]` table (`seed`, `num_draws`) names the seeds, through
`split_seed`. The output is `<--output-dir>/spectra-<key>-<digest>.h5`. The default is
`default_cache_dir() / "spectra"`, shared with the SNR analysis script across
worktrees: `~/Library/Caches/astrogwb/spectra` on macOS or
`~/.cache/astrogwb/spectra` on Linux, with `XDG_CACHE_HOME` taking precedence
on both. `--output-dir` overrides it. Cache locations are outside the
scientific metadata and its key; workflow catalog outputs remain under
`outputs/catalogs`. A second invocation with the same layers is a cache hit;
`--force` draws again and replaces the file:

```bash
uv run --extra paper python scripts/simulate_spectra.py \
    --config config/defaults.toml --config config/waveforms.toml \
    --config config/populations.toml --config config/detectors.toml \
    --config config/simulations/spectrum/poisson.toml
```

For 100 fixed-count realizations at the shared fiducials, each with
`${catalog.num_samples}` sources:

```bash
uv run --extra paper python scripts/simulate_spectra.py \
    --config config/defaults.toml --config config/waveforms.toml \
    --config config/populations.toml --config config/detectors.toml \
    --config config/simulations/spectrum/fixed.toml
```

The spectrum layers are not run layers: no chain reads `[spectra]`. Callers
using custom source-model compositions can still build a
`PopulationSimulator` from a registered population and reduce its draws with
`ChunkedPowerSum` (or `SpectraSimulator.reduce`). Sampled inclination is already part of the
registered BNS population and needs no custom composition.

The `outputs/`
group of a spectra file holds:

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `frequencies` | `(F,)` | the generating backend's grid, as in a power catalog |
| `spectral_density` | `(draws, F)` | one realization of the selected forward model per row |
| `n_events` | `(draws,)` | that draw's Poisson count or the recorded fixed `num_events` |
| `total_merger_rate` | `(draws,)` | that draw's observer-frame total rate |
| `hyperparameters/<name>` | `(draws,)` | the value each row was drawn at, one dataset per name |

The root `metadata` attribute is the complete `SpectraMetadata.model_dump_json()`
record; `inputs/seeds` holds the seeds the draws were made at.

The two artifacts differ in what a row is, and that is the whole difference. A
power catalog's sample axis indexes *sources* drawn once at one set of
hyperparameters, recorded as `fiducials` in the metadata JSON. A
spectral-density catalog's row axis indexes *draws* of the whole forward model,
so its hyperparameters are a column per name. A fixed hyperparameter's column
must repeat its value; a sampled one's holds each row's draw. In fixed mode,
every `n_events` entry must match the metadata's `num_events`.

`SpectralDensityCatalog.from_arrays` validates the same way its sibling does:
shapes and that every column agrees on the draw count.

## What is *not* in the file: the analysis window

The recorded settings are the *generation* window, `[0.0, 20.0]`. The analysis
window is narrower — `analysis.population.model_kwargs.minimum_redshift = 0.3`
in the shared `[analysis]` table — so the per-sample log density cannot be
baked into the catalog: it depends on a truncation the run chooses, not on
anything generation knows.

`restrict_redshift(data, metadata, minimum_redshift, maximum_redshift)` narrows both halves
together, and that is the whole reason it is one function. Dropping samples without narrowing
the recorded model would leave the density normalized over a window the samples
no longer span, and every importance weight would be off by that
normalization. Draws truncated to a sub-window follow the same law as draws
made directly from it, so only the support changes.
