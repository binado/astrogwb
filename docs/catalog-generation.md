# Catalog generation

A **catalog** is one persisted waveform draw: a population drawn from a
registered NumPyro model, with its frequency-domain polarization power reduced
and written to `outputs/catalogs/polarization_power-<key>-<digest>.h5`.
Catalogs are the expensive artifact of the analysis.

## Catalogs are content-addressed

A catalog is named by a hash of what it draws, and at which seed:

```text
(CatalogMetadata, seed)
    -> outputs/catalogs/polarization_power-<metadata.key()>-<digest of seed>.h5
```

There is no catalog config tree and no name for a catalog. Two callers that ask
for the same draw resolve to the same file name and share one file; changing
anything about a draw -- a seed, a kwarg, a fiducial -- gives a new one.

### What a catalog declares

`CatalogMetadata` (in `astrogwb.simulators.polarization_power`) carries:

1. `waveform` -- a `WaveformMetadata`, by default the named
   `[waveforms.default]` from `config/waveforms.toml`. The stored band matches
   `[analysis]`'s `minimum_frequency` and `maximum_frequency`: the catalog grid
   *is* the array every model is evaluated on, and an analysis band selects
   bins on it with a mask rather than compressing it.
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
2. `population` -- a `PopulationMetadata`: `model_name`, a key in the
   `astrogwb.populations` registry, `model_kwargs`, the construction settings
   bound into it. The named populations live in `config/populations.toml`.
   There is no seed in it: a seed picks a realization of the density, so it is
   an input beside the metadata.
3. `fiducials` -- the hyperparameters the draw is made at, `"${fiducials}"`.
4. `num_samples` -- the draw's size.

The shared default draw is `[catalog]` in `config/defaults.toml`;
`astrogwb.paper.config.waveform_metadata()` and `population_metadata()` return
its records.

### The key, and what invalidates it

`CatalogMetadata` is the waveform settings, the population record, the
fiducials, the sample count, and the `astrogwb` version. Its `key()` is the
first 16 hex digits of a SHA-256 over its canonical JSON; the file name adds
the same 16 digits of a SHA-256 over the seed input. Anything in the record or
the seed invalidates the file by renaming it.

The version is in the key so that *code* changes invalidate catalogs too -- but
only if it is bumped. **Bump `version` in `pyproject.toml` whenever a change
alters what a population draw or a waveform generator produces.** Every key
changes with it, and the next draw regenerates everything.

## The population is a registered model

A catalog names a population by its key in the `astrogwb.populations`
registry, and supplies the construction kwargs it takes. The redshift window
and grid resolution are referenced from `[populations.cosmological]` and the
hyperparameters from the shared `[fiducials]`; `model_kwargs` is one mapping, passed whole to the factory. The
population declares its density factors and source outputs.

**A registry key, not an import path.** Registry keys change only on purpose;
module paths move as collateral whenever a module is reorganized, so a
persisted `module:function` string is a reference that silently rots. An
unknown key fails when the population is built, listing what is registered.

A registered population is a *factory*: it takes the construction kwargs and
returns a callable `parameters -> (merger_rate, model)`. One call builds the
redshift distribution once, so the rate and the source density come from the
same grid; `model()` is a no-argument NumPyro model. Hyperparameters are what
the callable is called with, and source arrays are what `evaluate_sources`
conditions in.

```python
from astrogwb.populations import DEFAULT_DENSITY_SITES, build_population
from astrogwb.populations.evaluation import evaluate_sources, sample_sources

population = build_population(
    "bns_coba",
    mass_model="uniform",
    minimum_redshift=0.0,
    maximum_redshift=20.0,
    n_grid=4096,
)
total_merger_rate, model = population(params)  # rate: shape ()

sources = sample_sources(model, key, num_samples=1024)
log_prob, outputs = evaluate_sources(
    model, sources, density_sites=DEFAULT_DENSITY_SITES
)  # log_prob: shape (1024,); outputs["luminosity_distance"]: shape (1024,)
```

One call, not two. The source model and its merger rate are both
normalizations of the same redshift law, so they are built together. The one
exception is a guard mixture: its returned rate is still the Madau-Dickinson
total rate, which does not normalize the mixture density. A guard mixture is a
proposal, and the caller is trusted never to use one as an injection or an
analysis target.

`bns_coba` is the one shipped population, and its variants are construction
kwargs: `mass_model` (`"uniform"` or `"gaussian"`), `time_delay` (with
`minimum_delay`, `maximum_formation_redshift`, `n_delay_nodes`) and
`uniform_mixing_fraction`. Modified GW propagation is selected by the
hyperparameters, not a kwarg: it applies whenever `xi_0` is among them, and is
the identity at `xi_0 = 1`. The contracts each setting carries -- which
hyperparameters it needs, which only rescale the spectrum (`H0` stops being one
when `time_delay` is on), what a proposal may be used for -- are documented in
the `bns_coba_population_fn` docstring and are not checked.

The factory's signature *is* the construction-settings schema. A key the named
population does not take raises `TypeError` naming the population and what it
accepts, rather than being silently dropped.

- The source model's returned mapping defines the stored columns, including
  spins, detector-frame masses, and `luminosity_distance`. Its sample sites are
  exactly the inputs needed to replay it; a missing one raises `KeyError`.
- `density_sites` selects the density factors included in importance
  weighting. It is an argument, not a stored field: no sample depends on it, so
  the analysis that reweights a draw states it — `DEFAULT_DENSITY_SITES`,
  redshift and the ordered mass pair -- and one value
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

A population hashes by identity: build it once per run and close a
JIT-compiled function over it, while hyperparameters and source arrays are
traced. To compile sampling, keep `num_samples` static.

### Mass models

Component masses are an ordered pair: `source_frame_mass_1` is the larger one.
Two mass laws share the rest of the BNS Madau-Dickinson declaration:

- **Ordered uniforms** (`mass_model = "uniform"`). Parameters `minimum_mass` and `mass_width`; the
  fiducial support is `[1.0, 2.5]` solar masses, with constant joint density
  `2 / width**2` on the ordered triangle. That triangle is compact, so a NUTS
  step that moves the edges can send catalog samples outside the support and
  drop their importance weights to zero.
- **Ordered Gaussians** (`mass_model = "gaussian"`). Both components are i.i.d. `Normal(mass_mean, mass_sigma)`, then ordered.
  The joint density `2 N(m1) N(m2)` lives on the half-plane `m1 >= m2`, with
  no compact mass support, so moving `(mass_mean, mass_sigma)` never zeros a
  weight. Galactic BNS masses motivate the shape (a Gaussian around
  `1.33 Msun` with width `~0.09 Msun`). The default catalog still
  uses the uniform triangle.

### Guard mixtures are one density, not two draws

`uniform_mixing_fraction = ε` blends a fraction ε of uniform-in-redshift draws
into the Madau-Dickinson density with `numpyro.distributions.MixtureGeneral`. The
same mixture that draws the redshifts evaluates their log density, so the
recorded guard fraction can never be something other than what was drawn. It
replaced a pair of gwmock graphs differing only in their redshift block, a
weighted `MixtureSimulator`, and a hand-written `logaddexp` mixture density in
the analysis layer.

A guard mixture is a sampling density, not a physical population, and the
Madau-Dickinson total rate is the normalization of the Madau-Dickinson redshift
density, not of a mixture of it with a uniform component. Nothing reads a rate
off a proposal -- importance weighting takes the *target's* -- so this costs
nothing, but a guard mixture used as an injection or an analysis target would
silently pair a rate that does not normalize it. Nothing checks this; it is the
caller's to honour.

### Prefix stability across sizes

Draws of different sizes at one seed are exact nested draws of the *source
parameters*, which makes a catalog-size series a clean sweep rather than
unrelated draws.
`Predictive` allocates its per-draw keys with `jax.random.split`, which is
prefix-stable -- a property of the installed JAX rather than an API promise, so
`tests/core/test_populations.py` checks it directly.

Prefix stability is a property of the population draw, not of the waveforms.
Generation is exactly reproducible at a fixed sample count, but the same source
generated in a batch of 8192 and a batch of 32768 gets polarization power
agreeing only to ~1e-14 relative: XLA picks different reduction orders at
different batch sizes and floating-point addition is not associative. The
differences land on the deep tail of the spectrum -- values some three orders of
magnitude below the array peak -- so they are numerically irrelevant, but the
catalogs are not byte-identical to one another.

## Generate a catalog

From Python, the cached node draws on a miss and reads on a hit:

```python
import numpy as np

from astrogwb.paper.config import fiducials, population_metadata, waveform_metadata
from astrogwb.simulators.polarization_power import (
    CatalogMetadata,
    polarization_power,
)

metadata = CatalogMetadata(
    waveform=waveform_metadata(),
    population=population_metadata(),
    fiducials=fiducials(),
    num_samples=32768,
)
data = polarization_power(
    {"seed": np.uint64(41)}, metadata, cache_dir="outputs/catalogs"
)
```

A hit is read and checked against the metadata it was asked for; a miss is
generated and saved under the name `polarization_power.path` gives, written
atomically so an interrupted job never leaves a partial file.

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
the `metadata` half of the `(data, metadata)` pair -- which a loader checks
against the request it was handed.

The density factors are deliberately *not* part of the record. They change no
sample: the stored columns and power are the same whichever of their densities
a later weight counts, so the choice belongs to the analysis rather than to the
file. What still matters is that one value covers both sides of a ratio — a
proposal density computed with the mass factors excluded, reweighted against a
target that includes them, gives silently wrong weights with no shape error
anywhere — which is why `build_importance_spectrum` takes it once and threads
that one value into both callables it returns.

### The file format

Every simulator's data uses the same layout, through
`astrogwb.simulators.core.write`:

| Where | What |
| --- | --- |
| root attribute `metadata` | the metadata's `model_dump_json()` |
| root attributes `version`, `created`, `host` | which `astrogwb`, when and where |
| optional root attributes `seed`, `batch_size` | the caller's draw settings |
| root datasets and groups | the data: a nested dict maps to subgroups, an array to a dataset |

For a catalog, the root holds `frequencies` `(F,)`, `polarization_power`
`(F, N)` and a `source_parameters/` group of `(N,)` columns. There are no
format names or version files: the path says which node, metadata and inputs a
file answers, and `load(path, MetadataType)` returns `(data, metadata, attrs)` to a
caller that was handed a file by path.
The arrays are used as read (the writer is the only source of files, so they
are not re-validated), and `metadata.population.build()` rebuilds the recorded population from the registry: an
unknown name raises `KeyError` listing what is registered, and a construction
setting the population does not take raises `TypeError`. Bin widths are not
stored; they are derived from `frequencies`.

## The cache

Persistence and cache locations belong to the callers. The core `write` and
`load` functions neither generate draws nor choose paths:

| | simulator | metadata | input | data |
| --- | --- | --- | --- | --- |
| catalogs | `draw_catalog` | `CatalogMetadata` | one JAX key | `PolarizationPowerData` |
| reference catalogs | `reference_catalog` | `CatalogMetadata` | one JAX key | `PolarizationPowerData` |
| spectra | `BackgroundSpectralDensitySimulator` | `BackgroundSpectralDensityMetadata` | one JAX key | `BackgroundSpectralDensityData` |

Catalog files are `polarization_power-<key>-<seed>.h5`; spectra files are
`spectra-<key>-<seed>-<num_draws>.h5`. A miss generates data and writes atomically;
a hit is loaded and its recorded metadata compared with the request.
The SNR notebook supports `cache_only`, which serves hits only. Settings such as
`chunk_size` are bound in the simulator and kept out of the path.

Seeds are inputs because they pick a realization rather than describe a
distribution: `batch_keys(seed, n)` folds the draw indices into a JAX key,
producing prefix-stable draw keys. Build a simulator once and call it for each
key so its compiled stages are reused.

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

The shipped BNS population always samples isotropic inclination:
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

Earlier versions could omit the inclination site and column
(`sample_inclination = false`) and rescale face-on power by the analytic `2/5`
factor. That path is gone: inclination is always sampled, and contraction never
rescales. Catalogs drawn without the column are not reused, since the
population record and the version are part of the key.

Version **0.6.0** replaces the `bns_md_*` populations with `bns_coba`. Both
catalog and spectrum artifacts therefore receive new keys; older artifacts are
not reused. Regenerate spectra with the existing CLI, choosing either the `fixed` or
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

## The population simulator

A spectrum is a population draw reduced through a waveform, and the first half
is a simulator of its own: `astrogwb.simulators.population.PopulationSimulator`.

- **metadata** -- a `PopulationDrawMetadata`: the population, each
  hyperparameter's fixed value or prior, `observation_time`, `count`,
  `num_events` and the version. No waveform, so two waveforms' spectra records
  share one population key (`BackgroundSpectralDensityMetadata.sources.key()`).
- **input** -- one JAX key per draw, from `batch_keys(seed, n)`.
- **outputs** -- one draw per call: `count`, `total_merger_rate` and
  `hyperparameters/<name>` are 0-d and each `source_parameters/<name>` column
  is `(count,)` (`PopulationData`).

Draw once and reduce through several waveforms -- the same events, so the
differences are the waveform's alone:

```python
population = PopulationSimulator(metadata.sources)
simulator_a, simulator_b = BackgroundSpectralDensitySimulator(metadata_a), BackgroundSpectralDensitySimulator(metadata_b)
parts_a, parts_b = [], []
for key in batch_keys(seed, n):
    draw = population(key)
    parts_a.append(simulator_a.reduce(draw))
    parts_b.append(simulator_b.reduce(draw))
spectra_a, spectra_b = stack_spectra(parts_a), stack_spectra(parts_b)
```

Persisting a population pays when it is reused like this; a simulation loop
calls `BackgroundSpectralDensitySimulator(...)(key)` per draw and keeps nothing.

## The background spectral-density simulator

The sibling data is `BackgroundSpectralDensityData`. It holds the forward
model's *contraction* rather than the power it contracts, so a run that only
needs predicted spectra never materializes `(F, N)` waveforms.

It is produced by `astrogwb.simulators.spectra.BackgroundSpectralDensitySimulator`:

- **metadata** -- a `BackgroundSpectralDensityMetadata`: the waveform, the population, each
  hyperparameter's fixed value *or* prior, `observation_time`, `count`,
  `num_events`, and the `astrogwb` version. It extends the waveform-free
  `PopulationDrawMetadata` (see [the population simulator](#the-population-simulator)); its
  `.sources` is that part. No seed and no draw count.
- **input** -- one JAX key, usually from `batch_keys(seed, num_draws)`.
  Each key is one draw
  (hyperparameters and sources alike), so a draw depends on its own seed alone,
  not on its batchmates, and the same key gives the same spectrum in any call.
- **settings** -- `chunk_size` chunks the waveform reduction;
  `source_chunk_size` (default `chunk_size`) sets the size of the pieces
  Poisson-count sources are drawn in. Neither consumes randomness, so neither is in the
  path. Memory is one draw's sources.

```python
from astrogwb.paper.cache import default_cache_dir
from astrogwb.paper.config import fiducials, population_metadata, waveform_metadata
from astrogwb.distributions.config import DistributionConfig
from astrogwb.simulators.core import batch_keys, load, write
from astrogwb.simulators.spectra import (
    BackgroundSpectralDensityMetadata,
    BackgroundSpectralDensitySimulator,
    stack_spectra,
)

metadata = BackgroundSpectralDensityMetadata(
    waveform=waveform_metadata(),
    population=population_metadata(),
    hyperparameters={
        **fiducials(),
        "local_merger_rate": DistributionConfig(
            dist="Normal", kwargs={"loc": 770.0, "scale": 7.7}
        ),
    },
    observation_time=1.0,
    count="poisson",
)
simulator = BackgroundSpectralDensitySimulator(metadata, chunk_size=1024)
seed, num_draws = 41, 64
data = stack_spectra([simulator(key) for key in batch_keys(seed, num_draws)])
path = (
    default_cache_dir() / "spectra"
    / f"spectra-{metadata.key()}-{seed}-{num_draws}.h5"
)
write(path, data, metadata, seed=seed)
data, metadata, attrs = load(path, BackgroundSpectralDensityMetadata)
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
waveforms than the counts need. A simulator call or `.reduce(population)`
returns `BackgroundSpectralDensityData` with one row: `(1, F)` spectral density
and `(1,)` event counts, rates and hyperparameter columns. `stack_spectra`
concatenates a non-empty sequence along that existing draw axis. Its inputs
must share a frequency grid and hyperparameter names; it keeps the first grid
without re-validating the arrays. A single spectrum is `data["spectral_density"][0]`.
`BackgroundSpectralDensitySimulator` owns the built population, generator and
jitted stages, so reusing one instance in a loop reuses its compilations.

`count="poisson"` (the default) draws
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
the key sequence is the number of independent realizations. `observation_time` remains positive
in both modes and is recorded in the key. It controls Poisson counts and
cancels from fixed-count normalization.

The complete normalized metadata is hashed. `n_max_sigma` is gone from the
record, so earlier spectrum cache addresses are unreachable; rerun spectrum
generation under the new keys. There is no key migration. Polarization-power
catalog draws and waveform algorithms are unchanged, so this does not bump the
package version or invalidate polarization-power catalog caches.

`scripts/simulate_spectra.py` runs the same simulator from a shell. It takes
config layers on argv -- the four shared `config/*.toml` layers,
then `config/simulations/spectrum/<name>.toml` -- and validates the merged
`[spectra]` table as the `BackgroundSpectralDensityMetadata`. In that table a hyperparameter is a
`"${fiducials.X}"` reference (fixed) or a `"${priors.X}"` one (sampled). The
sibling `[draws]` table (`seed`, `num_draws`) names the seeds, through
`batch_keys`. The output is `<--output-dir>/spectra-<key>-<seed>-<num_draws>.h5`. The default is
`default_cache_dir() / "spectra"`, shared with the SNR analysis script across
worktrees: `~/Library/Caches/astrogwb/spectra` on macOS or
`~/.cache/astrogwb/spectra` on Linux, with `XDG_CACHE_HOME` taking precedence
on both. `--output-dir` overrides it. Cache locations are outside the
scientific metadata and its key; catalogs live under `outputs/catalogs`. A second invocation with the same layers is a cache hit;
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

Callers
using custom source-model compositions can still build a
`PopulationSimulator` from a registered population and reduce its draws with
`ChunkedPowerSum` (or `BackgroundSpectralDensitySimulator.reduce`). Sampled inclination is already part of the
registered BNS population and needs no custom composition.

The root of a spectra file holds:

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `frequencies` | `(F,)` | the generating backend's grid, as in a power catalog |
| `spectral_density` | `(draws, F)` | one realization of the selected forward model per row |
| `n_events` | `(draws,)` | that draw's Poisson count or the recorded fixed `num_events` |
| `total_merger_rate` | `(draws,)` | that draw's observer-frame total rate |
| `hyperparameters/<name>` | `(draws,)` | the value each row was drawn at, one dataset per name |

The root `metadata` attribute is the complete `BackgroundSpectralDensityMetadata.model_dump_json()`
record; the root `seed` attribute records the seed passed to `batch_keys`.

The two artifacts differ in what a row is, and that is the whole difference. A
power catalog's sample axis indexes *sources* drawn once at one set of
hyperparameters, recorded as `fiducials` in the metadata JSON. A
spectral-density catalog's row axis indexes *draws* of the whole forward model,
so its hyperparameters are a column per name. A fixed hyperparameter's column
must repeat its value; a sampled one's holds each row's draw. In fixed mode,
every `n_events` entry must match the metadata's `num_events`.

Construction guarantees this layout. Consumers use the data dictionary and
metadata separately, without a catalog wrapper or a data-validation layer.
Observation time and population construction come directly from metadata;
draw count is `data["spectral_density"].shape[0]`, and bin widths are derived
from `data["frequencies"]` using `astrogwb.frequency.bin_widths`.

The `BackgroundSpectralDensity*` names and unified single-draw layout preserve
metadata JSON, content keys and existing batch files. The package version
remains unchanged because the scientific draws and waveform values are unchanged.

## What is *not* in the file: the analysis window

The recorded settings are the *generation* window, `[0.0, 20.0]`. The analysis
window is narrower — `analysis.population.model_kwargs.minimum_redshift = 0.3`
in the analysis setup — so the per-sample log density cannot be
baked into the catalog: it depends on a truncation the run chooses, not on
anything generation knows.

`restrict_redshift(data, metadata, minimum_redshift, maximum_redshift)` narrows both halves
together, and that is the whole reason it is one function. Dropping samples without narrowing
the recorded model would leave the density normalized over a window the samples
no longer span, and every importance weight would be off by that
normalization. Draws truncated to a sub-window follow the same law as draws
made directly from it, so only the support changes.
