# Catalog generation

A **catalog** is one persisted waveform draw: a population drawn from a
registered NumPyro model, with its frequency-domain polarization power reduced
and written to `outputs/catalogs/<key>.h5`. Catalogs are the only expensive
artifact in the workflow and the only generated input a run consumes.

There is no separate "bank" any more. Catalogs used to be split in two: banks
were persisted single-component draws, and a catalog was a cheap in-memory
prefix or mixture over them, composed at run time. Storing only polarization
power made every catalog cheap enough to persist, so the composition step and
its plumbing went away.

## Catalogs are content-addressed

A run declares *what* each of its two catalogs draws, and the file is named by
a hash of that declaration:

```text
config/runs/<experiment>/<run>.json [analysis.catalog.<role>]
    -> CatalogRequest  ->  outputs/catalogs/<request.key()>.h5
```

There is no catalog config tree and no name for a catalog. Two runs that ask
for the same draw resolve to the same key and share one file; a run that
changes anything about its draw -- a seed, a kwarg, a fiducial -- gets a new
one. `just catalogs` maps the keys back to what they draw and which runs use
them.

### What a run declares

Each role in `[analysis.catalog]` is a partial spec. `seed` and `num_samples`
belong to the draw and are always stated; everything else is inherited from the
run's own blocks, and a role overrides only what differs:

```json
"catalog": {
  "injection": {"seed": 41, "num_samples": 32768},
  "proposal": {
    "seed": 61,
    "num_samples": 16384,
    "population": {
      "model_name": "bns_md_uniform_mixture",
      "model_kwargs": {"uniform_mixing_fraction": 0.1}
    }
  }
}
```

The inherited blocks are three run layers:

1. `config/waveform.json` — the `[waveform]` settings every catalog shares.
   Only `waveform-approximant/TaylorF2` overrides anything here (the
   approximant). The stored band matches `config/analysis.json`'s
   `minimum_frequency` and `maximum_frequency`: the catalog grid *is* the array
   every model is evaluated on, and a run's band selects bins on it with a mask
   rather than compressing it. `sampling_frequency` is the waveform backend's
   Nyquist, not the stored grid. `approximant="AnalyticInspiral"` selects the
   closed-form inspiral, and is the only approximant accepting the optional
   `alpha` key (the inspiral termination constant, defaulting to the
   Schwarzschild ISCO value); naming it alongside a Ripple approximant is
   rejected. `astrogwb.paper.config.waveform_generator()` builds the same
   generator for a notebook.
2. `config/population.json` — the population a catalog is drawn from unless a
   role overrides it: `model_name`, a key in the `astrogwb.populations`
   registry, and `model_kwargs`, the construction settings bound into it. This
   top-level `[population]` is the *draw* default; the analysis target is
   `analysis.population`, a separate block. It declares no `seed`.
3. `config/fiducials.json` — the hyperparameters the draw is made at: the run's
   own merged `[fiducials]`, so the injection is drawn at exactly the values the
   run initializes at. `time-delay` sets `delay_slope = -1` once, as a run
   fiducial, and its injection inherits it.

Overrides are recursive merges, so a guarded proposal that names another
population keeps the shared redshift window and grid and adds the one setting
it takes. `astrogwb.paper.config.runs.resolve_catalog_blocks` is the one
implementation of that resolution; the workflow, `RunConfig.catalog_request`
and the notebooks all go through it, and a test pins the workflow's keys and
`RunConfig`'s agreeing for every run.

Drawing at the run's fiducials has one visible cost: a run-level fiducial
override also changes the proposal's key, even for a population that never
reads it. `time-delay`'s proposal is therefore its own copy of the eps = 0.1
guard catalog. The draws are identical; only the recorded fiducials differ.

### The key, and what invalidates it

`CatalogRequest` (in `astrogwb.metadata`) is the waveform settings, the
population record with its seed, the fiducials, the sample count, and the
`astrogwb` version. Its `key()` is the first 16 hex digits of a SHA-256 over
its canonical JSON. Anything in the request invalidates the file by renaming
it, so the workflow's catalog rule declares no config inputs at all.

The version is in the key so that *code* changes invalidate catalogs too -- but
only if it is bumped. **Bump `version` in `pyproject.toml` whenever a change
alters what a population draw or a waveform generator produces.** Every key
changes with it, so the next `snakemake catalogs` regenerates everything.
`just catalogs --orphans` lists the files no run asks for any more.

## The population is a registered model

A catalog names a population by its key in the `astrogwb.populations`
registry, and supplies the construction kwargs it takes -- the guarded proposal
above, for example. The redshift window and grid resolution are inherited from
`config/population.json` and the hyperparameters from `config/fiducials.json`;
`model_kwargs` is one mapping, deep-merged across layers and passed whole to the
factory. The population declares its density factors and source outputs.

A role inherits every block it does not name, whether or not the population it
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
from astrogwb.utils.sampling import evaluate_sources, sample_sources

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

# one catalog, by key (`just catalogs` lists them)
snakemake --snakefile Snakefile --cores 1 --allowed-rules waveform_catalog \
  outputs/catalogs/<key>.h5
```

`rule waveform_catalog` hands `scripts/generate_catalog.py` the resolved request
as JSON and the output path. The script refuses a path whose stem is not the
request's key, so one draw cannot be filed under another's address, and writes
atomically so an interrupted job never leaves a partial file.

The `--allowed-rules` filter keeps catalog generation explicit. MCMC commands
omit these rules, so a missing catalog stops the run with a
`MissingInputException` rather than silently scheduling waveform generation.

From Python the same generator sits behind a cache lookup:

```python
from astrogwb.catalog import load_or_generate
from astrogwb.paper.catalogs import run_catalog

# a committed run's catalog: resolved from its config, generated on a miss
proposal = run_catalog("variable-proposal-guard", "eps1e-2", "proposal")

# or any request, against any cache directory
catalog = load_or_generate(request, "outputs/catalogs")
```

A hit is loaded and checked against the request it was asked for; a miss is
generated and saved under the request's key.

## Catalogs record the density that drew them

Each catalog stores the complete population declaration, as HDF5 attributes
written once at generation time:

```text
population_model         = "bns_md_cosmological"
population_model_kwargs  = '{"n_grid": 4096, "maximum_redshift": 20.0, "minimum_redshift": 0.0}'
population_params        = '{"H0": 67.66, "Omega_m": 0.3096, "gamma": 1.42,
                             "kappa": 4.62, "local_merger_rate": 770.0,
                             "z_peak": 1.84}'
population_seed          = 41
population_num_samples   = 32768
```

HDF5 attributes are flat scalars, so the mappings travel as JSON strings. The
`population_*` block is one `PopulationMetadata`, the same record a
spectral-density catalog carries, which is what keeps the two formats spelling
these fields identically.

What is *not* stored is a callable: `PolarizationPowerCatalog.get_population()`
looks the name up in the registry and binds the recorded settings, returning
both callables at once (they hash by identity, so two getters would force a
recompile on every call).

That is enough to reconstruct the exact map from hyperparameters to source
density. With the version, it is also enough to reconstruct the request the
file answers, which is what `run_mcmc` checks each catalog against before
sampling. Before this, three partial records described one run —
the merged run TOML, a catalog attribute naming only the *shape* of the
redshift proposal, and a config object derived from those two — reconciled by
exact float equality over five hard-coded parameter names. Three more
parameters that change the answer (`xi_0`, `xi_n`, `local_merger_rate`) were
checked by nothing at all.

The density factors are deliberately *not* part of the record. They change no
sample: the stored columns and power are the same whichever of their densities
a later weight counts, so the choice belongs to the analysis rather than to the
file. What still matters is that one value covers both sides of a ratio — a
proposal density computed with the mass factors excluded, reweighted against a
target that includes them, gives silently wrong weights with no shape error
anywhere — which is why `build_importance_spectrum` takes it once and threads
that one value into both callables it returns. Catalogs written before this
moved still carry a `population_density_sites` attribute; the reader names the
attributes it wants, so it is simply not read.

### What loading checks

`PolarizationPowerCatalog.load` validates the HDF5 layout, array shapes and
serialized dtypes, then reconstructs the recorded population from the registry.
Reconstruction is the whole check: an unknown name raises `KeyError` listing
what is registered, and a construction setting the population does not take
raises `TypeError`. It does not serialize a callable or require the analysis
run configuration.

The format is `astrogwb_catalog_v8`, a direct HDF5 file. Root attributes hold
the waveform and population metadata (JSON is used for mappings and ordered
lists) and `astrogwb_version`, the package version that generated the arrays;
`frequency`, `polarization_power`, and `source_parameters` are HDF5
datasets. Earlier formats require regeneration.

## The catalog cache

A `CatalogRequest` (`astrogwb.metadata`) is everything that determines a
catalog: the waveform settings, the population record with its seed, the
hyperparameters and size of the draw, and the `astrogwb` version. Its `key()`
is a 16-hex-digit SHA-256 of the record's canonical JSON, and
`astrogwb.catalog.load_or_generate(request, cache_dir)` keeps each catalog at
`<cache_dir>/<key>.h5`: a miss generates and writes atomically, a hit is loaded
and checked against the request it was asked for.

The version is in the key so that code changes invalidate the cache -- but
only if it is bumped. Bump `version` in `pyproject.toml` whenever a change
alters what a population draw or a waveform generator produces. The recorded density-site and
source-parameter order is preserved on load and when narrowing the redshift
window.

The waveform attributes record `frequency_resolution` -- what was *requested*
of the generating backend -- while the bin width used in every integral is
measured from the `frequency` dataset itself (`PolarizationPowerCatalog.df`);
the backend chooses the actual grid, so the two can differ.

## The spectral-density format

`scripts/simulate_spectra.py` writes the sibling artifact: a
`SpectralDensityCatalog`, format `astrogwb_spectral_density_v3`. It persists
the forward model's *contraction* rather than the power it contracts, so a
run that only needs predicted spectra never materializes `(F, N)` waveforms.

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `frequency` | `(F,)` | the generating backend's grid, as in a power catalog |
| `spectral_density` | `(draws, F)` | one `gwb_forward_model` draw per row |
| `n_events` | `(draws,)` | that draw's Poisson event count |
| `total_merger_rate` | `(draws,)` | that draw's observer-frame total rate |
| `hyperparameters` | `(draws, P)` | one column per name, ordered by `source_parameter_names` |

Its root attributes are the same three blocks a power catalog stamps -- format
identity, the six waveform attributes, and the `PopulationMetadata` -- plus three
the draws cannot be read back from:

```
n_max_sigma      = 5.0   # sized the static plate the Poisson count was capped against
observation_time = 1.0   # years; set the Poisson mean
```

The two formats differ in what a row is, and that is the whole difference. A
power catalog's sample axis indexes *sources* drawn once at one set of
hyperparameters, recorded as scalar `population_params` fiducials. A
spectral-density catalog's row axis indexes *draws* of the whole forward model,
so its hyperparameters are a column per name -- free to vary from row to row,
even though the simulator holds them fixed today.

`SpectralDensityCatalog.load` validates the same way its sibling does: layout,
shapes, serialized dtypes, then reconstruction of the recorded population from
the registry. Earlier formats require regeneration.

## What is *not* in the file: the analysis window

The recorded settings are the *generation* window, `[0.0, 20.0]`. The analysis
window is narrower — `analysis.population.model_kwargs.minimum_redshift = 0.3`
in `config/analysis.json` — so the per-sample log density cannot be
baked into the catalog: it depends on a truncation the run chooses, not on
anything generation knows.

`PolarizationPowerCatalog.restrict_redshift(minimum_redshift, maximum_redshift)` narrows both halves
together, and that is the whole reason it is one method. Dropping samples without narrowing
the recorded model would leave the density normalized over a window the samples
no longer span, and every importance weight would be off by that
normalization. Draws truncated to a sub-window follow the same law as draws
made directly from it, so only the support changes.
