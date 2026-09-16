# Catalog generation

A **catalog** is one persisted waveform draw: a population drawn from a
registered NumPyro model, with its frequency-domain polarization power reduced
and written to `outputs/catalogs/<name>.h5`. Catalogs are the only expensive
artifact in the workflow and the only generated input a run consumes.

There is no separate "bank" any more. Catalogs used to be split in two: banks
were persisted single-component draws, and a catalog was a cheap in-memory
prefix or mixture over them, composed at run time from a five-field spec each
run declared inline. Storing only polarization power made every catalog cheap
enough to persist, so the composition step, the inline spec, and the
`--bank NAME=PATH` plumbing all went away. A run names two catalogs; each
catalog is a file.

## Filenames are the mapping

```text
config/catalogs/<name>.json  ->  outputs/catalogs/<name>.h5
```

No registry translates between them. Adding a catalog means adding a JSON file;
the `waveform_catalog` rule and the `catalogs` target pick it up by globbing.

A catalog config is four layers merged in order -- three shared files, then one
file per named thing, the same shape a run config has. The shared three sit in
`config/` beside the run tables rather than in a `base/` subdirectory, and they
are named rather than globbed, because `config/priors.json`,
`config/networks.json` and `config/plotting.json` are in that directory too and
must not enter a catalog merge:

1. `config/waveform.json` — the `[waveform]` block every catalog shares.
   A catalog layer, not a run layer. Only
   `config/catalogs/md-taylorf2-s41-n32768.json` overrides anything here
   (the approximant). The stored band matches
   `config/analysis/base/model.toml`'s `[analysis]` `f_min` and `f_max`: the
   catalog grid *is* the array every model is evaluated on, and a run's band
   selects bins on it with a mask rather than compressing it.
   `sampling_frequency` is the waveform backend's Nyquist, not the stored grid.
   `approximant="AnalyticInspiral"` selects the closed-form inspiral, and is the only
   approximant accepting the optional `alpha` key (the inspiral termination
   constant, defaulting to the Schwarzschild ISCO value); naming it alongside a
   Ripple approximant is rejected. `WaveformMetadata.build()` constructs the
   generator, and `astrogwb.paper.config.waveform_generator()` is the same path
   for a notebook reading this file directly.
2. `config/population.json` — the population every catalog is drawn from:
   `model_name`, a key in the `astrogwb.populations` registry, and
   `model_kwargs`, the construction settings bound into it. Those two keys are
   exactly `PopulationMetadata`'s configurable half, so the block validates
   straight into the record the `.h5` persists. It declares no `seed` and no
   density sites: a seed belongs to a particular draw, and the density factors
   follow from the registered population rather than from a file.
3. `config/fiducials.json` — the hyperparameters the draw is made at. The same
   table the runs initialize at, stated once. It used to be restated as a
   `[population.params]` block here, which was an exact copy of the eight
   fiducials a source model reads; the copy is gone. The propagation entries
   `xi_0` / `xi_n` ride along and are inert during generation, because a source
   model indexes `params` by name and the GR population never reads them —
   `xi_0 = 1.0` is the value the injection is drawn at, so recording it is
   accurate rather than misleading.
4. `config/catalogs/<name>.json` — the seed, the sample count, and any
   population or waveform override. Nothing else: `CatalogDefinition` is
   `extra="forbid"`, so what each committed catalog is *for* is documented in
   [`config/catalogs/README.md`](../config/catalogs/README.md) rather than in a
   field that would rot separately from it.

The cost of sourcing the hyperparameters from `config/fiducials.json` is that
editing *any* fiducial now invalidates all eight catalogs — GPU jobs — including
an edit to the analysis-only `xi_0` / `xi_n`. That is the price of the two
tables being one: while they were separate, keeping them in step was a manual
two-file discipline that nothing checked. A def that wants an injection away
from the fiducials overrides the `[fiducials]` block like any other layer, so
the freedom the separate table provided is preserved rather than lost.

All three shared layers are declared as workflow inputs of every catalog, so
editing any of them correctly invalidates all of them.

The eight committed catalogs — what each is for, and which experiment needs it
— are listed in [`config/catalogs/README.md`](../config/catalogs/README.md),
next to the files themselves. There is one inventory, not two.

`seed` and `num_samples` are catalog-level, not population-level: `s41` and
`s42` are the *same* population drawn twice, so pushing either into the shared
population layer would mean near-identical layer files. `CatalogDefinition`
folds the seed into the population record during validation, which is where it
belongs once a particular draw exists.

## The population is a registered model

A def names a population by its key in the `astrogwb.populations` registry, and
supplies the construction kwargs it takes:

```json
{
  "num_samples": 16384,
  "seed": 61,
  "population": {
    "model_name": "bns_md_uniform_mixture",
    "model_kwargs": {
      "uniform_mixing_fraction": 0.1
    }
  }
}
```

The redshift window and grid resolution are inherited from
`config/population.json` and the hyperparameters from `config/fiducials.json`;
`model_kwargs` is one mapping, deep-merged across layers and passed whole to the
factory. The population declares its density factors and source outputs.

A def inherits every block it does not name, whether or not the population it
names reads all of it: this one inherits `[fiducials]` whole,
`local_merger_rate` included, even though `bns_md_uniform_mixture` declares no
merger rate. [`config/catalogs/README.md`](../config/catalogs/README.md) says
why that is right for a guard mixture.

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
- `density_sites` selects the density factors included in importance weighting.
  Generation records `DEFAULT_DENSITY_SITES` — redshift and the ordered mass
  pair — and the catalog's recorded tuple is the one both sides of every weight
  are evaluated with. Omitting factors does not marginalize variables.
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

The three `md-imrphenom-s42-n*` catalogs are generated independently and their
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

One command does the whole thing — population draw, waveform generation, power
reduction, write.

Every layer is JSON, so the fold is one `jq` pass into a merged file, and the
generator is handed blocks read out of it rather than a list of paths:

```bash
layers="config/waveform.json config/population.json config/fiducials.json \
  config/catalogs/md-imrphenom-s41-n32768.json"
merged=outputs/catalogs/md-imrphenom-s41-n32768.merged.json

jq -s 'reduce .[] as $layer ({}; . * $layer)' $layers > "$merged"

uv run --extra paper python scripts/generate_catalog.py \
  --name md-imrphenom-s41-n32768 \
  --population "$(jq -c .population "$merged")" \
  --fiducials "$(jq -c .fiducials "$merged")" \
  --waveform "$(jq -c .waveform "$merged")" \
  --seed "$(jq -r .seed "$merged")" \
  --num-samples "$(jq -r .num_samples "$merged")" \
  --output outputs/catalogs/md-imrphenom-s41-n32768.h5
```

`jq`'s `*` is a recursive merge, which is `astrogwb.paper.utils.deep_merge`
exactly; the catalog layers carry no `[priors]` block, so the shallow-merge rule
the run path needs never applies here. `tests/paper/test_runs.py` pins the two
merges agreeing.

The workflow splits those two commands into two rules. `merge_catalog_config`
owns the fold and declares the layer files; `waveform_catalog` reads five keys
out of its `temp()` output. Folding once and reading keys, rather than folding
per flag, is the whole reason the merged file exists — it is a build
intermediate, not an artifact, and the `.h5` records its own provenance either
way. `waveform_catalog` refuses to overwrite an existing catalog unless
`--force` is passed.

Through the workflow, from the repository root:

```bash
# every catalog
snakemake --snakefile Snakefile --cores 1 \
  --allowed-rules waveform_catalog catalogs --dry-run catalogs

# one catalog
snakemake --snakefile Snakefile --cores 1 --allowed-rules waveform_catalog \
  outputs/catalogs/md-taylorf2-s41-n32768.h5
```

The `--allowed-rules` filter keeps catalog generation explicit. MCMC commands
omit these rules, so a missing catalog stops the run with a
`MissingInputException` rather than silently scheduling hours of waveform
generation.

## Catalogs record the density that drew them

Each catalog stores the complete population declaration, as HDF5 attributes
written once at generation time:

```text
population_model         = "bns_md_cosmological"
population_model_kwargs  = '{"n_grid": 4096, "maximum_redshift": 20.0, "minimum_redshift": 0.0}'
population_params        = '{"H0": 67.66, "Omega_m": 0.3096, "gamma": 1.42,
                             "kappa": 4.62, "local_merger_rate": 770.0,
                             "z_peak": 1.84}'
population_density_sites = '["redshift"]'
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
recompile on every call); `PolarizationPowerCatalog.density_sites` carries the
ordered density selection.

That is enough to reconstruct the exact map from hyperparameters to source
density, which is why the run config no longer restates any of it and nothing
has to be cross-checked. Before this, three partial records described one run —
the merged run TOML, a catalog attribute naming only the *shape* of the
redshift proposal, and a config object derived from those two — reconciled by
exact float equality over five hard-coded parameter names. Three more
parameters that change the answer (`xi_0`, `xi_n`, `local_merger_rate`) were
checked by nothing at all.

`population_density_sites` is part of the record for a reason that is easy to
miss: a catalog whose proposal density was computed with the mass factors
excluded, reweighted against a target that includes them, gives silently wrong
weights with no shape error anywhere.

### What loading checks

`PolarizationPowerCatalog.load` validates the HDF5 layout, array shapes and
serialized dtypes, then reconstructs the recorded population from the registry.
Reconstruction is the whole check: an unknown name raises `KeyError` listing
what is registered, and a construction setting the population does not take
raises `TypeError`. It does not serialize a callable or require the analysis
run configuration.

The format is `astrogwb_catalog_v7`, a direct HDF5 file. Root attributes hold
the waveform and population metadata (JSON is used for mappings and ordered
lists); `frequency`, `polarization_power`, and `source_parameters` are HDF5
datasets. Earlier formats require regeneration. The recorded density-site and
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
window is narrower — `minimum_redshift = 0.3` in
`config/analysis/base/model.toml` — so the per-sample log density cannot be
baked into the catalog: it depends on a truncation the run chooses, not on
anything generation knows.

`PolarizationPowerCatalog.restrict_redshift(minimum_redshift, maximum_redshift)` narrows both halves
together, and that is the whole reason it is one method. Dropping samples without narrowing
the recorded model would leave the density normalized over a window the samples
no longer span, and every importance weight would be off by that
normalization. Draws truncated to a sub-window follow the same law as draws
made directly from it, so only the support changes.
