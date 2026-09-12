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
config/catalogs/defs/<name>.toml  ->  outputs/catalogs/<name>.h5
```

No registry translates between them. Adding a catalog means adding a TOML file;
the `waveform_catalog` rule and the `catalogs` target pick it up by globbing.

A catalog config is three layers merged in order, the same shape as a run
config:

1. `config/catalogs/base/population.toml` — the population every catalog is
   drawn from, and the hyperparameters it is drawn at.
2. `config/catalogs/base/waveform.toml` — the `[waveform]` block every catalog
   shares.
3. `config/catalogs/defs/<name>.toml` — the seed, the sample count, and any
   population or waveform override.

Both base layers are declared as workflow inputs of every catalog, so editing
either correctly invalidates all of them.

The eight committed catalogs:

| Catalog | Population | Seed | Samples | Approximant |
| --- | --- | ---: | ---: | --- |
| `md-imrphenom-s41-n32768` | `bns_md_cosmological` | 41 | 32768 | `IMRPhenomXAS_NRTidalv3` |
| `md-imrphenom-s42-n8192` | `bns_md_cosmological` | 42 | 8192 | `IMRPhenomXAS_NRTidalv3` |
| `md-imrphenom-s42-n16384` | `bns_md_cosmological` | 42 | 16384 | `IMRPhenomXAS_NRTidalv3` |
| `md-imrphenom-s42-n32768` | `bns_md_cosmological` | 42 | 32768 | `IMRPhenomXAS_NRTidalv3` |
| `md-taylorf2-s41-n32768` | `bns_md_cosmological` | 41 | 32768 | `TaylorF2` |
| `md-uniform-imrphenom-s61-n16384-eps1e-1` | `bns_md_uniform_mixture` (ε = 0.1) | 61 | 16384 | `IMRPhenomXAS_NRTidalv3` |
| `md-uniform-imrphenom-s62-n16384-eps1e-2` | `bns_md_uniform_mixture` (ε = 0.01) | 62 | 16384 | `IMRPhenomXAS_NRTidalv3` |
| `md-uniform-imrphenom-s63-n16384-eps1e-3` | `bns_md_uniform_mixture` (ε = 0.001) | 63 | 16384 | `IMRPhenomXAS_NRTidalv3` |

Eight, not nine: `md-imrphenom-s41-n32768` serves as both the shared injection
and `waveform-approximant/IMRPhenom`'s proposal, and the ε = 0.1 guard catalog
serves both `astrophysical-parameters` runs and
`variable-proposal-guard/eps1e-1`.

`seed` and `num_samples` are catalog-level, not population-level: `s41` and
`s42` are the *same* population drawn twice, so pushing either into the shared
population layer would mean near-identical layer files.

## The population is a registered model

A def names a population by its key in the `astrogwb.populations` registry, and
supplies the construction settings it takes:

```toml
num_samples = 16384
seed = 61

[population]
model = "bns_md_uniform_mixture"

[population.kwargs]
uniform_mixing_fraction = 0.1
```

The redshift window, grid resolution, and hyperparameters are inherited from
`config/catalogs/base/population.toml`. The population class declares its density
factors and source outputs.

**A registry key, not an import path.** Registry keys change only on purpose;
module paths move as collateral whenever a module is reorganized, so a
persisted `module:function` string is a reference that silently rots. An
unknown key fails pre-flight, in `snakemake validate`, listing what is
registered — before a GPU job is queued.

Source models are plain registered NumPyro functions. `build_source_model`
binds their construction settings into a `functools.partial`; hyperparameters
and source arrays remain arguments. The merger rate is a separate registered
function, bound the same way from the same flat settings:

```python
from astrogwb.populations import (
    DEFAULT_DENSITY_SITES,
    build_merger_rate_fn,
    build_source_model,
)
from astrogwb.utils.sampling import evaluate_sources, sample_sources

settings = {"z_min": 0.0, "z_max": 20.0, "n_grid": 4096}
source_model = build_source_model("bns_md_cosmological", settings=settings)
merger_rate_fn = build_merger_rate_fn(settings=settings)

sources = sample_sources(source_model, key, params, num_samples=1024)
log_prob, outputs = evaluate_sources(
    source_model, params, sources, density_sites=DEFAULT_DENSITY_SITES
)  # log_prob: shape (1024,); outputs["luminosity_distance"]: shape (1024,)
total_merger_rate = merger_rate_fn(params)  # shape ()
```

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

```bash
uv run --extra paper python scripts/generate_catalog.py \
  --config config/catalogs/base/population.toml \
  --config config/catalogs/base/waveform.toml \
  --config config/catalogs/defs/md-imrphenom-s41-n32768.toml \
  --output outputs/catalogs/md-imrphenom-s41-n32768.h5
```

Layers arrive as repeated `--config` flags, in merge order, and the last one's
filename stem names the catalog. It refuses to overwrite an existing catalog
unless `--force` is passed.

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
population_model_kwargs  = '{"n_grid": 4096, "z_max": 20.0, "z_min": 0.0}'
population_params        = '{"H0": 67.66, "Omega_m": 0.3096, "gamma": 1.42,
                             "kappa": 4.62, "local_merger_rate": 770.0,
                             "z_peak": 1.84}'
population_density_sites = '["redshift"]'
population_seed          = 41
population_num_samples   = 32768
```

netCDF attributes are flat scalars, so the mappings travel as JSON strings.
What is *not* stored is a callable: `Catalog.get_source_model()` and
`Catalog.get_merger_rate_fn()` look the names up in the registries and bind
their recorded settings; `Catalog.density_sites` carries the ordered density
selection.

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

`Catalog.load` validates the HDF5 layout, array shapes and serialized dtypes,
then reconstructs the recorded population from the registry. It does not
serialize a callable or require the analysis run configuration.

The format is `astrogwb_catalog_v5`, a direct HDF5 file. Root attributes hold
the waveform and population metadata (JSON is used for mappings and ordered
lists); `frequency`, `polarization_power`, and `source_parameters` are HDF5
datasets. Earlier formats require regeneration. The recorded density-site and
source-parameter order is preserved on load and when narrowing the redshift
window.

The waveform attributes record `frequency_resolution` -- what was *requested*
of the generating backend -- while the bin width used in every integral is
measured from the `frequency` dataset itself (`Catalog.df`); the backend
chooses the actual grid, so the two can differ.

## What is *not* in the file: the analysis window

The recorded settings are the *generation* window, `[0.0, 20.0]`. The analysis
window is narrower — `minimum_redshift = 0.3` in
`config/analysis/base/model.toml` — so the per-sample log density cannot be
baked into the catalog: it depends on a truncation the run chooses, not on
anything generation knows.

`Catalog.restrict_redshift(z_min, z_max)` narrows both halves together, and
that is the whole reason it is one method. Dropping samples without narrowing
the recorded model would leave the density normalized over a window the samples
no longer span, and every importance weight would be off by that
normalization. Draws truncated to a sub-window follow the same law as draws
made directly from it, so only the support changes.
