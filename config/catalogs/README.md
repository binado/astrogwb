# Committed catalogs

One file per catalog, and the filename is the mapping:

```text
config/catalogs/<name>.json  ->  outputs/catalogs/<name>.h5
```

A def carries only what distinguishes this catalog: `seed`, `num_samples`, and
any `population` or `waveform` override. Everything else is inherited from
`config/waveform.json`, `config/population.json` and `config/fiducials.json`.
`CatalogDefinition` is `extra="forbid"`, so a def cannot carry a `description`
key — what a catalog is *for* is documented here, next to the files, rather
than as a field that duplicates this page and rots separately from it.

See [`docs/catalog-generation.md`](../../docs/catalog-generation.md) for the
layer model, the registry, and how to generate one.

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
serves both `astrophysical-parameters` runs and `variable-proposal-guard/eps1e-1`.

## What each one is for

### `md-imrphenom-s41-n32768`

The fiducial injection: the "observed" catalog all 26 runs are compared
against. Also serves as the proposal for `waveform-approximant/IMRPhenom`,
which measures IMRPhenom against itself as the systematics baseline.

### `md-imrphenom-s42-n8192`, `md-imrphenom-s42-n16384`, `md-imrphenom-s42-n32768`

One seed-42 draw at three sizes, which is the `variable-catalog-size`
experiment. The middle one is also the default proposal, used by 21 of the 26
runs; the other two exist only for that experiment's other two arms.

`Predictive` allocates its per-draw keys with `jax.random.split`, which is
prefix-stable, so each of these is an exact prefix of the next: the three sizes
are nested draws, not three unrelated ones, which is what makes the experiment a
clean series. That is a property of the installed JAX rather than something the
API promises, so `test_a_smaller_catalog_is_a_prefix_of_a_larger_one` in
`tests/core/test_populations.py` checks it rather than assuming it.

### `md-taylorf2-s41-n32768`

The waveform-systematics proposal: the same seed-41 population as the
injection, generated with a different approximant. Only `waveform.approximant`
differs from `md-imrphenom-s41-n32768`, so the comparison isolates the
waveform.

### `md-uniform-imrphenom-s6{1,2,3}-n16384-eps1e-{1,2,3}`

Guarded proposals at ε = 1e-1, 1e-2 and 1e-3: the Madau-Dickinson redshift law
with a uniform-in-redshift component mixed in, so the importance weights do not
degenerate when NUTS moves the posterior away from the proposal.

The mixture is one density, not two draws blended after the fact: the same
`MixtureGeneral` that draws the redshifts evaluates their log density, so the
recorded guard fraction can never be something other than what was drawn. That
replaced a pair of gwmock graphs whose only difference was the redshift block,
plus a hand-written mixture log-density in the analysis layer.

Each inherits `[fiducials]` whole, `local_merger_rate` included, even though
`bns_md_uniform_mixture` declares no merger rate. A guard mixture is a sampling
density, not a physical population: the Madau-Dickinson total rate normalizes
the Madau-Dickinson redshift density, not a mixture of it with a uniform
component. Nothing reads a rate off a proposal — importance weighting takes the
target's — and a catalog drawn from this population fails by name if it is used
as an injection.

## Adding one

Add a JSON file; the `waveform_catalog` rule and the `catalogs` target pick it
up by globbing, and its stem becomes the output path. Then add a row above and
a section saying what it is for. `snakemake validate` checks the population
name and its construction kwargs before any GPU job is queued.
