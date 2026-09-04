# Catalog generation

A **catalog** is one persisted waveform draw: a population drawn from one or
more graphs, with its frequency-domain polarization power reduced and written
to `outputs/catalogs/<name>.h5`. Catalogs are the only expensive artifact in
the workflow and the only generated input a run consumes.

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

A catalog config is two layers merged in order, the same shape as a run config:

1. `config/catalogs/base/*.toml` — the `[waveform]` block every catalog shares.
2. `config/catalogs/defs/<name>.toml` — the draw, and any waveform override.

The base layer is declared as a workflow input of every catalog, so editing it
correctly invalidates all of them.

The eight committed catalogs:

| Catalog | Components | Samples | Approximant |
| --- | --- | ---: | --- |
| `md-imrphenom-s41-n32768` | `madau-dickinson` @ s41 | 32768 | `IMRPhenomXAS_NRTidalv3` |
| `md-imrphenom-s42-n8192` | `madau-dickinson` @ s42 | 8192 | `IMRPhenomXAS_NRTidalv3` |
| `md-imrphenom-s42-n16384` | `madau-dickinson` @ s42 | 16384 | `IMRPhenomXAS_NRTidalv3` |
| `md-imrphenom-s42-n32768` | `madau-dickinson` @ s42 | 32768 | `IMRPhenomXAS_NRTidalv3` |
| `md-taylorf2-s41-n32768` | `madau-dickinson` @ s41 | 32768 | `TaylorF2` |
| `md-uniform-imrphenom-s61-n16384-eps1e-1` | MD @ s42 × 0.9 + uniform @ s51 × 0.1 | 16384 | `IMRPhenomXAS_NRTidalv3` |
| `md-uniform-imrphenom-s62-n16384-eps1e-2` | MD @ s42 × 0.99 + uniform @ s51 × 0.01 | 16384 | `IMRPhenomXAS_NRTidalv3` |
| `md-uniform-imrphenom-s63-n16384-eps1e-3` | MD @ s42 × 0.999 + uniform @ s51 × 0.001 | 16384 | `IMRPhenomXAS_NRTidalv3` |

Eight, not nine: `md-imrphenom-s41-n32768` serves as both the shared injection
and `waveform-approximant/IMRPhenom`'s proposal, and the eps = 0.1 guard
catalog serves both `astrophysical-parameters` runs and
`variable-proposal-guard/eps1e-1`.

`seed` and `num_samples` are catalog-level, not population-level: `s41` and
`s42` are the *same* graph drawn twice, so pushing either into the population
config would mean near-identical graph files.

## The draw: one component or several

A def declares its components as an array of tables:

```toml
num_samples = 16384
mixture_seed = 61

[[components]]
population = "madau-dickinson"
seed = 42
weight = 0.9

[[components]]
population = "uniform-redshift"
seed = 51
weight = 0.1
```

`weight` defaults to `1.0`, so a single-component def is four lines and carries
no `mixture_seed` — a lone component has no assignment draw to seed.

**One component draws straight through `GraphSimulator`; two or more go through
`MixtureSimulator`.** That dispatch is load-bearing, not an optimization.
`MixtureSimulator` splits its seed to draw component assignments, so routing a
lone graph through it would change that graph's RNG stream. Keeping the direct
path is what makes single-component catalogs prefix-stable across sizes: the
three `md-imrphenom-s42-n*` catalogs are generated independently and their
*source parameters* are still exact nested draws, which is what makes
`variable-catalog-size` a clean series rather than three unrelated runs.

Prefix stability is a property of the population draw, not of the waveforms.
Generation is exactly reproducible at a fixed sample count, but the same source
generated in a batch of 8192 and a batch of 32768 gets polarization power
agreeing only to ~1e-14 relative: XLA picks different reduction orders at
different batch sizes and floating-point addition is not associative. The
differences land on the deep tail of the spectrum -- values some three orders
of magnitude below the array peak -- so they are numerically irrelevant, but
the catalogs are not byte-identical to one another.

Each component keeps its own seed because a component's draw comes from its
*construction* seed: `MixtureSimulator` passes each component a derived `seed`
keyword, and `GraphSimulator._simulate_impl` discards it (`del kwargs`). The
`mixture_seed` and the component seeds are nonetheless independent, because
`MixtureSimulator` splits its key before drawing assignments — so unlike the
old in-memory composition, they need not be distinct.

## Population graphs

[`config/populations/`](../config/populations/) holds two complete, standalone
graphs — [`madau-dickinson.yaml`](../config/populations/madau-dickinson.yaml)
and [`uniform-redshift.yaml`](../config/populations/uniform-redshift.yaml).
There is no base/overlay split and no merge rule: the ~80 duplicated lines are
visible and diffable, and there are only two files.

They differ only in `parameters.redshift`. Keep the parameter *order* in sync
between them as well: `GraphSimulator` draws RNG keys in topological order,
which breaks ties by declaration order, so reordering a block silently changes
every generated catalog.

## Generate a catalog

One command does the whole thing — population draw, waveform generation, power
reduction, write. The population never lands on disk; it was previously a
`temp()` node with exactly one consumer.

```bash
uv run --extra paper python scripts/generate_catalog.py \
  --config config/catalogs/base/waveform.toml \
  --config config/catalogs/defs/md-imrphenom-s41-n32768.toml \
  --output outputs/catalogs/md-imrphenom-s41-n32768.h5
```

Layers arrive as repeated `--config` flags, in merge order, and the last one's
filename stem names the catalog. It refuses to overwrite an existing catalog
unless `--force` is passed. Source-frame masses are converted to detector-frame
masses by multiplying by `1 + z` immediately before waveform generation.

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

## Catalogs record their own provenance

Each catalog stores how it was made, as HDF5 attributes written once at
generation time:

```text
population_name        = "md-imrphenom-s41-n32768"
population_seed        = 41
population_num_samples = 32768
redshift_proposal      = '{"components": [{"weight": 1.0, "density":
                           {"kind": "madau_dickinson", "z_min": 0.0,
                            "z_max": 20.0, "gamma": 1.42, "kappa": 4.62,
                            "z_peak": 1.84, "H0": 67.66, "Omega_m": 0.3096,
                            "n_grid": 4096}}]}'
```

netCDF attributes are flat scalars, so the descriptor travels as one JSON
string. It is always a *mixture*, even for a single-component catalog, so the
analysis reads one shape regardless of how the catalog was built; a guard
catalog records two components with their normalized mixing fractions.

This is the density the importance weights divide by, and it is what lets the
run config drop `uniform_mixing_fraction` entirely — the catalog owns it. It is
extracted from the population graphs exactly once, by
`astrogwb.paper.config.catalogs.extract_mixture_proposal`, and every later
consumer reads it back with `CatalogProvenance.from_file` instead of re-parsing
a config that may have drifted since. `scripts/run_mcmc.py` also checks the
run's `[fiducials]` against what the catalog recorded, so a drifted fiducial
fails before a device is claimed rather than silently reweighting against the
wrong denominator.

A file written before this metadata existed carries no readable
`redshift_proposal` attribute. Reading it fails with *"generated before proposal
metadata; regenerate it"* — deliberately, rather than falling back to parsing.

## What is *not* in the file: the analysis window

The recorded descriptor is the *generation* density, over `[0.0, 20.0]`. The
analysis window is narrower — `minimum_redshift = 0.3` in
`config/analysis/base/model.toml` — and `compute_proposal_logprob` renormalizes
both the uniform component and the Madau-Dickinson grid to *that* window.

So the per-sample log-density cannot be baked into the catalog: it depends on a
truncation the run chooses, not on anything generation knows. The file records
the descriptor; `astrogwb.paper.config.catalogs.resolve_proposal` narrows it
with the run's own window at load time, before JAX claims a device.
