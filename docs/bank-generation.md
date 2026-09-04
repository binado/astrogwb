# Bank generation

A **bank** is one persisted, single-component waveform catalog: a population
drawn from one graph at one seed, with its frequency-domain polarization power
reduced and written to `outputs/banks/<bank>.h5`. Banks are the only expensive
artifact in the workflow and the only generated input a run consumes.

A **catalog** is no longer generated. It is a cheap, in-memory draw over one or
two banks, composed at run time from the `[catalog.injection]` and
`[catalog.proposal]` tables of a run config. Nothing writes one to disk.

## Filenames are the mapping

```text
config/banks/<bank>.toml  ->  outputs/banks/<bank>.h5
```

No registry translates between them. Adding a bank means adding a TOML file;
the `waveform_bank` rule and the `banks` target pick it up by globbing.

The four committed banks:

| Bank | Population | Seed | Samples | Approximant |
| --- | --- | ---: | ---: | --- |
| `md-imrphenom-s41` | `madau-dickinson` | 41 | 32768 | `IMRPhenomXAS_NRTidalv3` |
| `md-imrphenom-s42` | `madau-dickinson` | 42 | 32768 | `IMRPhenomXAS_NRTidalv3` |
| `md-taylorf2-s41` | `madau-dickinson` | 41 | 32768 | `TaylorF2` |
| `uniform-imrphenom-s51` | `uniform-redshift` | 51 | 8192 | `IMRPhenomXAS_NRTidalv3` |

`seed` and `num_samples` are bank-level, not population-level: `s41` and `s42`
are the *same* graph drawn twice, so pushing either into the population config
would mean two near-identical graph files.

## Population graphs

[`config/populations/`](../config/populations/) holds two complete, standalone
graphs -- [`madau-dickinson.yaml`](../config/populations/madau-dickinson.yaml)
and [`uniform-redshift.yaml`](../config/populations/uniform-redshift.yaml).
There is no base/overlay split and no merge rule: the ~80 duplicated lines are
visible and diffable, and there are only two files.

They differ only in `parameters.redshift`. Keep the parameter *order* in sync
between them as well: `GraphSimulator` draws RNG keys in topological order,
which breaks ties by declaration order, so reordering a block silently changes
every generated bank.

## Generate a bank

One command does the whole thing -- population draw, waveform generation, power
reduction, write. The population never lands on disk; it was previously a
`temp()` node with exactly one consumer.

```bash
uv run --extra paper python scripts/generate_bank.py \
  --config config/banks/md-imrphenom-s41.toml \
  --output outputs/banks/md-imrphenom-s41.h5
```

It refuses to overwrite an existing bank unless `--force` is passed.
Source-frame masses are converted to detector-frame masses by multiplying by
`1 + z` immediately before waveform generation.

Through the workflow, from the repository root:

```bash
# every bank
snakemake --snakefile Snakefile --cores 1 --allowed-rules waveform_bank banks \
  --dry-run banks

# one bank
snakemake --snakefile Snakefile --cores 1 --allowed-rules waveform_bank \
  outputs/banks/uniform-imrphenom-s51.h5
```

The `--allowed-rules` filter keeps bank generation explicit. MCMC commands omit
these rules, so a missing bank stops the run with a `MissingInputException`
rather than silently scheduling hours of waveform generation.

## Banks record their own provenance

Each bank stores how it was made, as HDF5 attributes written once at generation
time:

```text
population_name    = "madau-dickinson"
population_seed    = 41
population_num_samples = 32768
redshift_proposal  = '{"kind": "madau_dickinson", "z_min": 0.0, "z_max": 20.0,
                       "gamma": 1.42, "kappa": 4.62, "z_peak": 1.84,
                       "H0": 67.66, "Omega_m": 0.3096, "n_grid": 4096}'
```

netCDF attributes are flat scalars, so the redshift-proposal descriptor travels
as one JSON string. A uniform bank writes
`{"kind": "uniform_redshift", "z_min": ..., "z_max": ...}`.

This is the density the importance weights divide by. It is extracted from the
population graph exactly once, by
`astrogwb.paper.config.banks.extract_redshift_proposal`, and every later
consumer reads it back with `BankConfig.from_file` instead of re-parsing a
config that may have drifted since. `scripts/run_mcmc.py` also checks the run's `[fiducials]` against
what the bank recorded, so a drifted fiducial fails before a device is claimed
rather than silently reweighting against the wrong denominator.

A bank generated before this metadata existed carries no `redshift_proposal`
attribute. Reading it fails with *"generated before proposal metadata;
regenerate it"* -- deliberately, rather than falling back to parsing.

## Composition

`astrogwb.paper.catalogs.CatalogSource.compose` draws a run's catalog from its
banks:

- `uniform_mixing_fraction == 0` short-circuits to a bank *prefix* with no RNG
  draw at all, so the composed catalog is bit-identical to the first `n` rows
  of the bank;
- otherwise per-sample component assignments are drawn once from
  `mixture_seed`, and each component contributes its next-in-sequence bank
  samples.

Bank draws are prefix-stable -- the construction-time RNG stream does not depend
on how many samples are later requested -- so a prefix of a composed catalog is
itself a valid composition. That is what makes `num_samples` a free parameter:
`variable-catalog-size`'s three runs share one bank file, not three.
