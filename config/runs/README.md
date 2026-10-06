# Committed runs

One file per run, and the filename is the mapping:

```text
config/runs/<experiment>/<run>.toml  ->  outputs/chains/<experiment>/<run>.nc
```

The packaged detector geometry and sensitivity tables use the same
`[detectors.<name>]` layout as the shared registry and are merged first with
pyknf. A run is six layers merged over those defaults in order:

```text
config/defaults.toml  ->  config/waveforms.toml  ->  config/populations.toml
  ->  config/detectors.toml  ->  <experiment>/_base.toml  ->  <run>.toml
```

`config/defaults.toml` declares shared scientific values and the default
catalog draw; `config/waveforms.toml` and `config/populations.toml` declare the
named waveforms and populations a catalog or target refers to by
`"${...}"` reference; `config/detectors.toml` declares networks and optional
detector overrides. All four are inherited by every run.
A run file carries only what distinguishes it. `_base.toml` is the experiment
override and is required in every experiment directory -- a conditional
Snakemake input would complicate the DAG for no gain. It is not a run, so it
never becomes a chain.

**What each file is for is written in the file.** Every layer opens with a
comment. Each `_base.toml` describes its experiment, each run says how it
differs, and each catalog override explains itself where it is declared. This
page is the index, plus the one thing that spans experiments: which catalogs
exist and who shares them.

See [`docs/running-inference.md`](../../docs/running-inference.md) for the layer
model, the merge rules, and how to run one.

## The experiments

| Experiment | Runs | Question |
| --- | --- | --- |
| [`cosmological-parameters`](cosmological-parameters/_base.toml) | 8 | H0 -- or one parameter with H0 marginalized -- across detector networks |
| [`modified-propagation`](modified-propagation/_base.toml) | 8 | Xi_0 and n across the same networks, plus two narrowed problems |
| [`astrophysical-parameters`](astrophysical-parameters/_base.toml) | 2 | the Madau-Dickinson rate shape, H0 marginalized |
| [`variable-catalog-size`](variable-catalog-size/_base.toml) | 3 | how H0 tightens with proposal size |
| [`variable-proposal-guard`](variable-proposal-guard/_base.toml) | 3 | how the guard fraction eps affects the rate-shape problem |
| [`waveform-approximant`](waveform-approximant/_base.toml) | 2 | waveform systematics: IMRPhenom vs TaylorF2 proposal |
| [`time-delay`](time-delay/_base.toml) | 1 | the delay-time slope of a time-delayed population |

## The catalogs

Each run declares what its two catalogs draw as `[analysis.injection]` and
`[analysis.proposal]`, each a `CatalogMetadata` once its references resolve. A
run sets only the fields that differ: a size, a seed, or a named population or
waveform. The file is `outputs/catalogs/polarization_power-<key>-<seed>.h5`,
named by the hash of the resolved request and the seed it is drawn at
(`[analysis.seeds]`), so runs that ask for the same draw share one file.
`just catalogs` lists every file, what it draws, and which runs use it. See
[`docs/catalog-generation.md`](../../docs/catalog-generation.md).

`config/defaults.toml` sets the default for both roles, `[catalog]`: a
seed-41, 32768-source draw from `[populations.cosmological]` with the
IMRPhenom `[waveforms.default]`. That one catalog is the injection -- the "observed" data -- of every
run except `time-delay`, and the proposal of `cosmological-parameters`,
`modified-propagation` and `waveform-approximant/IMRPhenom`.

The overrides, and who uses each; the file that declares a draw says why:

| Draw | Role | Used by |
| --- | --- | --- |
| seed 42, n = 8192 / 16384 / 32768 | proposal | `variable-catalog-size` |
| `[populations.guard]` (`bns_md_uniform_mixture`, eps = 0.1), seed 61, n = 16384 | proposal | `astrophysical-parameters`, `variable-proposal-guard/eps1e-1`, `time-delay` (a separate copy, drawn at its own fiducials) |
| `[populations.guard]` at eps = 0.01 / 0.001, seeds 62 / 63, n = 16384 | proposal | `variable-proposal-guard` |
| `[waveforms.TaylorF2]`, seed 41 | proposal | `waveform-approximant/TaylorF2` |
| `[populations.time_delayed]` (`bns_md_time_delayed_cosmological`), seed 71 | injection | `time-delay` |

## Adding one

Add a `.toml` file under an experiment directory, opening with a comment that
says what it measures and how it differs from its `_base.toml`.
`discover_runs` picks it up by globbing, and its stem becomes the chain path;
add a row above if it starts a new experiment. Run `snakemake validate` first:
it merges and catalog-checks all runs, and builds every named population,
before any GPU job is queued.
