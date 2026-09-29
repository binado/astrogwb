# Repository Guidelines

## Layout

One package, `astrogwb`, with the reproducibility application inside it:

- `src/astrogwb/`: publishable scientific library. It owns cosmology, detector
  geometry/data, GWB contractions, source populations declared as NumPyro
  models, importance weighting, the generic NumPyro model, waveform power
  reduction, and HDF5 catalog serialization.
- `src/astrogwb/paper/`: the reproducibility application. It owns
  configuration, runtime setup, prior adaptation, console commands, plotting
  style, and the SNR/inference helpers the figures use.
- `config/`, `scripts/`, `notebooks/`, `profiles/`, `docs/`, `Snakefile`: the
  application's committed assets, at the repository root.
- `tests/core/` and `tests/paper/`.



## Commands

Every check is a `just` recipe, and CI runs the same string:

- `uv sync --extra notebook --group dev`: full development environment.
- `just lint`, `just typecheck`, `just fmt`.
- `just test-core`, `just test-paper`, `just test-integration`.
- `uv run --group workflow snakemake --snakefile Snakefile --dry-run --cores 1 experiments`:
  production workflow entrypoint (omit `--dry-run` to execute). Chains come
  from `run_experiment_<name>` targets; figures are opt-in via the `plot_*`
  rules. Run `snakemake validate` first: it merges and catalog-checks all 27 runs
  without building anything.

## Configuration

Run workflow and application entrypoints from the repository root; configuration paths are relative to the caller's current working directory, and no checkout discovery is performed.

A catalog records the density that drew it: its file carries the registered
population model, that model's construction settings, the hyperparameters it
was drawn at, and the `astrogwb` version that generated it. Which of those
density factors enter an importance weight is *not* part of the record -- no
sample depends on it -- so it is declared by the analysis that reweights the
draw. Adding a population means adding a registered source-model function under
`src/astrogwb/populations/`, never an import path in a config.

Catalogs are content-addressed. A run declares what each role draws as
`[analysis.injection]` and `[analysis.proposal]`, each a complete
`CatalogMetadata` (`astrogwb.metadata`) once the merge resolves its `${...}`
references; both default, field by field, to the shared draw `[catalog]`.
`CatalogMetadata` validates the role and its `key()` -- a hash of the
canonical record -- names
`outputs/catalogs/<key>.h5` (`astrogwb.metadata.artifact_path`). The same
record is what a `PolarizationPowerCatalog` carries as `.metadata`. The
Snakefile keys every run's roles at parse time (`resolve_run_catalogs`),
`rule waveform_catalog` hands the generator the metadata as JSON and declares
no config inputs, and `run_mcmc` and the notebooks alike reach the files through
`astrogwb.catalog.simulate(metadata, CatalogGenerator(), cache_dir)` (notebooks
via `astrogwb.paper.catalogs.run_catalog`), which checks a hit against the
request and generates a miss -- unless `generate=False`, which the workflow's
`run_mcmc --cached-only` uses so a job never generates. There is
no `config/catalogs/` and no catalog name. The version is part of the key, so **bump `version` in
`pyproject.toml` whenever a change alters what a population draw or a waveform
generator produces**, or stale catalogs keep being served. `just catalogs` maps
keys back to what they draw and which runs use them.
Forward-model spectra go through the same `simulate` outside the workflow: a
`SpectraMetadata` (each hyperparameter a fixed number or a prior spec) keys
`outputs/spectra/<key>.h5`, and `astrogwb.catalog.simulate(metadata,
SpectrumGenerator(), cache_dir)` serves or generates it.

A run config is six TOML layers merged in order -- the four shared layers
`config/defaults.toml`, `config/waveforms.toml`, `config/populations.toml` and
`config/detectors.toml`, then the experiment
`config/runs/<experiment>/_base.toml`, then the run.
`config/defaults.toml` declares the shared scientific defaults and the default
draw `[catalog]`; `config/waveforms.toml` and `config/populations.toml` declare
named `[waveforms.<name>]` / `[populations.<name>]` records (each population
with a default seed) that catalogs and the analysis target refer to;
`config/detectors.toml` declares `[networks]` and optional `[detectors]` overrides. Every layer opens with a comment saying what it is
for, so what each committed run -- and each catalog override -- is for lives in
its own file; `config/runs/README.md` indexes the experiments and the catalogs
they share. The analysis target is `analysis.population`, a
`PopulationMetadata` whose seed is unused. `[fiducials]` is both
where NUTS initializes and what a run's catalogs are drawn at, so editing the
shared `[fiducials]` re-keys every catalog. `[fiducials]`, `[priors]` and
`[networks]` are also read directly by the notebooks and figure scripts through
`astrogwb.paper.config.fiducials()` / `priors()` / `networks()`, and
`waveform_generator()` / `population_model()` read `[catalog]`'s waveform and
population, so a copy cannot drift from what the runs sample.
`merge_config_layers` folds the layers with `knf` (pyknf): a deep merge, except
that each `priors.<param>` table replaces the inherited one (`PRIOR_SHALLOW =
"priors.*"`), then resolves every `"${a.b}"` reference against the merged
result. A reference is atomic -- setting a key under one replaces it whole --
and cannot be reached *through* (`${x.y}` fails when `x` is itself a
reference), so a table that runs override field by field is spelled one
reference per field; to change a named variant for one run, override it at its
source. `[catalog]`, `[waveforms]` and `[populations]` exist only to be
referenced, and `RunConfig` drops them. `knf src/astrogwb/detector/{geometry,sensitivity}.toml <layers> --shallow 'priors.*' --interpolate`
prints the same merge in the shell. The packaged detector files use the same
`[detectors.<name>]` format and are merged before the six run layers. A run names a detector network (`analysis.network`) rather than listing
detectors. `config/plotting.toml` is presentation -- LaTeX parameter labels and
savefig settings, reached through `astrogwb.paper.plotting` -- and is
deliberately *not* a run layer. No entrypoint is handed an assembled config.
Every one -- `run_mcmc` and the figure and diagnostic scripts alike -- takes the
layer paths as repeated `--config` flags and merges them in process, and the
workflow rule declares the same files as its `input:`, so the dependency edge
and the data path are one list. `run_mcmc` writes the resolved config next to
the chain as JSON (an output, not a layer) and stamps both catalog keys into it.

## Coding and testing

- Target Python `>=3.12`, use explicit public type hints, `pathlib.Path`, Ruff
  formatting, snake_case functions, PascalCase classes, and uppercase constants.
- Runtime configuration must run before the XLA *backend* is initialized. Importing JAX
  or NumPyro does not initialize it; creating an array or querying devices
  does. `tests/paper/test_cli.py` asserts both halves.
- Add core tests for scientific interfaces. Mark dependency-heavy tests with
  `@pytest.mark.integration`.

## Commits and pull requests

Use focused Conventional Commits (`feat:`, `fix:`, `chore:`, `refactor:`).
PRs should state purpose, key changes, commands run, and data/config impacts.
