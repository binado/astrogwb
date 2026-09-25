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

Catalogs are content-addressed. A run declares what each role draws in
`[analysis.catalog]` -- a partial spec per role (`seed`, `num_samples`, and any
`waveform` / `population` / `fiducials` override) over the run's own blocks of
the same names. `resolve_catalog_blocks` (stdlib, in `config/runs.py`) is the
one resolution; `CatalogRequest` (`astrogwb.metadata`) validates the result and
its `key()` -- a hash of the canonical request -- names
`outputs/catalogs/<key>.h5`. The Snakefile keys every run's roles at parse time
(`resolve_run_catalogs`), `rule waveform_catalog` hands the generator the
request as JSON and declares no config inputs, `run_mcmc` checks each loaded
file against its own `RunConfig.catalog_request(role)`, and notebooks reach the
same files through `astrogwb.paper.catalogs.run_catalog` /
`astrogwb.catalog.load_or_generate`. There is no `config/catalogs/` and no
catalog name. The version is part of the key, so **bump `version` in
`pyproject.toml` whenever a change alters what a population draw or a waveform
generator produces**, or stale catalogs keep being served. `just catalogs` maps
keys back to what they draw and which runs use them.

A run config is three layers merged in order --
`config/{analysis,fiducials,networks,priors,sampler,waveform,population}.json`,
then the experiment `config/runs/<experiment>/_base.json`, then the run. The
shared layers are one file per top-level block of a run config, each a
single-key object whose key is its own stem; that convention is what lets an
entrypoint take one flag per block and `jq` fold a block in the shell, and a
test pins it. The top-level `[population]` is the default a run's catalogs are
drawn from; the analysis target is `analysis.population`. `[fiducials]` is both
where NUTS initializes and what a run's catalogs are drawn at, so editing
`config/fiducials.json` re-keys every catalog. Three of the layers are also
read directly by the notebooks and figure scripts through
`astrogwb.paper.config.fiducials()` / `priors()` / `networks()`, and
`waveform_generator()` / `population_model()` read the other two, so a copy
cannot drift from what the runs sample. Every layer is JSON, so `load_mapping`
is one parser and `jq` reads any of them without importing the package. A run
names a detector network (`analysis.network`) rather than listing detectors.
What each committed run -- and each catalog override -- is for is documented in
`config/runs/README.md`, next to the files. `config/plotting.json` is
presentation -- LaTeX parameter labels and savefig settings, reached through
`astrogwb.paper.plotting` -- and is deliberately *not* a run layer. No
entrypoint is handed an assembled config. `run_mcmc` takes the blocks `jq`
folded out of its layers, one flag per block -- one operator per block, `*`
everywhere and `+` for `priors`, spelled once in `BLOCK_FOLDS` beside the
Python fold a test pins it against. The figure and diagnostic scripts take the
layer paths instead, as repeated `--config` flags, and merge them in process.
Either way the workflow rule declares the layer files as its `input:`, so the
dependency edge and the data path are one list. `run_mcmc` writes the resolved
config next to the chain and stamps the ordered layer paths and both catalog
keys into it.

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
