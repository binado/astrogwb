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
  rules. Run `snakemake validate` first: it merges and catalog-checks all 26 runs
  without building anything.

## Configuration

Run workflow and application entrypoints from the repository root; configuration paths are relative to the caller's current working directory, and no checkout discovery is performed.

A catalog records the density that drew it: its file carries the registered
population model, that model's construction settings, and the hyperparameters
it was drawn at. No run config restates any of it, and nothing cross-checks the
two. Which of those density factors enter an importance weight is *not* part of
the record -- no sample depends on it -- so it is declared by the analysis that
reweights the draw. Adding a population means adding a registered source-model
function under `src/astrogwb/populations/`, never an import path in a config.

A catalog config is four layers: `config/waveform.json`,
`config/population.json`, `config/fiducials.json`, then
`config/catalogs/<name>.json`, whose stem is the catalog name and its output
path. The shared three are named rather than globbed, because they sit beside
run tables that must not enter a catalog merge. `config/fiducials.json` is
deliberately both a run layer and a catalog layer: the hyperparameters a
catalog is drawn at and the ones a run initializes at are one table, so editing
it invalidates every catalog as well as every run. `config/population.json`
declares only `model_name` and `model_kwargs` -- the seed belongs to a
particular draw, so `CatalogDefinition` supplies it during validation and holds
the result as a `PopulationMetadata`, the same record the `.h5` persists.

Every catalog layer is JSON, so the fold is `jq`, not Python. `rule
merge_catalog_config` folds exactly the files it declares as `input:` into one
`temp()` merged JSON, and `rule waveform_catalog` reads a key per block out of
that and hands them to the generator. Two rules rather than one so the fold
happens once per catalog instead of once per flag; the merged file is a build
intermediate, not an artifact, and the layer files remain the dependency edge
through it. `jq`'s `*` is `deep_merge`; catalog layers carry no `[priors]`
block, so the shallow-merge rule the run path needs never applies. A test pins
the two merges agreeing.

A run config is three layers merged in order --
`config/{analysis,fiducials,networks,priors,sampler}.json`, then the experiment
`config/runs/<experiment>/_base.json`, then the run. The shared layers are one
file per top-level block of a run config, each a single-key object whose key is
its own stem; that convention is what lets an entrypoint take one flag per
block and `jq` fold a block in the shell, and a test pins it. Three of them are
also read directly by the notebooks and figure scripts through
`astrogwb.paper.config.fiducials()` / `priors()` / `networks()`, so a copy
cannot drift from what the runs sample. Every layer is JSON, so
`load_mapping` is one parser and `jq` reads any of them without importing the
package. A run names a detector network (`analysis.network`) rather than
listing detectors, and its target population, redshift grid and two catalogs
all live in its `[analysis]` block. What each committed run is for is
documented in `config/runs/README.md`, next to the files. `config/plotting.json` is presentation -- LaTeX parameter
labels and savefig settings, reached through `astrogwb.paper.plotting` -- and
is deliberately *not* a run layer. `config/waveform.json` and
`config/population.json` are catalog layers, reached through
`astrogwb.paper.config.waveform_generator()` / `population_model()`, and are
likewise not run layers. No entrypoint is handed an assembled config. `run_mcmc` and
`generate_catalog.py` both take the blocks `jq` folded out of their layers, one
flag per block -- for runs the fold is one operator per block, `*` everywhere
and `+` for `priors`, spelled once in `BLOCK_FOLDS` beside the Python fold a
test pins it against. The figure and diagnostic scripts take the layer paths
instead, as repeated `--config` flags, and merge them in process. The catalog path does materialize that merge as a `temp()` file, but it
is a workflow build intermediate that no entrypoint reads as config. Either way
the workflow rule declares the layer files as its `input:`, so the dependency
edge and the data path are one list. `run_mcmc` writes the resolved
config next to the chain and stamps the ordered layer paths into it.

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
