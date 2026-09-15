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
population model, that model's construction settings, the hyperparameters it
was drawn at, and the density factors included in importance weighting. No
run config restates any of it, and nothing cross-checks the two. Adding a
population means adding a registered source-model function under
`src/astrogwb/populations/`, never an import path in a config.

A run config is four layers merged in order -- `config/{fiducials,priors,networks}.json`,
then `config/analysis/base/*`, then the experiment `_base.toml`, then the run.
The three JSON files are layer 0: the shared scientific values, which the
notebooks and figure scripts also read directly through
`astrogwb.paper.config.fiducials()` / `priors()` / `networks()`, so a copy
cannot drift from what the runs sample. `jq` can read them without importing
the package. A run names a detector network (`analysis.network`) rather than
listing detectors. `config/plotting.json` is presentation -- LaTeX parameter
labels and savefig settings, reached through `astrogwb.paper.plotting` -- and
is deliberately *not* a run layer. `config/waveform.json` is catalog layer 0,
reached through `astrogwb.paper.config.waveform_generator()`, and is likewise
not a run layer. There is no assembled-config artifact: every entrypoint takes the layers on argv as repeated `--config`
flags, and the workflow rule declares those same files as its `input:`, so the
dependency edge and the data path are one list. `run_mcmc` writes the resolved
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
