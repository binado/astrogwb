# Repository Guidelines

## Layout

One package, `astrogwb`, with the reproducibility application inside it:

- `src/astrogwb/`: publishable scientific library (cosmology, detectors, GWB
  contractions, NumPyro source populations, importance weighting, waveform power
  reduction, HDF5 catalog serialization).
- `src/astrogwb/paper/`: the reproducibility application (configuration, runtime
  setup, console commands, plotting style, SNR/inference helpers).
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
  rules. The dry run merges and catalog-checks every run without building
  anything.
- `just catalogs`: maps catalog stems back to what they draw, at which seed, and
  which runs use them.

## Configuration

- Run workflow and application entrypoints from the repository root;
  configuration paths are relative to the caller's current working directory.
- A run config is six TOML layers merged in order: the four shared layers
  (`config/defaults.toml`, `waveforms.toml`, `populations.toml`, `detectors.toml`),
  then `config/runs/<experiment>/_base.toml`, then the run. Every entrypoint
  takes the layer paths as repeated `--config` flags. Each layer's header comment
  says what it is for; `config/runs/README.md` indexes the experiments.
- Editing the shared `[fiducials]` re-keys every catalog, and the `astrogwb`
  version is part of the catalog key.
- Adding a population means adding a registered factory under
  `src/astrogwb/populations/`, never an import path in a config.
- `batch_keys` needs `jax_enable_x64`.
- Pass the rescaled-spectrum builder's pytree callables *as arguments* to jitted
  functions, not through a closure, so the catalog stays a traced input.
- `run_mcmc --cached-only` never generates catalogs.

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
