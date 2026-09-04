# Repository Guidelines

## Layout

One package, `astrogwb`, with the reproducibility application inside it:

- `src/astrogwb/`: publishable scientific library. It owns cosmology, detector
  geometry/data, GWB contractions, importance weighting, the generic NumPyro
  model, waveform power reduction, and HDF5 catalog serialization.
- `src/astrogwb/paper/`: the reproducibility application. It owns
  configuration, runtime setup, prior adaptation, console commands, plotting
  style, and the SNR/inference helpers the figures use.
- `config/`, `scripts/`, `notebooks/`, `profiles/`, `docs/`, `Snakefile`: the
  application's committed assets, at the repository root.
- `tests/core/` and `tests/paper/`.
- `out/`, `outputs/`, `chains/`, `figures/`, `grids/`, `logs/`: generated
  artifacts at the repository root, which is Snakemake's execution `cwd`. They
  are gitignored and must not be committed.

The dependency runs one way: `astrogwb.paper` may import `astrogwb`, never the
reverse. Two independent things enforce it, because they catch different
mistakes:

- ruff `TID251` bans `astrogwb.paper` inside core (`[tool.ruff.lint]`
  `extend-select = ["TID"]` plus the `banned-api` entry). This is what catches
  a pure-stdlib helper drifting across.
- `just smoke-wheel` installs the built wheel with no extras and asserts the
  application cannot run. The application's dependencies live behind the
  `paper` extra, so `pip install astrogwb` gets none of them.

Library code names no absolute path and never looks for a checkout: paths like
`config/analysis` are relative to the caller's cwd, which for the workflow and
every script is the repository root.

## Commands

Every check is a `just` recipe, and CI runs the same string:

- `uv sync --extra notebook --group dev`: full development environment.
- `just lint`, `just typecheck`, `just fmt`.
- `just test-core`, `just test-paper`, `just test-integration`.
- `just test-notebooks` (set `ASTROGWB_NOTEBOOK_SMOKE=1` to shrink it).
- `just build-core`, `just smoke-wheel`: publication build and its boundary
  assertions.
- `uv run --group workflow snakemake --snakefile Snakefile --dry-run --cores 1 experiments`:
  production workflow entrypoint (omit `--dry-run` to execute). Chains come
  from `run_experiment_<name>` targets; figures are opt-in via the `plot_*`
  rules. Run `snakemake validate` first: it merges and bank-checks all 26 runs
  without building anything.

## Configuration

A run config is three TOML layers merged in order -- `config/analysis/base/*`,
then the experiment `_base.toml`, then the run. There is no assembled-config
artifact: every entrypoint takes the layers on argv as repeated `--config`
flags, and the workflow rule declares those same files as its `input:`, so the
dependency edge and the data path are one list. `run_mcmc` writes the resolved
config next to the chain and stamps the ordered layer paths into it.

## Coding and testing

- Target Python `>=3.12`, use explicit public type hints, `pathlib.Path`, Ruff
  formatting, snake_case functions, PascalCase classes, and uppercase constants.
- `astrogwb.paper.config.runs` is stdlib-only and `astrogwb.paper.config.banks`
  reaches `astrogwb.catalog` only inside function bodies. The `Snakefile`
  imports both, so keeping them JAX-free is what keeps `--dry-run` cheap;
  `tests/paper/test_cli.py` asserts it.
- Runtime configuration must run before imports that can initialize JAX or
  NumPyro.
- Add core tests for scientific interfaces and paper tests for configuration,
  CLI, runtime, I/O, and workflows. Mark dependency-heavy tests with
  `@pytest.mark.integration`.
- Preserve detector TOML/noise data inside the core module tree and verify it
  from built wheels.

## Commits and pull requests

Use focused Conventional Commits (`feat:`, `fix:`, `chore:`, `refactor:`).
PRs should state purpose, key changes, commands run, and data/config impacts.
