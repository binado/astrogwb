# Repository Guidelines

## Workspace layout

- `packages/astrogwb/`: publishable scientific library. It owns cosmology,
  detector geometry/data, GWB contractions, importance weighting, the generic
  NumPyro model, and waveform power reduction.
- `packages/astrogwb-paper/`: private reproducibility application. It owns
  configuration, runtime setup, prior adaptation, console commands, notebooks,
  Snakemake workflows, profiles, examples, and paper documentation.
- `out/`, `chains/`, `figures/`, and `logs/` are generated artifacts that
  live under `packages/astrogwb-paper/` (Snakemake's execution `cwd` is
  anchored there); they must not be committed or duplicated elsewhere in
  the workspace.

The dependency direction is `astrogwb-paper -> astrogwb`. Core must never
import `astrogwb_paper` or know the repository checkout layout.

## Commands

- `uv sync --all-packages --all-groups`: full workspace environment.
- `uv run --frozen --isolated --no-default-groups --package astrogwb --group test pytest packages/astrogwb/tests`:
  isolated core tests.
- `uv run --frozen --isolated --no-default-groups --package astrogwb-paper --group test --group workflow pytest packages/astrogwb-paper/tests`:
  isolated paper tests.
- `uv run --group dev ruff check` and `uv run --group dev ruff format`:
  lint and format.
- `uv run --group dev ty check`: type check.
- `uv build --package astrogwb --no-sources`: publication build.
- `uv run astrogwb-workflow --help`: production workflow entrypoint.

## Coding and testing

- Target Python `>=3.12`, use explicit public type hints, `pathlib.Path`, Ruff
  formatting, snake_case functions, PascalCase classes, and uppercase constants.
- Keep `astrogwb_paper` imports before JAX initialization lightweight. Runtime
  configuration must run before imports that can initialize JAX or NumPyro.
- Add core tests for scientific interfaces and paper tests for configuration,
  CLI, runtime, I/O, paths, and workflows. Mark dependency-heavy tests with
  `@pytest.mark.integration`.
- Preserve detector TOML/noise data inside the core module tree and verify it
  from built wheels.

## Commits and pull requests

Use focused Conventional Commits (`feat:`, `fix:`, `chore:`, `refactor:`).
PRs should state purpose, key changes, commands run, and data/config impacts.
