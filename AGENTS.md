# Repository Guidelines

## Project Structure & Module Organization
- `src/astrogwb/`: Python package code.
  - `detector/`: ORF, effective PSD, `load_sensitivities_for_network`, `geometry.toml`, `sensitivity.toml`, and bundled noise curves.
  - `gwb.py`: JAX SGWB spectral-density contractions and Omega_GW conversions.
  - `waveform/`: polarization-power catalog generation and persistence.
  - `sampling/`: thin NumPyro model for caller-prepared arrays.
- `tests/`: Pytest suite with unit and integration coverage; committed `tests/fixtures/*.npz` for regression locks.
- `notes/`: LaTeX notes/manuscript support files (`notes/justfile` for PDF build helpers).

gwmock-signal owns preset detector networks; gwmock-noise owns bundled PSD presets and interpolation. astrogwb owns the frequency-dependent ORF, out-of-band PSD policy, supplemental geometry/noise tables for str-named detectors, and SGWB detector utilities.

## Build, Test, and Development Commands
- `uv sync --group dev`: create/update the local environment with dev dependencies.
- `uv run pytest`: run all tests.
- `uv run pytest -m "not integration"`: run fast unit tests only.
- `uv run ruff check . --fix`: lint and apply safe fixes.
- `uv run ruff format .`: format Python files.
- `uvx ty check`: type check.
- `pre-commit run --all-files`: run all configured quality hooks before pushing.

## Coding Style & Naming Conventions
- Target Python `>=3.12`; use 4-space indentation and explicit type hints for public APIs.
- Follow Ruff defaults for linting and formatting; do not hand-format against the formatter.
- Use `snake_case` for modules/functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Keep file/path handling via `pathlib.Path` and prefer config-driven behavior over hardcoded paths.

## Testing Guidelines
- Framework: `pytest` (configured in `pyproject.toml`).
- Mark expensive or dependency-heavy tests with `@pytest.mark.integration`.
- Test files should be named `test_*.py`; test functions should start with `test_`.
- Add unit tests for new logic paths and integration tests when touching gwmock-signal or gwmock-noise interop.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (as seen in history): `feat:`, `fix:`, `chore:`, `refactor:`; optional scope is encouraged (for example, `fix(slurm): ...`).
- Keep commits focused and atomic; include tests/docs updates with behavior changes.
- PRs should include: purpose, key changes, commands run (tests/lint), and any data or config impacts.
