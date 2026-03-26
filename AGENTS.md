# Repository Guidelines

## Project Structure & Module Organization
- `src/asgwb/`: Python package code. Core detector logic lives in `src/asgwb/detector/` (`detector.py`, `psd.py`, `detectors.toml`, and bundled noise curves).
- `tests/`: Pytest suite (`test_detector.py`) with unit and integration coverage.
- `scripts/`: Research workflows (waveform generation, merge utilities, SLURM submission via `submit.sh`).
- `config/`: Runtime TOML defaults (for example, `generate_injection_waveforms.toml`).
- `data/`: Input catalogs and reference documents. Treat as runtime assets, not active source code.
- `notes/`: LaTeX notes/manuscript support files (`notes/justfile` for PDF build helpers).

## Build, Test, and Development Commands
- `uv sync --group dev`: create/update the local environment with dev dependencies.
- `uv run pytest`: run all tests.
- `uv run pytest -m "not integration"`: run fast unit tests only.
- `uv run ruff check . --fix`: lint and apply safe fixes.
- `uv run ruff format .`: format Python files.
- `pre-commit run --all-files`: run all configured quality hooks before pushing.
- `uv run python scripts/generate_injection_waveforms.py --help`: inspect script CLI options.

## Coding Style & Naming Conventions
- Target Python `>=3.11`; use 4-space indentation and explicit type hints for public APIs.
- Follow Ruff defaults for linting and formatting; do not hand-format against the formatter.
- Use `snake_case` for modules/functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Keep file/path handling via `pathlib.Path` and prefer config-driven behavior over hardcoded paths.

## Testing Guidelines
- Framework: `pytest` (configured in `pyproject.toml`).
- Mark expensive or dependency-heavy tests with `@pytest.mark.integration`.
- Test files should be named `test_*.py`; test functions should start with `test_`.
- Add unit tests for new logic paths and integration tests when touching Bilby/LAL interop.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (as seen in history): `feat:`, `fix:`, `chore:`, `refactor:`; optional scope is encouraged (for example, `fix(slurm): ...`).
- Keep commits focused and atomic; include tests/docs updates with behavior changes.
- PRs should include: purpose, key changes, commands run (tests/lint), and any data or config impacts.
- For pipeline/cluster changes, include a concrete invocation example and expected output location.

## Cursor Cloud specific instructions

- **No services to start.** This is a pure Python scientific computing library with no servers, databases, or background daemons.
- **uv is the only package manager.** All commands should be run via `uv run ...`. The venv lives at `.venv/`.
- **Integration test fixtures are not committed to git.** Before running `uv run pytest -m integration`, generate them once with `uv run --script scripts/generate_orf_fixtures.py`. The generated `.npz` files land in `tests/fixtures/`.
- **Build-system version warning is benign.** `uv sync` may emit `warning: build_system.requires = ["uv_build>=0.10.2,<0.11.0"] does not contain the current uv version ...` — this does not affect functionality.
- Refer to the "Build, Test, and Development Commands" section above for standard lint/test/format commands.
