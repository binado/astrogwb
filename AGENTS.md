# Repository Guidelines

## Project Structure & Module Organization
- `src/astrogwb/`: Python package code.
  - `detector/`: ORF, effective PSD, `load_sensitivities_for_network`, `geometry.toml`, `sensitivity.toml`, and bundled noise curves.
  - `gwb/`: `spectral.py` (SGWB spectral-density contractions, Omega_GW conversions, and `gaussian_bin_scale`; note: `observation_time` is in **years**), `snr.py` (matched-filter SNR).
  - `waveform/`: reduces loaded `pluscross` waveform catalogs to polarization power for inference (`polarization_power`); catalog IO and generation live in `pluscross` and `scripts/generate_waveform_catalog.py`.
  - `sampling/`: thin NumPyro model for caller-prepared arrays; the model takes a single `merger_rate_and_log_weights_fn(params, samples) -> (total_merger_rate, log_weights)` callback.
  - `utils.py`: unit-conversion helpers (`SECONDS_PER_YEAR`, `years_to_seconds`) and `repo_root()` for resolving the repository from notebooks or scripts.
- `tests/`: Pytest suite with unit and integration coverage; committed `tests/fixtures/*.npz` for regression locks.
- `notebooks/`: runnable, version-controlled workflows. `mcmc.py` (py:percent) is the NumPyro port of ASGWB.jl's importance-weighted NUTS run; `mcmc_plotting.py` loads saved chains and produces corner and diagnostic plots.
- `docs/`: topic guides linked from the root README (catalog generation, inference, paper figures, Snakemake/SLURM).
- `notes/`: LaTeX notes/manuscript support files (`notes/justfile` for PDF build helpers).

gwmock-signal owns preset detector networks; gwmock-noise owns bundled PSD presets and interpolation. astrogwb owns the frequency-dependent ORF, out-of-band PSD policy, supplemental geometry/noise tables for str-named detectors, and SGWB detector utilities.

## Build, Test, and Development Commands
- `uv sync --group dev`: create/update the local environment with dev dependencies (includes the `plotting` group: arviz, corner, h5netcdf).
- `uv sync --extra mcmc`: install the headless MCMC runner dependencies: pydantic for validated `RunConfig` parsing and ArviZ with NetCDF output support (`scripts/run_mcmc.py`, config generation, related tests).
- `uv sync --extra mcmc --group dev`: typical full local/CI install for MCMC scripts + notebooks + tests.
- `uv sync --group plotting`: install notebook plotting dependencies only (arviz, corner, h5netcdf).
- `uv run --extra mcmc --group dev pytest`: run all tests (include `--extra mcmc` when exercising `RunConfig`).
- `uv run --extra mcmc --group dev pytest -m "not integration"`: run fast unit tests only.
- `uv run ruff check . --fix`: lint and apply safe fixes.
- `uv run ruff format .`: format Python files.
- `uvx ty check`: type check.
- `pre-commit run --all-files`: run all configured quality hooks before pushing.
- `uv run jupyter nbconvert --to notebook --execute notebooks/mcmc.py`: run the MCMC notebook end-to-end (or open in JupyterLab after `uv sync --group dev` or `uv sync --group plotting`, which bundle arviz, corner, and matplotlib).

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
