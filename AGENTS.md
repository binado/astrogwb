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

## Code conventions and styleguide

- Use numpy-style docstrings
- Don't use unicode symbols in comments or docstrings. Prefer rst math directives for the latter
- Use jax.typing.ArrayLike for annotating array inputs and jax.Array for array outputs
- When manipulating numpyro models, prefer desigining useful custom effect handlers
- When designing methods that will be jit compiled, avoid capturing large constants or
closing over arguments which may trigger many re-compilations


## Coding and testing

- Target Python `>=3.12`, use explicit public type hints, `pathlib.Path`, Ruff
  formatting, snake_case functions, PascalCase classes, and uppercase constants.
- Runtime configuration must run before the XLA *backend* is initialized. Importing JAX
  or NumPyro does not initialize it; creating an array or querying devices
  does. `tests/paper/test_cli.py` asserts both halves.
- Add core tests for scientific interfaces.
- Use pytest fixtures and `pytest.mark.parametrize`
- Avoid tests which re-implement functions
- Avoid tests which require brittle `atol` and `rtol` in `np.testing.assert_allclose`
- Mark dependency-heavy tests with
  `@pytest.mark.integration`.

## Commits and pull requests

Use focused Conventional Commits (`feat:`, `fix:`, `chore:`, `refactor:`).
PRs should state purpose, key changes, commands run, and data/config impacts.
