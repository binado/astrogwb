# Repository Guidelines

## Layout

`src/astrogwb/` contains the scientific library and CLI. Tests live in
`tests/core/`; `tests/parity/` compares the library against established
packages (bilby directly, pygwb through the frozen arrays in
`tests/parity/references/`). Paper workflows live in the separate `experiments-astrogwb` repository
and must not be imported by this package.

## Commands

- `uv sync --extra cli --extra simulation --group dev`
- `just lint`, `just typecheck`, `just fmt`, `just test-core`
- `just test-integration`, `just build-core`
- `just test-parity`; `just parity-references` regenerates the pygwb arrays

## Code conventions and styleguide

- Keep imports at the top level of the module, except where required for optional dependencies, circular imports, or execution ordering.
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
  does. `tests/core/test_cli.py` and `tests/core/test_runtime.py` assert both halves.
- Add core tests for scientific interfaces.
- Use pytest fixtures and `pytest.mark.parametrize`
- Avoid tests which re-implement functions
- Avoid tests which require brittle `atol` and `rtol` in `np.testing.assert_allclose`
- Mark dependency-heavy tests with
  `@pytest.mark.integration`.

## Commits and pull requests

Use focused Conventional Commits (`feat:`, `fix:`, `chore:`, `refactor:`).
PRs should state purpose, key changes, commands run, and data/config impacts.
