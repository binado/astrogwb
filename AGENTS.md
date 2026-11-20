# Repository Guidelines

## Layout

`src/astrogwb/` contains the scientific library and CLI. Tests live in
`tests/`; `tests/parity/` compares the library against established
packages (bilby directly, pygwb through the frozen arrays in
`tests/parity/references/`). Paper workflows live in the separate `experiments-astrogwb` repository
and must not be imported by this package.

## Commands

- `uv sync --extra cli --extra simulation --group dev`
- `just lint`, `just typecheck`, `just fmt`, `just test-core`
- `just test-integration`
- `just test-parity`; `just parity-references` regenerates the pygwb arrays

## Code conventions and styleguide

- Keep imports at the top level of the module, except where required for optional dependencies, circular imports, or execution ordering.
- Use numpy-style docstrings
- Don't use unicode symbols in comments or docstrings. Prefer rst math directives for the latter
- Use jax.typing.ArrayLike for annotating array inputs and jax.Array for array outputs
- When manipulating numpyro models, prefer desigining useful custom effect handlers
- When designing methods that will be jit compiled, avoid capturing large constants or
closing over arguments which may trigger many re-compilations
- Don't use comments for enforcing patterns or invariants. Those should be enforced naturally by the API design. Use comments for documenting behavior or gotchas.


## Python conventions

- Target Python `>=3.12`, use explicit public type hints, `pathlib.Path`, Ruff
  formatting, snake_case functions, PascalCase classes, and uppercase constants.

## Commits and pull requests

Use focused Conventional Commits (`feat:`, `fix:`, `chore:`, `refactor:`).
PRs should state purpose, key changes, commands run, and data/config impacts.
