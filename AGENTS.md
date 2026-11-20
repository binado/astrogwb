# Repository Guidelines

## Layout

`src/astrogwb/` contains the scientific library and CLI.

## Commands

- `uv sync --extra cli --group dev`
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
- Don't use comments for enforcing patterns or invariants. Those should be enforced naturally by the API design. Use comments for documenting non-trivial behavior or gotchas.

## Commits and pull requests

- Use focused Conventional Commits (`feat:`, `fix:`, `chore:`, `refactor:`).
- Make sure to check whether the docs are up to date with the changes in the PR
