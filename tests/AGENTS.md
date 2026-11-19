# Testing guidelines

- Runtime configuration must run before the XLA *backend* is initialized. Importing JAX
  or NumPyro does not initialize it; creating an array or querying devices
  does. `test_cli.py` and `test_runtime.py` assert both halves.
- Add core tests for scientific interfaces.
- Use pytest fixtures and `pytest.mark.parametrize`.
- Avoid tests which re-implement functions.
- Use `assert_close` from `tolerances` for approximate array comparisons, relying on its dtype-aware defaults instead of hand-picked `atol` and `rtol`.
- Mark dependency-heavy tests with `@pytest.mark.integration`.
