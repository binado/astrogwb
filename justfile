# Every recipe below is byte-identical to the command CI runs, so there is
# exactly one definition of each. See .github/workflows/ci.yml.

# The checks a change should pass before it is pushed.
default: lint typecheck test

# Lockfile, lint, and formatting -- the CI `lint` job.
lint:
    uv lock --check
    uv run --group dev ruff check
    uv run --group dev ruff format --check

# Apply formatting and the safe lint fixes. Not run by CI.
fmt:
    uv run --group dev ruff check --fix
    uv run --group dev ruff format

# The CI `typecheck` job. Needs the application's own dependencies on the path
# -- ty resolves imports against the environment, and src/astrogwb/paper,
# scripts/ and tests/paper all import the `notebook` extra's stack.
typecheck:
    uv run --extra notebook --group dev ty check

# The core suite installs astrogwb WITHOUT the `paper` extra, so a core test
# that reaches across the boundary fails here rather than passing by accident
# on a developer's fully-synced venv. `io` is the one extra core owns.

# Core library tests.
test-core:
    uv run --frozen --isolated --no-default-groups \
        --extra io --group test \
        pytest tests/core -m "not integration"

# Paper application tests: configuration, CLI, runtime, IO, paths, workflows.
test-paper:
    uv run --frozen --isolated --no-default-groups \
        --extra notebook --group test --group workflow \
        pytest tests/paper -m "not integration"

# The end-to-end NUTS runs, slow enough to be their own CI job. These are also
# the tests that cross-check the cosmology against gwmock-pop, so they need
# `simulation` -- which test-core deliberately does not have, since core must
# work without it.
#
# The second line is the paper half: the pipeline's parity checks against the
# hand-written grid formula, and the generation tests that actually run Ripple.
# They were dark before -- marked `integration` but reachable from no recipe --
# so a change to catalog generation or to the prepared estimator could pass CI
# with nothing having exercised either end to end.
test-integration:
    uv run --frozen --isolated --no-default-groups \
        --extra io --extra simulation --group test \
        pytest tests/core -m integration
    uv run --frozen --isolated --no-default-groups \
        --extra notebook --group test --group workflow \
        pytest tests/paper -m integration

# Both fast suites.
test: test-core test-paper

# Execution *is* the test: nbclient fails the run on any raised exception, so
# the notebooks carry no assertion cells. Set ASTROGWB_NOTEBOOK_SMOKE=1 to
# shrink the chains, the catalog, and the convergence sweeps.

# Convert and execute the root notebooks.
test-notebooks:
    uv run --extra notebook --group jupyter \
        jupytext --to notebook --execute notebooks/catalog_convergence.py

# Convert notebook from py:percent format to .ipynb
convert-notebooks:
    uv run --group jupyter jupytext --to notebook notebooks/*.py

# The publishable distribution. `astrogwb.paper` ships inside it but is
# unimportable without the `paper` extra, whose dependencies stay out of the
# core requirement set.
build-core:
    uv build --no-sources
