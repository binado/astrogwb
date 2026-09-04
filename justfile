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

# The end-to-end NUTS runs, slow enough to be their own CI job. These are the
# tests that cross-check against gwmock-pop, so they need `simulation` -- which
# test-core deliberately does not have, since core must work without it.
test-integration:
    uv run --frozen --isolated --no-default-groups \
        --extra io --extra simulation --group test \
        pytest tests/core -m integration

# Both fast suites.
test: test-core test-paper

# Execution *is* the test: nbclient fails the run on any raised exception, so
# the notebooks carry no assertion cells. Set ASTROGWB_NOTEBOOK_SMOKE=1 to
# shrink the chains, the catalog, and the convergence sweeps.

# Convert and execute the root notebooks.
test-notebooks:
    uv run --extra simulation --extra io --group jupyter \
        jupytext --to notebook --execute notebooks/catalog_convergence.py

# Convert notebook from py:percent format to .ipynb
convert-notebooks:
    uv run --group jupyter jupytext --to notebook notebooks/*.py

# The publishable distribution. `astrogwb.paper` ships inside it but is
# unimportable without the `paper` extra -- proved by the CI wheel smoke test.
build-core:
    uv build --no-sources

# Regenerate the committed core mock-population fixture.
generate-mock-population-fixture:
    uv run --frozen --isolated --no-default-groups \
        --group fixture \
        python scripts/generate_mock_population_fixture.py \
        --population tests/core/fixtures/mock_bns_population.yaml \
        --output tests/core/fixtures/mock_bns_population.csv \
        --num-samples 1024 \
        --seed 41

# Build the wheel, install it WITHOUT extras, and prove the boundary holds.
# This is the packaging half of the one-way dependency; ruff's TID251 ban is
# the source half. Run by the CI `build-core` job.
smoke-wheel: build-core
    rm -rf .wheel-venv
    uv venv .wheel-venv
    uv pip install --python .wheel-venv/bin/python dist/*.whl
    .wheel-venv/bin/python scripts/check_wheel.py
