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
    uv run --extra notebook --extra blackjax --extra cli --group dev ty check

# The core suite installs astrogwb WITHOUT the `paper` extra, so a core test
# that reaches across the boundary fails here rather than passing by accident
# on a developer's fully-synced venv. `io` and `cli` are the extras core owns.

# Core library tests.
test-core:
    uv run --frozen --isolated --no-default-groups \
        --extra cli --group test \
        pytest tests/core -m "not integration"

# Paper application tests: configuration, CLI, runtime, plotting.
test-paper:
    uv run --frozen --isolated --no-default-groups \
        --extra notebook --group test \
        pytest tests/paper -m "not integration"

# The end-to-end NUTS runs, slow enough to be their own CI job. These are also
# the tests that cross-check the cosmology against gwmock-pop, so they need
# `simulation` -- which test-core deliberately does not have, since core must
# work without it.
#
# The second line is the paper half: its integration-marked tests.
test-integration:
    uv run --frozen --isolated --no-default-groups \
        --extra cli --extra simulation --extra blackjax --group test \
        pytest tests/core -m integration
    uv run --frozen --isolated --no-default-groups \
        --extra notebook --group test \
        pytest tests/paper -m integration

# Both fast suites.
test: test-core test-paper

run-notebook notebook:
    uv run --group notebook marimo edit {{ notebook }}

# The publishable distribution. `astrogwb.paper` ships inside it but is
# unimportable without the `paper` extra, whose dependencies stay out of the
# core requirement set.
build-core:
    uv build --no-sources
