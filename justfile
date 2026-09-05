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

# The CI `typecheck` job.
typecheck:
    uv run --group dev ty check

# The isolated environment has astrogwb and nothing else -- no astrogwb_paper,
# no matplotlib -- so an accidental plotting import in a core test fails here.

# Core library tests.
test-core:
    uv run --frozen --isolated --no-default-groups \
        --package astrogwb --group test \
        pytest packages/astrogwb/tests -m "not integration"

# Paper application tests: configuration, CLI, runtime, IO, paths, workflows.
test-paper:
    uv run --frozen --isolated --no-default-groups \
        --package astrogwb-paper --group test --group workflow \
        pytest packages/astrogwb-paper/tests -m "not integration"

# The end-to-end NUTS runs, slow enough to be their own CI job.
test-integration:
    uv run --frozen --isolated --no-default-groups \
        --package astrogwb --group test \
        pytest packages/astrogwb/tests -m integration

# Both fast suites.
test: test-core test-paper

# The publishable core distribution.
build-core:
    uv build --package astrogwb --no-sources
