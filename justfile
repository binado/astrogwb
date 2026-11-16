default: lint typecheck test-core

lint:
    uv lock --check
    uv run --group dev ruff check
    uv run --group dev ruff format --check

fmt:
    uv run --group dev ruff check --fix
    uv run --group dev ruff format

typecheck:
    uv run --extra cli --extra simulation --group dev ty check

test-core:
    uv run --frozen --isolated --no-default-groups --extra cli --group test python -m pytest tests/core -m "not integration"

test: test-core

test-integration:
    uv run --frozen --isolated --no-default-groups --extra cli --extra simulation --group test python -m pytest tests/core -m integration

build-core:
    uv build --no-sources
