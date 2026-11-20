default: lint typecheck test-core
mod docs "docs.just"

lint:
    uv lock --check
    uv run --group dev ruff check
    uv run --group dev ruff format --check

fmt:
    uv run --group dev ruff check --fix
    uv run --group dev ruff format

typecheck:
    uv run --extra cli --group dev ty check

test-core:
    uv run --frozen --isolated --no-default-groups --extra cli --group test python -m pytest -m "not integration"

test: test-core

# The core suite on the lowest version each direct dependency allows. Re-resolves
# instead of reading uv.lock (no --frozen), so it rewrites the lock: run it in CI.
test-floor:
    uv run --isolated --resolution lowest-direct --no-default-groups --extra cli --group test python -m pytest -m "not integration"

test-integration:
    uv run --frozen --isolated --no-default-groups --extra cli --group test python -m pytest -m integration

# Comparisons against bilby (live) and pygwb (frozen references).
test-parity:
    uv run --frozen --isolated --no-default-groups --group parity python -m pytest tests/parity

# Rewrite tests/parity/references/pygwb.npz; pygwb runs in its own environment.
parity-references:
    uv run python tests/parity/references/draw_coba_events.py
    uv run --no-project tests/parity/references/generate_pygwb.py
