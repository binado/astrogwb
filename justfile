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
    uv run --extra notebook --extra cli --group dev ty check

# The core suite installs astrogwb WITHOUT the `paper` extra, so a core test
# that reaches across the boundary fails here rather than passing by accident
# on a developer's fully-synced venv. `io` and `cli` are the extras core owns.

# Core library tests.
test-core:
    uv run --frozen --isolated --no-default-groups \
        --extra cli --group test \
        pytest tests/core -m "not integration"

# Paper application tests: configuration, CLI, runtime.
test-paper:
    uv run --frozen --isolated --no-default-groups \
        --extra notebook --group test \
        pytest tests/paper -m "not integration"

# The end-to-end NUTS runs, slow enough to be their own CI job. These are also
# the tests that cross-check the cosmology against gwmock-pop, so they need
# `simulation` -- which test-core deliberately does not have, since core must
# work without it.
#
# The second line is the paper half: its integration-marked tests. Pytest
# exits 5 if that suite has no integration tests; only that empty selection
# is accepted in addition to success. Test failures still fail the recipe.
test-integration:
    uv run --frozen --isolated --no-default-groups \
        --extra cli --extra simulation --group test \
        pytest tests/core -m integration
    uv run --frozen --isolated --no-default-groups \
        --extra notebook --group test \
        pytest tests/paper -m integration || test "$?" -eq 5

# Both fast suites.
test: test-core test-paper

run-notebook notebook:
    uv run --group notebook marimo edit {{ notebook }}

# Paper outputs, not run by CI. Each writes under outputs/; Snakemake skips
# files that exist, so delete one to redraw it.

# The reference catalogs, locally: seconds each. Seeds live in Snakefile.
catalogs cores="4":
    uv run --no-sync snakemake catalogs --cores {{ cores }}

# One ensemble draw locally, e.g. `just draw fiducial 41 0`, the grid
# notebooks' injection: about 20 min on 8 cores.
draw ensemble="fiducial" seed="41" index="0" cores="8":
    uv run --no-sync snakemake outputs/ensembles/{{ ensemble }}/{{ seed }}-{{ index }}.h5 \
        --cores {{ cores }} --config threads={{ cores }}

# Every ensemble draw and catalog on SLURM; extra arguments go to Snakemake,
# e.g. `just cluster --dry-run` or `just cluster --config stop=10`.
cluster *args:
    uv run --no-sync snakemake --profile workflow/profiles/slurm {{ args }}

# Run a marimo notebook headlessly, writing its figures under
# outputs/figures/<name>/, e.g. `just notebook fiducial_spectrum`.
notebook name:
    uv run --extra notebook python notebooks/{{ name }}.py

# The paper outputs that need only the catalogs and draw 41-0, one at a time:
# the grid notebooks each hold several GB, too much to run side by side.
figures: (notebook "fiducial_spectrum") (notebook "cosmology_grid_posteriors") (notebook "madau_dickinson_grid")
    uv run --extra notebook python scripts/shot_noise_budget.py

# The publishable distribution. `astrogwb.paper` ships inside it but is
# unimportable without the `paper` extra, whose dependencies stay out of the
# core requirement set.
build-core:
    uv build --no-sources
