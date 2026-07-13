# astrogwb

This is a package for Bayesian inference of cosmological and astrophysical parameters
with the stochastic gravitational-wave background (SGWB) from stellar-mass compact binary coalescences (CBCs) detectable by ground-based interferometer networks such as LIGO, Virgo, KAGRA, Einstein Telescope and Cosmic Explorer.

## Installation

The easiest way to install the package is using the `uv` package manager:

```bash
git clone git@github.com:binado/astrogwb.git
uv sync --all-groups --all-extras
```

Core inference libraries install without the headless runner dependencies. The
optional ``mcmc`` extra adds pydantic for validated ``RunConfig`` parsing and
ArviZ's NetCDF output support used by [`scripts/run_mcmc.py`](scripts/run_mcmc.py)
and related tools:

```bash
uv sync --extra mcmc
# or with notebook/dev tooling:
uv sync --extra mcmc --group dev
```

## Documentation

Typical path: [catalogs](docs/catalog-generation.md) → [inference](docs/running-inference.md) → [Snakemake](docs/snakemake-workflow.md). Paper builds are covered in [Paper figures](docs/paper-figures.md).

- [Generating catalogs](docs/catalog-generation.md): building a population of CBCs with `gwmock-pop` and its waveform catalog with `gwmock-signal`.
- [Running inference](docs/running-inference.md): running `scripts/run_mcmc.py` / `notebooks/mcmc.py`, generating sweep configs, and interpreting chain outputs / diagnostics.
- [Paper figures](docs/paper-figures.md): the analysis notebooks, `configs/paper.toml`, and dry-run/build commands for the paper's figures.
- [Snakemake workflow](docs/snakemake-workflow.md): the catalog, MCMC, and paper Snakefiles, batch manifests, and SLURM/local deployment.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)

## Development

Install dependencies (the dev group also bundles Jupyter, arviz, corner, and
matplotlib so the MCMC notebook in `notebooks/` runs out of the box). The
`mcmc` extra alone is sufficient for the headless runner, including ArviZ
NetCDF output; include it with dev tools for related tests:

```bash
uv sync --group dev
uv sync --extra mcmc --group dev
```

Run tests:

```bash
uv run --extra mcmc --group dev pytest
uv run --extra mcmc --group dev pytest -m "not integration"   # fast unit tests only
```

Regression fixtures under `tests/fixtures/` are committed; integration tests
cross-check gwfast and skip if optional fixtures are missing.

Format and lint:

```bash
uv run ruff format .
uv run ruff check .
```

Type check:

```bash
uvx ty check
```
