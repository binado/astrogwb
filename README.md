# astrogwb workspace

This repository contains two Python projects managed through one uv workspace:

- [`astrogwb`](packages/astrogwb/): the publishable scientific library for
  compact-binary stochastic gravitational-wave background inference.
- [`astrogwb-paper`](packages/astrogwb-paper/): the private application owning
  MCMC runners, catalog workflows, campaign configuration, cluster profiles,
  and manuscript notebooks.

The dependency direction is one-way: `astrogwb-paper` depends on `astrogwb`.
The workspace uses one lockfile and installs both members editably for local
development.

## Core library quick start

Install the released scientific library from PyPI:

```bash
python -m pip install astrogwb
```

For core development in this checkout:

```bash
uv sync --package astrogwb --group test
uv run --package astrogwb --group test pytest packages/astrogwb/tests
```

## Paper workflow quick start

```bash
uv sync --package astrogwb-paper --group dev
uv run astrogwb-workflow --help
uv run astrogwb-workflow paper
```

Build the library exactly as it will be published, without workspace source
overrides:

```bash
uv build --package astrogwb --no-sources
```

Generated catalogs, chains, figures, and logs remain in root-level `out/`,
`chains/`, `figures/`, and `logs/` directories. Detailed workflow documentation
lives under [`packages/astrogwb-paper/docs`](packages/astrogwb-paper/docs/).
