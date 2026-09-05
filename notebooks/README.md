# Notebooks

Workflows in this directory are stored as plain `.py` files in [Jupytext](https://jupytext.readthedocs.io/) **py:percent** format — a way to represent Jupyter notebooks as Python source instead of `.ipynb` JSON. That keeps diffs readable and lets normal Python tooling (Ruff, `ty`) work on notebook code. The `.py` is the source of truth; `*.ipynb` is gitignored.

## Two kinds, one directory

They used to live in two trees, one per workspace package. The packages merged;
the distinction did not, and it is worth knowing which kind you are opening.

**Self-contained** — demonstrations of the core `astrogwb` library. They carry
their population graph inline and build their own catalog through
`astrogwb.catalog`, so they run against a clean checkout with
`astrogwb[simulation,io]` installed and nothing else. They read no file outside
themselves and import nothing from `tests/core`.

- **`catalog_convergence.py`** — how the catalog contraction approaches the
  analytic spectrum as the number of sources grows ($\propto N^{-1/2}$), and
  how the SNR and the log-likelihood *ratio* converge as the frequency
  resolution $\Delta f$ is refined. The companion to
  `tests/core/test_frequency_resolution.py`, which owns the
  tolerances; the notebook owns the picture.

**Paper analyses** — these drive real waveform banks through the
`astrogwb.paper` configuration layer and are not self-contained by design. They
need built banks under `outputs/banks/`, the `notebook` extra, and the
repository root as the working directory.

- **`mcmc.py`** — importance-weighted NUTS inference (NumPyro port of ASGWB.jl)
- **`mcmc_plotting.py`** — load saved chains and produce diagnostic and corner
  plots
- **`logposterior_grid.py`** — evaluate the log-posterior on a parameter grid

The three paper notebooks merge a run's config layers with
`assemble_run(*REFERENCE_RUN)` — the by-name convenience wrapper over the same
merge the workflow performs by passing layer paths on argv. Neither reads an
intermediate artifact, so these run against a fresh clone.

## Running them

The self-contained notebook converts and executes through `just`:

```bash
just test-notebooks
```

That recipe installs the `simulation` and `io` extras and the `jupyter` group,
and nothing else — which is what keeps the notebook's self-containment honest. The notebook keeps its
scientifically significant population graph and luminosity-distance
recomputation inline, then passes the prepared parameters through
`Catalog.from_generator(..., generator=AnalyticInspiralGenerator(...))`.

`ASTROGWB_NOTEBOOK_SMOKE=1` shrinks the catalog and the convergence sweeps. It
changes only how long the notebook runs, never which
cells execute — there is one code path in both modes:

```bash
ASTROGWB_NOTEBOOK_SMOKE=1 just test-notebooks
```

## Opening in Jupyter

To use the classic notebook UI, convert a `.py` file to `.ipynb`:

```bash
uvx jupytext --to ipynb notebooks/catalog_convergence.py
```

Jupytext writes a sibling `.ipynb` you can open in JupyterLab. Install the
environment first — the `notebook` extra is the library stack the notebooks
import, the `jupyter` group is the tooling that runs them:

```bash
uv sync --extra notebook --group jupyter
```

## The catalog cache

The self-contained notebook writes the catalog it builds to a
`notebooks/*.h5` file
(gitignored) and reuses it on the next run. The file is a cache, not an input:
delete it and the notebook rebuilds from its own inline population graph.
Persistence is not part of the core dependency set: `astrogwb` builds an
array-native `Catalog`, and `astrogwb.catalog.io` writes it to HDF5 behind the
`io` extra, so the notebook needs `astrogwb[simulation,io]` and nothing else.

The reuse is guarded on more than the file existing. The notebook compares
the stored catalog's attributes — `df`, `population_num_samples`,
`population_seed`, the
band, and the fiducials — against its configuration cell, and rebuilds on any
mismatch. Without that, editing `CATALOG_DF` and re-running would silently
analyse the old frequency grid, which in `catalog_convergence.py` would
invalidate the entire result while looking perfectly healthy.

Because the catalogs are regenerated live rather than read from the committed
`tests/core/fixtures/mock_bns_population.csv`, a `gwmock-pop`
version bump can shift the draw: `GraphSimulator` derives its RNG keys from
the graph. The notebook prints the installed version alongside the seed and
the grid, so a changed plot is explainable rather than mysterious.
