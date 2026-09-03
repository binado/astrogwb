# Notebooks

Workflows in this directory are stored as plain `.py` files in [Jupytext](https://jupytext.readthedocs.io/) **py:percent** format — a way to represent Jupyter notebooks as Python source instead of `.ipynb` JSON. That keeps diffs readable and lets normal Python tooling (Ruff, `ty`) work on notebook code. The `.py` is the source of truth; `*.ipynb` is gitignored.

## What belongs where

- **This directory** — self-contained demonstrations of the core `astrogwb`
  library. The notebook carries its own population graph inline and builds
  its own catalog through `astrogwb.catalog`, so it runs against a clean
  checkout with `astrogwb[simulation]` and the plotting extra installed. It
  reads no file outside itself and does not import from
  `packages/astrogwb/tests`.
- **[`packages/astrogwb-paper/notebooks/`](../packages/astrogwb-paper/notebooks/)** —
  paper analyses. Those drive real waveform banks through the
  `astrogwb-paper` configuration layer and are not self-contained by design.

## Notebooks

- **`catalog_convergence.py`** — how the catalog contraction approaches the
  analytic spectrum as the number of sources grows ($\propto N^{-1/2}$), and
  how the SNR and the log-likelihood *ratio* converge as the frequency
  resolution $\Delta f$ is refined. The companion to
  `packages/astrogwb/tests/test_frequency_resolution.py`, which owns the
  tolerances; the notebook owns the picture.

## Running them

The notebook converts and executes through the `notebook` dependency group:

```bash
just test-notebooks
```

That group explicitly installs `astrogwb[simulation]`. The notebook keeps its
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
environment first:

```bash
uv sync --group notebook
```

## The catalog cache

The notebook writes the catalog it builds to a `notebooks/*.h5` file
(gitignored) and reuses it on the next run. The file is a cache, not an input:
delete it and the notebook rebuilds from its own inline population graph.
Persistence itself is not part of the core library: `astrogwb` builds an
array-native `Catalog`, and `astrogwb_paper.catalog_io` is what writes it to
HDF5, so the notebook reaches for the paper package only for the cache.

The reuse is guarded on more than the file existing. The notebook compares
the stored catalog's attributes — `df`, `population_num_samples`,
`population_seed`, the
band, and the fiducials — against its configuration cell, and rebuilds on any
mismatch. Without that, editing `CATALOG_DF` and re-running would silently
analyse the old frequency grid, which in `catalog_convergence.py` would
invalidate the entire result while looking perfectly healthy.

Because the catalogs are regenerated live rather than read from the committed
`packages/astrogwb/tests/fixtures/mock_bns_population.csv`, a `gwmock-pop`
version bump can shift the draw: `GraphSimulator` derives its RNG keys from
the graph. The notebook prints the installed version alongside the seed and
the grid, so a changed plot is explainable rather than mysterious.
