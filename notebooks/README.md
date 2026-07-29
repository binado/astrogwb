# Notebooks

Workflows in this directory are stored as plain `.py` files in [Jupytext](https://jupytext.readthedocs.io/) **py:percent** format — a way to represent Jupyter notebooks as Python source instead of `.ipynb` JSON. That keeps diffs readable and lets normal Python tooling (e.g. Ruff) work on notebook code.

## Opening in Jupyter

To use the classic notebook UI, convert a `.py` file to `.ipynb`:

```bash
uvx jupytext --to ipynb notebooks/mcmc.py
```

Replace `mcmc.py` with any notebook below; Jupytext writes a sibling `.ipynb` you can open in JupyterLab.

Install Jupyter and plotting dependencies first:

```bash
uv sync --group dev
```

## Notebooks

- **`mcmc.py`** — importance-weighted NUTS inference (NumPyro port of ASGWB.jl)
- **`paper/amplitude_toy_model.py`** — smoke-test sibling of `mcmc.py` used by the paper workflow: same pipeline scaffolding, but the cosmology/population callback is replaced by a trivial single-parameter amplitude model, to sanity-check that NUTS recovers a known injection
- **`mcmc_plotting.py`** — load saved chains and produce diagnostic and corner plots
- **`logposterior_grid.py`** — evaluate the log-posterior on a parameter grid
