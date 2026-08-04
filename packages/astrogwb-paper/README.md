# astrogwb-paper

Private reproducibility application for the `astrogwb` research project. It
owns the MCMC runners, campaign configuration, Snakemake workflows, cluster
profiles, and paper notebooks.

The application intentionally runs only from a monorepo checkout. Install its
local development and workflow dependencies from the workspace root:

```bash
uv sync --package astrogwb-paper --group dev
uv run astrogwb-workflow --help
uv run astrogwb-run-mcmc --help
```

Use `uv run astrogwb-workflow catalog ...`, `mcmc ...`, and `paper` for local
dry-runs; add `--submit` to execute. The committed `local`, `slurm`, and
`slurm-cpu` profiles live under `profiles/`. Install the SLURM executor with
`uv sync --package astrogwb-paper --group slurm`, adding `--extra cuda` for
GPU jobs.

For notebooks, install the `notebook` extra and open or convert the Jupytext
sources under `notebooks/`:

```bash
uv sync --package astrogwb-paper --extra notebook
uvx jupytext --to ipynb packages/astrogwb-paper/notebooks/mcmc.py
```

All user paths are resolved from the workspace root. Generated populations and
catalogs remain under `out/`, chains under `chains/`, figures under `figures/`,
and scheduler/runtime logs under `logs/`; none are written inside this member.

See [`docs/`](docs/) for catalog generation, inference, paper figures, and
Snakemake/SLURM workflows.
