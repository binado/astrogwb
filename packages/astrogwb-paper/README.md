# astrogwb-paper

Private reproducibility application for the `astrogwb` research project. It
owns the MCMC runners, experiment configuration, Snakemake workflows, cluster
profiles, paper figure scripts, and exploratory notebooks.

The application intentionally runs only from a monorepo checkout. Install its
local development and workflow dependencies from the workspace root:

```bash
uv sync --package astrogwb-paper --group dev
uv run astrogwb-run-mcmc --help
```

Run the Snakemake workflows directly from this package's root (`uv run --group
workflow snakemake` from the workspace root also works); preview with
`--dry-run`, omit it to execute:

```bash
cd packages/astrogwb-paper
snakemake --snakefile Snakefile \
  --allowed-rules assemble_config run_mcmc plot_cosmological_parameters \
  --profile profiles/local --cores 8 --dry-run plot_cosmological_parameters
```

The committed `local`, `slurm`, and
`slurm-cpu` profiles live under `profiles/`. Install the SLURM executor with
`uv sync --package astrogwb-paper --group slurm`, adding `--extra cuda` for
GPU jobs.

For notebooks, install the `notebook` extra and open or convert the Jupytext
sources under `notebooks/`:

```bash
uv sync --package astrogwb-paper --extra notebook
uvx jupytext --to ipynb packages/astrogwb-paper/notebooks/mcmc.py
```

All user paths are resolved from this package's root. Generated populations,
catalogs, canonical configs, chains, and figures live under `outputs/`;
scheduler/runtime logs remain under `logs/`. All live inside this member.
The shared MCMC base and all curated experiment runs are declared in
`inputs/config.yaml`. The shared catalog base and named recipes are declared
in `inputs/catalogs.yaml`.

See [`docs/`](docs/) for catalog generation, inference, paper figures, and
Snakemake/SLURM workflows.
