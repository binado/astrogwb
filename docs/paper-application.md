# The paper application

The reproducibility application for the `astrogwb` research project. It owns
the MCMC runners, experiment configuration, Snakemake workflows, cluster
profiles, paper figure scripts, and exploratory notebooks.

The code ships inside the `astrogwb` wheel as `astrogwb.paper`, behind the
`paper` extra. Its *assets* -- `config/`, `scripts/`, `notebooks/`,
`profiles/`, and the `Snakefile` -- are not packaged, so the application runs
only from a checkout, with the repository root as the working directory:

```bash
uv sync --extra notebook --group dev
uv run astrogwb-run-mcmc --help
```

Run the workflow from the repository root; preview with `--dry-run`, omit it to
execute:

```bash
uv run --group workflow snakemake --snakefile Snakefile \
  --allowed-rules validate run_mcmc plot_cosmological_parameters \
  --profile profiles/local --cores 8 --dry-run plot_cosmological_parameters
```

The committed `local`, `slurm`, and `slurm-cpu` profiles live under
`profiles/`. Install the SLURM executor with `uv sync --extra paper --group
slurm`, adding `--extra cuda` for GPU jobs.

For notebooks, install the `notebook` extra and open or convert the Jupytext
sources under `notebooks/`:

```bash
uv sync --extra notebook --group jupyter
uvx jupytext --to ipynb notebooks/mcmc.py
```

Every path is relative to the repository root -- library code names no absolute
path and does not go looking for a checkout. Committed configuration lives
under `config/`; generated banks, chains, and figures under `outputs/`;
scheduler and runtime logs under `logs/`.

Filenames are the mapping, so there is no registry file:

```text
config/populations/<population>.yaml            one complete population graph
config/banks/<bank>.toml         -> outputs/banks/<bank>.h5
config/analysis/base/*.toml                     settings every run shares
config/analysis/runs/<experiment>/_base.toml    the experiment override
config/analysis/runs/<experiment>/<run>.toml
  -> outputs/chains/<experiment>/<run>.nc       the chain
  -> outputs/chains/<experiment>/<run>.json     the config it was sampled with
```

There is no intermediate assembled config. The three layers are merged in
process by whatever runs -- the workflow passes them on argv as repeated
`--config` flags, and declares those same files as the rule's `input:` -- and
`run_mcmc` writes the resolved config next to the chain, stamping the ordered
layer paths into the chain itself.

See [`docs/`](.) for bank generation, inference, paper figures, and
Snakemake/SLURM workflows.
