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
uv run --extra paper python scripts/run_mcmc.py --help
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
under `config/`; generated catalogs, chains, and figures under `outputs/`;
scheduler and runtime logs under `logs/`.

Filenames are the mapping, so there is no registry file:

```text
config/catalogs/<name>.json      ->  outputs/catalogs/<name>.h5
config/population.json                          [population] every catalog shares
                                                (a catalog layer; not a run layer)
config/fiducials.json                           fiducial value of every parameter
                                                (a run layer AND a catalog layer:
                                                the values a catalog is drawn at)
config/priors.json                              prior on every parameter
config/networks.json                            each detector network, by name
config/waveform.json                            [waveform] every catalog shares
                                                (a catalog layer; not a run layer)
config/plotting.json                            LaTeX labels and savefig settings
                                                (presentation; not a run layer)
config/analysis/base/*.toml                     the remaining shared settings
config/analysis/runs/<experiment>/_base.toml    the experiment override
config/analysis/runs/<experiment>/<run>.toml
  -> outputs/chains/<experiment>/<run>.nc       the chain
  -> outputs/chains/<experiment>/<run>.json     the config it was sampled with
```

The three JSON files are layer 0. They are top-level and JSON because more
than the workflow reads them: the notebooks and figure scripts consume the same
bytes through `astrogwb.paper.config.fiducials()` / `priors()` / `networks()`,
and `jq` reads them without importing the package. Each accessor takes keyword
overrides merged over the file, so a notebook can vary one value without
editing JSON or retyping the table.

There is no intermediate assembled config. The four layers are merged in
process by whatever runs -- the workflow passes them on argv as repeated
`--config` flags, and declares those same files as the rule's `input:` -- and
`run_mcmc` writes the resolved config next to the chain, stamping the ordered
layer paths into the chain itself.

See [`docs/`](.) for catalog generation, inference, paper figures, and
Snakemake/SLURM workflows.
