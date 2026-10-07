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
  --allowed-rules run_mcmc plot_cosmological_parameters \
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

`notebooks/fiducial_spectrum.py` is a marimo notebook. Open it from the repository root with `uv run --extra notebook --group jupyter marimo edit notebooks/fiducial_spectrum.py`.

Every path is relative to the repository root -- library code names no absolute
path and does not go looking for a checkout. Committed configuration lives
under `config/`; generated catalogs, chains, and figures under `outputs/`;
scheduler and runtime logs under `logs/`.

Filenames are the mapping for runs, and catalogs are named by the content
hash of what a run asks for, so there is no registry file:

```text
config/defaults.toml                            shared scientific defaults:
                                                [analysis] band, target population,
                                                  the injection and proposal catalogs
                                                [catalog] the default draw, shaped
                                                  like CatalogMetadata
                                                [fiducials] every parameter (also
                                                  what catalogs are drawn at)
                                                [priors] every parameter's prior
                                                [sampler] RNG seed, NUTS defaults
config/waveforms.toml                           [waveforms.<name>] named waveforms
config/populations.toml                         [populations.<name>] named populations
config/detectors.toml                           [networks] membership and optional
                                                [detectors] geometry/PSD/label overrides
config/plotting.toml                            LaTeX labels and savefig settings
                                                (presentation; not a run layer)
config/runs/<experiment>/_base.toml             the experiment override
config/runs/<experiment>/<run>.toml             the run override
  -> outputs/chains/<experiment>/<run>.nc       the chain
  -> outputs/chains/<experiment>/<run>.json     the config it was sampled with
  -> outputs/catalogs/<key>.h5                  one per distinct catalog it asks for
```

A run is six TOML layers: the four shared ones (`config/defaults.toml`,
`config/waveforms.toml`, `config/populations.toml`, `config/detectors.toml`),
its experiment's `_base.toml`, and its own file, merged by `knf` (which the
`knf` CLI also exposes in the shell). The packaged `geometry.toml` and
`sensitivity.toml` are merged first, using the same `[detectors.<name>]` tables
as the shared registry file. After the merge, every `"${a.b}"` string resolves
to the merged value at `a.b`, which is how a catalog names its waveform and
population. Each file opens with a comment saying what it is for.
Several shared blocks are read by more than the workflow: the notebooks and
figure scripts consume the same bytes through
`astrogwb.paper.config.fiducials()` / `priors()` / `networks()` / `detector_registry()`. Each accessor
takes keyword overrides merged over the file, so a notebook can vary one value
without editing TOML or retyping the table.

There is no intermediate assembled config. The layers are merged in process by
whatever runs -- the workflow passes them on argv as repeated `--config` flags,
and declares those same files as the rule's `input:` -- and `run_mcmc` writes
the resolved config next to the chain, stamping the ordered layer paths into
the chain itself.

See [`docs/`](.) for catalog generation, inference, paper figures, and
Snakemake/SLURM workflows.
