# The paper application

The reproducibility application for the `astrogwb` research project. It owns
the shared configuration, the notebooks that produce the paper's figures and
tables, and the spectrum-simulation script.

The code ships inside the `astrogwb` wheel as `astrogwb.paper`, behind the
`paper` extra. Its *assets* -- `config/`, `scripts/` and `notebooks/` -- are not
packaged, so the application runs only from a checkout, with the repository
root as the working directory:

```bash
uv sync --extra notebook --group dev
uv run --extra paper python scripts/simulate_spectra.py --help
```

The notebooks are marimo notebooks (and one Jupytext `py:percent` source).
Open one from the repository root:

```bash
uv sync --extra notebook --group jupyter
uv run --extra notebook --group jupyter marimo edit notebooks/fiducial_spectrum.py
```

Every path is relative to the repository root -- library code names no absolute
path and does not go looking for a checkout. Committed configuration lives
under `config/`; generated catalogs and figures under `outputs/`.

The MCMC pipeline (Snakemake workflow, `run_mcmc`, SLURM profiles and the
`config/runs/` experiments) was removed; it remains available at the
`mcmc-pipeline-final` git tag.

## Configuration

```text
config/defaults.toml                            shared scientific defaults:
                                                [analysis] observation time, band
                                                [catalog] the default draw, shaped
                                                  like CatalogMetadata
                                                [fiducials] every parameter (also
                                                  what catalogs are drawn at)
                                                [priors] every parameter's prior
config/waveforms.toml                           [waveforms.<name>] named waveforms
config/populations.toml                         [populations.<name>] named populations
config/detectors.toml                           [networks] membership and optional
                                                [detectors] geometry/PSD/label overrides
config/plotting.toml                            LaTeX labels and savefig settings
                                                (presentation; not a config layer)
config/simulations/spectrum/<name>.toml         a spectrum simulation layer
```

The four shared layers (`defaults`, `waveforms`, `populations`, `detectors`)
are merged by `knf` (which the `knf` CLI also exposes in the shell), optionally
followed by a simulation layer. The packaged `geometry.toml` and
`sensitivity.toml` are merged first, using the same `[detectors.<name>]` tables
as the shared registry file. After the merge, every `"${a.b}"` string resolves
to the merged value at `a.b`, which is how a catalog names its waveform and
population. Each file opens with a comment saying what it is for.

The notebooks consume the same bytes through
`astrogwb.paper.config.fiducials()` / `priors()` / `networks()` /
`detector_registry()`. Each accessor takes keyword overrides merged over the
file, so a notebook can vary one value without editing TOML or retyping the
table.

See [`docs/`](.) for catalog generation and paper figures.
