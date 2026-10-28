# The paper application

The reproducibility application for the `astrogwb` research project. It owns
the shared configuration and the notebooks that produce the paper's figures and
tables. Spectrum realizations are realization directories, written by the
package's `astrogwb simulate` command (see
[the simulator pipeline](simulator-pipeline.md)).

Generating data and reading it are separate steps. The notebooks never
simulate: [`workflow/coverage_draws.smk`](../workflow/coverage_draws.smk)
writes the ensembles, their detector noise and the reference catalogs under
`outputs/` (see [simulation inputs on SLURM](coverage-draws-on-slurm.md)), and
each notebook's configuration cell names the paths it reads. A missing input
is a `FileNotFoundError`.

The code ships inside the `astrogwb` wheel as `astrogwb.paper`, behind the
`paper` extra. Its *assets* -- `config/`, `scripts/` and `notebooks/` -- are not
packaged, so the application runs only from a checkout, with the repository
root as the working directory:

```bash
uv sync --extra notebook --extra cli --group dev
uv run astrogwb simulate --help
```

The notebooks are marimo notebooks. Open one from the repository root:

```bash
uv sync --extra notebook --group notebook
uv run --extra notebook --group notebook marimo edit notebooks/ensemble_spectrum.py
```

Every path is relative to the repository root -- library code names no absolute
path and does not go looking for a checkout. Committed configuration lives
under `config/`, including the simulation inputs `config/detectors.toml` and
`config/ensembles/<name>/`;
generated ensembles, catalogs and figures under `outputs/`.

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
```

The four shared layers (`defaults`, `waveforms`, `populations`, `detectors`)
are merged by `knf` (which the `knf` CLI also exposes in the shell). The packaged `geometry.toml` and
`sensitivity.toml` are merged first, using the same `[detectors.<name>]` tables
as the shared registry file. After the merge, every `"${a.b}"` string resolves
to the merged value at `a.b`, which is how a catalog names its waveform and
population. Each file opens with a comment saying what it is for.

The simulator registry loader lives in `astrogwb.detector.config`, so core simulation code does not import the paper application. Outputs store network
names only. The registry passed to a reader is trusted and must remain fixed
for that dataset; changing it requires regenerating outputs.

The notebooks consume the same bytes through
`astrogwb.paper.config.fiducials()` / `priors()` / `networks()` /
`detector_registry()`. Each accessor takes keyword overrides merged over the
file, so a notebook can vary one value without editing TOML or retyping the
table.

See [`docs/`](.) for catalog generation and paper figures.

## Reference catalogs

Generate an importance-integration catalog with the CLI:

```sh
uv run astrogwb model-catalog outputs/catalogs/reference-fiducial.h5 \
    --catalog-metadata config/catalogs/base.toml \
    --catalog-metadata config/catalogs/reference-fiducial.toml \
    --seed 42 --chunk-size 4096
```

Repeat `--catalog-metadata` to merge TOML layers in argument order using
pyknf. Later fields override earlier fields, tables merge recursively, and
`${...}` references and `extends` table inheritance are resolved by the shared
config loader. The merged result is a complete `CatalogMetadata` record: top-level `num_samples`,
`sampling`, optional `reference_redshift`, and tables for `waveform`,
`population` (including `population.model_kwargs`), and `fiducials`.
Omit optional null fields and `version`; the installed package supplies its
version. The seed and execution options (`--chunk-size`, `--platform`, and
`--cpu-threads`) are command-line arguments, outside the catalog record.

`config/catalogs/base.toml` holds the shared population, waveform, fiducials,
and default Sobol sample settings. Each `reference-*.toml` is a small override
layer: fiducial sets `reference_redshift = 1e-4` and serves the posterior,
coverage, and minimum-redshift-scan calculations; ladder changes the sample
count and sampling method. These files do not
inherit changes from the shared paper configuration. Regenerate affected
outputs after editing the base or variant. `workflow/coverage_draws.smk`
tracks both files as inputs and runs the command with the appropriate seeds;
notebooks read `reference-fiducial.h5` or `reference-ladder.h5`.
Regenerate `outputs/catalogs/reference-fiducial.h5` before running the notebooks
after this change; the former `reference-scan.h5` output is no longer used.
