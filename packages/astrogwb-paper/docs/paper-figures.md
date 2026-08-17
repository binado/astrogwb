# Paper figures

Experiment figures are part of the same DAG as their chains. Ordered run IDs,
labels, styles, and output paths live in the optional `[figure]` table of:

```text
experiments/<experiment>.toml
```

Those `output_*` paths are valid Snakemake targets. Shared scientific values
such as fiducials, frequency bounds, and cosmology grid settings come from
`inputs/mcmc.base.toml`.

## Experiment figures

These complete experiment targets include local post-processing:

- `H0-all-detectors`: detector posterior comparison and CSV/LaTeX constraint
  table;
- `modified-propagation-all-detectors`: propagation corners, marginal
  comparison, and CSV/LaTeX tables;
- `H0-merger-rate`: fixed-versus-sampled merger-rate comparison, corner, and
  table;
- `H0-omega-m`: standard and relative-ESS corner figures.

Preview or build one (from `packages/astrogwb-paper/`):

```bash
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/local --cores 8 --dry-run H0_all_detectors
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/slurm H0_all_detectors
```

Rebuild a single figure by its output path:

```bash
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/local --cores 8 \
  outputs/figures/H0-all-detectors/H0-by-detector.pdf
```

With a SLURM profile, sampling runs remotely and figure rules run locally on the
submit host after their chains finish. The submit host must remain attached,
share the output filesystem, and provide plotting dependencies.

Use the `<experiment>_chains` target when post-processing should happen in a
later invocation.

## Standalone figures

The amplitude toy model, fiducial spectrum, effective detector PSD comparison,
and importance-weight grids are explicit standalone rules in the unified
workflow. Their settings live in
[`inputs/figures/standalone.toml`](../inputs/figures/standalone.toml).

```bash
snakemake --snakefile workflow/mcmc.smk --cores 1 --dry-run standalone_figures
snakemake --snakefile workflow/mcmc.smk --cores 1 standalone_figures
```

Build one declared output directly:

```bash
snakemake --snakefile workflow/mcmc.smk --cores 1 \
  outputs/figures/standalone/fiducial_spectrum.pdf
```

All new figure products are written under `outputs/figures/`.

## Notebook use

The Jupytext sources under `notebooks/paper/` remain directly executable.
Snakemake passes the experiment TOML, the shared MCMC base, and the chain files
the DAG must declare. Labels, detector lists, and output paths come from the
experiment file.
