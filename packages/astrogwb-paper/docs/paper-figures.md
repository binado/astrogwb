# Paper figures

Experiment figures are part of the same DAG as their chains. Each figure's
presentation -- the ordered run IDs it compares and its LaTeX labels -- lives
in [`inputs/figures/`](../inputs/figures), one file per figure group:

```text
inputs/figures/<figure>.toml
inputs/figures/detector-networks.toml   # shared network label registry
```

Output paths are named literally in the rules' `output:` blocks in
[`workflow/mcmc.smk`](../workflow/mcmc.smk), and each is a valid Snakemake
target. Shared scientific values -- fiducials, frequency bounds, cosmology grid
settings -- stay in `inputs/mcmc.base.toml`.

Detector *lists* are never duplicated into a figure config: a figure names its
experiment and the runs it compares, and `astrogwb_paper.config.figures`
resolves each run's detectors from `experiments/<experiment>.toml`. Name, label,
and detectors therefore always travel together.

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
workflow. Only the fiducial spectrum needs presentation settings of its own, in
[`inputs/figures/fiducial-spectrum.toml`](../inputs/figures/fiducial-spectrum.toml);
it borrows the six detector networks of the `H0-all-detectors` experiment
rather than restating them. The rest read `inputs/mcmc.base.toml` directly.

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

## Scripts

Figure entry points are plain Python scripts under `scripts/`. Each takes
`--base-config` and, where it has one, `--figure-config`, and reads its own
fiducials, analysis grid, labels, and detector networks from them. Snakemake
passes only what it owns: the chain and catalog paths it built, and the output
paths it declared.

Both config files are declared inputs of the rule, so editing a label or a
fiducial rebuilds the figure. Config parsing stays free of JAX --
`astrogwb_paper.config.figures` is a stdlib-only leaf, guarded by a subprocess
test -- so `--help` and config errors stay cheap.
