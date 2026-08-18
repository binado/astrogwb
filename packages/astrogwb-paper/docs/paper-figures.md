# Paper figures

Experiment figures are part of the same DAG as their chains. Each figure's
presentation -- the ordered run IDs it compares and its LaTeX labels -- is
hard-coded in the script that draws it. There is no figure config to load:
changing a legend label is a code change, reviewed alongside the plot it
labels. The six detector networks compared by more than one figure are the one
shared piece, and they live in `astrogwb_paper.plotting.DETECTOR_NETWORKS` as
ordered `(run name, LaTeX label)` pairs.

Input and output paths are both named literally in
[`workflow/mcmc.smk`](../workflow/mcmc.smk), and every output is a valid
Snakemake target. Shared scientific values -- fiducials, frequency bounds,
cosmology grid settings -- stay in `inputs/mcmc.base.toml`.

Detector *lists* are never hard-coded next to a label: the script names its
experiment, and `astrogwb_paper.config.figures.resolve_networks` resolves each
run's detectors from `experiments/<experiment>.toml`. The detectors a figure
reports an SNR for are therefore always the ones its chain was sampled with.

The workflow imports that same tuple and expands its chain paths from it, so
chain order and legend order are one list rather than two that have to be kept
in step:

```python
from astrogwb_paper.plotting import DETECTOR_NETWORK_RUNS

chains=expand("outputs/chains/H0-all-detectors/{run}.nc",
              run=DETECTOR_NETWORK_RUNS),
```

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
workflow. The fiducial spectrum borrows the six detector networks of the
`H0-all-detectors` experiment rather than restating them, and keeps its
`OMEGA_GW_MIN` y-limit next to the axis it sets. All of them read
`inputs/mcmc.base.toml` directly.

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
`--base-config`, from which it reads its own fiducials and analysis grid, and
hard-codes its own labels and run order. Snakemake passes only what it owns:
the chain and catalog paths it built, and the output paths it declared.

The base config and the experiment TOML are declared inputs of each rule, so
editing a fiducial or a detector list rebuilds the figure; editing a label is a
code change and rebuilds it the same way. Config parsing stays free of JAX --
`astrogwb_paper.config.figures` is a stdlib-only leaf, guarded by a subprocess
test -- so `--help` and config errors stay cheap.
