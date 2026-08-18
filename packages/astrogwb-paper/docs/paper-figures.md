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

The detector-network, merger-rate, and Omega-m analyses form one paper section.
The `plot_cosmological_parameters` rule consumes all nine chains and produces their
five figures and two CSV/LaTeX table pairs in one script invocation. The
existing artifact paths remain grouped by their source experiment.

Preview or build the section (from `packages/astrogwb-paper/`):

```bash
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/local --cores 8 --dry-run plot_cosmological_parameters
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/slurm plot_cosmological_parameters
```

Every artifact remains a valid Snakemake target, but because the rule has
multiple outputs, requesting one builds the complete section:

```bash
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/local --cores 8 \
  outputs/figures/H0-all-detectors/H0-by-detector.pdf
```

With a SLURM profile, sampling runs remotely and figure rules run locally on the
submit host after their chains finish. The submit host must remain attached,
share the output filesystem, and provide plotting dependencies.

Use `H0_all_detectors_chains`, `H0_merger_rate_chains`, or
`H0_omega_m_chains` to sample one constituent experiment without running
post-processing. The three former complete experiment targets are not exposed.

`modified_propagation_all_detectors` remains a complete experiment target with
its own propagation figures and tables.

## Standalone figures

The amplitude toy model, fiducial spectrum, effective detector PSD comparison,
and importance-weight grids are explicit standalone rules in the unified
workflow. The fiducial spectrum borrows the six detector networks of the
`H0-all-detectors` experiment rather than restating them, and keeps its
`OMEGA_GW_MIN` y-limit next to the axis it sets. All of them read
`inputs/mcmc.base.toml` directly.

```bash
snakemake --snakefile workflow/mcmc.smk --cores 1 \
  outputs/figures/standalone/amplitude_toy_fisher_overlay.pdf \
  outputs/figures/standalone/fiducial_spectrum.pdf \
  outputs/figures/standalone/fiducial_effective_psd_by_detector.pdf \
  outputs/figures/standalone/importance_weights_grid_H0_Omega_m.pdf \
  outputs/figures/standalone/importance_weights_grid_Xi0_n.pdf
```

Or build any one of them directly:

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
