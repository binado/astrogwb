# Paper figures

The analysis notebooks
[`amplitude_toy_model.py`](../notebooks/amplitude_toy_model.py) and
[`snr_by_detector.py`](../notebooks/snr_by_detector.py) are self-contained: their
configuration cells hold editable scientific defaults, and every consumed setting
can be overridden with a command-line flag. They can therefore run directly in
Jupyter or from the shell without Snakemake or a configuration file.

For reproducible paper builds, scientific and presentation settings (fiducials,
detector networks, nested posterior plot entries) live in
[`configs/paper.toml`](../configs/paper.toml).
Reusable catalog recipes live in [`configs/catalogs/`](../configs/catalogs/),
while the catalog selected for the paper workflow lives in
[`configs/workflow.yaml`](../configs/workflow.yaml).
Figure-local knobs (output paths, dpi, sampler settings, which networks to plot),
catalog paths, and posterior chain paths are argparse defaults in each Jupytext
notebook. Edit them in Jupyter or override them with CLI flags headless. Keep the
analysis-notebook defaults aligned with `paper.toml` when promoting paper values;
the test suite guards against drift.

Snakemake reads [`configs/workflow.yaml`](../configs/workflow.yaml) for the paper
config path, selected catalog, and declared output paths. For the self-contained
analysis notebooks it translates `paper.toml` into explicit analysis, cosmology,
fiducial, path, and detector-network arguments. The structured posterior plotting
notebook remains config-driven and receives `--config` plus concrete chain inputs.

Preview the declared workflow:

```bash
uv run snakemake --snakefile workflow/paper.smk --dry-run paper_figures
```

Build all declared paper figures:

```bash
uv run snakemake --snakefile workflow/paper.smk --cores 1 paper_figures
```

Build one configured target:

```bash
uv run snakemake --snakefile workflow/paper.smk --cores 1 \
  figures/mcmc_compare_posteriors_H0.pdf
```

See [Snakemake workflow](./snakemake-workflow.md) for how the underlying
Snakefiles source catalogs, chains, and configuration.
