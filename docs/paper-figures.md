# Paper figures

The analysis notebooks
[`amplitude_toy_model.py`](../notebooks/amplitude_toy_model.py),
[`mcmc_cosmological_parameters.py`](../notebooks/paper/mcmc_cosmological_parameters.py),
and [`fiducial_spectrum.py`](../notebooks/paper/fiducial_spectrum.py)
hold editable scientific defaults and expose command-line overrides for scientific
inputs, paths, and labels. They can therefore run directly in Jupyter or from the shell.
The spectrum notebook plots fiducial $\Omega_{\mathrm{GW}}(f)$ and $S_h(f)$ on dual
$y$-axes. Configure the $\Omega_{\mathrm{GW}}$ floor via `omega_gw_min` in
`workflow.yaml` (or `--omega-gw-min`); $S_h$'s floor is inferred at the matching
frequency so both curves show the same band.
The cosmology notebook reads paper plot styling from `configs/paper.toml`;
its two ordered chain groups and their labels can be replaced independently with
`--detector-chains`/`--detector-labels` and
`--prior-chains`/`--prior-labels`.

For reproducible paper builds, scientific and presentation settings (fiducials,
detector networks, nested posterior plot entries) live in
[`configs/paper.toml`](../configs/paper.toml).
Reusable catalog recipes live in [`configs/catalogs/`](../configs/catalogs/),
while the catalog selected for the paper workflow lives in
[`configs/workflow.yaml`](../configs/workflow.yaml).
Figure-local knobs (output paths, dpi, sampler settings, which networks to plot),
catalog paths, and posterior chain paths are argparse defaults in each Jupytext
notebook. Edit them in Jupyter or override them with CLI flags headless. Keep the
analysis-notebook defaults aligned with `paper.toml` when promoting paper values.

Snakemake reads [`configs/workflow.yaml`](../configs/workflow.yaml) for the paper
config path, selected catalog, and declared output paths. It translates
`paper.toml` into explicit analysis, cosmology, fiducial, detector-network, chain,
label, and styling inputs. The `fiducial_spectrum` rule builds
`figures/fiducial_spectrum.pdf` from the shared catalog and fiducials. The unified
cosmology rule produces two marginalized $H_0$ comparisons, separate
$H_0$--$\mathcal{R}_0$ corner plots for the narrow and broad merger-rate priors,
an $H_0$--$\Omega_m$ corner plot, and a CSV/LaTeX SNR-and-constraint table.

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
  figures/fiducial_spectrum.pdf
```

```bash
uv run snakemake --snakefile workflow/paper.smk --cores 1 \
  figures/mcmc_cosmological_parameters_H0_by_detector.pdf
```

See [Snakemake workflow](./snakemake-workflow.md#paper-workflow) for a pipeline
overview of how the Snakefiles source catalogs, chains, and configuration.
