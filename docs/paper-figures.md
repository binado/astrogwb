# Paper figures

Paper figures are drawn by marimo notebooks under `notebooks/`. The split for
presentation is that *order and structure* are code, *values* are data.

Order and structure stay hard-coded in the notebook that draws each figure.
The six detector networks compared by more than one figure are the one shared
piece, and they live in `astrogwb.paper.plotting.DETECTOR_NETWORKS` as ordered
`(network name, LaTeX label)` pairs -- a network's label stays there because
nothing reads it without also needing the order it sits in. The detectors of
each network come from the shared `[networks]` table in
[`config/detectors.toml`](../config/detectors.toml), through
`astrogwb.config.load_config().registry`.

Values live in [`config/plotting.toml`](../config/plotting.toml): the LaTeX
label for each *parameter*, plus `figure_dpi` and `figure_format`. Notebooks
reach them through `astrogwb.paper.plotting.parameter_label`, `figure_dpi` and
`figure_format`. `use_paper_style()` applies the dpi and format as rcParams, so
`paper.mplstyle` declares neither.

Shared scientific values -- fiducials, priors, detector networks -- come from
`astrogwb.config.load_config()`, which reads the same files. Each notebook keeps its analysis window (`observation_time`, the
frequency band, the redshift bounds) and local plotting choices as literals in
its configuration cell. These mirror the `[analysis]` table in
`config/defaults.toml`, so editing that table does not change the figures --
update the notebook's configuration cell too.

The spectrum figure (characteristic amplitudes $h_c$ and $h_n$ with percentile
bands over a Poisson ensemble, stacked over $d\mathrm{SNR}^2/d\ln f$) and the
SNR tables (mean over draws of each draw's SNR, with the 16-84% range) come from
[`notebooks/ensemble_spectrum.py`](../notebooks/ensemble_spectrum.py).
The single-draw explorer, with the network $S_{\mathrm{eff}}$, cumulative SNR
and sliders, is
[`notebooks/fiducial_spectrum.py`](../notebooks/fiducial_spectrum.py).

## Grid posteriors

`notebooks/cosmology_grid_posteriors.py` is a marimo notebook that evaluates
the posteriors of `H0`, `(H0, Omega_m)` and `(xi_0, xi_n)` on a grid for all six
networks, with no sampler. The data are a zero-noise Poisson injection
(`data_seed = 41`); the model spectrum is the rescaled reference-redshift
spectrum of `astrogwb.gwb.importance` (`catalog_seed = 42`): `2^14` intrinsic
draws, each generated once at reference redshift `1e-4` by `reference_catalog`,
and `ImportanceGaussianLikelihood` rescaling their mean power to 32 Gauss-Legendre
nodes in the scale factor `1/(1 + z)` at each grid point, with no intrinsic
reweighting. The log densities are computed on every run. The notebook reads
its inputs and never generates them: the injection ensemble
(`outputs/ensembles/fiducial/`, whose background gives the detector noise) and
the reference catalog (`outputs/catalogs/reference-fiducial.h5`), all written by
`Snakefile` (see `docs/coverage-draws-on-slurm.md`).
Figures go to `outputs/figures/cosmology_grid_posteriors/` when the
`Write figures` switch is on:

```bash
uv run --extra notebook --group dev marimo check --strict notebooks/cosmology_grid_posteriors.py
```

## Importance convergence

`notebooks/importance_convergence.py` measures the two errors of the
rescaled reference-redshift spectrum separately, to choose the grid posteriors'
`(N, Z)`. The rescaling itself holds for the (2,2)-mode aligned-spin waveforms
used; `tests/core/test_gwb_importance.py` checks it against waveforms generated
at every node. The redshift quadrature runs a `Z = 2^i` ladder (4 to 1024 nodes) on one
reference catalog -- the node count is a builder argument, so the ladder costs
no waveforms. The Monte Carlo noise is the scatter of `M`
catalogs at fixed seeds about their mean; each `N = 2^k` is the first `N`
draws of every catalog, so the `N` ladder costs no extra waveforms. Errors are
reported in units of the most sensitive network's per-bin `sigma` and as each
parameter's likelihood-peak shift in Fisher widths, at the fiducials and the
grid windows' edges. A recommendation cell reports the cheapest pair below
tolerance and the `N` the `1/N` variance law extrapolates to. The ladder's
catalog is read from `outputs/catalogs/reference-ladder.h5`; the Monte Carlo
catalogs are drawn in memory. Figures go to
`outputs/figures/importance_convergence/` when `Write figures` is on:

```bash
uv run --extra notebook --group dev marimo check --strict notebooks/importance_convergence.py
```
