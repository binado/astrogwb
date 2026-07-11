# astrogwb

This is a package for Bayesian inference of cosmological and astrophysical parameters
with the stochastic gravitational-wave background (SGWB) from stellar-mass compact binary coalescences (CBCs) detectable by ground-based interferometer networks such as LIGO, Virgo, KAGRA, Einstein Telescope and Cosmic Explorer.

## Installation

The easiest way to install the package is using the `uv` package manager:

```bash
git clone git@github.com:binado/astrogwb.git
uv sync --all-groups --all-extras
```

Core inference libraries install without the headless runner dependencies. The
optional ``mcmc`` extra adds pydantic for validated ``RunConfig`` parsing and
ArviZ's NetCDF output support used by [`scripts/run_mcmc.py`](scripts/run_mcmc.py)
and related tools:

```bash
uv sync --extra mcmc
# or with notebook/dev tooling:
uv sync --extra mcmc --group dev
```

## Generating a population of CBCs

Our inference framework uses an importance sampling scheme to calculate the spectral density of the astrophysical SGWB with a fixed population of CBCs. To generate the population, we suggest using the excellent [`gwmock-pop`](https://leuven-gravity-institute.github.io/gwmock-pop/) package.

An example BNS population is defined declaratively in [`examples/bns_population.yaml`](examples/bns_population.yaml).
The complete default recipe lives in
[`configs/catalogs/bns-n16384-df1.toml`](configs/catalogs/bns-n16384-df1.toml).
Its equivalent explicit population command is:

```bash
uv run gwmock-pop simulate \
  --config examples/bns_population.yaml \
  --n 16384 \
  --output out/populations/bns-n16384-df1.h5 \
  --seed 42
```

`--output` accepts `.csv`, `.h5`, or `.hdf5`; use `.h5` for the structured
output the downstream catalog step consumes. The result is a table of `n`
intrinsic samples — one column per parameter, using gwmock-pop canonical
names (`source_frame_mass_1/2`, `luminosity_distance`, `spin_1z/2z`,
`lambda_1/2`, `inclination`, `coa_phase`, `coa_time`).

The committed config encodes a BNS population with a Madau–Dickinson-like redshift
distribution (converted to luminosity distance), uniform source-frame component
masses ordered so `mass_1 >= mass_2`, aligned spins (in-plane components zero),
uniform tidal deformabilities, and inclination/coalescence phase/time fixed at
zero. Edit the `arguments` blocks to retune ranges. The aligned-spin + tidal
parameters suit a non-precessing NRTidal approximant downstream.
The mass priors are defined in the source frame.

## Generating waveforms for the population catalog

We provide a [helper script](./scripts/generate_waveform_catalog.py) which wraps the [`gwmock-signal`](https://github.com/Leuven-Gravity-Institute/gwmock-signal) package for generating the frequency-domain polarizations for a given population of CBCs which enter the spectral density calculation. The output is a [`pluscross`](https://pypi.org/project/pluscross/) HDF5 catalog of complex polarizations, which the inference consumers reduce to polarization power at load time.

The corresponding explicit waveform command is:

```bash
uv run python scripts/generate_waveform_catalog.py \
--population out/populations/bns-n16384-df1.h5 \
--output out/catalogs/bns-n16384-df1.h5 \
--approximant IMRPhenomXAS_NRTidalv3 \
--sampling-frequency 8192 \
--minimum-frequency 2 \
--maximum-frequency 4096 \
--reference-frequency 20 \
--frequency-resolution 1 \
--chunk-size 2048
```

The local catalog workflow joins the population and waveform steps. Requesting
a waveform catalog builds its missing or stale population first:

```bash
uv run snakemake \
  --snakefile workflow/catalog.smk \
  --cores 1 \
  out/catalogs/bns-n16384-df1.h5
```

Each catalog has an independent `configs/catalogs/<catalog-id>.toml` recipe, so
editing one recipe cannot invalidate another catalog. Catalog generation is not
part of the MCMC submission workflow.

## Running inference

There are two working examples provided in the repo:

- The [`run_mcmc.py` script](./scripts/run_mcmc.py) accepts a TOML or JSON configuration file and is suitable for running MCMC on a cluster.
- The [`mcmc.py` notebook](./notebooks/mcmc.py) has the same functionality as the script but can be run interactively in a Jupyter notebook. You can use `uvx jupytext --to ipynb notebooks/mcmc.py` to convert it to a Jupyter notebook in your local machine.

Both examples assume a population of binary neutron star (BNS) mergers following the example described above. The headless runner takes the proposal catalog explicitly:

```bash
uv run --extra mcmc python scripts/run_mcmc.py \
  --config configs/mcmc.example.toml \
  --catalog out/catalogs/bns-n16384-df1.h5
```

Generate sweep configs explicitly on the local machine or cluster submit host;
existing configs are skipped unless `--force` is supplied:

```bash
uv run --extra mcmc python scripts/generate_mcmc_configs.py
```

Batch submission uses an explicit manifest such as
[`configs/mcmc.batch.example.yaml`](configs/mcmc.batch.example.yaml). The
manifest selects one existing catalog and lists every MCMC config to submit.
Chains remain namespaced by catalog ID. A missing catalog or config stops the
workflow instead of triggering preprocessing.

Dry-run the selected batch locally:

```bash
uv run snakemake \
  --snakefile workflow/mcmc.smk \
  --configfile configs/mcmc.batch.example.yaml \
  --dry-run mcmc
```

On a SLURM cluster, install the executor plugin and submit through the committed
profile. Each `run_mcmc` job gets a GPU and compatible jobs land in an array;
logs go to `.snakemake/slurm_logs/`:

```bash
uv sync --extra mcmc --group slurm
uv run snakemake \
  --snakefile workflow/mcmc.smk \
  --profile profiles/slurm \
  --configfile /home/user/batches/paper-h0.yaml \
  mcmc
```

Reproducibility is layered on committed catalog recipes, fixed population seeds,
and content hashes instead of lock files. Each chain sidecar records the exact
catalog path and SHA-256 used; its MCMC config hash excludes output routing and
is independent of catalog selection. Snakemake reruns selected chains after
their catalog or config changes.
Chains and sidecars are written `protected()` (read-only); before intentionally
redoing a run, `chmod +w` its outputs and rerun with `--forcerun`.

Existing non-namespaced chains are left untouched. Always dry-run before real
cluster submissions. The MCMC workflow cannot generate catalogs, and the paper
workflow cannot generate catalogs or chains.

### Outputs

Each workflow run writes an ArviZ `InferenceData` to
`chains/<catalog-id>/<campaign>/<run>.nc`
(ad-hoc unlabelled runs keep the timestamped
`chains/mcmc-<params>-det=<det>-seed<n>-<ts>.nc` convention) alongside a
sibling `.json` sidecar recording the run's provenance: catalog path and
`catalog_sha256`, the resolved `config_sha256`, detectors, seed, fiducials,
priors, sampler settings, and the git revision.
Diagnostics surface the model's `importance_relative_ess` (the key proposal
health check — should stay close to 1) and `total_merger_rate`.

See the [plotting notebook](./notebooks/mcmc_plotting.py) for examples of how to visualize the results.

## Paper figures

The analysis notebooks
[`amplitude_toy_model.py`](notebooks/amplitude_toy_model.py) and
[`snr_by_detector.py`](notebooks/snr_by_detector.py) are self-contained: their
configuration cells hold editable scientific defaults, and every consumed setting
can be overridden with a command-line flag. They can therefore run directly in
Jupyter or from the shell without Snakemake or a configuration file.

For reproducible paper builds, scientific and presentation settings (fiducials,
detector networks, nested posterior plot entries) live in
[`configs/paper.toml`](configs/paper.toml).
Reusable catalog recipes live in [`configs/catalogs/`](configs/catalogs/),
while the catalog selected for the paper workflow lives in
[`configs/workflow.yaml`](configs/workflow.yaml).
Figure-local knobs (output paths, dpi, sampler settings, which networks to plot),
catalog paths, and posterior chain paths are argparse defaults in each Jupytext
notebook. Edit them in Jupyter or override them with CLI flags headless. Keep the
analysis-notebook defaults aligned with `paper.toml` when promoting paper values;
the test suite guards against drift.

Snakemake reads [`configs/workflow.yaml`](configs/workflow.yaml) for the paper
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

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)

## Development

Install dependencies (the dev group also bundles Jupyter, arviz, corner, and
matplotlib so the MCMC notebook in `notebooks/` runs out of the box). The
`mcmc` extra alone is sufficient for the headless runner, including ArviZ
NetCDF output; include it with dev tools for related tests:

```bash
uv sync --group dev
uv sync --extra mcmc --group dev
```

Run tests:

```bash
uv run pytest
uv run pytest -m "not integration"   # fast unit tests only
```

Regression fixtures under `tests/fixtures/` are committed; integration tests
cross-check gwfast and skip if optional fixtures are missing.

Format and lint:

```bash
uv run ruff format .
uv run ruff check .
```

Type check:

```bash
uvx ty check
```
