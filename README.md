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

An example BNS population is defined declaratively in [`examples/bns_population.yaml`](examples/bns_population.yaml)
and drawn with the `gwmock-pop simulate` CLI. The paper workflow uses the
sample count and seed in [`configs/paper.toml`](configs/paper.toml); its
equivalent explicit command is:

```bash
uv run gwmock-pop simulate \
  --config examples/bns_population.yaml \
  --n 16384 \
  --output out/bns_n=16384.h5 \
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
--population out/bns_n=16384.h5 \
--output out/bns_waveform_catalog_n=16384_df=1Hz.h5 \
--approximant IMRPhenomXAS_NRTidalv3 \
--sampling-frequency 8192 \
--minimum-frequency 2 \
--maximum-frequency 4096 \
--reference-frequency 20 \
--frequency-resolution 1 \
--chunk-size 2048
```

These two commands are also Snakemake dependencies: requesting the catalog or
an MCMC target automatically builds any missing or stale upstream files.

## Running inference

There are two working examples provided in the repo:

- The [`run_mcmc.py` script](./scripts/run_mcmc.py) accepts a TOML or JSON configuration file and is suitable for running MCMC on a cluster.
- The [`mcmc.py` notebook](./notebooks/mcmc.py) has the same functionality as the script but can be run interactively in a Jupyter notebook. You can use `uvx jupytext --to ipynb notebooks/mcmc.py` to convert it to a Jupyter notebook in your local machine.

Both examples assume a population of binary neutron star (BNS) mergers following the example described above. See the notebook for more details on how the inference is set up.

Generate the detector × sample-parameter sweep configs (from
[`configs/mcmc.example.toml`](configs/mcmc.example.toml)) into the three
sweep directories under `configs/mcmc/{cosmology,modified-propagation,astrophysical}/`:

```bash
uv run --extra mcmc python scripts/generate_mcmc_configs.py --force
```

Batch runs are driven by the Snakemake `run_mcmc` rule: the chain
`chains/<campaign>/<run>.nc` (plus its `.json` sidecar) is built from
`configs/mcmc/curated/<campaign>/<run>.json` — sweep campaigns
(`cosmology`, `modified-propagation`, `astrophysical`) resolve directly to
`configs/mcmc/<campaign>/<run>.json` instead. Request a single run by its
output path, or a whole campaign via the aggregate targets:

```bash
uv run snakemake -n mcmc_paper_h0                    # dry-run: shows pending work
uv run snakemake chains/paper-h0/et-triangular.nc    # one run
uv run snakemake mcmc_paper_h0                       # the six paper-h0 runs
uv run snakemake mcmc_sweeps                         # all generated sweep configs
```

On a SLURM cluster, install the executor plugin and submit through the
committed profile (each `run_mcmc` job gets a GPU and lands in one job array;
logs go to `.snakemake/slurm_logs/`):

```bash
uv sync --extra mcmc --group slurm
uv run snakemake --profile profiles/slurm mcmc_paper_h0
```

Reproducibility is layered on committed recipes, fixed population seeds, and
content hashes instead of lock files: Snakemake rebuilds chains after catalog
or config drift, while each chain sidecar records the catalog's actual SHA-256.
Chains and sidecars are written `protected()` (read-only); before intentionally
redoing a run, `chmod +w` its outputs and rerun with `--forcerun`.

**One-time migration on existing cluster checkouts** (chains produced by the
retired `submit_mcmc.py` flow have no Snakemake metadata): register them once
with `uv run snakemake --touch mcmc_paper_h0` while they are still writable.
For figure-only builds on machines with pre-existing chains,
`--rerun-triggers mtime` is the escape hatch to suppress metadata-based
reruns. Always dry-run (`-n`) before real runs on the cluster — note that
`paper_figures` pulls `run_mcmc` into its DAG, so on a machine without chains
it will schedule MCMC runs.

### Outputs

Each run writes an ArviZ `InferenceData` to `chains/<campaign>/<run>.nc`
(ad-hoc unlabelled runs keep the timestamped
`chains/mcmc-<params>-det=<det>-seed<n>-<ts>.nc` convention) alongside a
sibling `.json` sidecar recording the run's provenance: catalog path and
`catalog_sha256`, the resolved `config_sha256`, detectors, seed, fiducials,
priors, sampler settings, and the git revision.
Diagnostics surface the model's `importance_relative_ess` (the key proposal
health check — should stay close to 1) and `total_merger_rate`.

See the [plotting notebook](./notebooks/mcmc_plotting.py) for examples of how to visualize the results.

## Paper figures

Shared analysis settings (catalog path, fiducials, detector networks, nested
posterior plot entries) live in [`configs/paper.toml`](configs/paper.toml).
Figure-local knobs (output paths, dpi, sampler settings, which networks to
plot) are argparse defaults in each Jupytext notebook — edit them in Jupyter,
override with CLI flags headless, and promote happy values by updating those
defaults (and `paper.toml` for shared/nested data).

Snakemake reads [`configs/workflow.yaml`](configs/workflow.yaml) for the paper
config path and declared output paths, then passes `--config` plus output-path
flags into each notebook.

Preview the declared workflow:

```bash
uv run snakemake --dry-run paper_figures
```

Build all declared paper figures:

```bash
uv run snakemake --cores 1 paper_figures
```

Build one configured target:

```bash
uv run snakemake --cores 1 figures/mcmc_compare_posteriors_H0.pdf
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
