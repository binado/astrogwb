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

Add `--write-manifests` to also (re)generate a full-sweep batch manifest per
campaign —
`configs/mcmc/manifests/mcmc.batch.{cosmology,astrophysical,modified-propagation}.json`
— each holding only `chains_dir` and every config just written for that
campaign; manifests carry no catalog field (see below). Existing manifests
are skipped unless `--force` is supplied, same as the JSON configs. Like the
JSON configs, generated manifests are gitignored (`configs/mcmc/manifests/`):
they are fully reproducible from `configs/mcmc.sweeps.toml`, so there is
nothing to commit.

```bash
uv run --extra mcmc python scripts/generate_mcmc_configs.py --write-manifests
```

Batch submission uses an explicit manifest such as
[`configs/mcmc.batch.example.json`](configs/mcmc.batch.example.json) — plain
JSON, since Snakemake's `--configfile` loader tries JSON before YAML
regardless of extension. The manifest lists `chains_dir` and every MCMC
config to submit; it carries no catalog field. `workflow/mcmc.smk` sources
the catalog separately from [`configs/workflow.yaml`](configs/workflow.yaml)
(shared with the paper workflow) via a `configfile:` directive, deriving
`out/catalogs/<id>.h5` from `catalog.id` unless `catalog.path` is set
explicitly. This decouples which runs make up a campaign (the manifest, fully
reproducible from `configs/mcmc.sweeps.toml`) from which data file to
reweight (the catalog) — the same manifest runs against any catalog by
editing `configs/workflow.yaml` or passing an extra `--configfile` that
overrides `catalog`, with no manifest regeneration required. Chains remain
namespaced by catalog ID. A missing catalog or config stops the workflow
instead of triggering preprocessing.

Note the manifest schema has no `jax_platforms` field: which JAX backend to
initialize is a runtime concern owned by the Snakemake profile you run with
(`profiles/local`, `profiles/slurm`, `profiles/slurm-cpu`), not a property of
an MCMC campaign, so it is never written by the generator. `workflow/mcmc.smk`
itself defaults to `cuda` when a manifest doesn't set it.

Dry-run the selected batch locally:

```bash
uv run snakemake \
  --snakefile workflow/mcmc.smk \
  --configfile configs/mcmc.batch.example.json \
  --dry-run mcmc
```

### Deploying on a SLURM cluster

The workflow submits to SLURM through a committed Snakemake *profile* — a bundle
of executor flags and rule-specific resource overrides. The Snakefile itself is
backend-agnostic; the profile decides *where* jobs land. Two profiles ship with
the repo:

| Profile | Partition | GPU | JAX backend |
| --- | --- | --- | --- |
| `profiles/slurm` | `gpu` | `--gres=gpu:1` | `cuda` |
| `profiles/slurm-cpu` | `cpu` | none | `cpu` |

Both request 12 h wall-clock (`runtime: 720`) and batch compatible `run_mcmc`
jobs into a SLURM array. The CPU profile also selects `--platform cpu` so JAX
does not try to initialize CUDA on a CPU node.

1. **Install the executor plugin on the submit host.** The `slurm` group pulls
   in `snakemake-executor-plugin-slurm`:

   ```bash
   uv sync --extra mcmc --group slurm
   ```

2. **Prepare the batch manifest.** Generate sweep configs and their manifests
   with `uv run --extra mcmc python scripts/generate_mcmc_configs.py --write-manifests`
   if you haven't already — this writes
   `configs/mcmc/manifests/mcmc.batch.{cosmology,astrophysical,modified-propagation}.json`
   (gitignored — regenerate them on whichever host needs them), each listing
   every config for that campaign. Copy the one you want (or
   [`configs/mcmc.batch.example.json`](configs/mcmc.batch.example.json) for a
   hand-picked selection). Manifests carry no catalog field, so which catalog
   to use is set separately: `workflow/mcmc.smk` sources it from
   [`configs/workflow.yaml`](configs/workflow.yaml). Point that file's
   `catalog.id` (and `catalog.path`, if the catalog doesn't live at the
   default `out/catalogs/<id>.h5`) at one existing catalog before submitting.

3. **Dry-run before every real submission** to see the job graph without touching
   the scheduler:

   ```bash
   uv run snakemake \
     --snakefile workflow/mcmc.smk \
     --profile profiles/slurm \
     --configfile /home/user/batches/paper-h0.json \
     --dry-run mcmc
   ```

4. **Submit.** Drop `--dry-run` and pick the profile for your target partition —
   this is the only change needed to switch between GPU and CPU:

   ```bash
   # GPU nodes
   uv run snakemake \
     --snakefile workflow/mcmc.smk \
     --profile profiles/slurm \
     --configfile /home/user/batches/paper-h0.json \
     mcmc

   # CPU nodes
   uv run snakemake \
     --snakefile workflow/mcmc.smk \
     --profile profiles/slurm-cpu \
     --configfile /home/user/batches/paper-h0.json \
     mcmc
   ```

   Keep the Snakemake process alive for the duration of the run (submit inside
   `tmux`/`screen` or as a lightweight batch job); it stays up submitting and
   polling the array. Per-job SLURM logs land under `.snakemake/slurm_logs/`.

The partition names (`gpu`, `cpu`) are cluster-specific — verify them with
`sinfo -s` and edit the `slurm_partition` in the relevant profile if they
differ. Adjust `set-threads` for the CPU allocation, and `mem_mb` / `runtime`
under the profile's `set-resources.run_mcmc` entry.

### Running a batch locally on multiple cores

For smaller sweeps you can skip SLURM entirely and let Snakemake pack
independent `run_mcmc` jobs across the cores of your own machine with
[`profiles/local`](profiles/local/config.yaml). It forces the CPU JAX backend
and confines each run to its Snakemake thread allocation — both the BLAS pools
and, via `XLA_FLAGS intra_op_parallelism_threads`, JAX's XLA CPU pool — so
concurrent runs do not oversubscribe. The core budget lives only in the profile;
it never enters a run's config hash.

As a worked example, run the **cosmology sweep** locally. First generate the
sweep configs and their batch manifest (writes
`configs/mcmc/cosmology/<network>__<params>.json` and
`configs/mcmc/manifests/mcmc.batch.cosmology.json`):

```bash
uv run --extra mcmc python scripts/generate_mcmc_configs.py --write-manifests
```

`configs/mcmc/manifests/mcmc.batch.cosmology.json` now lists all 24 cosmology
sweep points (6 networks x 4 sample-label combos). To run only a subset
locally — e.g. the ET-triangular `H0`, `H0`+`Omega_m`, and `H0`+merger-rate
points — copy it somewhere and trim the `runs` list rather than editing the
generated file in place (the next `--write-manifests --force` run overwrites
it):

```bash
cp configs/mcmc/manifests/mcmc.batch.cosmology.json configs/mcmc.batch.cosmology-local.json
# then edit configs/mcmc.batch.cosmology-local.json down to the runs you want
```

Dry-run, then submit on (say) 8 cores. With the profile's `run_mcmc=4` thread
override, Snakemake runs `floor(8 / 4) = 2` sweep points at a time:

```bash
uv run snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/local --configfile configs/mcmc.batch.cosmology-local.json \
  --cores 8 --dry-run mcmc

uv run snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/local --configfile configs/mcmc.batch.cosmology-local.json \
  --cores 8 mcmc
```

To trade per-run speed for more concurrency, lower the `set-threads` value in
`profiles/local`: `run_mcmc=2` runs four sweep points at once on the same 8
cores. Because each step is dominated by a large
catalog contraction that BLAS already parallelizes, more independent runs
usually beats more threads per run — keep the sweep configs single-chain
(`num_chains = 1`) and let the job level do the work.

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
priors, sampler settings, and the git revision. Runtime controls
(`--platform`, `--chain-method`, etc.) affect only how a run executes, not
its scientific result, so they are not recorded.
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
