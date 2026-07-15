# Snakemake workflow

Three Snakefiles orchestrate the pipeline stages, each backend-agnostic and
driven by a committed Snakemake *profile* that decides where jobs run:

- [`workflow/catalog.smk`](../workflow/catalog.smk): joins population and
  waveform generation (see [Generating catalogs](./catalog-generation.md)).
- [`workflow/mcmc.smk`](../workflow/mcmc.smk): submits MCMC sweep configs
  (see [Running inference](./running-inference.md)).
- [`workflow/paper.smk`](../workflow/paper.smk): builds paper figures
  (see [Paper figures](./paper-figures.md)).

## Catalog workflow

Requesting a waveform catalog builds its missing or stale population first:

```bash
uv run snakemake \
  --snakefile workflow/catalog.smk \
  --cores 1 \
  out/catalogs/bns-n16384-df1.h5
```

## MCMC workflow

### Batch manifests

Generate sweep configs as described in
[Generating sweep configs](./running-inference.md#generating-sweep-configs).
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
[`configs/mcmc.batch.example.json`](../configs/mcmc.batch.example.json) — plain
JSON, since Snakemake's `--configfile` loader tries JSON before YAML
regardless of extension. The manifest lists `chains_dir` and every MCMC
config to submit; it carries no catalog field. `workflow/mcmc.smk` sources
the catalog separately from [`configs/workflow.yaml`](../configs/workflow.yaml)
(shared with the paper workflow) via a `configfile:` directive, deriving
`out/catalogs/<id>.h5` from `catalog.id` unless `catalog.path` is set
explicitly. This decouples which runs make up a campaign (the manifest, fully
reproducible from `configs/mcmc.sweeps.toml`) from which data file to
reweight (the catalog) — the same manifest runs against any catalog by
editing `configs/workflow.yaml` or passing an extra `--configfile` that
overrides `catalog`, with no manifest regeneration required. Chains remain
namespaced by catalog ID. A missing catalog or config stops the workflow
instead of triggering preprocessing.

To select a catalog in the standard `out/catalogs/<id>.h5` location, override
only its ID. For a catalog stored elsewhere, provide both the ID used to
namespace chains and the catalog path:

```bash
# Uses out/catalogs/my-catalog.h5.
--config "catalog={'id':'my-catalog'}"

# Uses an externally stored catalog.
--config "catalog={'id':'my-catalog','path':'/data/catalogs/my-catalog.h5'}"
```

Append the appropriate `--config` argument to the Snakemake commands below.
Use the nested dictionary syntax shown here rather than a flat
`catalog.id=...` key.

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
jobs into a SLURM array. The CPU profile also sets `jax_platforms=cpu` (passed
through as `--platform cpu`) so JAX does not try to initialize CUDA on a CPU
node.

1. **Install the executor plugin on the submit host.** The `slurm` group pulls
   in `snakemake-executor-plugin-slurm`. Add `--extra cuda` for the GPU
   profile (omit it for CPU-only):

   ```bash
   uv sync --extra mcmc --extra cuda --group slurm   # profiles/slurm
   uv sync --extra mcmc --group slurm                # profiles/slurm-cpu
   ```

2. **Prepare the batch manifest.** Generate sweep configs and their manifests
   with `uv run --extra mcmc python scripts/generate_mcmc_configs.py --write-manifests`
   if you haven't already — this writes
   `configs/mcmc/manifests/mcmc.batch.{cosmology,astrophysical,modified-propagation}.json`
   (gitignored — regenerate them on whichever host needs them), each listing
   every config for that campaign. Copy the one you want (or
   [`configs/mcmc.batch.example.json`](../configs/mcmc.batch.example.json) for a
   hand-picked selection). Manifests carry no catalog field, so which catalog
   to use is set separately: `workflow/mcmc.smk` sources it from
   [`configs/workflow.yaml`](../configs/workflow.yaml). Point that file's
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
[`profiles/local`](../profiles/local/config.yaml). It forces the CPU JAX backend
and confines each run to its Snakemake thread allocation — both the BLAS pools
and, via `XLA_FLAGS intra_op_parallelism_threads`, JAX's XLA CPU pool — so
concurrent runs do not oversubscribe. The core budget lives only in the profile;
it never enters a run's config hash.

As a worked example, run the **cosmology sweep** locally. First generate the
sweep configs and their batch manifest (writes
`configs/mcmc/cosmology/<network>__<analysis>__<observation>.json` and
`configs/mcmc/manifests/mcmc.batch.cosmology.json`):

```bash
uv run --extra mcmc python scripts/generate_mcmc_configs.py --write-manifests
```

`configs/mcmc/manifests/mcmc.batch.cosmology.json` now lists all 24 cosmology
sweep points (6 networks × 4 analyses × 1 observation). To run only a subset
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

## Paper workflow

[`workflow/paper.smk`](../workflow/paper.smk) builds paper figures from
[`configs/paper.toml`](../configs/paper.toml) and the catalog selected in
[`configs/workflow.yaml`](../configs/workflow.yaml). Dry-run and build commands
live in [Paper figures](./paper-figures.md); this page covers only how the
Snakefiles source catalogs, chains, and configuration.

## Reproducibility

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
