# Snakemake workflow

Three Snakefiles orchestrate the pipeline stages, each backend-agnostic and
driven by a committed Snakemake *profile* that decides where jobs run:

- [`workflow/catalog.smk`](../workflow/catalog.smk): joins population and
  waveform generation (see [Generating catalogs](./catalog-generation.md)).
- [`workflow/mcmc.smk`](../workflow/mcmc.smk): submits MCMC sweep configs
  (see [Running inference](./running-inference.md)).
- [`workflow/paper.smk`](../workflow/paper.smk): builds paper figures
  (see [Paper figures](./paper-figures.md)).

The [`scripts/workflow.py`](../scripts/workflow.py) CLI wraps common invocations
(`uv run python scripts/workflow.py --help`). Snakemake subcommands default to
`--dry-run`; pass `--submit` for a real run. The CLI expands to the same
`uv run snakemake …` commands shown below.

## Catalog workflow

Requesting a waveform catalog builds its missing or stale population first:

```bash
uv run python scripts/workflow.py catalog out/catalogs/bns-n16384-df1.h5 --submit
# expands to:
# uv run snakemake --snakefile workflow/catalog.smk --cores 1 \
#   out/catalogs/bns-n16384-df1.h5
```

## MCMC workflow

### Batch manifests

Generate sweep configs as described in
[Generating sweep configs](./running-inference.md#generating-sweep-configs).
Add `--write-manifests` to also (re)generate a full-sweep batch manifest per
campaign —
`configs/mcmc/manifests/mcmc.batch.{cosmology,cosmology-all-detectors,astrophysical,astrophysical-all-detectors,modified-propagation-all-detectors}.json`
— each holding only `chains_dir` and every config just written for that
campaign; manifests carry no catalog field (see below). Existing manifests
are skipped unless `--force` is supplied, same as the JSON configs. Like the
JSON configs, generated manifests are gitignored (`configs/mcmc/manifests/`):
they are fully reproducible from `configs/mcmc.sweeps.toml`, so there is
nothing to commit.

```bash
uv run python scripts/workflow.py gen-configs
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

Append the appropriate `--config` argument after `--` on
`scripts/workflow.py mcmc …` (or directly on snakemake). Use the nested
dictionary syntax shown here rather than a flat `catalog.id=...` key.

Note the manifest schema has no `jax_platforms` field: which JAX backend to
initialize is a runtime concern owned by the Snakemake profile you run with
(`profiles/local`, `profiles/slurm`, `profiles/slurm-cpu`), not a property of
an MCMC campaign, so it is never written by the generator. `workflow/mcmc.smk`
itself defaults to `cuda` when a manifest doesn't set it.

Dry-run the selected batch (default; omit `--submit`):

```bash
uv run python scripts/workflow.py mcmc configs/mcmc.batch.example.json --profile local
```

### Deploying on a SLURM cluster

The workflow submits to SLURM through a committed Snakemake *profile* — a bundle
of executor flags and rule-specific resource overrides. The Snakefile itself is
backend-agnostic; the profile decides *where* jobs land. Two profiles ship with
the repo:

| Profile | Partition | GPU / CPUs | Mem / walltime | JAX backend |
| --- | --- | --- | --- | --- |
| `profiles/slurm` | `gpu` | `gpu: 1`, `cpus_per_gpu: 4` | 8 GB / 12 h (`runtime: 720`) | `cuda` |
| `profiles/slurm-cpu` | `cpu` | 4 CPUs via `set-threads` | 16 GB / 24 h (`runtime: 1440`) | `cpu` |

Both batch compatible `run_mcmc` jobs into a SLURM array. The GPU profile
requests one GPU and ties four CPUs to it via `cpus_per_gpu` (plugin-native
`gpu` / `cpus_per_gpu` resources, not `--gres`). The CPU profile sets
`jax_platforms=cpu` (passed through as `--platform cpu`) so JAX does not try
to initialize CUDA on a CPU node, and spends rule threads on
`--host-device-count` (one logical device per concurrent chain, with
`--cpu-threads` pinned to 1).

1. **Install the executor plugin on the submit host.** The `slurm` group pulls
   in `snakemake-executor-plugin-slurm`. Add `--extra cuda` for the GPU
   profile (omit it for CPU-only):

   ```bash
   uv sync --extra mcmc --extra cuda --group slurm   # profiles/slurm
   uv sync --extra mcmc --group slurm                # profiles/slurm-cpu
   ```

2. **Prepare the batch manifest.** Generate sweep configs and their manifests
   with `uv run python scripts/workflow.py gen-configs` if you haven't already —
   this writes one gitignored manifest per campaign under
   `configs/mcmc/manifests/` (regenerate them on whichever host needs them),
   including the quick ET-2L campaigns and the `*-all-detectors` campaigns.
   Copy the one you want (or
   [`configs/mcmc.batch.example.json`](../configs/mcmc.batch.example.json) for a
   hand-picked selection). Manifests carry no catalog field, so which catalog
   to use is set separately: `workflow/mcmc.smk` sources it from
   [`configs/workflow.yaml`](../configs/workflow.yaml). Point that file's
   `catalog.id` (and `catalog.path`, if the catalog doesn't live at the
   default `out/catalogs/<id>.h5`) at one existing catalog before submitting.

3. **Dry-run before every real submission** to see the job graph without touching
   the scheduler (`scripts/workflow.py` defaults to `--dry-run`):

   ```bash
   uv run python scripts/workflow.py mcmc /home/user/batches/paper-h0.json --profile slurm
   ```

4. **Submit.** Pass `--submit` and pick `--profile` for your target partition —
   this is the only change needed to switch between GPU and CPU:

   ```bash
   # GPU nodes
   uv run python scripts/workflow.py mcmc /home/user/batches/paper-h0.json \
     --profile slurm --submit

   # CPU nodes
   uv run python scripts/workflow.py mcmc /home/user/batches/paper-h0.json \
     --profile slurm-cpu --submit
   ```

   Equivalent expanded form for the GPU path:

   ```bash
   uv run snakemake \
     --snakefile workflow/mcmc.smk \
     --profile profiles/slurm \
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
and passes rule threads as `--host-device-count` (with `--cpu-threads` pinned to
1) so each job gets one logical device per concurrent chain without
oversubscribing BLAS/XLA. The core budget lives only in the profile; it never
enters a run's config hash. Keep `set-threads.run_mcmc` ≥ the run config's
`[sampler] num_chains`.

As a worked example, run the **cosmology** (quick ET-2L) sweep locally. First
generate the sweep configs and their batch manifest (writes
`configs/mcmc/cosmology/<network>__<analysis>__<observation>.json` and
`configs/mcmc/manifests/mcmc.batch.cosmology.json`):

```bash
uv run python scripts/workflow.py gen-configs
```

`configs/mcmc/manifests/mcmc.batch.cosmology.json` now lists all 8 cosmology
sweep points (2 networks × 4 analyses × 1 observation). For the full six-network
grid use `mcmc.batch.cosmology-all-detectors.json` (24 points). To run only a
subset locally — e.g. a few ET-2L `H0` analyses — copy the manifest somewhere
and trim the `runs` list rather than editing the generated file in place (the
next `--write-manifests --force` run overwrites it):

```bash
cp configs/mcmc/manifests/mcmc.batch.cosmology.json configs/mcmc.batch.cosmology-local.json
# then edit configs/mcmc.batch.cosmology-local.json down to the runs you want
```

Dry-run, then submit on (say) 8 cores. With the profile's `run_mcmc=4` thread
override, Snakemake runs `floor(8 / 4) = 2` sweep points at a time:

```bash
uv run python scripts/workflow.py mcmc configs/mcmc.batch.cosmology-local.json --profile local
uv run python scripts/workflow.py mcmc configs/mcmc.batch.cosmology-local.json \
  --profile local --submit
```

To trade per-run speed for more concurrency, lower the `set-threads` value in
`profiles/local`: `run_mcmc=2` runs four sweep points at once on the same 8
cores (and must still be ≥ each run's `num_chains`). For a large sweep, prefer
single-chain configs (`num_chains = 1`) and let the job level do the work.

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
