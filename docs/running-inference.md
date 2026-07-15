# Running inference

There are two working examples provided in the repo:

- The [`run_mcmc.py` script](../scripts/run_mcmc.py) accepts a TOML or JSON configuration file and is suitable for running MCMC on a cluster.
- The [`mcmc.py` notebook](../notebooks/mcmc.py) has the same functionality as the script but can be run interactively in a Jupyter notebook. You can use `uvx jupytext --to ipynb notebooks/mcmc.py` to convert it to a Jupyter notebook in your local machine.

Both examples assume a population of binary neutron star (BNS) mergers following the example described in [Generating catalogs](./catalog-generation.md). The headless runner takes the proposal catalog explicitly:

```bash
uv run --extra mcmc python scripts/run_mcmc.py \
  --config configs/mcmc.example.toml \
  --catalog out/catalogs/bns-n16384-df1.h5
```

Add `--extra cuda` (or `--extra tpu`) when you need the matching JAX
accelerator plugin.

## Running on an accelerator (GPU/TPU, incl. Google Colab)

Both entrypoints resolve their JAX platform and chain method through
[`astrogwb.runtime.configure_runtime`](../src/astrogwb/runtime.py), which must
run before `jax`/`numpyro` are otherwise imported. `--platform` accepts
`auto` (default), `cpu`, `cuda`, or `tpu`. The chain method auto-resolves to
`"parallel"` whenever at least one visible device is available per chain,
including multi-device GPU/TPU runs. With fewer accelerator devices than
chains it uses `"vectorized"`; with too few CPU devices it uses
`"sequential"`. Explicit `--chain-method` values always take precedence. For
the headless runner:

```bash
uv run --extra mcmc --extra tpu python scripts/run_mcmc.py \
  --config configs/mcmc.example.toml \
  --catalog out/catalogs/bns-n16384-df1.h5 \
  --platform tpu
```

The `mcmc.py` notebook detects Google Colab automatically and uses four chains
for every Colab hardware type. It probes for TPU hardware with a stdlib-only
check *before* any `pip install`, then installs `astrogwb[mcmc,tpu]` or
`astrogwb[mcmc,cuda]`. The `mcmc` extra is the runner/plotting stack; `cuda`
and `tpu` are accelerator plugins (prefer one). TPU runs pass
`platform="tpu"`; CPU and GPU retain `platform="auto"`. The bootstrap also
installs plotting dependencies and mounts Google Drive for the waveform
catalog. Point `CATALOG_PATH`'s Colab branch at wherever you upload the
catalog on Drive. Off Colab the bootstrap cell is a no-op.

## Generating sweep configs

Generate sweep configs explicitly on the local machine or cluster submit host;
existing configs are skipped unless `--force` is supplied:

```bash
uv run --extra mcmc python scripts/generate_mcmc_configs.py
```

The committed [`configs/mcmc.sweeps.toml`](../configs/mcmc.sweeps.toml) is a
declarative product of named detector `[networks]`, likelihood
`[observations]`, inference `[analyses]`, and prior variants. Each `[runs.*]`
table selects lists from those collections and expands
`networks × analyses × observations`. Invariant cosmology, sampler, fiducial,
and output settings come from the sweep's explicitly declared
[`configs/mcmc.base.toml`](../configs/mcmc.base.toml).

An observation supplies a complete `observation_time`, `f_min`, and `f_max`.
An analysis selects a named prior variant for every sampled parameter and may
optionally override existing base fiducials inline:

```toml
[analyses.H0]
sampled_params = ["H0"]
priors = { H0 = "uniform" }
fiducials = { H0 = 67.66 }
```

Overrides affect the injected spectrum, proposal density, fixed constants,
and sampler initialization. They cannot introduce parameters absent from the
base fiducial table. Generated run IDs contain every product dimension:
`<network>__<analysis>__<observation>`.

To submit generated sweep configs as a batch (locally or on SLURM), see
[Snakemake workflow](./snakemake-workflow.md#mcmc-workflow).

## Outputs

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

See the [plotting notebook](../notebooks/mcmc_plotting.py) for examples of how to visualize the results.
