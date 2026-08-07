# Running inference

There are two inference entry points in the repository:

- The `astrogwb-run-mcmc` command accepts a TOML or JSON configuration file and is suitable for running MCMC on a cluster.
- The [`mcmc.py` notebook](../notebooks/mcmc.py) provides the same functionality interactively. Convert it locally with `uvx jupytext --to ipynb packages/astrogwb-paper/notebooks/mcmc.py`.

Both examples assume a population of binary neutron star (BNS) mergers following the example described in [Generating catalogs](./catalog-generation.md). The headless runner takes the proposal catalog explicitly:

```bash
uv run astrogwb-run-mcmc \
  --config configs/mcmc.example.toml \
  --catalog out/catalogs/bns-n16384-df1.h5
```

Select an accelerator while syncing and running the paper member with
`--package astrogwb-paper --extra cuda` (or `--extra tpu`).

## Running on an accelerator (GPU/TPU, incl. Google Colab)

Both entrypoints resolve their JAX platform and chain method through
[`astrogwb_paper.runtime.configure_runtime`](../src/astrogwb_paper/runtime.py), which must
run before `jax`/`numpyro` are otherwise imported. `--platform` accepts
`auto` (default), `cpu`, `cuda`, or `tpu`. The chain method auto-resolves to
`"parallel"` whenever at least one visible device is available per chain,
including multi-device GPU/TPU runs. With fewer accelerator devices than
chains it uses `"vectorized"`; with too few CPU devices it uses
`"sequential"`. Explicit `--chain-method` values always take precedence. For
the headless runner:

```bash
uv run --package astrogwb-paper --extra tpu astrogwb-run-mcmc \
  --config configs/mcmc.example.toml \
  --catalog out/catalogs/bns-n16384-df1.h5 \
  --platform tpu
```

The `mcmc.py` notebook detects Google Colab automatically and uses four chains
for every Colab hardware type. It probes for TPU hardware with a stdlib-only
check *before* installing dependencies, then clones the workspace and installs
the core member editably with `astrogwb[tpu]` or `astrogwb[cuda]` plus the
paper member's `notebook` extra. TPU runs pass
`platform="tpu"`; CPU and GPU retain `platform="auto"`. The bootstrap also
installs plotting dependencies and mounts Google Drive for the waveform
catalog. Point `CATALOG_PATH`'s Colab branch at wherever you upload the
catalog on Drive. Off Colab the bootstrap cell is a no-op.

## Generating sweep configs

Generate sweep configs explicitly on the local machine or cluster submit host;
existing configs are skipped unless `--force` is supplied:

```bash
uv run astrogwb-generate-mcmc-configs
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

### Amplitude-marginalized runs

Setting `analysis.likelihood = "amplitude_marginalized"` (see the commented
block in [`configs/mcmc.example.toml`](../configs/mcmc.example.toml))
integrates one multiplicative parameter -- `H0` or `local_merger_rate` -- out
of the likelihood analytically instead of sampling it with NUTS. The
resulting `.nc` is a drop-in replacement for a sampled chain: post-processing
draws the marginalized parameter and writes it into the `posterior` group
under its own physical name (e.g. `H0`), alongside the real
`total_merger_rate` (rescaled from the model's `template_merger_rate` by the
drawn amplitude), so every figure script and `paper.smk` path that reads
`total_merger_rate` or a sampled parameter by name works unchanged.

Two things are different from a sampled chain, though:

- There is **no `log_likelihood` group**. `az.from_numpyro` builds that group
  from observed sample sites, and the marginalized model has none -- the
  likelihood is a single `numpyro.factor`. `az.loo` and `az.waic` do not
  apply to these chains.
- The posterior additionally carries `quadrature_effective_nodes`, a
  per-draw diagnostic for how many quadrature grid points actually resolve
  the conditional posterior (should be comfortably above ~30; the runner logs
  a warning otherwise). If it is low, raise `analysis.amplitude_num_nodes`.
- **`sampled_params` no longer describes the chain.** It means "parameters
  NUTS has a latent for", and the marginalized parameter deliberately is not
  one: it must stay out of `sampled_params`, which drives `init_to_value` and
  the `set(priors) == set(sampled_params)` invariant. Use
  `RunConfig.posterior_params` for anything describing the saved chain --
  plot `var_names`, summaries, run records. The JSON sidecar records both.

See the [plotting notebook](../notebooks/mcmc_plotting.py) for examples of how to visualize the results.
