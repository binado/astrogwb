# Running inference

## Ad-hoc runs

`astrogwb-run-mcmc` accepts a complete validated configuration and a prebuilt
waveform catalog. The project ships no standalone example config: assemble one
first with `astrogwb-validate-config` (see [Curated experiment
runs](#curated-experiment-runs) for what it merges), then run it.

```bash
uv run astrogwb-validate-config \
  --base inputs/mcmc.base.toml \
  --run ET-2L-aligned-CE-Hanford \
  experiments/H0-all-detectors.toml \
  --output /tmp/adhoc.json

uv run astrogwb-run-mcmc \
  --config /tmp/adhoc.json \
  --catalog outputs/catalogs/bns-n16384-df1.h5
```

Any config the workflow has already assembled under
`outputs/configs/<experiment>/<run>.json` works the same way, which is the
usual choice for direct runner and profiling work.

## Curated experiment runs

Production runs use one TOML per experiment. Each `[runs.<id>]` table is one
chain overlay on the shared base:

```text
inputs/mcmc.base.toml
  + experiments/<experiment>.toml  [runs.<run>]
  -> outputs/configs/<experiment>/<run>.json
  -> outputs/chains/<experiment>/<run>.nc
  -> outputs/chains/<experiment>/<run>.json
```

The base owns settings shared across all experiments:

- random seed and observing time;
- frequency and cosmology grids;
- sampler settings;
- fiducial values;
- default priors;
- output defaults.

Top-level keys in the experiment file (except `runs`) overlay the base for
every run. Each `[runs.<id>]` table then overlays detector names,
`sampled_params`, likelihood settings, catalog path, and run-specific priors.

For example:

```toml
# experiments/H0-omega-m.toml
sampled_params = ["Omega_m"]

[analysis]
detectors = ["S1", "R1", "C1"]
likelihood = "amplitude_marginalized"
amplitude_parameter = "H0"
amplitude_num_nodes = 1024

[runs.H0-Omega_m]
```

There is no network/analysis/observation product and no fragment lookup.
Adding a chain means adding one `[runs.<id>]` table to the experiment file.

To assemble one config manually:

```bash
uv run astrogwb-validate-config \
    --base inputs/mcmc.base.toml \
    --run ET-2L-aligned \
    experiments/H0-all-detectors.toml \
    --output /tmp/ET-2L-aligned.json
```

The validator checks that every sampled parameter has a prior and fiducial and
that amplitude-marginalized runs define a valid amplitude parameter.

Run one experiment's chains through Snakemake (from
`packages/astrogwb-paper/`, adding `--dry-run` to preview):

```bash
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/local --cores 8 H0_all_detectors_chains
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/slurm H0_all_detectors_chains
```

The `H0-all-detectors`, `H0-merger-rate`, and `H0-omega-m` chains feed one
paper section. Build all of its figures and tables together with the
`plot_cosmological_parameters` target.

## Outputs and provenance

Every labelled experiment run has deterministic `.nc` and `.json` paths under
`outputs/chains/<experiment>/`. The JSON sidecar records the catalog path and
SHA-256, canonical config SHA-256, detectors, seed, fiducials, priors, sampler
settings, and git revision.

Runtime controls such as platform and chain method affect execution rather than
the scientific result and are not included in the config hash.

Ad-hoc unlabelled runs retain the timestamped
`mcmc-<params>-det=<detectors>-seed<n>-<timestamp>` convention.

## Catalog prerequisite

Experiments consume existing catalogs and never generate them implicitly.
Build the required catalogs first:

```bash
snakemake --snakefile workflow/catalog.smk --cores 1 \
  outputs/catalogs/bns-n16384-df1.h5
```

The variable-injection-size experiment additionally requires the 8192 and
32768 catalogs.
