# Running inference

## Ad-hoc runs

`astrogwb-run-mcmc` accepts a complete validated configuration and a prebuilt
waveform catalog:

```bash
uv run astrogwb-run-mcmc \
  --config configs/mcmc.example.toml \
  --catalog outputs/catalogs/bns-n16384-df1.h5
```

The standalone examples under `configs/` remain useful for direct runner and
profiling work.

## Curated experiment runs

Production runs use a one-file/one-chain model:

```text
inputs/mcmc.base.toml
  + experiments/<experiment>/mcmc.<run>.toml
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

Each run TOML owns every scientific difference from the base:

- `sampled_params`;
- detector names;
- likelihood and amplitude-marginalization settings;
- run-specific prior overrides.

For example:

```toml
# experiments/H0-omega-m/mcmc.H0-Omega_m.toml
sampled_params = ["Omega_m"]

[analysis]
detectors = ["S1", "R1", "C1"]
likelihood = "amplitude_marginalized"
amplitude_parameter = "H0"
amplitude_num_nodes = 1024
```

There is no network/analysis/observation product and no fragment lookup.
Adding a chain means adding one run TOML and declaring its stable run ID in
`astrogwb_paper.config.experiments`.

To assemble one config manually:

```bash
knf inputs/mcmc.base.toml \
    experiments/H0-all-detectors/mcmc.ET-2L-aligned.toml \
    --strict -f json \
  | uv run astrogwb-validate-config - \
      --output /tmp/ET-2L-aligned.json
```

The validator checks that every sampled parameter has a prior and fiducial and
that amplitude-marginalized runs define a valid amplitude parameter.

Run the corresponding experiment through Snakemake:

```bash
uv run astrogwb-workflow mcmc H0-all-detectors --profile local
uv run astrogwb-workflow mcmc H0-all-detectors \
  --profile slurm --submit
```

Use `--chains-only` to omit local post-processing.

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
uv run astrogwb-workflow \
  catalog outputs/catalogs/bns-n16384-df1.h5 --submit
```

The variable-injection-size experiment additionally requires the 8192 and
32768 catalogs.
