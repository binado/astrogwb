# Running inference

## Ad-hoc runs

`astrogwb-run-mcmc` accepts a complete validated configuration and a prebuilt
waveform catalog. The project ships no standalone example config. Assemble the
curated inventory first:

```bash
uv run astrogwb-validate-config \
  inputs/config.yaml \
  --output-dir outputs/configs

uv run astrogwb-run-mcmc \
  --config outputs/configs/cosmological-parameters/ET-2L-aligned-CE-Hanford.json \
  --catalog outputs/catalogs/bns-n16384-df1.h5
```

Any generated config under `outputs/configs/<experiment>/<run>.json` works for
direct runner and profiling work.

## Curated experiment runs

[`inputs/config.yaml`](../inputs/config.yaml) is the sole MCMC configuration
source. Its `base` mapping owns settings shared by all runs:

- random seed and observing time;
- frequency and cosmology grids;
- sampler settings;
- fiducial values;
- default priors;
- output defaults.

Its `experiments` mapping declares four groups and all 22 runs:

```text
inputs/config.yaml
  base + experiments.<experiment> + runs.<run>
  -> outputs/configs/<experiment>/<run>.json
  -> outputs/chains/<experiment>/<run>.nc
  -> outputs/chains/<experiment>/<run>.json
```

YAML anchors and aliases remove repeated detector and likelihood mappings.
Experiment-level settings overlay `base`, then a run mapping overlays both.
Nested mappings are merged, lists replace inherited lists, and each overridden
prior specification replaces that parameter's inherited prior table wholesale.
The optional `catalog` run key routes the prebuilt catalog and is not written
into the scientific run config.

The groups are:

| Experiment | Runs |
| --- | ---: |
| `cosmological-parameters` | six detector networks, `fixed`, `sampled`, and `H0-Omega_m` |
| `astrophysical-parameters` | `Madau-Dickinson` and `z_peak` |
| `modified-propagation` | six detector networks, `Xi_0`, and `Xi_0-H0` |
| `variable-injection-size` | `n8192`, `n16384`, and `n32768` |

`assemble_config` is one local Snakemake job. A change to
`inputs/config.yaml` validates the complete inventory and regenerates all 22
canonical JSON files together. Each MCMC job then consumes its own JSON.

Run one experiment's chains through Snakemake from
`packages/astrogwb-paper/`:

```bash
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/local --cores 8 cosmological_parameters_chains
snakemake --snakefile workflow/mcmc.smk \
  --profile profiles/slurm modified_propagation_chains
```

Build all cosmological chains, figures, and tables with
`plot_cosmological_parameters`.

## Outputs and provenance

Every labelled run has deterministic `.nc` and `.json` paths under
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
snakemake --snakefile workflow/catalog.smk \
  --profile profiles/local --cores 8 catalogs
```

If a required catalog is absent, the MCMC workflow fails with a
`MissingInputException` instead of silently scheduling catalog generation.
