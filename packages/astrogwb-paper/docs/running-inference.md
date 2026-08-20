# Running inference

## Ad-hoc runs

`astrogwb-run-mcmc` accepts a complete validated configuration and a prebuilt
waveform catalog. The project ships no standalone example config. Assemble the
curated inventory first:

```bash
uv run astrogwb-validate-config \
  inputs/experiments.yaml \
  --output-dir outputs/configs

uv run astrogwb-run-mcmc \
  --config outputs/configs/cosmological-parameters/ET-2L-aligned-CE-Hanford.json \
  --injection-catalog outputs/catalogs/injection-bns-n32768-eps=0-df1.h5 \
  --proposal-catalog outputs/catalogs/bns-n16384-eps=0.1-df1.h5
```

Any generated config under `outputs/configs/<experiment>/<run>.json` works for
direct runner and profiling work.

## Curated experiment runs

[`inputs/experiments.yaml`](../inputs/experiments.yaml) is the sole MCMC configuration
source. Its `base` mapping owns settings shared by all runs:

- random seed and observing time;
- frequency and cosmology grids;
- sampler settings;
- fiducial values;
- default priors;
- output defaults.

Its `experiments` mapping declares four groups and all 21 runs:

```text
inputs/experiments.yaml
  base + experiments.<experiment> + runs.<run>
  -> outputs/configs/<experiment>/<run>.json
  -> outputs/chains/<experiment>/<run>.nc
  -> outputs/chains/<experiment>/<run>.json
```

YAML anchors and aliases remove repeated detector and likelihood mappings.
Experiment-level settings overlay `base`, then a run mapping overlays both.
Nested mappings are merged, lists replace inherited lists, and each overridden
prior specification replaces that parameter's inherited prior table wholesale.
The optional `catalog` run key routes the prebuilt proposal catalog and is not
written into the scientific run config. Every run separately consumes the
shared independent fiducial injection catalog.

The groups are:

| Experiment | Runs |
| --- | ---: |
| `cosmological-parameters` | six detector networks, `H0-Omega_m`, and `H0-merger-rate` |
| `astrophysical-parameters` | `Madau-Dickinson` and `z_peak` |
| `modified-propagation` | six detector networks, `Xi_0`, and `Xi_0-H0` |
| `variable-proposal-size` | `n8192`, `n16384`, and `n32768` |
| `variable-proposal-guard` | `eps1e-1`, `eps1e-2`, and `eps1e-3` |

`assemble_config` is one local Snakemake job. A change to
`inputs/experiments.yaml` validates the complete inventory and regenerates all 24
canonical JSON files together. Each MCMC job then consumes its own JSON.

Run one experiment's chains through Snakemake from
`packages/astrogwb-paper/`:

```bash
snakemake --snakefile Snakefile \
  --allowed-rules assemble_config run_mcmc run_experiment_cosmological_parameters \
  --profile profiles/local --cores 8 run_experiment_cosmological_parameters
snakemake --snakefile Snakefile \
  --allowed-rules assemble_config run_mcmc run_experiment_modified_propagation \
  --profile profiles/slurm run_experiment_modified_propagation
```

Build all cosmological chains, figures, and tables with
`plot_cosmological_parameters`.

## Outputs

Every labelled run has a deterministic `.nc` path under
`outputs/chains/<experiment>/`. The assembled config at
`outputs/configs/<experiment>/<run>.json` is the record of the settings that
produced it; the run itself writes no provenance sidecar.

Ad-hoc unlabelled runs retain the timestamped
`mcmc-<params>-det=<detectors>-seed<n>-<timestamp>` convention.

## Catalog prerequisite

Experiments consume existing catalogs and never generate them implicitly.
Build the required catalogs first:

```bash
snakemake --snakefile Snakefile \
  --allowed-rules population_config population waveform_catalog \
  --profile profiles/local --cores 8 \
  outputs/catalogs/injection-bns-n32768-eps=0-df1.h5 \
  outputs/catalogs/bns-n16384-eps=0.1-df1.h5
```

If a required catalog is absent, the MCMC workflow fails with a
`MissingInputException` instead of silently scheduling catalog generation.
