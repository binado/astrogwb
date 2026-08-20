# Catalog generation

Catalog generation has two stages:

1. draw a fresh population from an MD/uniform-redshift mixture;
2. generate and persist its frequency-domain waveform polarizations.

The intermediate population is a Snakemake `temp()` output. The waveform
catalog under `outputs/catalogs/` is the durable artifact.

## Population graphs

The population definition is split into three fragments:

- [`population.base.yaml`](../inputs/populations/population.base.yaml) contains
  every non-redshift sampler and transform;
- [`population.md.yaml`](../inputs/populations/population.md.yaml) contains only
  the Madau-Dickinson redshift sampler;
- [`population.uniform-redshift.yaml`](../inputs/populations/population.uniform-redshift.yaml)
  contains only the uniform-redshift sampler.

The workflow recursively merges the base with each redshift fragment. The two
complete graphs therefore differ only in redshift by construction.

## Generate a population

`astrogwb-generate-population` loads the two complete graph configurations with
`GraphSimulator`. For an interior mixing fraction it draws directly from
`MixtureSimulator` with weights `[1 - epsilon, epsilon]`. The endpoints invoke
only the selected graph.

```bash
uv run astrogwb-generate-population \
  --md-config outputs/population-configs/md.yaml \
  --uniform-redshift-config outputs/population-configs/uniform-redshift.yaml \
  --uniform-mixing-fraction 0.1 \
  --num-samples 16384 \
  --seed 42 \
  --output outputs/populations/bns-n16384-eps=0.1-df1.h5
```

The command refuses to overwrite an existing output unless `--force` is
provided. It stores only physical population columns, not component labels or
proposal-density columns.

## Generate waveforms

```bash
uv run astrogwb-generate-waveform-catalog \
  --population outputs/populations/bns-n16384-eps=0.1-df1.h5 \
  --output outputs/catalogs/bns-n16384-eps=0.1-df1.h5 \
  --approximant IMRPhenomXAS_NRTidalv3 \
  --sampling-frequency 8192 \
  --minimum-frequency 2 \
  --maximum-frequency 4096 \
  --reference-frequency 20 \
  --frequency-resolution 1 \
  --chunk-size 2048
```

Source-frame masses are converted to detector-frame masses by multiplying by
`1 + z` immediately before waveform generation.

## Catalog inventory

[`inputs/catalogs.yaml`](../inputs/catalogs.yaml) contains one catalog list.
Each entry specifies a sample count, seed, and inclusive
`uniform_mixing_fraction`. Injection and proposal are inference roles rather
than different catalog types.

Names include the fraction for readable provenance, for example
`bns-n16384-eps=0.1-df1`, but inference never parses scientific settings from
the filename. Config assembly expands the selected catalog's complete proposal
definition into the canonical run JSON. Inference computes the analytic
proposal density in memory from that definition.

Build catalogs explicitly:

```bash
cd packages/astrogwb-paper
uv run --group workflow snakemake --snakefile Snakefile --cores 1 \
  --allowed-rules population_config population waveform_catalog \
  outputs/catalogs/injection-bns-n32768-eps=0-df1.h5 \
  outputs/catalogs/bns-n16384-eps=0.1-df1.h5
```
