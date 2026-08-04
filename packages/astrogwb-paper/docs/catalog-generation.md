# Generating catalogs

## Generating a population of CBCs

Our inference framework uses an importance sampling scheme to calculate the spectral density of the astrophysical SGWB with a fixed population of CBCs. To generate the population, we suggest using the excellent [`gwmock-pop`](https://leuven-gravity-institute.github.io/gwmock-pop/) package.

An example BNS population is defined declaratively in [`packages/astrogwb-paper/examples/bns_population.yaml`](../examples/bns_population.yaml).
The complete default recipe lives in
[`configs/catalogs/bns-n16384-df1.toml`](../configs/catalogs/bns-n16384-df1.toml).
Its equivalent explicit population command is:

```bash
uv run gwmock-pop simulate \
  --config packages/astrogwb-paper/examples/bns_population.yaml \
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

The `astrogwb-generate-waveform-catalog` command wraps the [`gwmock-signal`](https://github.com/Leuven-Gravity-Institute/gwmock-signal) package to generate the frequency-domain polarizations used in the spectral-density calculation. The output is a [`pluscross`](https://pypi.org/project/pluscross/) HDF5 catalog of complex polarizations, which inference consumers reduce to polarization power at load time.

The corresponding explicit waveform command is:

```bash
uv run astrogwb-generate-waveform-catalog \
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

Each catalog has an independent `configs/catalogs/<catalog-id>.toml` recipe, so
editing one recipe cannot invalidate another catalog. Catalog generation is not
part of the MCMC submission workflow.

To run the population and waveform steps together as a single reproducible
pipeline (rebuilding the population automatically if it's missing or stale),
see the local catalog workflow in [Snakemake workflow](./snakemake-workflow.md#catalog-workflow).
