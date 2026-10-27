# The simulator pipeline

`astrogwb.simulators` draws Poisson realizations of the background in three
stages. The stages run fused in memory, or one at a time into a single
realization directory that each later stage adds to.

```text
PopulationMetadata --PopulationSimulator(key)--> PopulationArtifact
WaveformMetadata   --WaveformSimulator(pop)----> Waveforms / WaveformArtifact
DetectorConfiguration --BackgroundSimulator(w)-> BackgroundArtifact
```

Only the population stage draws random numbers. The waveform and background
stages are deterministic, given their inputs.

## Stages

**Population.** `PopulationSimulator(metadata)(key)` draws the
hyperparameters (fixed, or drawn from their priors), a Poisson count
`N ~ Poisson(R T)`, and every source:

- the intrinsic parameters and the inclination, from the population model;
- the extrinsic angles: `polar_cosine` and `azimuth` (an isotropic,
  Earth-fixed direction), `polarization` and `coa_phase`.

Event `i` is drawn from `fold_in(source_key, i)` and
`fold_in(extrinsic_key, i)`, so `chunk_size` changes the cost and never the
realization. Source `i` has the stable ID `i`. Counts are Poisson only; the
fixed-count mode was removed. A fixed-size Monte Carlo sample of a population
is a reference catalog for importance weighting (`astrogwb.gwb.catalog`), not
a realization.

**Waveforms.** `WaveformSimulator(metadata)(population)` returns `Waveforms`,
the complex polarizations `(h+, hx)` of every source. They are in the
radiation frame: they depend on the intrinsic parameters, the inclination and
the coalescence phase. The sky position and polarization angle are applied
once, through the antenna patterns, at projection. `Waveforms` generate
lazily: `iter_chunks(chunk_size)` yields `(source_parameters, plus, cross)`
one fixed-size chunk at a time. The last chunk is padded with copies of its
first source, whose polarizations are zero, so every realization compiles
once. `Waveforms.write(path)` streams those chunks to an HDF5 file;
`WaveformArtifact.open` reads them back the same way.

**Background.** `BackgroundSimulator(detectors)(waveforms)` loops over the
chunks. For each chunk it projects the polarizations onto every detector pair
`a < b` and accumulates the real cross-power

```text
H[a,i]   = F+[a,i] h+[i] + Fx[a,i] hx[i]
K[a,b,i] = Re(exp(i phase[a,b,i]) conj(H[a,i]) H[b,i])
D[a,b]   = sum_i K[a,b,i]
Y[a,b]   = 5 D[a,b] / T
```

The product is computed through the Stokes contraction
(`polarization_stokes`, then `stokes_cross_power`), which keeps every
polarization term. The observation time `T` comes from the population. The
sum is never divided by the realized count.

The background does not model:

- interference between sources;
- detector noise;
- the Earth's rotation during a signal.

## The pair-spectrum convention

Averaged over isotropic sky positions and a uniform polarization angle,
`<K_ab> = I gamma_ab / 5`. Here `I = |h+|^2 + |hx|^2` and `gamma_ab` is the
normalized isotropic overlap reduction function. So, for an isotropic and
unpolarized population, `E[Y_ab] = gamma_ab S_h`.

Every `[background]` table records this convention as its `normalization`,
version 1, and `BackgroundArtifact.open` refuses any other version.
`combine_pair_spectra(Y, frequencies, detectors)` weights a network's pairs
optimally into one spectrum, using the noise curves in `psd_references`. It is
a separate step, so changing the weights needs no new projection. The mean of
the combined spectrum is `S_h`.

The expected, sky-averaged spectrum is the likelihood's model prediction. No
simulator draws it.

## Realization directories

A realization is one directory. It holds one manifest and at most one product
of each stage:

```text
<directory>/
  metadata.toml
  source_parameters.h5   # unless keep_sources=False / --no-keep-sources
  polarization.h5        # only with keep_polarizations / --keep-polarizations
  background.h5          # when a background is configured
```

The caller chooses the directory. Ensembles use `<root>/<seed>-<index>/`, so
paths can be computed without JAX. The realization ID is the population key.

There is one `metadata.toml`, rewritten atomically whenever a stage is added:

| Table | Contents |
| --- | --- |
| `[layout]` | `version = 1`, `astrogwb_version` |
| `[realization]` | `id` (the population key), `seed`, `index`, `rng_key`, `count`, `total_merger_rate` (`1 / s`), `observation_seconds`, `hyperparameters` |
| `[population]` | `metadata`, `key` |
| `[waveforms]` | `metadata`, `key`, `conventions` (frame, polarization angle, Fourier convention, units, source ordering) |
| `[background]` | `metadata` (the detectors), `key`, `normalization`, `units` |
| `[execution]` | `chunk_size`, `source_chunk_size`; recorded but never hashed |

A stage table means the stage is configured. A product file means it is
complete. Products are written to a temporary file and renamed into place, so
an interrupted run never leaves a partial product at its final path.

| Product | Datasets |
| --- | --- |
| `source_parameters.h5` | `source_id` `(N,)`, `source_parameters/<name>` `(N,)` |
| `polarization.h5` | `frequencies` `(F,)`, `source_id` `(N,)`, `plus`, `cross` `(N, F)` complex |
| `background.h5` | `frequencies` `(F,)`, `pairs` `(P, 2)`, `pair_spectral_density` `(P, F)` |

`RealizationSimulator(population, waveform, background)` applies one rule per
stage:

| Settings given | Recorded | Result |
| --- | --- | --- |
| yes | no | record them; run the stage if its product is wanted |
| yes | same key | no rerun; regenerate the product only if it is wanted and missing |
| yes | other key | `ValueError` |
| no | yes | use the recorded settings when a later stage needs them |
| no | no | the stage does not run; a later stage that needs it raises `ValueError` |

Upstream data that is not on disk is regenerated in memory. The sources come
from the recorded metadata and `rng_key`, and the polarizations are generated
chunk by chunk from the `[waveforms]` metadata. A directory without a manifest
needs population settings, or raises `FileNotFoundError`.

Keys are content hashes. The population key hashes its metadata, including the
schema version, together with its RNG key. The waveform key hashes the
population key and the waveform metadata. The background key hashes the
waveform key, the detector list and the normalization version. Population
schema version 2 moved the extrinsic draws into the population stage, so no
earlier spectrum cache is reused. The realization layout did not change any
key.

## Python

```python
from astrogwb.simulators import (
    BackgroundSimulator, PopulationSimulator, RealizationSimulator,
    WaveformSimulator, batch_keys, simulate_ensemble,
)

# Fused in memory: at most one chunk of polarizations
key = batch_keys(41, 1)[0]
population = PopulationSimulator(population_metadata)(key)
waveforms = WaveformSimulator(waveform_metadata, chunk_size=1024)
spectra = BackgroundSimulator(detectors, chunk_size=1024)(waveforms(population))

# One realization directory, stage by stage
RealizationSimulator(population_metadata)("draw/", seed=41)
realization = RealizationSimulator(waveform=waveform_metadata, background=detectors)(
    "draw/"
)
spectra = realization.background()

# An ensemble: draw i at root/41-i, keyed by batch_keys(41, n)[i]
simulator = RealizationSimulator(
    population_metadata, waveform_metadata, detectors, keep_sources=False
)
realizations = simulate_ensemble(simulator, "root/", 41, 100)
```

`RealizationSimulator` builds each stage simulator the first time it is
needed and reuses it, so an ensemble compiles each stage once and a fully
cached ensemble builds none. `Realization.population()`, `.waveforms()` and
`.background()` read a stage's product. When the product is missing, pass the
stage simulators to regenerate it.

For several networks, run the background once over the union of their
detectors (`union_configuration`). Then select each network's rows with
`BackgroundArtifact.pair_spectra(network)` and combine them.
`astrogwb.paper.ensemble.network_spectra_ensemble` does this on top of
`simulate_ensemble`.

## Command line

The `cli` extra installs `astrogwb`. A single `astrogwb simulate DIRECTORY`
runs every stage whose metadata is given:

```bash
# Fused: manifest, sources and background
astrogwb simulate draw/ --population-metadata population.toml --seed 41 \
    --waveform-metadata waveform.toml --background-metadata network.toml

# Stage by stage into the same directory
astrogwb simulate draw/ --population-metadata population.toml --seed 41
astrogwb simulate draw/ --waveform-metadata waveform.toml --keep-polarizations
astrogwb simulate draw/ --background-metadata network.toml
```

The metadata files hold the fields of `PopulationMetadata`, `WaveformMetadata`
and `DetectorConfiguration` at their top level. `--seed` goes with
`--population-metadata` and is refused without it. Realization `i` of a seed
is `--index i`, the key `batch_keys(seed, i + 1)[i]` that ensembles use.
`--no-keep-sources` skips `source_parameters.h5`. The runtime options
(`--platform cpu --cpu-threads 8`) are applied before the JAX backend
initializes.

## Migration

| Before | Now |
| --- | --- |
| `BackgroundSpectralDensitySimulator` (sky-averaged) | removed; the expected spectrum is the likelihood's prediction |
| `NetworkSpectralDensitySimulator`, `JointNetworkSpectralDensitySimulator` | `BackgroundSimulator` over the detector union, then `combine_pair_spectra` |
| `BackgroundSpectralDensityMetadata`, `NetworkSpectralDensityMetadata` | `PopulationMetadata` + `WaveformMetadata` + `DetectorConfiguration` |
| `PopulationDrawMetadata` (`count`, `num_events`) | `astrogwb.simulators.PopulationMetadata`, Poisson only |
| `astrogwb.simulators.polarization_power` | `astrogwb.gwb.catalog` |
| `astrogwb.simulators.spectra.poisson_counts_forward_model` | `astrogwb.inference.poisson_counts` |
| `validate_source_model` | `astrogwb.waveform.validation` |
| `astrogwb.simulators.core.{write,load,content_key}` | `astrogwb.utils.io` |
| `spectra_ensemble`, `network_spectra_ensemble` | `astrogwb.paper.ensemble.network_spectra_ensemble(EnsembleRecord, ...)` |
| `scripts/simulate_spectra.py`, `config/simulations/spectrum/` | `astrogwb simulate` |
| one artifact directory per stage (`metadata.toml` + `data.h5`, `[parent]` references) | one realization directory: `metadata.toml` + `source_parameters.h5`, `polarization.h5`, `background.h5` |
| `PopulationArtifact.write(dir)` / `.open(path)` | `RealizationSimulator`, `Realization.population()`; `write_sources`, `from_tables` |
| `Waveforms.write(dir)` / `WaveformArtifact.open(path)` | `Waveforms.write(path)` / `WaveformArtifact.open(path, metadata=, population=)` |
| `BackgroundArtifact.upstream`, `.manifest_path` | `BackgroundArtifact.waveform_key`; the manifest belongs to the `Realization` |
| `astrogwb simulate population` / `waveforms` / `background` | `astrogwb simulate DIRECTORY` with `--population-metadata`, `--waveform-metadata`, `--background-metadata` |
| `astrogwb.paper.ensemble.union_configuration` | `astrogwb.simulators.union_configuration` |
| ensemble caches of #389 (`<key>/<seed>-<i>/{metadata.toml,data.h5}`) | `scripts/migrate_ensemble_cache.py` converts them in place, without redrawing |
