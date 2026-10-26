# The simulator pipeline

`astrogwb.simulators` draws Poisson realizations of the background in three
stages. Each stage runs alone, writing an artifact directory that the next
stage opens, or fused in memory, where nothing but the final background is
written.

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
once. `Waveforms.write(directory)` streams those chunks to disk;
`WaveformArtifact.open` reads them back the same way. If the population was
never written, it is written first, to `directory/population`.

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

Every background manifest records this convention as `[normalization]`
version 1, and `BackgroundArtifact.open` refuses any other version.
`combine_pair_spectra(Y, frequencies, detectors)` weights a network's pairs
optimally into one spectrum, using the noise curves in `psd_references`. It is
a separate step, so changing the weights needs no new projection. The mean of
the combined spectrum is `S_h`.

The expected, sky-averaged spectrum is the likelihood's model prediction. No
simulator draws it.

## Artifacts

An artifact directory holds `metadata.toml` (its manifest) and `data.h5` (its
arrays). The manifest is an input: the next stage reads it to find its parent
and checks that the parent is the one it was made from.

| Table | Contents |
| --- | --- |
| `[artifact]` | `kind`, `schema_version`, `astrogwb_version`, the scientific `key`, the `data` file |
| `[metadata]` | the stage's scientific settings, which `key` hashes |
| `[parent]` | the parent's `key` and, when it is on disk, its `manifest` path relative to this file |
| `[realization]` | population and background: the realized count, rate (`1 / s`), hyperparameters, `T`, the RNG key |
| `[conventions]` | waveforms: frame, polarization angle, Fourier convention, units, source ordering |
| `[normalization]` | background: the pair-spectrum convention and its version |
| `[execution]` | `chunk_size`; recorded but never hashed |

| Artifact | `data.h5` |
| --- | --- |
| population | `source_id` `(N,)`, `source_parameters/<name>` `(N,)` |
| waveforms | `frequencies` `(F,)`, `source_id` `(N,)`, `plus`, `cross` `(N, F)` complex |
| background | `frequencies` `(F,)`, `pairs` `(P, 2)`, `pair_spectral_density` `(P, F)` |

Opening a waveform artifact resolves its population and checks:

- the population key;
- the source IDs and the array shapes.

A missing or mismatched parent raises. Opening reads no polarization.

Keys are content hashes. The population key hashes its metadata, including the
schema version, together with its RNG key. The waveform key hashes the
population key and the waveform metadata. The background key hashes the
waveform key, the detector list and the normalization version. Population
schema version 2 moved the extrinsic draws into the population stage, so no
earlier spectrum cache is reused.

## Python

```python
from astrogwb.simulators import (
    BackgroundSimulator, PopulationSimulator, WaveformSimulator,
    PopulationArtifact, WaveformArtifact, batch_keys,
)

key = batch_keys(41, 1)[0]

# Stage by stage
PopulationSimulator(population_metadata)(key).write("population/")
waveforms = WaveformSimulator(waveform_metadata, chunk_size=1024)
waveforms(PopulationArtifact.open("population/")).write("waveforms/")
background = BackgroundSimulator(detectors, chunk_size=1024)
background(WaveformArtifact.open("waveforms/")).write("background/")

# Fused: at most one chunk of polarizations in memory
population = PopulationSimulator(population_metadata)(key)
spectra = background(waveforms(population))
```

Build each simulator once and call it often; each holds its compiled chunk.
For several networks, run the background once over the union of their
detectors. Then select each network's rows with
`BackgroundArtifact.pair_spectra(network)` and combine them.
`astrogwb.paper.ensemble.network_spectra_ensemble` does this, caching one
background per draw.

## Command line

The `cli` extra installs `astrogwb`:

```bash
astrogwb simulate population population.toml --seed 41 --output population/
astrogwb simulate waveforms --population-file population/metadata.toml \
    --waveform-metadata waveform.toml --output waveforms/
astrogwb simulate background --waveform-file waveforms/metadata.toml \
    --detectors network.toml --output background/

# Fused: only the background is written
astrogwb simulate background --population-metadata population.toml --seed 41 \
    --waveform-metadata waveform.toml --detectors network.toml --output background/
```

The metadata files hold the fields of `PopulationMetadata`, `WaveformMetadata`
and `DetectorConfiguration` at their top level. Realization `i` of a seed is
`--index i`, the key `batch_keys(seed, i + 1)[i]` that ensembles use. Runtime
options (`astrogwb simulate --platform cpu --cpu-threads 8 ...`) are applied
before the JAX backend initializes.

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
