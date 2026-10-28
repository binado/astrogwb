# The simulator pipeline

`astrogwb.simulators` draws Poisson realizations of the background in three
stages. The stages run fused in memory, or one at a time into a single
realization directory that each later stage adds to.

```text
PopulationMetadata --PopulationSimulator(key)--> PopulationArtifact
WaveformMetadata   --WaveformSimulator(pop)----> Waveforms / WaveformArtifact
DetectorRegistry    --BackgroundSimulator(w)--> BackgroundArtifact
```

`DetectorRegistry` resolves detector geometry and PSD references once for
simulation and noise generation. Background and noise metadata record each
network's detector pairs, in the order of the pair axis of its arrays, so
readers need no registry.

The registry is frozen for the lifetime of an output dataset. Readers trust the
files they open and preserve the recorded artifact identity. Changing geometry,
PSD references, or membership requires regenerating affected realizations and
noise files.

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

**Background.** `BackgroundSimulator(networks)(waveforms)` loops over the
chunks. For each chunk it projects the polarizations onto every pair `a < b`
of the union of the networks' detectors, once, and accumulates the real
cross-power

```text
H[a,i]   = F+[a,i] h+[i] + Fx[a,i] hx[i]
K[a,b,i] = Re(exp(i phase[a,b,i]) conj(H[a,i]) H[b,i])
D[a,b]   = sum_i K[a,b,i]
Y[a,b]   = 5 D[a,b] / T
```

The product is computed through the Stokes contraction
(`polarization_stokes`, then `stokes_cross_power`), which keeps every
polarization term. The observation time `T` comes from the population. The
sum is never divided by the realized count. Each network then keeps its own
rows, `(P_n, F)`, and their combined spectrum, `(F,)` (see below):
`artifact.spectra[name]` returns both. A network sharing detectors with
another costs no extra projection.

Detectors are stored sorted, whatever order they are declared in, so a
network's pairs are always `a < b` in site-code order. That order is the pair
axis of every product and of `astrogwb.detector.pair_terms` on the same
detectors.

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
`combine_pair_spectra(Y, frequencies, registry, network_name)` weights a
network's pairs optimally into one spectrum, using the `psd_reference` of each
detector in the registry. The background stage stores it per network, so every
detector needs a noise curve, and the noise curves are part of the
background's key. The mean of the
combined spectrum is `S_h`.

## Detector noise

The background carries no noise. `detector_noise(frequencies, registry,
observation_time, network_names=None)` evaluates, per network, the pair ORFs and per-pair Gaussian
scales `(P_n, F)` and the effective PSD and its Gaussian scale `(F,)`, as a
`DetectorNoise`. `astrogwb noise REALIZATION OUTPUT` writes it for the
networks, frequencies and observation time a realization's background was
simulated with, so the two cannot drift apart. It is one file per ensemble:
`DetectorNoise.write`/`open`, datasets `frequencies` and
`networks/<name>/{pair_overlap,pair_scale,effective_psd,scale}`.

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
| `[background]` | `metadata` (the `networks`, by name), `key`, `normalization`, `units` |
| `[execution]` | `chunk_size`, `source_chunk_size`; recorded but never hashed |

A stage table means the stage is configured. A product file means it is
complete. Products are written to a temporary file and renamed into place, so
an interrupted run never leaves a partial product at its final path.

| Product | Datasets |
| --- | --- |
| `source_parameters.h5` | `source_id` `(N,)`, `source_parameters/<name>` `(N,)` |
| `polarization.h5` | `frequencies` `(F,)`, `source_id` `(N,)`, `plus`, `cross` `(N, F)` complex |
| `background.h5` | `frequencies` `(F,)`; per network, `networks/<name>/pair_spectral_density` `(P_n, F)` and `networks/<name>/spectral_density` `(F,)` |

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
waveform key, the networks it covers (their members, and the geometry and
`psd_reference` of the detectors they use, not the labels), the normalization
version and the schema version (3: per-network spectra, keyed by the registry).
Population schema version 2 moved the extrinsic draws into the population
stage, so no earlier spectrum cache is reused. Directories of realization
layout version 1 are refused and are regenerated.

## Python

```python
from astrogwb.simulators import (
    BackgroundSimulator, PopulationSimulator, RealizationSimulator,
    WaveformSimulator, batch_keys,
)

# Fused in memory: at most one chunk of polarizations
key = batch_keys(41, 1)[0]
population = PopulationSimulator(population_metadata)(key)
waveforms = WaveformSimulator(waveform_metadata, chunk_size=1024)
spectra = BackgroundSimulator(registry, chunk_size=1024)(waveforms(population))

# One realization directory, stage by stage
RealizationSimulator(population_metadata)("draw/", seed=41)
realization = RealizationSimulator(waveform=waveform_metadata, background=registry)(
    "draw/"
)
spectra = realization.background()

# An ensemble draw: root/41-i, keyed by batch_keys(41, i + 1)[i]
simulator = RealizationSimulator(
    population_metadata, waveform_metadata, registry, keep_sources=False
)
simulator("root/41-7", seed=41, index=7)
```

`RealizationSimulator` builds the stage simulators of each realization it
runs, so a stage that is already recorded builds none. `Realization.population()`, `.waveforms()` and
`.background()` read a stage's product. When the product is missing, pass the
stage simulators to regenerate it.

`astrogwb.simulators.network_spectra_ensemble(root, seed, n, names)` reads
an ensemble's draws and stacks each network's spectra. It never simulates;
`workflow/coverage_draws.smk` writes the paper's ensembles (see
`docs/coverage-draws-on-slurm.md`).

## Command line

The `cli` extra installs `astrogwb`. A single `astrogwb simulate DIRECTORY`
runs every stage whose metadata is given:

```bash
# Fused: manifest, sources and background
astrogwb simulate root/41-0 --population-metadata population.toml --seed 41 \
    --index 0 --waveform-metadata waveform.toml \
    --background --detector-config config/detectors.toml

# Stage by stage into the same directory
astrogwb simulate draw/ --population-metadata population.toml --seed 41
astrogwb simulate draw/ --waveform-metadata waveform.toml --keep-polarizations
astrogwb simulate draw/ --background --detector-config config/detectors.toml

# The ensemble's detector noise, from one of its realizations
astrogwb noise root/41-0 root/detector_noise.h5 \
    --detector-config config/detectors.toml
```

The population and waveform files hold the fields of `PopulationMetadata` and
`WaveformMetadata` at their top level. Repeat `--detector-config PATH` to merge
registry layers in argument order. `simulate` runs the background stage with
`--background`, which goes with `--detector-config`; the background manifest records each network's detector
pairs. Noise uses the same option and selects the networks recorded in the
realization. The supplied registry is treated as frozen. Readers trust the
files they open and preserve the recorded artifact identity. Changing a
registry does not migrate old outputs; regenerate affected realizations and
noise files.

`--seed` goes with
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
| `BackgroundSpectralDensityMetadata`, `NetworkSpectralDensityMetadata` | `PopulationMetadata` + `WaveformMetadata` + `DetectorRegistry` |
| `PopulationDrawMetadata` (`count`, `num_events`) | `astrogwb.simulators.PopulationMetadata`, Poisson only |
| `astrogwb.simulators.polarization_power` | `astrogwb.gwb.catalog` |
| `astrogwb.simulators.spectra.poisson_counts_forward_model` | `astrogwb.inference.poisson_counts` |
| `astrogwb.simulators.core.{write,load,content_key}` | `astrogwb.utils.io` |
| `spectra_ensemble`, `network_spectra_ensemble(EnsembleRecord, ...)` | `workflow/coverage_draws.smk` writes the ensemble; `network_spectra_ensemble(root, seed, n, names)` reads it |
| `scripts/simulate_spectra.py`, `config/simulations/spectrum/` | `astrogwb simulate` |
| one artifact directory per stage (`metadata.toml` + `data.h5`, `[parent]` references) | one realization directory: `metadata.toml` + `source_parameters.h5`, `polarization.h5`, `background.h5` |
| `PopulationArtifact.write(dir)` / `.open(path)` | `RealizationSimulator`, `Realization.population()`; `write_sources`, `from_tables` |
| `Waveforms.write(dir)` / `WaveformArtifact.open(path)` | `Waveforms.write(path)` / `WaveformArtifact.open(path, metadata=, population=)` |
| `BackgroundArtifact.upstream`, `.manifest_path` | `BackgroundArtifact.waveform_key`; the manifest belongs to the `Realization` |
| `astrogwb simulate population` / `waveforms` / `background` | `astrogwb simulate DIRECTORY` with `--population-metadata`, `--waveform-metadata`, `--detector-config` |
| per-run network metadata snapshots | core `DetectorRegistry`, supplied to simulation and readers |
| `astrogwb.paper.ensemble.network_spectra_ensemble`, `NetworkSpectra` | `astrogwb.simulators.network_spectra_ensemble`, `NetworkSpectra` |
| `scripts/write_ensemble_configs.py`, `paper_networks`, `ensemble_metadata` | `config/detectors.toml` and `config/ensembles/<name>/`, edited by hand |
| ensemble caches of #389 and #390 (`<cache>/ensembles/<key>/<seed>-<i>/`) | not read; regenerate them with the workflow |
| one background over the detector union; `BackgroundArtifact.pair_spectra(network)`, `.detectors`, `.pairs` | per-network spectra; `BackgroundArtifact.spectra[name]`, `.pairs[name]` |
| `Realization.detectors` | `Realization.networks` |
| `--detector-config` with network and detector layers | repeated `--detector-config PATH` options |
| `ensure_reference_catalog`, `settings_digest`, `noise_digest`, posterior grid caches | `astrogwb model-catalog`; grids are computed on every run |
| `astrogwb.paper.cache.default_cache_dir` | explicit paths under `outputs/` |
