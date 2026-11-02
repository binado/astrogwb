# The simulator pipeline

`astrogwb.simulators` draws a Poisson realization of the background in one
pass: a `SimulationConfig` and a key in, one `BackgroundArtifact` out, written
to one HDF5 file.

```text
SimulationConfig = RealizationMetadata + WaveformMetadata + DetectorRegistry
                   (+ segment_duration)

simulate_population(RealizationMetadata, key)                  -> PopulationArtifact
simulate_background(PopulationArtifact, Waveform..., Registry) -> BackgroundArtifact
simulate(config, key)                                          = both, composed
```

`SimulationConfig` is a frozen pydantic model: everything that determines the
run, nothing about the draw. The draw, a seed and an index, is the key the
config is simulated at. Execution details (`chunk_size`, `segment_batch_size`)
change cost, never the result, and are arguments of `simulate`, not config.

`DetectorRegistry` resolves detector geometry and PSD references once for
simulation and noise generation. The artifact derives each network's detector
pairs, in the order of the pair axis of its arrays, from the registry in its
config, so a reader needs nothing else.

Only the population stage draws random numbers. The background stage is
deterministic given the population and its config.

## Stages

**Population.** `simulate_population(metadata, key)` draws the
hyperparameters (fixed, or drawn from their priors), a Poisson count
`N ~ Poisson(R T)`, and every source:

- the intrinsic parameters and the inclination, from the population model;
- the extrinsic angles: `right_ascension` uniform on `[0, 2 pi)`,
  `declination = arcsin(uniform(-1, 1))` (an isotropic celestial direction),
  `polarization_angle` uniform on `[0, pi)`, and `coa_phase`. Angles are radians.

All sources are drawn in one batch through `simulate_sources`, which the
importance catalogs of `astrogwb.importance` share, with independent
random streams for hyperparameters, count, and the joint source model.
The joint model samples intrinsic parameters and extrinsic angles together.
Counts are Poisson only; the
fixed-count mode was removed. A fixed-size Monte Carlo sample of a population
is a reference catalog for importance weighting (`astrogwb.importance.catalog`), not
a realization.

**Background.** `simulate_background(population, waveform_metadata, registry)`
loops over fixed-size source chunks (the last chunk padded with copies of its
first source whose polarizations are masked to zero, so every realization
compiles once; the compiled kernels are cached per waveform and pair layout,
so every call with the same waveform reuses them). For each chunk it evaluates the complex radiation-frame
polarizations `(h+, hx)` (which depend on the intrinsic parameters, inclination
and coalescence phase), projects them onto every pair `a < b` of the union of
the networks' detectors once, and accumulates the real cross-power

```text
delay[a,i] = -dot(source_direction[i], location[a]) / c
H[a,i]   = (F+[a,i] h+[i] + Fx[a,i] hx[i]) exp(-2 pi i f delay[a,i])
K[a,b,i] = Re(conj(H[a,i]) H[b,i])
D[a,b]   = sum_i K[a,b,i]
Y[a,b]   = 5 D[a,b] / T
```

`DetectorGeometry` owns WGS-84 vertex locations in metres and tilted-arm
response tensors computed from local north/east/up vectors. `Detector` owns a
name, geometry and lazy sensitivity. Registry builders return tuples of these
objects: `build_detectors` preserves supplied order; `build_network` sorts it.
Projection uses the gwmock/LAL plus/cross basis and delay sign. Background
simulation fixes `gmst=0`. Right
ascension and declination must be in the celestial frame compatible with the
supplied sidereal angle; no GPS, epoch or precession conversion occurs.
Tensors and locations are stacked outside JIT and passed into the chunk kernel.

The detector strains are projected directly from both polarizations before
the cross-power is formed. The observation time `T` comes from the population. The
sum is never divided by the realized count. Each network then keeps its own
rows, `(P_n, F)`: `artifact.spectra[name]` returns them. A network sharing detectors with
another costs no extra projection.

Detectors are stored sorted, whatever order they are declared in, so a
network's pairs are always `a < b` in site-code order. That order is the pair
axis of every spectrum and of `astrogwb.detector.pair_terms` on the same
detectors.

The background does not model:

- interference between sources (outside the noisy mode below);
- detector noise, unless `segment_duration` is set;
- the Earth's rotation during a signal.

## The pair-spectrum convention

Averaged over isotropic sky positions and a uniform polarization angle,
`<K_ab> = I gamma_ab / 5`. Here `I = |h+|^2 + |hx|^2` and `gamma_ab` is the
normalized isotropic overlap reduction function of the native WGS-84 geometry
(elevations and arm tilts included), so this relationship,
`E[Y_ab] = gamma_ab S_h`, and the unit-mean network-response normalization
are exact.
The factor-five pair normalization and intrinsic waveform conventions remain unchanged.

A change of this convention ships with a new package version, which
`BackgroundArtifact.open` checks (see below).
`combine_pair_spectra(Y, frequencies, registry, network_name)` weights a
network's pairs optimally into one spectrum `(F,)`, using the `psd_reference`
of each detector in the registry. It is computed on demand and not stored;
the likelihood scores the pair spectra directly. Every detector needs a noise
curve. The mean of the
combined spectrum is `S_h`.

## Detector noise

By default the background carries no noise. With `segment_duration` (seconds,
e.g. 2048; `--segment-duration` of `astrogwb simulate`)
`simulate_background` cuts the observation into `N = floor(T / tau)`
segments of length `tau_eff = T / N` and assigns every source to a uniformly
drawn segment (sources are Poisson in time). Each segment has its own coherent
strain `h_a^s = sum_{i in s} H_{a,i}` and fresh independent noise
`n_a^s ~ CN(0, tau_eff S_a / 2)` per detector and bin; empty segments still
carry noise. With `C_ab = sum_s Re[conj(h_a^s + n_a^s)(h_b^s + n_b^s)]` and
`r = 1 / sqrt(tau_eff * df)`,

    Y_ab = (5 / T) * [(1 - r) K_ab + r C_ab],   K_ab = sum_i K_ab,i.

The mean is the incoherent signal; the zero-mean part is scaled by `r`, so the
noise-by-noise term has the standard deviation of `pair_bin_scale` for any
grid and any `tau`. The model is valid while signals are mostly shorter than
`tau` and bins hold `tau * df >= 1` modes (shorter segments raise). The segment
labels and noise are folded from the realization's key (streams 2 and 1), so
the background stays a deterministic function of it, and
`pair_spectral_density` then holds signal plus noise. The cost is about 0.2 ms per segment, so
`tau = 2048` s (about 1.5e4 segments a year) adds seconds to the waveforms'
tens of seconds; `segment_batch_size` only trades memory for speed.

The likelihood inputs are not stored. `BackgroundArtifact.noise()`
derives, per network, the pair ORFs and per-pair Gaussian scales `(P_n, F)` and
the effective PSD and its Gaussian scale `(F,)`, as a `NetworkNoise`, from the
background's grid, the registry of its config and its observation time.

The expected, sky-averaged spectrum is the likelihood's model prediction. No
simulator draws it.

## The file

One run writes one HDF5 file, atomically (a temporary file renamed into
place, so a file at its final path is complete):

| Item | Contents |
| --- | --- |
| `frequencies` | `(F,)`, Hz |
| `networks/<name>/pair_spectral_density` | `(P_n, F)`, `1 / Hz`, the pairs in the order of `DetectorRegistry.pairs(name)` |
| attr `metadata` | `SimulationConfig` as JSON |
| attr `version` | the astrogwb version that wrote it |
| attr `realization` | JSON of `rng_key`, `count`, `total_merger_rate` (`1 / s`) and `hyperparameters` |

`BackgroundArtifact.open(path)` rebuilds the artifact, config included, from
the file alone. It raises `ValueError` ("regenerate with astrogwb simulate") if
the `version` attr is missing or is not the installed version: a release that
changes a convention invalidates the files of the one before it, and nothing is
converted. There are no content keys, no manifest and no sources file; the
sources are drawn, projected and discarded in the same pass.

Ensembles use `<root>/<seed>-<index>.h5`, so paths can be computed without
JAX.

## Python

```python
from astrogwb.simulators import (
    BackgroundArtifact, SimulationConfig, realization_key, simulate,
)

config = SimulationConfig(
    population=population_metadata,
    waveform=waveform_metadata,
    detectors=registry,
    segment_duration=None,  # or seconds, for sampled detector noise
)

# At most one chunk of polarizations is held in memory.
background = simulate(config, realization_key(41, 7), chunk_size=1024)
background.write("root/41-7.h5")

background = BackgroundArtifact.open("root/41-7.h5")
background.config.population.observation_time
background.noise()  # likelihood inputs, from the config's detectors
```

`simulate` composes `simulate_population(config.population, key)` and
`simulate_background(population, config.waveform, config.detectors, ...)`,
which remain available as building blocks.

`astrogwb.simulators.network_spectra_ensemble(root, seed, n, names)` reads an
ensemble's files and stacks each network's spectra. It never simulates;
`Snakefile` writes the paper's ensembles (see
`docs/coverage-draws-on-slurm.md`).

## Command line

The `cli` extra installs `astrogwb`. `astrogwb simulate OUTPUT.h5` simulates
one draw; every config option is required:

```bash
astrogwb simulate root/41-0.h5 --population-metadata population.toml \
    --waveform-metadata waveform.toml --detector-config config/detectors.toml \
    --seed 41 --index 0

# With sampled detector noise, in segments of 2048 s
astrogwb simulate root/41-0.h5 --population-metadata population.toml \
    --waveform-metadata waveform.toml --detector-config config/detectors.toml \
    --seed 41 --segment-duration 2048
```

The population and waveform files hold the fields of `RealizationMetadata` and
`WaveformMetadata` at their top level. Repeat `--detector-config PATH` to merge
registry layers in argument order. Draw `i` of a seed is `--index i`, the key
`realization_key(seed, i)` that ensembles use. The runtime options
(`--platform cpu --cpu-threads 8`) are applied before the JAX backend
initializes. To change a setting, regenerate: files are neither converted nor
mixed.

## Migration

| Before | Now |
| --- | --- |
| `BackgroundSpectralDensitySimulator` (sky-averaged) | removed; the expected spectrum is the likelihood's prediction |
| `NetworkSpectralDensitySimulator`, `JointNetworkSpectralDensitySimulator` | `simulate_background` over the detector union, then `combine_pair_spectra` |
| `BackgroundSpectralDensityMetadata`, `NetworkSpectralDensityMetadata` | `RealizationMetadata` + `WaveformMetadata` + `DetectorRegistry` |
| `PopulationDrawMetadata` (`count`, `num_events`) | `astrogwb.simulators.RealizationMetadata`, Poisson only |
| `PopulationSimulator(metadata)(key)`, `BackgroundSimulator(waveform, registry)(population)` | `simulate_population(metadata, key)`, `simulate_background(population, waveform, registry)` |
| `astrogwb.simulators.PopulationMetadata` | `astrogwb.simulators.RealizationMetadata` |
| `astrogwb.simulators.polarization_power` | `astrogwb.importance.catalog` |
| `astrogwb.simulators.core.{write,load,content_key}` | `astrogwb.utils.io` |
| `spectra_ensemble`, `network_spectra_ensemble(EnsembleRecord, ...)` | `Snakefile` writes the ensemble; `network_spectra_ensemble(root, seed, n, names)` reads it |
| `scripts/simulate_spectra.py`, `config/simulations/spectrum/` | `astrogwb simulate` |
| one artifact directory per stage (`metadata.toml` + `data.h5`, `[parent]` references), then realization directories (`metadata.toml` + `source_parameters.h5` + `background.h5`) | one HDF5 file per draw carrying its `SimulationConfig` and version |
| `PopulationArtifact.write(dir)` / `.open(path)`, `Realization`, `RealizationSimulator`, `simulate_realization` | `simulate(config, key)`; no sources file |
| `WaveformSimulator`, `Waveforms`, `WaveformArtifact`, `polarization.h5` | removed; `simulate_background(population, waveform_metadata, registry)` fuses waveform evaluation and pair projection |
| `content_key`, `ContentKeyed`, `Keyed`, `catalog_stem`, `reference_catalog_stem` | removed; files are named by the caller |
| `BackgroundArtifact.upstream`, `.manifest_path`, `.waveform_key`, `.key()`, `population_key`, `waveform_key`, `background_key` | `BackgroundArtifact.config`; no content keys |
| `astrogwb simulate population` / `waveforms` / `background`, then `astrogwb simulate DIRECTORY --background --keep-sources` | `astrogwb simulate OUTPUT.h5` with `--population-metadata`, `--waveform-metadata`, `--detector-config`, `--seed` |
| per-run network metadata snapshots | core `DetectorRegistry`, supplied to simulation and readers |
| `astrogwb.paper.ensemble.network_spectra_ensemble`, `NetworkSpectra` | `astrogwb.simulators.network_spectra_ensemble`, `NetworkSpectra` |
| `scripts/write_ensemble_configs.py`, `paper_networks`, `ensemble_metadata` | `config/detectors.toml` and `config/ensembles/<name>/`, edited by hand |
| ensemble caches of #389 and #390 (`<cache>/ensembles/<key>/<seed>-<i>/`) | not read; regenerate them with the workflow |
| one background over the detector union; `BackgroundArtifact.pair_spectra(network)`, `.detectors`, `.pairs` | per-network spectra; `BackgroundArtifact.spectra[name]`, `.pairs[name]` |
| `Realization.open(dir).background()`, `background.noise(registry)` | `BackgroundArtifact.open(file)`, `background.noise()` |
| `--detector-config` with network and detector layers | repeated `--detector-config PATH` options |
| `ensure_reference_catalog`, `settings_digest`, `noise_digest`, posterior grid caches | `astrogwb model-catalog`; grids are computed on every run |
| `astrogwb.paper.cache.default_cache_dir` | explicit paths under `outputs/` |
