# Catalog generation

A **catalog** is one persisted waveform draw: a population drawn from a
registered NumPyro model, with its frequency-domain intensity reduced
and written to an explicit path.
Catalogs are the expensive artifact of the analysis.

## Catalogs are written to explicit paths

The caller names the file: `astrogwb model-catalog OUTPUT.h5`, or the names the
Snakefile uses. The file describes itself -- it records the `CatalogMetadata`
JSON, the seed and the `astrogwb` version -- so a reader gets the request back
from the file with `astrogwb.utils.io.load`.

```text
(CatalogMetadata, seed) -> caller-chosen path.h5
```

There is no catalog config tree and no catalog naming scheme. Two draws at different
settings or seeds need different paths; the recorded metadata tells which is which.

### What a catalog declares

`CatalogMetadata` (in `astrogwb.importance.catalog`) carries:

1. `waveform` -- a `WaveformMetadata`, by default the named
   `[waveforms.default]` from `config/waveforms.toml`. The stored band matches
   `[analysis]`'s `minimum_frequency` and `maximum_frequency`: the catalog grid
   *is* the array every model is evaluated on, and an analysis band selects
   bins on it with a mask rather than compressing it.
   The grid is `frequency_spacing` over that band: `"loglinear"` (the default) is
   uniform at `frequency_resolution` up to `turnover_frequency` and geometric
   above it, with the bin width continuous at the turn, which is ~400 bins for
   2--2048 Hz at 1 Hz and 100 Hz against 2047 for `"linear"`. `"log"` is
   geometric throughout, `frequency_resolution` wide at `minimum_frequency`.
   `turnover_frequency` belongs to `"loglinear"` alone: required there and
   rejected elsewhere, so one grid has one record. The turn falls on
   the last uniform point not above `turnover_frequency`. Optional
   `dlogf` independently bounds the natural-log step of the geometric
   tail; endpoint alignment may make the actual step slightly smaller. Linear
   points and the effective turnover stay unchanged, but spacing may jump at
   the turn. It must be finite and positive, and is rejected for `"linear"`
   and `"log"`. Omitting it keeps the current grid. For a `"linear"` Ripple
   waveform `sampling_frequency` is the backend's Nyquist, not the stored grid;
   the other spacings are built directly to `maximum_frequency` and do not use
   it. `[waveforms.linear]` is the default on the uniform grid, for comparing the
   two.
   `approximant="AnalyticInspiral"` selects the closed-form inspiral, and is
   the only approximant accepting the optional `alpha` key (the inspiral
   termination constant, defaulting to the Schwarzschild ISCO value); naming it
   alongside a Ripple approximant is rejected.
   `astrogwb.waveform.waveform_generator(config.catalog.waveform)` builds the
   default draw's generator for a notebook.
2. `population` -- an `AnyPopulation`: a `model_name` and that model's
   construction settings as fields, validated into its `PopulationMetadata`
   subclass (`BNSCobaPopulation` for `"bns_coba"`). The named populations live
   in `config/populations.toml`.
   There is no seed in it: a seed picks a realization of the density, so it is
   an input beside the metadata.
3. `fiducials` -- the hyperparameters the draw is made at, `"${fiducials}"`.
4. `num_samples` -- the draw's size.

The shared default draw is `[catalog]` in `config/defaults.toml`;
`astrogwb.config.load_config().catalog` contains its records.

### What the record holds, and the version

`CatalogMetadata` is the waveform settings, the population record, the
fiducials, the sample count, and the `astrogwb` version. All of it is stored in
the file next to the seed.

The version is recorded so that *code* changes are caught too -- but only if it
is bumped. **Bump `version` in `pyproject.toml` whenever a change alters what a
population draw or a waveform generator produces.** `simulate_catalog` and
`reference_catalog` refuse to draw if the metadata's version differs from the
installed one; regenerate the catalog.

## The population is a typed record

A catalog names a population by its `model_name` and states the construction
settings that model takes, as fields of its `PopulationMetadata` subclass. The
redshift window (`minimum_redshift`, `maximum_redshift`) is a field of every
population, declared on the base class. The window and grid resolution are
referenced from `[populations.cosmological]` and the hyperparameters from the
shared `[fiducials]`. The population declares its density factors and source
outputs.

**A tag, not an import path.** `model_name` values change only on purpose;
module paths move as collateral whenever a module is reorganized, so a
persisted `module:function` string is a reference that silently rots.
`AnyPopulation` is a pydantic discriminated union on `model_name`: an unknown
tag, an unknown setting, a wrong literal (`mass_model = "flat"`) or an
inconsistent pair (`time_delay = true` without `minimum_delay`) fails when the
record is validated, when the config loads, not when the population is built.

A record's `build()` returns a callable
`parameters -> (redshift_distribution, source_model)`. Equal records return
the same callable, so a jitted function closing over it compiles once. One
call builds the redshift distribution once, so the rate and the redshift
density come from the same grid. `source_model()` is a no-argument NumPyro
model of the intrinsic parameters alone, in the source frame; it never sees
redshift. `joint_model` composes the two halves into the model a catalog is
drawn from. Hyperparameters are what the callable is called with, and source
arrays are what `sample_and_logprob` conditions in.

```python
from astrogwb.populations import (
    DEFAULT_DENSITY_SITES,
    BNSCobaPopulation,
    joint_model,
)
from astrogwb.utils.numpyro import sample_and_logprob

population = BNSCobaPopulation(
    mass_model="uniform",
    minimum_redshift=0.0,
    maximum_redshift=20.0,
    n_grid=4096,
).build()
redshift_distribution, source_model = population(params)
total_merger_rate = redshift_distribution.total_merger_rate()  # shape ()
model = joint_model(redshift_distribution, source_model)

sources, log_prob = sample_and_logprob(
    model, key, num_samples=1024, density_sites=DEFAULT_DENSITY_SITES
)
outputs, log_prob = sample_and_logprob(
    model, values=sources, density_sites=DEFAULT_DENSITY_SITES
)  # log_prob: shape (1024,); outputs["luminosity_distance"]: shape (1024,)
```

One call, not two. The redshift density and the merger rate are both
normalizations of the same redshift law, so they come from one
`~astrogwb.distributions.redshift.base.RedshiftDistribution`. Its
`distance_model()` samples `redshift` and registers the GW `luminosity_distance`
(modified propagation included), and the quadrature in `astrogwb.importance.spectrum`
reads the rate and the distances off it directly.

`bns_coba` is the one shipped population, and its variants are construction
settings: `mass_model` (`"uniform"` or `"gaussian"`), `time_delay` (with
`minimum_delay`, `maximum_formation_redshift`, `n_delay_nodes`). Modified GW propagation is selected by the
hyperparameters, not a setting: it applies whenever `xi_0` is among them, and is
the identity at `xi_0 = 1`. The contracts each setting carries -- which
hyperparameters it needs, which only rescale the spectrum (`H0` stops being one
when `time_delay` is on) -- are documented in
the `astrogwb.populations.bns_coba` module docstring and are not checked.

The record's fields *are* the construction-settings schema: a setting the
named population does not take is rejected by validation rather than being
silently dropped. To add a population, subclass `PopulationMetadata` with a
`model_name` literal, implement `_build`, and add the class to
`AnyPopulation`.

- The joint model's sample and deterministic sites define the stored columns, including
  redshift, source-frame masses, spins, and `luminosity_distance`. Detector-frame
  masses are not stored: the waveform layer derives them from the source-frame
  masses and redshift. Its sample sites are
  exactly the inputs needed to replay it; a missing one raises `KeyError`.
- `density_sites` selects the density factors included in importance
  weighting. It is an argument, not a stored field: no sample depends on it, so
  the analysis that reweights a draw states it — `DEFAULT_DENSITY_SITES`,
  redshift and the ordered mass pair -- and one value
  is used for both sides of every weight. Omitting factors does not
  marginalize variables.
- `sample_and_logprob` runs the model once under a `batch` plate and returns
  `(outputs, log_prob)` from its trace. Model return values are ignored, and
  auxiliary sample sites such as factors are excluded from output columns.

A key allows free sites to be sampled; without a key every sample site must be
conditioned through `values`. The source count is inferred from supplied columns
or given explicitly as `num_samples`. Deterministic values are recomputed.
Sampling and density evaluation use the same batched execution, preserving exact
self-reweighting. Execution is isolated from enclosing inference models with
`handlers.block`. Sobol sampling uses `sample_sources_qmc` and the same output
collection from a trace.

A population hashes by identity: build it once per run and close a
JIT-compiled function over it, while hyperparameters and source arrays are
traced. To compile sampling, keep `num_samples` static.

### Mass models

Component masses are an ordered pair: `source_frame_mass_1` is the larger one.
Two mass laws share the rest of the BNS Madau-Dickinson declaration:

- **Ordered uniforms** (`mass_model = "uniform"`). Parameters `minimum_mass` and `mass_width`; the
  fiducial support is `[1.0, 2.5]` solar masses, with constant joint density
  `2 / width**2` on the ordered triangle. That triangle is compact, so a NUTS
  step that moves the edges can send catalog samples outside the support and
  drop their importance weights to zero.
- **Ordered Gaussians** (`mass_model = "gaussian"`). Both components are i.i.d. `Normal(mass_mean, mass_sigma)`, then ordered.
  The joint density `2 N(m1) N(m2)` lives on the half-plane `m1 >= m2`, with
  no compact mass support, so moving `(mass_mean, mass_sigma)` never zeros a
  weight. Galactic BNS masses motivate the shape (a Gaussian around
  `1.33 Msun` with width `~0.09 Msun`). The default catalog still
  uses the uniform triangle.

## Generate a catalog

From Python, the cached node draws on a miss and reads on a hit:

```python
import numpy as np

from astrogwb.config import load_config
from astrogwb.importance.catalog import (
    CatalogMetadata,
    intensity,
)

config = load_config()
metadata = CatalogMetadata(
    waveform=config.catalog.waveform,
    population=config.catalog.population,
    fiducials=config.fiducials,
    num_samples=32768,
)
data = intensity(
    {"seed": np.uint64(41)}, metadata, cache_dir="outputs/catalogs"
)
```

A hit is read and checked against the metadata it was asked for; a miss is
generated and saved under the name `intensity.path` gives, written
atomically so an interrupted job never leaves a partial file.

## Catalogs record the density that drew them

Each cached file stores its complete `CatalogMetadata` in a root HDF5
attribute, written once at generation time:

```python
handle.attrs["metadata"] = metadata.model_dump_json()
```

The JSON record nests `waveform` and `population`, and includes `fiducials`,
`num_samples`, and `version`. The population records its
`model_name` and its construction settings as fields. Reading the file back, a
caller validates the attribute with `CatalogMetadata.model_validate_json()`
(or `read_catalog(path)`), and the discriminated union restores the
population's subclass; Pydantic handles the serialization and validation
without a field-by-field HDF5 codec.

What is *not* stored is a callable: `metadata.population.build()` returns
both halves at once, cached on the record, so equal records share one
callable and a jitted consumer compiles once.

That is enough to reconstruct the exact map from hyperparameters to source
density. With the version, it is the file's whole `CatalogMetadata` --
the `metadata` half of the `(data, metadata)` pair -- which a loader checks
against the request it was handed.

The density factors are deliberately *not* part of the record. They change no
sample: the stored columns and power are the same whichever of their densities
a later weight counts, so the choice belongs to the analysis rather than to the
file. What still matters is that one value covers both sides of a ratio — a
proposal density computed with the mass factors excluded, reweighted against a
target that includes them, gives silently wrong weights with no shape error
anywhere — which is why `ImportanceGaussianLikelihood.from_catalog` takes it once and uses
that one value for both the proposal density and the weights.

### The file format

Every catalog uses the same layout, through `astrogwb.utils.io.write`:

| Where | What |
| --- | --- |
| root attribute `metadata` | the metadata's `model_dump_json()` |
| root attributes `version`, `created`, `host` | which `astrogwb`, when and where |
| optional root attributes `seed`, `batch_size` | the caller's draw settings |
| root datasets and groups | the data: a nested dict maps to subgroups, an array to a dataset |

For a catalog, the root holds `frequencies` `(F,)`, `intensity`
`(F, N)` and a `source_parameters/` group of `(N,)` columns. The intensity
dataset was called `polarization_power` before the rename to `intensity`, and
there is no legacy fallback: catalogs written under the old key must be
regenerated. There are no
format names or version files: the path says which node, metadata and inputs a
file answers, and `load(path, MetadataType)` returns `(data, metadata, attrs)` to a
caller that was handed a file by path; `astrogwb.importance.catalog.read_catalog(path)`
is that call for a catalog, returning `(data, metadata, seed)`.
The arrays are used as read (the writer is the only source of files, so they
are not re-validated), and `metadata.population.build()` rebuilds the recorded population: an unknown
`model_name` or a construction setting the population does not take is a
validation error when the metadata is read. Bin widths are not
stored; they are derived from `frequencies`.

## The cache

Persistence and cache locations belong to the callers. The core `write` and
`load` functions neither generate draws nor choose paths:

| | simulator | metadata | input | data |
| --- | --- | --- | --- | --- |
| catalogs | `simulate_catalog` | `CatalogMetadata` | one JAX key | `IntensityData` |
| reference catalogs | `reference_catalog` | `CatalogMetadata` | one JAX key | `IntensityData` |

Catalog files are written to caller-chosen paths, atomically, with the metadata,
seed and version recorded inside. Settings such as `chunk_size` are bound in the
call and are not recorded.

Seeds are inputs because they pick a realization rather than describe a
distribution: `realization_key(seed, index)` folds a draw index into a JAX
key, producing a deterministic key for each realization. Build a simulator once and call it for each
key so its compiled stages are reused.

Bump `version` in `pyproject.toml` whenever a change alters what a population
draw or a waveform generator produces. Version **0.4.0** moved the seed out of
the metadata and changed the file layout, so files from earlier versions are
unusable; remove the old `outputs/catalogs/` and `outputs/spectra/` files and
regenerate.

The waveform metadata records `frequency_spacing`, `frequency_resolution`,
`turnover_frequency` and `dlogf` -- what was *requested* of the
generating backend -- while
the bin widths used in every integral are derived from the `frequency` dataset
itself (`astrogwb.frequency.bin_widths`, used directly on
`data["frequencies"]`); the backend chooses the actual grid, so
the two can differ. The grid need only be strictly increasing: each bin's width
is half the distance between its neighbours, which is the grid spacing on a
uniform grid and grows with frequency above the turn of a `"loglinear"` one.

`dlogf` is serialized even when unset (`null`). Catalogs written
before it existed lack the field; no migration is provided. Configured defaults and notebook grids are unchanged.

The version was bumped with the grid fields (0.2.0), so files from earlier
versions are refused; regenerate them.

## Inclination convention

Physical simulated populations and plain `simulate_catalog` draws sample isotropic inclination:
`cos(iota)` is uniform on `[-1, 1]`, and the returned `inclination` sample site
is in radians. Intensity catalogs store this column, and density
reconstruction conditions on the recorded values even though inclination is
excluded from the default importance-weight factors. The common,
hyperparameter-independent inclination law cancels in those weights.

Poisson realizations of the background -- a population, its waveforms and the
spectra a network records -- are not catalogs: `astrogwb.simulators` simulates
them in one pass into one file, described in [the simulator pipeline](simulator-pipeline.md).

Reference/integration catalogs instead fix `inclination = 0` (face-on), and
serialize `reference_inclination = 0.0` in their metadata. In both cases,
`intensity` remains the literal `|h_plus|^2 + |h_cross|^2`.
For supported aligned-spin, quasi-circular dominant-quadrupole models, the
population mean multiplies face-on power by `2/5`, and the raw same-source
power product multiplies the face-on product by `71/315`. The latter is not
`(2/5)^2`; the relative second moment is still `355/252`. Each raw moment
uses one intrinsic importance weight. Redshift and distance rescaling are
unchanged.

Network response draws already include inclination scatter and have unit
isotropic mean (or the pair overlap for unnormalized pair responses). With
these draws, each source contribution uses `(2/5) P0 r`; the covariance uses
`(2/5)^2 E[P0(f) P0(g)] E[r(f) r(g)]`, without another inclination average
or division by a pair overlap. Compound-Poisson points use the same convention.
These analytic conversions are rejected for higher-mode or precessing models.

The new metadata changes catalog cache keys; existing configuration files need no edits.

Version **0.6.0** replaces the `bns_md_*` populations with `bns_coba`. Older
catalog and spectrum artifacts are not compatible and must be regenerated.

## What is *not* in the file: the analysis window

The recorded settings are the *generation* window, `[0.0, 20.0]`. The analysis
window is narrower — `analysis.population.minimum_redshift = 0.3`
in the analysis setup — so the per-sample log density cannot be
baked into the catalog: it depends on a truncation the run chooses, not on
anything generation knows.
