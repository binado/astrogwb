# Repository Guidelines

## Layout

One package, `astrogwb`, with the reproducibility application inside it:

- `src/astrogwb/`: publishable scientific library. It owns cosmology, detector
  geometry/data, GWB contractions, source populations declared as NumPyro
  models, importance weighting, the generic NumPyro model, waveform power
  reduction, and HDF5 catalog serialization.
- `src/astrogwb/paper/`: the reproducibility application. It owns
  configuration, runtime setup, prior adaptation, console commands, plotting
  style, and the SNR/inference helpers the figures use.
- `config/`, `scripts/`, `notebooks/`, `profiles/`, `docs/`, `Snakefile`: the
  application's committed assets, at the repository root.
- `tests/core/` and `tests/paper/`.



## Commands

Every check is a `just` recipe, and CI runs the same string:

- `uv sync --extra notebook --group dev`: full development environment.
- `just lint`, `just typecheck`, `just fmt`.
- `just test-core`, `just test-paper`, `just test-integration`.
- `uv run --group workflow snakemake --snakefile Snakefile --dry-run --cores 1 experiments`:
  production workflow entrypoint (omit `--dry-run` to execute). Chains come
  from `run_experiment_<name>` targets; figures are opt-in via the `plot_*`
  rules. The dry run above merges and catalog-checks all 27 runs without
  building anything.

## Configuration

Run workflow and application entrypoints from the repository root; configuration paths are relative to the caller's current working directory, and no checkout discovery is performed.

A catalog records the density that drew it: its file carries the registered
population model, that model's construction settings, the hyperparameters it
was drawn at, and the `astrogwb` version that generated it. Which of those
density factors enter an importance weight is *not* part of the record -- no
sample depends on it -- so it is declared by the analysis that reweights the
draw. A population is a registered factory returning a callable
`parameters -> (merger_rate, model)`, where `model()` is a no-argument NumPyro
model whose sample sites are the columns a catalog stores; the rate and the
density come from one call. `bns_coba` is the one shipped population, its
variants (mass law, time delay, guard mixture) being construction kwargs.
Adding a population means adding a registered factory under
`src/astrogwb/populations/`, never an import path in a config. Which
hyperparameters a population needs, which of them only rescale the spectrum
(`analysis.amplitude_parameter`), and that a guard mixture is a proposal and
never an analysis target or injection are contracts documented on the factory
and trusted, not checked.

Simulators are partials implementing one protocol, `astrogwb.simulators.core.Simulator[**P, D, M]`:
built from a validated metadata record `M` (it names itself with `key()`) plus
cost-only settings (`chunk_size`), then called on the inputs that
vary per item. `__call__` is the whole interface: its signature and the layout of
its output `D` are the simulator's own contract. `PopulationSimulator` and
`BackgroundSpectralDensitySimulator` take one key and return one draw (a
population, a background spectral density), already vectorized over its
events; callers loop over `batch_keys`. Waveform power itself is the
generator's (`WaveformMetadata.build().generate_batch(sources, chunk_size=)`,
trace-safe, chunked with `lax.map`); `polarization_power_data(generator,
sources)` checks the sources once, jits it and packs `PolarizationPowerData`.
`draw_catalog(metadata, key)` is the plain catalog (sources drawn at the
fiducials, then that packaging), and `astrogwb.gwb.importance` places the same
draw at redshift nodes:
`importance_catalog(ImportanceCatalogMetadata, key)` places every intrinsic
draw at each Gauss-Legendre node in the scale factor `1/(1+z)` at a fixed effective
inclination, and `build_importance_spectrum` integrates redshift on those nodes
and reweights only the intrinsic draws. It is meant to replace the
redshift-sampled `astrogwb.importance.spectral` estimator.
`reference_catalog(CatalogMetadata, key)` makes one waveform per draw instead,
at the population's minimum redshift (file stem `reference_catalog-<key>-<seed>`,
apart from the plain catalog's), and `build_rescaled_spectrum` rescales the
weighted mean power to the nodes in each call -- `s**4` and `f * s`, with
`s = (1+z)/(1+z_min)`, exact for a (2,2)-only aligned-spin waveform -- so the
node count is a builder argument and costs no waveforms. Both builders return
pytree callables (`jax.tree_util.Partial` over a by-value static half): pass one
*as an argument* to a jitted function -- `GaussianGWBBatchedLikelihood` takes
`spectral_density_fn` per call for this -- and the catalog is a traced input,
held once and shared by every compilation, instead of a constant each closure
bakes in.
Anything static is bound in the
constructor. The output `D`
is a per-simulator `TypedDict` of array-like leaves (`PopulationData`,
`BackgroundSpectralDensityData`, `PolarizationPowerData`), layouts documented
on it.
A **stochastic simulator takes a JAX key**, not metadata: it picks one
realization of the density the metadata describes. Batched keys come from
`batch_keys(seed, n) = fold_in(key(seed), arange(n))`, which is prefix-stable and
needs `jax_enable_x64` (a 64-bit seed would otherwise truncate). Persistence is
the caller's: `write(path, data, metadata, *, seed=, batch_size=)` writes the data
tree as an HDF5 group tree with the metadata JSON, seed and batch size as
top-level attributes (atomically), and `load(path, MetadataType)` returns
`(data, metadata, attrs)`. Nothing caches or addresses by content.

A run declares what each role draws as `[analysis.injection]` and
`[analysis.proposal]`, each a complete `CatalogMetadata`
(`astrogwb.simulators.polarization_power`) once the merge resolves its `${...}`
references, plus the seed of each role in `[analysis.seeds]` (default 41 for
both); both roles default, field by field, to the shared draw `[catalog]`.
`RunConfig.catalog_request(role)` returns `(CatalogMetadata, np.uint64 seed)`.
By convention the file is `outputs/catalogs/polarization_power-<key>-<seed>.h5`
(`catalog_stem(metadata, seed)`). The Snakefile names every run's roles at parse
time (`resolve_run_catalogs`, which maps each file stem to `(metadata, seed)`),
`rule waveform_catalog` hands `scripts/generate_catalog.py` the metadata as JSON
plus `--seed` and declares no config inputs, and `run_mcmc` and the notebooks
alike reach the files through `astrogwb.paper.catalogs.ensure_catalog(metadata,
seed, directory)` (notebooks via `run_catalog`), which loads a hit, checks it
against the request, and on a miss draws it with `draw_catalog`
at `batch_keys(seed, 1)[0]` -- unless `generate=False`, which the workflow's
`run_mcmc --cached-only` uses so a job never generates.
`ensure_catalog` and `run_catalog` return the `(data, metadata)` pair as is; there
is no catalog wrapper class, and `restrict_redshift(data, metadata, zmin, zmax)`
(`astrogwb.simulators.polarization_power`) narrows samples and recorded
population window together. There is
no `config/catalogs/` and no catalog name. The version is part of the key, so
**bump `version` in `pyproject.toml` whenever a change alters what a population
draw or a waveform generator produces**, or stale catalogs keep being served.
`just catalogs` maps stems back to what they draw, at which seed, and which runs
use them.
Forward-model spectra: a `BackgroundSpectralDensityMetadata` (each hyperparameter
a fixed number or a prior spec) plus one key per draw, through
`BackgroundSpectralDensitySimulator(metadata, chunk_size=...)(key)`, returning
`BackgroundSpectralDensityData` with a leading draw axis even for one draw:
`frequencies` is `(F,)`, `spectral_density` is `(1, F)`, and event counts, rates
and hyperparameter columns are `(1,)`.
`stack_spectra([simulator(k) for k in batch_keys(seed, n)])` concatenates that
axis into `(n, F)` and `(n,)` columns, keeping the first frequency grid.
Its inputs must share a grid and hyperparameter names. Construction owns data
correctness; there is no data validator or catalog wrapper. Metadata remains
separate, and files use the same draw-first layout.
Each key is one draw, so a draw's events depend on its own key alone (event `i`
comes from a key folded with `i`, so `chunk_size` changes cost, not the draw). A spectrum is a population draw reduced through a waveform:
`PopulationSimulator` (`astrogwb.simulators.population`) draws hyperparameters,
an exact `count` and flat `source_parameters` (a `PopulationDrawMetadata`, which
`BackgroundSpectralDensityMetadata` extends; `.sources` is the waveform-free
part), and
`ChunkedPowerSum` reduces the draw's events chunk by chunk with a masked tail.
`BackgroundSpectralDensitySimulator.reduce(population_data)` pushes an
already-drawn population through another waveform. Poisson and fixed counts share that path
and differ only in the per-draw count and normalization. Build a simulator once
and reuse it in a loop -- it owns the compiled stages.
The spectrum scripts default to `astrogwb.paper.cache.default_cache_dir() / "spectra"`,
shared across worktrees. `platformdirs` honors `XDG_CACHE_HOME` on Linux and macOS,
otherwise using the platform's user cache directory. CLI directory overrides
take precedence; cache locations are outside the scientific metadata/key.
Workflow catalog outputs remain under `outputs/catalogs`.
`scripts/simulate_spectra.py` builds the record from the `[spectra]` table of
`--config` layers (the four shared layers, then
`config/simulations/spectrum/<name>.toml`) and the draw keys from its sibling
`[draws]` table (`seed`, `num_draws`); those files are not run layers.

A run config is six TOML layers merged in order -- the four shared layers
`config/defaults.toml`, `config/waveforms.toml`, `config/populations.toml` and
`config/detectors.toml`, then the experiment
`config/runs/<experiment>/_base.toml`, then the run.
`config/defaults.toml` declares the shared scientific defaults and the default
draw `[catalog]`; `config/waveforms.toml` and `config/populations.toml` declare
named `[waveforms.<name>]` / `[populations.<name>]` records that catalogs and the analysis target refer to;
`config/detectors.toml` declares `[networks]` and optional `[detectors]` overrides. Every layer opens with a comment saying what it is
for, so what each committed run -- and each catalog override -- is for lives in
its own file; `config/runs/README.md` indexes the experiments and the catalogs
they share. The analysis target is `analysis.population`, a
`PopulationMetadata`, evaluated rather than drawn. `[fiducials]` is both
where NUTS initializes and what a run's catalogs are drawn at, so editing the
shared `[fiducials]` re-keys every catalog. `[fiducials]`, `[priors]` and
`[networks]` are also read directly by the notebooks and figure scripts through
`astrogwb.paper.config.fiducials()` / `priors()` / `networks()`, and
`waveform_generator()` / `population_model()` read `[catalog]`'s waveform and
population, so a copy cannot drift from what the runs sample.
`merge_config_layers` folds the layers with `knf` (pyknf): a deep merge, except
that each `priors.<param>` table replaces the inherited one (`PRIOR_SHALLOW =
"priors.*"`), then resolves every `"${a.b}"` reference against the merged
result. A reference to a table merges as that table -- setting a key under one
overrides that field and keeps the rest, so a run changes one role's population
size alone -- and can be reached *through* (`${x.y}` resolves when `x` is
itself a reference). A table that is a base plus additions is written
`extends = "${a.b}"` (`MERGE_KEY = "extends"`); it takes one base. To change a
named variant for every role that names it in one run, override it at its
source. `[catalog]`, `[waveforms]` and `[populations]` exist only to be
referenced, and `RunConfig` drops them. `knf src/astrogwb/detector/{geometry,sensitivity}.toml <layers> --shallow 'priors.*' --interpolate --merge-key extends`
prints the same merge in the shell. The packaged detector files use the same
`[detectors.<name>]` format and are merged before the six run layers. A run names a detector network (`analysis.network`) rather than listing
detectors. `config/plotting.toml` is presentation -- LaTeX parameter labels and
savefig settings, reached through `astrogwb.paper.plotting` -- and is
deliberately *not* a run layer. No entrypoint is handed an assembled config.
Every one -- `run_mcmc` and the figure and diagnostic scripts alike -- takes the
layer paths as repeated `--config` flags and merges them in process, and the
workflow rule declares the same files as its `input:`, so the dependency edge
and the data path are one list. `run_mcmc` writes the resolved config next to
the chain as JSON (an output, not a layer) and stamps both catalog keys into it.

## Coding and testing

- Target Python `>=3.12`, use explicit public type hints, `pathlib.Path`, Ruff
  formatting, snake_case functions, PascalCase classes, and uppercase constants.
- Runtime configuration must run before the XLA *backend* is initialized. Importing JAX
  or NumPyro does not initialize it; creating an array or querying devices
  does. `tests/paper/test_cli.py` asserts both halves.
- Add core tests for scientific interfaces. Mark dependency-heavy tests with
  `@pytest.mark.integration`.

## Commits and pull requests

Use focused Conventional Commits (`feat:`, `fix:`, `chore:`, `refactor:`).
PRs should state purpose, key changes, commands run, and data/config impacts.
