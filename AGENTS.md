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
  rules. Run `snakemake validate` first: it merges and catalog-checks all 27 runs
  without building anything.

## Configuration

Run workflow and application entrypoints from the repository root; configuration paths are relative to the caller's current working directory, and no checkout discovery is performed.

A catalog records the density that drew it: its file carries the registered
population model, the registered redshift and mass sub-models it is composed
from and their construction settings, the hyperparameters it was drawn at, and
the `astrogwb` version that generated it. Which of those
density factors enter an importance weight is *not* part of the record -- no
sample depends on it -- so it is declared by the analysis that reweights the
draw. Adding a variant means registering a sub-model -- a redshift model in
`src/astrogwb/populations/redshift.py` or a mass model in
`populations/mass.py` -- and naming it in a population record, never an import
path in a config. A redshift model builds a `RedshiftFn` (hyperparameters in, a
`RedshiftLaw` out), so the merger rate is read off the same law that is sampled.

Simulators are cached functions. `astrogwb.simulators` holds nodes
`fn(inputs, metadata, **settings) -> arrays`, wrapped by `@cached`
(`astrogwb.simulators.core`): `inputs` and the returned arrays are nested dicts
of arrays, `metadata` is a validated record that names itself with `key()`, and
`settings` (a `chunk_size`) change cost, not the result. With a `cache_dir` the
result lives at `<cache_dir>/<fn.__name__>-<metadata.key()>-<digest(inputs)>.h5`
(`fn.path(inputs, metadata, cache_dir)`), the body runs only on a miss, and
`generate=False` makes a miss raise `FileNotFoundError`. A **seed is an input,
not metadata**: it picks one realization of the density the metadata describes
(a 0-d `uint64` for `polarization_power`, a 1-d `uint64` array of per-draw
`seeds` for `spectra`, from `split_seed(seed, n)`). `read(path)` returns the
`(inputs, outputs, metadata_json)` of a file handed over by path.

A run declares what each role draws as `[analysis.injection]` and
`[analysis.proposal]`, each a complete `CatalogMetadata`
(`astrogwb.simulators.polarization_power`) once the merge resolves its `${...}`
references, plus the seed of each role in `[analysis.seeds]` (default 41 for
both); both roles default, field by field, to the shared draw `[catalog]`.
`RunConfig.catalog_request(role)` returns `(CatalogMetadata, np.uint64 seed)`.
The file is `outputs/catalogs/polarization_power-<key>-<digest>.h5`. The
Snakefile names every run's roles at parse time (`resolve_run_catalogs`, which
maps each file stem to `(metadata, seed)`), `rule waveform_catalog` hands
`scripts/generate_catalog.py` the metadata as JSON plus `--seed` and declares no
config inputs, and `run_mcmc` and the notebooks alike reach the files through
`astrogwb.simulators.polarization_power.polarization_power({"seed": seed},
metadata, cache_dir=...)` (notebooks via `astrogwb.paper.catalogs.run_catalog`),
which checks a hit against the request and generates a miss -- unless
`generate=False`, which the workflow's `run_mcmc --cached-only` uses so a job
never generates. `PolarizationPowerCatalog.from_arrays(outputs, metadata)` wraps
the arrays. There is no `config/catalogs/` and no catalog name. The version is
part of the key, so **bump `version` in `pyproject.toml` whenever a change alters
what a population draw or a waveform generator produces**, or stale catalogs
keep being served. `just catalogs` maps stems back to what they draw, at which
seed, and which runs use them.
Forward-model spectra use the same cache: a `SpectraMetadata` (each
hyperparameter a fixed number or a prior spec) plus one seed per draw, through
`astrogwb.simulators.spectra.spectra({"seeds": split_seed(seed, n)}, metadata,
cache_dir=..., chunk_size=...)`. Each seed is one draw, so a draw depends on its
own seed alone.
The spectrum scripts default to `astrogwb.paper.cache.default_cache_dir() / "spectra"`,
shared across worktrees. `platformdirs` honors `XDG_CACHE_HOME` on Linux and macOS,
otherwise using the platform's user cache directory. CLI directory overrides
take precedence; cache locations are outside the scientific metadata/key.
Workflow catalog outputs remain under `outputs/catalogs`.
`scripts/simulate_spectra.py` builds the record from the `[spectra]` table of
`--config` layers (the four shared layers, then
`config/simulations/spectrum/<name>.toml`) and the seeds from its sibling
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
