# Simulation inputs on SLURM

The paper notebooks read their simulation inputs and never generate them. A
missing input is a `FileNotFoundError`, not a simulation.
[`workflow/coverage_draws.smk`](../workflow/coverage_draws.smk) writes every
input, per ensemble:

| Output | Command | Read by |
| --- | --- | --- |
| `outputs/ensembles/<name>/<seed>-<i>/` | `astrogwb simulate`, one realization per job | every notebook with Poisson draws |
| `outputs/ensembles/<name>/detector_noise.h5` | `astrogwb noise` of draw 0 | the grid and coverage notebooks, `scripts/shot_noise_*.py` |
| `outputs/catalogs/reference-*.h5` | `scripts/draw_reference_catalog.py` | the posterior notebooks, `importance_convergence`, `ensemble_spectrum` |

The ensembles are `fiducial` (the default waveform: the grid notebooks'
injection and `spectrum_snrs`'s Poisson draws) and `coverage`
(`IMRPhenomXAS_NRTidalv3`: `shot_noise_coverage`'s injections and
`ensemble_spectrum`'s draws). Each also has a `-smoke` variant observed for
`1e-4` yr, read under `ASTROGWB_NOTEBOOK_SMOKE=1`. Their settings are the
committed `config/ensembles/<name>/{population,waveform}.toml` and
`config/networks.toml`, which
[`scripts/write_ensemble_configs.py`](../scripts/write_ensemble_configs.py)
writes from the shared config layers. `tests/paper/test_ensemble.py` fails
while they are stale.

Every realization's background holds every paper network's pair spectra and
combined spectrum. The pairs of all networks are projected once per draw, so
adding a network that shares their detectors costs no waveforms. Draw `i` of a
seed is fixed by `(seed, i)` alone, so any split of the indices across
machines gives the same directories, and they merge by copying.

## Regenerating after a change

Nothing is migrated. If you change a detector, a noise curve, the fiducials,
the population or the waveform, rewrite the configs and regenerate the
ensemble:

```bash
uv run python scripts/write_ensemble_configs.py
rm -r outputs/ensembles/coverage
uv run --no-sync snakemake -s workflow/coverage_draws.smk --profile workflow/profiles/slurm --config ensembles=coverage
```

Rerunning against a realization recorded with other settings fails on the
conflict (`ValueError: ... holds background ...`) rather than mixing
settings. Ensembles drawn before per-network backgrounds (one union background,
under `<cache>/ensembles/<key>/`) are not read by anything any more; regenerate
them on the cluster.

## Job sizing

Draw jobs share the Snakemake group `draws`. The profile's `group-components:
draws=5` packs five draws into one SLURM job, run one after another.

| Setting | Value | Why |
| --- | --- | --- |
| draws per SLURM job (`group-components`) | 5 | short jobs that schedule easily and lose little on failure; JIT is a negligible share of a ~80 s draw |
| CPUs per draw (`threads`) | 8 | XLA spreads each waveform chunk over the job's CPUs |
| `mem_mb` | 8000 | two draws on an M4 peaked at 2.1 GB RSS (`/usr/bin/time -l`), with `chunk_size=8192` and `source_chunk_size=65_536`; headroom for per-thread buffers |
| `runtime` | 60 min | the first cluster batch, 10 draws on 4 cores, took 50-90 min; 8 cores should bring 5 draws to ~15-30 min |
| `jobs` | 100 | 500 draws in flight at once |

After a job finishes, `sacct` gives its time and CPU use:

```bash
sacct -u $USER -S today --format=JobID,JobName%40,AllocCPUS,Elapsed,TotalCPU,MaxRSS,State
```

`TotalCPU / (Elapsed x AllocCPUS)` is how busy the cores were: near 1, more
`threads` would still help; well below, prefer more jobs with fewer threads.
Change `threads` with `--config`, and `group-components`, `runtime` and
`mem_mb` in the profile.

## One-time setup on the cluster

Same commit as the Macs, CPU-only JAX (the default wheel):

```bash
git clone git@github.com:binado/astrogwb.git && cd astrogwb
git checkout <commit>
uv sync --extra notebook --extra cli --group dev --group cluster
```

The `cluster` group puts Snakemake and its SLURM executor plugin in the
project venv, so the controller and every job run one interpreter; `cli`
installs the `astrogwb` command and `notebook` the catalog script's stack.
Every command below uses `uv run --no-sync`, and so do the jobs: a plain
`uv run` would sync the venv back to the default groups and remove Snakemake,
and a hundred concurrent jobs must never race to sync it.

Jobs go to the `short` partition, whose time limit must cover the profile's
`runtime`. If the cluster needs an account, uncomment `slurm_account` in the
profile. Change either one in the profile rather than with
`--default-resources` on the command line, which would replace the profile's
whole list, `mem_mb` and `runtime` included.

## Submit

The Snakemake controller stays on the login node for the whole run, so start
it in `tmux` (or `screen`). Calibrate on a few draws first (`stop` caps the
draws of the ensemble's first seed), and check them with `sacct`:

```bash
uv run --no-sync snakemake -s workflow/coverage_draws.smk --profile workflow/profiles/slurm --config ensembles=coverage stop=10
```

Then dry-run and submit the rest:

```bash
uv run --no-sync snakemake -s workflow/coverage_draws.smk --profile workflow/profiles/slurm --config ensembles=coverage --dry-run
```

```bash
uv run --no-sync snakemake -s workflow/coverage_draws.smk --profile workflow/profiles/slurm --config ensembles=coverage
```

The default, `ensembles="fiducial coverage"`, writes both. `threads=<n>` and
`outputs=<path>` go in the same `--config` group.

Each job exports `INPUT="*"` and `OUTPUT="*"` and runs its command under
`job-nanny` whenever it is on `PATH`, as the cluster requires. `job-nanny`
runs the job in a node-local copy of the working directory, which has no
`.venv`, so jobs run `uv run --active --no-sync`: `--active` takes the shared
venv from `VIRTUAL_ENV`, which `uv run` sets when it starts the controller.
Start the controller with `uv run` as above, never with a bare `snakemake`.
If outputs written in the node-local copy do not come back, pass
`outputs=<absolute path on the shared filesystem>` and link it to `outputs/`.

## Monitor and resume

```bash
squeue -u $USER
```

```bash
tail -f logs/ensembles/coverage/41-510.log
```

SLURM's own logs land under `.snakemake/slurm_logs/`. Rerunning the same
command after an interruption, a failure or a killed controller resubmits only
the draws with a missing `background.h5`; it is written last and atomically,
and Snakemake deletes a failed job's outputs, so no half-written draw
survives.

## Bring the draws back

From the Mac, merge the cluster's ensembles into the local outputs:

```bash
rsync -av --ignore-existing <cluster>:astrogwb/outputs/ensembles/ outputs/ensembles/
rsync -av --ignore-existing <cluster>:astrogwb/outputs/catalogs/ outputs/catalogs/
```

The trailing slashes merge directory by directory; `--ignore-existing` leaves
the Macs' own draws untouched.

## Linux versus macOS

Draws on Linux x86 and macOS arm64 take the same random keys and the same
sources, but XLA's CPU kernels differ between the two, so spectra can differ
in their last floating-point digits. That is acceptable: a realization's key
names its settings, not its floating-point output, and a coverage test does not
resolve differences at that level.
