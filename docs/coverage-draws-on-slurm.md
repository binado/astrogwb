# Simulation inputs on SLURM

The paper notebooks read their simulation inputs and never generate them. A
missing input is a `FileNotFoundError`, not a simulation.
[`Snakefile`](../Snakefile) writes every
input, per ensemble:

| Output | Command | Read by |
| --- | --- | --- |
| `outputs/ensembles/<name>/<seed>-<i>.h5` | `astrogwb simulate`, one file per job | every notebook with Poisson draws |
| `outputs/catalogs/reference-*.h5` | `astrogwb model-catalog` | the posterior notebooks, `importance_convergence`, `ensemble_spectrum` |

The one ensemble is `fiducial`, the default waveform and grid: the grid
notebooks' injection, `shot_noise_coverage`'s injections and the draws of
`ensemble_spectrum`. Its settings are `[ensembles.fiducial]` in the committed
`config/ensembles.toml` and `config/detectors.toml`, which supplies the shared frozen detector registry.

Every file holds each network's pair spectra and carries its configuration. The pairs are projected once per draw, so
adding a network that shares their detectors costs no waveforms. Draw `i` of a
seed is fixed by `(seed, i)` alone, so any split of the indices across
machines gives the same files, and they merge by copying.

## Regenerating after a change

Nothing is migrated. If you change a detector, a noise curve, the fiducials,
the population or the waveform, edit the committed simulation inputs to match
and regenerate the ensemble:

```bash
rm -r outputs/ensembles/fiducial
uv run --no-sync snakemake --profile workflow/profiles/slurm
```

A file records the configuration it was simulated with, but nothing compares
it with the committed inputs: an existing file is never redrawn. After editing
them, delete the outputs and rerun, as above. Files of another astrogwb
version are refused when read; regenerate them too.

## Job sizing

Each draw is one SLURM job: one `astrogwb simulate` process, one output file.
Grouping several draws into a shared job was tried first, to amortize
job-nanny's node-local staging, but the bundled request (the sum of the
draws' CPUs and memory) scheduled poorly on a busy partition, where the
queue, not the compute, is the bottleneck.
The executor submits each ready simulation as its own SLURM job. It does not
use job arrays: `job-nanny` keys its staging directory on `$SLURM_JOBID`,
which every task of an array shares, so array tasks would clobber each other's
staging.

Keep the number of ready jobs at or below `jobs`. When Snakemake hands the
executor fewer simulations than are ready, the executor waits for the rest
without submitting them, and the controller can stall. The cluster limit of
1000 jobs applies to these jobs too, so `jobs` cannot be raised; instead build
everything that is not one of the 1000 draws before the cluster run (see
Submit).

| Setting | Value | Why |
| --- | --- | --- |
| CPUs per draw (`threads`) | 8 | XLA spreads each waveform chunk over the job's CPUs |
| `mem_mb` | 4000 | two draws on an M4 peaked at 2.1 GB RSS (`/usr/bin/time -l`), with `chunk_size=8192`; an out-of-memory kill loses a single draw, redrawn by `retries` |
| `runtime` | 60 min | the slurm-test draws took 26-27 min on the cluster (8 min 10 s on an M-series Mac); 20 min timed out. 60 is about twice the cluster time. Backfill places jobs against their time limit, so the shorter the better |
| `jobs` | 1000 | up to the cluster's limit of 1000 draws in flight |

The queue is the bottleneck, not the compute: if jobs sit in `squeue` with
reason `Priority`, the partition is busy with higher-priority work, and the
way through is requests the scheduler can slot into leftover capacity -- few
CPUs, little memory, a short time limit -- rather than more queued jobs.

After a job finishes, `sacct` gives its time and CPU use:

```bash
sacct -u $USER -S today --format=JobID,JobName%40,AllocCPUS,Elapsed,TotalCPU,MaxRSS,State
```

`TotalCPU / (Elapsed x AllocCPUS)` is how busy the cores were: near 1, more
`threads` would still help; well below, prefer more jobs with fewer threads.
Change `threads` with `--config`, and `runtime` and `mem_mb` in the profile.

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
and concurrent tasks must never race to sync it.

Jobs go to the `short` partition, whose time limit must cover the profile's
`runtime`. If the cluster needs an account, uncomment `slurm_account` in the
profile. Change either one in the profile rather than with
`--default-resources` on the command line, which would replace the profile's
whole list, `mem_mb` and `runtime` included.

## Submit

The Snakemake controller stays on the login node for the whole run, so start
it in `tmux` (or `screen`). First probe staging with a two-task run in a separate
test directory:

```bash
just slurm-test
```

Check `logs/slurm-test/fiducial/41-{0,1}.log` and the resulting
`outputs/slurm-test/fiducial/41-{0,1}.h5`. The forced run repeats both tasks
even if their outputs already exist; `--retries 0` disables Snakemake retries.
Calibrate on a few draws next (`stop` caps the
draws of the ensemble's first seed), and check them with `sacct`:

```bash
uv run --no-sync snakemake --profile workflow/profiles/slurm --config stop=10
```

The ensemble has 1001 draws (seed 41 index 0-999 and the seed-42 reference
draw) and the run has two reference catalogs, more than `jobs`. The draws
read only `config/`, not the catalogs, so build the three extras locally
first and the cluster run is exactly 1000 draws:

```bash
just catalogs
just draw fiducial 42 0
```

Then dry-run and submit the rest:

```bash
uv run --no-sync snakemake --profile workflow/profiles/slurm --dry-run
```

```bash
uv run --no-sync snakemake --profile workflow/profiles/slurm
```

The default is `ensembles=fiducial`. `threads=<n>` and
`outputs=<path>` go in the same `--config` group.

Each job exports `INPUT="config"` and `OUTPUT="*"` and runs its command under
`job-nanny` whenever it is on `PATH`, as the cluster requires. `job-nanny`
runs the job in a node-local copy of the working directory, which has no
`.venv`, so jobs run `uv run --active --no-sync`: `--active` takes the shared
venv from `VIRTUAL_ENV`, which `uv run` sets when it starts the controller.
The editable installation also requires the checkout's `src/` directory to be
available on the shared filesystem from compute nodes.
Start the controller with `uv run` as above, never with a bare `snakemake`.
If outputs written in the node-local copy do not come back, pass
`outputs=<absolute path on the shared filesystem>` and link it to `outputs/`.

## Monitor and resume

```bash
squeue -u $USER
```

```bash
tail -f logs/ensembles/fiducial/41-510.log
```

SLURM's own logs land under `.snakemake/slurm_logs/`. Rerunning the same
command after an interruption, a failure or a killed controller resubmits only
the draws with a missing file; each is written atomically,
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
in their last floating-point digits. That is acceptable: a draw's key and
config fix its settings, not its floating-point output, and a coverage test does not
resolve differences at that level.
