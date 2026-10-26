# Coverage draws on SLURM

[`notebooks/shot_noise_coverage.py`](../notebooks/shot_noise_coverage.py) reads
1000 Poisson BNS catalogs, each recorded by the six detector networks.
[`scripts/draw_coverage_injections.py`](../scripts/draw_coverage_injections.py)
draws a range of them into `<cache>/ensembles/<key>/41-<i>/`, one background
artifact per draw over every network's detectors. Draw `i` is fixed by the seed
(41) and `i` alone, so any split of the indices across machines gives the same
directories, and they merge by copying.

[`workflow/coverage_draws.smk`](../workflow/coverage_draws.smk) runs that
script as SLURM jobs, one per batch of draws, through
[`workflow/profiles/slurm`](../workflow/profiles/slurm/config.yaml). Its
outputs are the per-draw manifests themselves, so a rerun skips every batch whose
files exist and only resubmits what is missing.

## Job sizing

| Setting | Value | Why |
| --- | --- | --- |
| draws per job (`batch`) | 5 | short jobs that schedule easily and lose little on failure, while the ~1 min of import and JIT per job stays a small share |
| CPUs per job (`threads`) | 8 | XLA spreads each waveform chunk over the job's CPUs |
| `mem_mb` | 8000 | two draws on an M4 peaked at 2.1 GB RSS (`/usr/bin/time -l`), with `chunk_size=8192` and `source_chunk_size=65_536`; headroom for per-thread buffers |
| `runtime` | 60 min | the first cluster batch, 10 draws on 4 cores, took 50-90 min; 8 cores should bring 5 draws to ~15-30 min |
| `jobs` | 100 | all of 500 draws in flight at once |

After a batch finishes, `sacct` gives its time and CPU use:

```bash
sacct -u $USER -S today --format=JobID,JobName%40,AllocCPUS,Elapsed,TotalCPU,MaxRSS,State
```

`TotalCPU / (Elapsed x AllocCPUS)` is how busy the cores were: near 1, more
`threads` would still help; well below, prefer more jobs with fewer threads.
The draw directories' timestamps give the time per draw:

```bash
ls -l --time-style=+%H:%M ~/.cache/astrogwb/ensembles/0295b914397ec3c8/
```

Change `batch` and `threads` with `--config`, and `runtime` and `mem_mb` in
the profile.

## One-time setup on the cluster

Same commit as the Macs, CPU-only JAX (the default wheel):

```bash
git clone git@github.com:binado/astrogwb.git && cd astrogwb
git checkout <commit>
uv sync --extra notebook --group dev --group cluster
```

The `cluster` group puts Snakemake and its SLURM executor plugin in the
project venv, so the controller and every job run the interpreter the script
does. Every command below uses `uv run --no-sync`, and so do the jobs: a plain
`uv run` would sync the venv back to the default groups and remove Snakemake,
and the hundred concurrent jobs must never race to sync it.

Check the key. It names the output directory, so it must match the Macs'
exactly:

```bash
uv run --no-sync python scripts/draw_coverage_injections.py --keys
```

```text
0295b914397ec3c8
```

The same key is written into the workflow; if the script ever drew under
another key, each job would fail on a `MissingOutputException` rather than
pass.

Jobs go to the `short` partition, whose time limit must cover the profile's
90-minute `runtime`. If the cluster needs an account, uncomment
`slurm_account` in the profile. Change either one in the profile rather than
with `--default-resources` on the command line, which would replace the
profile's whole list, `mem_mb` and `runtime` included.

## Submit

The Snakemake controller stays on the login node for the whole run, so start
it in `tmux` (or `screen`). Calibrate on one batch first, and check it with
`sacct` as in [Job sizing](#job-sizing):

```bash
uv run --no-sync snakemake -s workflow/coverage_draws.smk --profile workflow/profiles/slurm --config start=510 stop=515
```

Then dry-run and submit the rest:

```bash
uv run --no-sync snakemake -s workflow/coverage_draws.smk --profile workflow/profiles/slurm --config start=500 stop=1000 --dry-run
```

```bash
uv run --no-sync snakemake -s workflow/coverage_draws.smk --profile workflow/profiles/slurm --config start=500 stop=1000
```

For the whole ensemble, use `start=0 stop=1000`. `batch=<n>`, `threads=<n>`
and `cache_dir=<path>` go in the same `--config` group (the default cache is
`$XDG_CACHE_HOME/astrogwb`, else `~/.cache/astrogwb`, where
`default_cache_dir()` puts it on Linux).

Each job exports `INPUT="*"` and `OUTPUT="*"` and runs the script under
`job-nanny` whenever it is on `PATH`, as the cluster requires. `job-nanny`
runs the job in a node-local copy of the working directory, which has no
`.venv`, so jobs run `uv run --active --no-sync`: `--active` takes the shared
venv from `VIRTUAL_ENV`, which `uv run` sets when it starts the controller.
Start the controller with `uv run` as above, never with a bare `snakemake`.
The draws themselves go straight to `cache_dir`, an absolute path on the
shared filesystem, so only the log comes back through `OUTPUT`.

## Monitor and resume

```bash
squeue -u $USER
```

```bash
tail -f logs/coverage_draws/0500-0510.log
```

SLURM's own logs land under `.snakemake/slurm_logs/`. Rerunning the same
command after an interruption, a failure or a killed controller resubmits only
the batches with a missing file; Snakemake deletes a failed job's outputs, so
no half-written draw survives. The profile sets `rerun-triggers: mtime`, so
changing `batch` between runs does not redraw files that already exist.

## Bring the draws back

From the Mac, merge the cluster's per-key directories into the local cache:

```bash
rsync -av --ignore-existing <cluster>:.cache/astrogwb/ensembles/ ~/Library/Caches/astrogwb/ensembles/
```

The trailing slashes merge directory by directory; `--ignore-existing` leaves
the Macs' own draws untouched. The notebook then finds all 1000 draws cached.

## Linux versus macOS

Draws on Linux x86 and macOS arm64 take the same random keys and the same
sources, but XLA's CPU kernels differ between the two, so spectra can differ
in their last floating-point digits. That is acceptable: the loader checks only
the key and seed a file records, and a coverage test does not resolve
differences at that level.
