# Coverage draws on SLURM

[`notebooks/shot_noise_coverage.py`](../notebooks/shot_noise_coverage.py) reads
1000 Poisson BNS catalogs, each recorded by the six detector networks.
[`scripts/draw_coverage_injections.py`](../scripts/draw_coverage_injections.py)
draws a range of them into `<cache>/ensembles/<key>/41-<i>.h5`, one file per
network. Draw `i` is fixed by the seed (41) and `i` alone, so any split of the
indices across machines gives the same files, and they merge by copying.

[`workflow/coverage_draws.smk`](../workflow/coverage_draws.smk) runs that
script as SLURM jobs, one per batch of draws, through
[`workflow/profiles/slurm`](../workflow/profiles/slurm/config.yaml). Its
outputs are the per-draw files themselves, so a rerun skips every batch whose
files exist and only resubmits what is missing.

## Job sizing

| Setting | Value | Why |
| --- | --- | --- |
| draws per job (`batch`) | 10 | long enough that queue wait and the ~1 min of import and JIT per job do not dominate, short enough that a failed job loses little |
| CPUs per job (`threads`) | 4 | XLA's CPU threads within one waveform pass |
| `mem_mb` | 6000 | two draws on an M4 peaked at 2.1 GB RSS (`/usr/bin/time -l`), with `chunk_size=8192` and `source_chunk_size=65_536`; ~3x headroom |
| `runtime` | 90 min | ~7 CPU-min per draw on an M4, so ~20-30 min for 10 draws on 4 cluster cores; a job killed at its limit loses its whole batch |
| `jobs` | 100 | all of 1000 draws in flight at once |

Calibrate on the first batch (below) and change `batch`, `threads` or the
profile's resources before submitting the rest.

## One-time setup on the cluster

Same commit as the Macs, CPU-only JAX (the default wheel):

```bash
git clone git@github.com:binado/astrogwb.git && cd astrogwb
git checkout <commit>
uv sync --extra notebook --group dev
uv tool install 'snakemake>=9.24' --with 'snakemake-executor-plugin-slurm>=1.5'
```

Snakemake lives in its own tool environment: the workflow imports nothing from
`astrogwb`, and the jobs run the project venv with `uv run --frozen --no-sync`,
so the hundred concurrent jobs never race to sync it.

Check the keys. They name the output directories, so they must match the
Macs' exactly:

```bash
uv run --no-sync python scripts/draw_coverage_injections.py --keys
```

```text
c6644e48729b2472  ET-triangular
5d8ba6ffa0854f0e  ET-triangular-CE-Hanford
ed5877c73ffe9c84  ET-2L-aligned
d5913ea9df8a959c  ET-2L-aligned-CE-Hanford
e5a046788dc068d1  ET-2L-misaligned
348897d7e32aff06  ET-2L-misaligned-CE-Hanford
```

The same six are written into the workflow; if the script ever drew under
other keys, each job would fail on a `MissingOutputException` rather than
pass.

If the cluster needs an account or partition, uncomment `slurm_account` /
`slurm_partition` in the profile, or pass
`--default-resources slurm_account=<account> slurm_partition=<partition>`.

## Submit

The Snakemake controller stays on the login node for the whole run, so start
it in `tmux` (or `screen`). Calibrate on one batch first:

```bash
snakemake -s workflow/coverage_draws.smk --profile workflow/profiles/slurm --config start=500 stop=510
```

```bash
sacct -u $USER -S today --format=JobID,JobName%40,Elapsed,MaxRSS,State
```

Then dry-run and submit the rest:

```bash
snakemake -s workflow/coverage_draws.smk --profile workflow/profiles/slurm --config start=500 stop=1000 --dry-run
```

```bash
snakemake -s workflow/coverage_draws.smk --profile workflow/profiles/slurm --config start=500 stop=1000
```

For the whole ensemble, use `start=0 stop=1000`. `batch=<n>`, `threads=<n>`
and `cache_dir=<path>` go in the same `--config` group (the default cache is
`$XDG_CACHE_HOME/astrogwb`, else `~/.cache/astrogwb`, where
`default_cache_dir()` puts it on Linux).

Each job exports `INPUT="*"` and `OUTPUT="*"` and runs the script under
`job-nanny` whenever it is on `PATH`, as the cluster requires.

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
