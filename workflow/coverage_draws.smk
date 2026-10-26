# Coverage-notebook injection draws, one job per batch of draw indices.
#
#   uv run --no-sync snakemake -s workflow/coverage_draws.smk \
#       --profile workflow/profiles/slurm --config start=500 stop=1000 --dry-run
#
# Each job runs scripts/draw_coverage_injections.py --start i --stop j, which
# writes draw k, one background over every network's detectors, to
# <cache_dir>/ensembles/<key>/41-<k>/.
# Draw k is fixed by (seed, k) alone, so the batches are independent, finished
# files merge by copying, and a batch whose files all exist is skipped: a
# rerun resumes, and files copied in from another machine are not redrawn.
#
# Snakemake comes from the project venv (the `cluster` dependency group), so
# the controller and its jobs share the interpreter the script runs in. The
# key below is what the script prints with --keys; it names the outputs. A
# script that drew under another key would leave these paths missing, and
# Snakemake fails that job on a MissingOutputException rather than reporting
# it done.
#
# Config (override with --config key=value):
#   start, stop  draw indices [start, stop)               default 500, 1000
#   batch        draws per job                            default 5
#   cache_dir    cache root, the script's --cache-dir     default $XDG_CACHE_HOME/astrogwb
#   threads      CPUs per job                             default 8

import os
from pathlib import Path

SEED = 41
KEY = "0295b914397ec3c8"  # the six DETECTOR_NETWORKS, one background per draw

# platformdirs' user cache on Linux; on macOS pass cache_dir explicitly.
_XDG_CACHE = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
CACHE_DIR = Path(config.get("cache_dir", Path(_XDG_CACHE) / "astrogwb"))
START = int(config.get("start", 500))
STOP = int(config.get("stop", 1000))
BATCH = int(config.get("batch", 5))
THREADS = int(config.get("threads", 8))

if not 0 <= START < STOP or BATCH < 1:
    raise ValueError(f"need 0 <= start < stop and batch >= 1, got {START}, {STOP}, {BATCH}")


def draw_files(start, stop):
    """The manifest of every draw in [start, stop)."""
    return [
        str(CACHE_DIR / "ensembles" / KEY / f"{SEED}-{index}" / "metadata.toml")
        for index in range(start, stop)
    ]


BATCHES = [(start, min(start + BATCH, STOP)) for start in range(START, STOP, BATCH)]


localrules:
    all,


rule all:
    input:
        draw_files(START, STOP),


# One rule per batch: its outputs are a range of indices, which a wildcard
# rule cannot name file by file.
for _start, _stop in BATCHES:

    rule:
        name:
            f"draws_{_start:04d}_{_stop:04d}"
        output:
            draw_files(_start, _stop),
        log:
            f"logs/coverage_draws/{_start:04d}-{_stop:04d}.log",
        params:
            start=_start,
            stop=_stop,
            cache_dir=str(CACHE_DIR),
        threads: THREADS
        # INPUT, OUTPUT and job-nanny are the cluster's job requirements, as in
        # the retired run_mcmc rule; job-nanny is used wherever it is on PATH.
        # job-nanny runs the command in a node-local copy of the working
        # directory without .venv, so `uv run` alone would create an empty
        # venv there. --active uses VIRTUAL_ENV instead, the shared venv the
        # controller was started from with `uv run`. --no-sync: that venv is
        # synced once on the login node; a hundred jobs each syncing it at
        # once would race on the filesystem.
        shell:
            """
            export INPUT="*"
            export OUTPUT="*"
            NANNY=
            if command -v job-nanny >/dev/null 2>&1; then
                NANNY=job-nanny
            fi
            $NANNY uv run --active --frozen --no-sync \
                python scripts/draw_coverage_injections.py \
                --cache-dir {params.cache_dir:q} \
                --start {params.start} --stop {params.stop} > {log:q} 2>&1
            """
