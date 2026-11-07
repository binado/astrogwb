# The simulation inputs the paper notebooks read, generated explicitly.
#
#   uv run --no-sync snakemake \
#       --profile workflow/profiles/slurm --dry-run
#
# Notebooks never simulate: they read what this writes and fail with a
# FileNotFoundError when it is missing. Per ensemble <name> (the
# [ensembles.<name>] table of the shared config and config/detectors.toml):
#
#   outputs/ensembles/<name>/<seed>-<index>.h5   one file per `astrogwb
#       simulate` job: every paper network's pair spectra, with the
#       configuration it was simulated from;
#
# and the reference catalogs the notebooks name, the [catalogs.<name>] tables,
# by `astrogwb model-catalog`, under outputs/catalogs/.
#
# A draw is fixed by (seed, index) alone, so jobs are independent, finished
# files merge by copying, and a file that exists is not redrawn. Each file is
# written atomically, so a file at its path is the job's output. Editing a
# committed config does not migrate anything: delete the outputs and rerun.
#
# Snakemake comes from the project venv, synced with `uv sync --extra
# notebook --extra cli --group cluster`, so the controller and its jobs share
# one interpreter.
#
# Config (override with --config key=value):
#   ensembles  space-separated names           default "fiducial"
#   stop       draws of the ensembles' first   default per ensemble (DRAWS)
#              seed, overriding DRAWS
#   threads    CPUs per draw job               default 8
#   outputs    output root; an absolute path   default "outputs", where the
#              on a shared filesystem if jobs   notebooks read
#              run in node-local copies

OUTPUTS = str(config.get("outputs", "outputs")).rstrip("/")
ENSEMBLES_ROOT = f"{OUTPUTS}/ensembles"
CATALOGS_ROOT = f"{OUTPUTS}/catalogs"

# Draws per seed of each ensemble: what the notebooks read.
#   fiducial: the grid notebooks' injection (41-0), shot_noise_coverage's
#             injections and the draws of ensemble_spectrum (41), and an
#             independent reference draw (42-0).
DRAWS = {
    "fiducial": {41: 1000, 42: 1},
}

# The [catalogs.<name>] tables of the shared config, with draw seeds and
# execution-only chunk sizes.
CATALOGS = {
    "fiducial": {
        "reference-fiducial": "--seed 42 --chunk-size 4096",
        "reference-ladder": "--seed 100",
    },
}
CATALOG_OPTIONS = {
    name: options for catalogs in CATALOGS.values() for name, options in catalogs.items()
}

SELECTED = str(config.get("ensembles", "fiducial")).split()
THREADS = int(config.get("threads", 8))
unknown = set(SELECTED) - set(DRAWS)
if unknown:
    raise ValueError(f"unknown ensembles {sorted(unknown)}; known: {sorted(DRAWS)}")

CATALOG_FILES = [
    f"{CATALOGS_ROOT}/{catalog}.h5" for name in SELECTED for catalog in CATALOGS[name]
]


def draws(name):
    """Every draw of the ensemble `name`, overridden by --config stop."""
    seeds = dict(DRAWS[name])
    if "stop" in config:
        seeds[next(iter(seeds))] = int(config["stop"])
    return [(seed, index) for seed, count in seeds.items() for index in range(count)]


wildcard_constraints:
    ensemble=r"[A-Za-z0-9_-]+",
    seed=r"\d+",
    index=r"\d+",
    catalog=r"[A-Za-z0-9_-]+",


localrules:
    all,
    catalogs,
    slurm_test,


rule all:
    input:
        [
            f"{ENSEMBLES_ROOT}/{name}/{seed}-{index}.h5"
            for name in SELECTED
            for seed, index in draws(name)
        ],
        CATALOG_FILES,


# The reference catalogs alone: seconds each, so `just catalogs` runs them
# locally rather than through the cluster profile.
rule catalogs:
    input:
        CATALOG_FILES,


# INPUT, OUTPUT and job-nanny are the cluster's job requirements; job-nanny is
# used wherever it is on PATH. It runs the command in a node-local copy of the
# working directory without .venv, so `uv run` alone would create an empty venv
# there. --active uses VIRTUAL_ENV instead, the shared venv the controller was
# started from with `uv run`. --no-sync: that venv is synced once on the login
# node; a hundred jobs each syncing it at once would race on the filesystem.
_RUN = """
    export INPUT="config"
    export OUTPUT="*"
    NANNY=
    if command -v job-nanny >/dev/null 2>&1; then
        NANNY=job-nanny
    fi
    $NANNY uv run --active --frozen --no-sync"""


rule simulate:
    input:
        detectors="config/detectors.toml",
    output:
        f"{ENSEMBLES_ROOT}/{{ensemble}}/{{seed}}-{{index}}.h5",
    log:
        "logs/ensembles/{ensemble}/{seed}-{index}.log",
    threads: THREADS
    shell:
        _RUN
        + """ astrogwb simulate {output:q} \
            --ensemble {wildcards.ensemble:q} \
            --seed {wildcards.seed} --index {wildcards.index} \
            --detector-config {input.detectors:q} \
            --chunk-size 8192 \
            --cpu-threads {threads} > {log:q} 2>&1
        """


# A separate output lets the staging probe run without touching paper draws.
rule slurm_test:
    input:
        f"{OUTPUTS}/slurm-test/fiducial/41-0.h5",


use rule simulate as simulate_slurm_test with:
    output:
        f"{OUTPUTS}/slurm-test/{{ensemble}}/{{seed}}-{{index}}.h5",
    log:
        "logs/slurm-test/{ensemble}/{seed}-{index}.log",


rule reference_catalog:
    output:
        f"{CATALOGS_ROOT}/{{catalog}}.h5",
    log:
        "logs/catalogs/{catalog}.log",
    params:
        options=lambda wildcards: CATALOG_OPTIONS[wildcards.catalog],
    threads: THREADS
    shell:
        _RUN
        + """ astrogwb model-catalog {output:q} {params.options} \
            --catalog {wildcards.catalog:q} \
            --cpu-threads {threads} > {log:q} 2>&1
        """
