# The simulation inputs the paper notebooks read, generated explicitly.
#
#   uv run --no-sync snakemake -s workflow/coverage_draws.smk \
#       --profile workflow/profiles/slurm --dry-run
#
# Notebooks never simulate: they read what this writes and fail with a
# FileNotFoundError when it is missing. Per ensemble <name> (the committed
# config/ensembles/<name>/{population,waveform}.toml and config/networks.toml):
#
#   outputs/ensembles/<name>/<seed>-<index>/   one realization per
#       `astrogwb simulate` job, its manifest and background.h5: every paper
#       network's pair spectra and combined spectrum;
#   outputs/ensembles/<name>/detector_noise.h5 `astrogwb noise` of draw 0,
#       the networks' noise on the same grid and observation time;
#
# and the reference catalogs the notebooks name, by
# scripts/draw_reference_catalog.py, under outputs/catalogs/.
#
# A realization is fixed by (seed, index) alone, so jobs are independent,
# finished directories merge by copying, and a background that exists is not
# redrawn. The background is written last and atomically, so its file is the
# job's output. Editing a committed config does not migrate anything: a rerun
# against a realization recorded with other settings fails on the conflict, so
# delete the ensemble and regenerate it.
#
# Draw jobs share the `draws` group; the profile's group-components packs
# several into one SLURM job. Snakemake comes from the project venv, synced
# with `uv sync --extra notebook --extra cli --group cluster`, so the
# controller and its jobs share one interpreter.
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
#             injections and the draws of ensemble_spectrum and spectrum_snrs
#             (41), and spectrum_snrs' independent reference (42-0).
DRAWS = {
    "fiducial": {41: 1000, 42: 1},
}

# The reference catalogs each ensemble's notebooks read, and their options.
CATALOGS = {
    "fiducial": {
        # the grid and coverage notebooks, shot_noise_* scripts
        "reference-fiducial": "--num-samples 16384 --chunk-size 4096",
        # importance_convergence's redshift ladder
        "reference-ladder": "--num-samples 256 --sampling random --seed 100",
        # ensemble_spectrum's minimum-redshift scan
        "reference-scan": (
            "--num-samples 16384 --reference-distance-mpc 1 --chunk-size 4096"
        ),
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


rule all:
    input:
        [
            f"{ENSEMBLES_ROOT}/{name}/{seed}-{index}/background.h5"
            for name in SELECTED
            for seed, index in draws(name)
        ],
        [f"{ENSEMBLES_ROOT}/{name}/detector_noise.h5" for name in SELECTED],
        [f"{CATALOGS_ROOT}/{catalog}.h5" for name in SELECTED for catalog in CATALOGS[name]],


# INPUT, OUTPUT and job-nanny are the cluster's job requirements; job-nanny is
# used wherever it is on PATH. It runs the command in a node-local copy of the
# working directory without .venv, so `uv run` alone would create an empty venv
# there. --active uses VIRTUAL_ENV instead, the shared venv the controller was
# started from with `uv run`. --no-sync: that venv is synced once on the login
# node; a hundred jobs each syncing it at once would race on the filesystem.
_RUN = """
    export INPUT="*"
    export OUTPUT="*"
    NANNY=
    if command -v job-nanny >/dev/null 2>&1; then
        NANNY=job-nanny
    fi
    $NANNY uv run --active --frozen --no-sync"""


rule simulate:
    input:
        population="config/ensembles/{ensemble}/population.toml",
        waveform="config/ensembles/{ensemble}/waveform.toml",
        networks="config/networks.toml",
    output:
        f"{ENSEMBLES_ROOT}/{{ensemble}}/{{seed}}-{{index}}/background.h5",
    log:
        "logs/ensembles/{ensemble}/{seed}-{index}.log",
    params:
        directory=f"{ENSEMBLES_ROOT}/{{ensemble}}/{{seed}}-{{index}}",
    group:
        "draws"
    threads: THREADS
    shell:
        _RUN
        + """ astrogwb simulate {params.directory:q} \
            --population-metadata {input.population:q} \
            --seed {wildcards.seed} --index {wildcards.index} \
            --waveform-metadata {input.waveform:q} \
            --background-metadata {input.networks:q} \
            --no-keep-sources --chunk-size 8192 --source-chunk-size 65536 \
            --cpu-threads {threads} > {log:q} 2>&1
        """


def first_draw(wildcards):
    seed, index = draws(wildcards.ensemble)[0]
    return f"{ENSEMBLES_ROOT}/{wildcards.ensemble}/{seed}-{index}/background.h5"


rule noise:
    input:
        first_draw,
    output:
        f"{ENSEMBLES_ROOT}/{{ensemble}}/detector_noise.h5",
    log:
        "logs/ensembles/{ensemble}/detector_noise.log",
    shell:
        _RUN
        + """ astrogwb noise $(dirname {input:q}) {output:q} > {log:q} 2>&1
        """


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
        + """ python scripts/draw_reference_catalog.py {params.options} \
            --output {output:q} > {log:q} 2>&1
        """
