import re
from pathlib import Path


CATALOG_ID = config["catalog"]["id"]
CATALOG_PATH = config["catalog"]["path"]
CHAINS_DIR = Path(config["chains_dir"])
JAX_PLATFORMS = config.get("jax_platforms", "cuda")

RUN_CONFIGS = {}
for entry in config["runs"]:
    campaign = entry["campaign"]
    config_path = Path(entry["config"])
    run = config_path.stem
    key = (campaign, run)
    if key in RUN_CONFIGS:
        raise ValueError(f"duplicate MCMC run {campaign}/{run}")
    RUN_CONFIGS[key] = str(config_path)

CHAIN_PATTERN = str(CHAINS_DIR / CATALOG_ID / "{campaign}" / "{run}.nc")
SIDECAR_PATTERN = str(CHAINS_DIR / CATALOG_ID / "{campaign}" / "{run}.json")
CHAINS = [
    CHAIN_PATTERN.format(campaign=campaign, run=run)
    for campaign, run in RUN_CONFIGS
]


def run_config_path(wildcards):
    key = (wildcards.campaign, wildcards.run)
    if key not in RUN_CONFIGS:
        raise ValueError(
            f"MCMC run {wildcards.campaign}/{wildcards.run} is not selected "
            "in the batch manifest"
        )
    return RUN_CONFIGS[key]


wildcard_constraints:
    campaign="|".join(re.escape(campaign) for campaign, _ in RUN_CONFIGS),
    run="|".join(re.escape(run) for _, run in RUN_CONFIGS),


localrules:
    mcmc,


rule mcmc:
    input:
        CHAINS,


rule run_mcmc:
    input:
        config=run_config_path,
        catalog=CATALOG_PATH,
    output:
        chain=protected(CHAIN_PATTERN),
        sidecar=protected(SIDECAR_PATTERN),
    params:
        jax_platforms=JAX_PLATFORMS,
        outdir=lambda wc: str(CHAINS_DIR / CATALOG_ID / wc.campaign),
    resources:
        cpus_per_task=4,
        mem_mb=8000,
        runtime=720,
    shell:
        """
        # Deployment-level core budget: cpus_per_task is the single source of
        # truth, injected by the executor profile. Cap the BLAS thread pools and
        # -- crucially -- XLA's CPU Eigen pool, the knob JAX actually respects
        # (OMP_NUM_THREADS alone does not bound XLA compute). Harmless on GPU.
        # Kept out of the RunConfig so it never enters config_sha256.
        export OMP_NUM_THREADS={resources.cpus_per_task}
        export OPENBLAS_NUM_THREADS={resources.cpus_per_task}
        export MKL_NUM_THREADS={resources.cpus_per_task}
        export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={resources.cpus_per_task}"
        export JAX_PLATFORMS={params.jax_platforms:q}
        # job-nanny I/O conventions; harmless when the wrapper is absent.
        export INPUT="*"
        export OUTPUT="*"
        # Prefix run_mcmc with the job-nanny wrapper only when it is installed.
        NANNY=
        if command -v job-nanny >/dev/null 2>&1; then
            NANNY=job-nanny
        fi
        $NANNY uv run --extra mcmc python scripts/run_mcmc.py \
            --config {input.config:q} --outdir {params.outdir:q} \
            --label {wildcards.run:q} --catalog {input.catalog:q} --force
        """
