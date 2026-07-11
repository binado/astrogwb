import re

from astrogwb.config.batches import parse_mcmc_batch


BATCH = parse_mcmc_batch(config)
RUN_CONFIGS = {
    (run.campaign, run.run): str(run.config_path)
    for run in BATCH.runs
}
CHAIN_PATTERN = str(
    BATCH.chains_dir / BATCH.catalog_id / "{campaign}" / "{run}.nc"
)
SIDECAR_PATTERN = str(
    BATCH.chains_dir / BATCH.catalog_id / "{campaign}" / "{run}.json"
)
CHAINS = [
    str(BATCH.chains_dir / BATCH.catalog_id / run.campaign / f"{run.run}.nc")
    for run in BATCH.runs
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
    campaign="|".join(re.escape(run.campaign) for run in BATCH.runs),
    run="|".join(re.escape(run.run) for run in BATCH.runs),


localrules:
    mcmc,


rule mcmc:
    input:
        CHAINS,


rule run_mcmc:
    input:
        config=run_config_path,
        catalog=str(BATCH.catalog_path),
    output:
        chain=protected(CHAIN_PATTERN),
        sidecar=protected(SIDECAR_PATTERN),
    params:
        jax_platforms=BATCH.jax_platforms,
        outdir=lambda wc: str(
            BATCH.chains_dir / BATCH.catalog_id / wc.campaign
        ),
    resources:
        cpus_per_task=4,
        mem_mb=8000,
        runtime=240,
    shell:
        """
        export OMP_NUM_THREADS={resources.cpus_per_task}
        export JAX_PLATFORMS={params.jax_platforms:q}
        # job-nanny I/O conventions; harmless when the wrapper is absent.
        export INPUT="*"
        export OUTPUT="*"
        if command -v job-nanny >/dev/null 2>&1; then
            job-nanny uv run --extra mcmc python scripts/run_mcmc.py \
                --config {input.config:q} --outdir {params.outdir:q} \
                --label {wildcards.run:q} --catalog {input.catalog:q} --force
        else
            uv run --extra mcmc python scripts/run_mcmc.py \
                --config {input.config:q} --outdir {params.outdir:q} \
                --label {wildcards.run:q} --catalog {input.catalog:q} --force
        fi
        """
