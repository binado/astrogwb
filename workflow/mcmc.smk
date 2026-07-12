import re
from pathlib import Path


CATALOG_ID = config["catalog"]["id"]
CATALOG_PATH = config["catalog"]["path"]
CHAINS_DIR = Path(config["chains_dir"])
JAX_PLATFORM = {"cuda": "gpu"}.get(
    config.get("jax_platforms", "cuda"), config.get("jax_platforms", "cuda")
)

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
        platform=JAX_PLATFORM,
        outdir=lambda wc: str(CHAINS_DIR / CATALOG_ID / wc.campaign),
    threads: 4
    resources:
        mem_mb=8000,
        runtime=720,
    shell:
        """
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
            --label {wildcards.run:q} --catalog {input.catalog:q} \
            --platform {params.platform:q} --cpu-threads {threads} --force
        """
