import re
from pathlib import Path


configfile: "packages/astrogwb-paper/configs/workflow.yaml"


try:
    _CATALOG_CONFIG = config["catalog"]
    CATALOG_ID = _CATALOG_CONFIG["id"]
except KeyError as exc:
    raise KeyError(
        "missing 'catalog' config: define catalog.id (and optionally "
        "catalog.path) in packages/astrogwb-paper/configs/workflow.yaml, "
        "or pass a --configfile "
        "that sets it"
    ) from exc
CATALOG_PATH = _CATALOG_CONFIG.get("path") or f"out/catalogs/{CATALOG_ID}.h5"
CHAINS_DIR = Path(config["chains_dir"])
JAX_PLATFORM = config.get("jax_platforms", "cuda")

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
        if [ "{params.platform}" = "cpu" ]; then
            # CPU jobs: chain_method resolves to "parallel", which needs one
            # logical host device per concurrent chain. Spend the allocated
            # threads on --host-device-count (must be >= the run config's
            # num_chains) instead of also fanning each chain out over
            # BLAS/XLA intra-op threads, which would oversubscribe the cores
            # actually granted by --cores/cpus-per-task.
            RUNTIME_FLAGS="--host-device-count {threads} --cpu-threads 1"
            UV_EXTRAS="--package astrogwb-paper"
        else
            # GPU jobs: chain_method resolves to "vectorized" on the single
            # GPU device, so host-device-count is irrelevant; spend the
            # allocated CPUs on host-side BLAS/data-loading threads instead.
            RUNTIME_FLAGS="--cpu-threads {threads}"
            UV_EXTRAS="--package astrogwb-paper --extra cuda"
        fi
        $NANNY uv run $UV_EXTRAS astrogwb-run-mcmc \
            --config {input.config:q} --outdir {params.outdir:q} \
            --label {wildcards.run:q} --catalog {input.catalog:q} \
            --platform {params.platform:q} $RUNTIME_FLAGS --force
        """
