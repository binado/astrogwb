import re
from pathlib import Path

from astrogwb_paper.config.sweeps import load_sweep_config, run_fragments


configfile: "configs/workflow.yaml"


try:
    _CATALOG_CONFIG = config["catalog"]
    CATALOG_ID = _CATALOG_CONFIG["id"]
except KeyError as exc:
    raise KeyError(
        "missing 'catalog' config: define catalog.id (and optionally "
        "catalog.path) in configs/workflow.yaml, "
        "or pass a --configfile "
        "that sets it"
    ) from exc
CATALOG_PATH = _CATALOG_CONFIG.get("path") or f"out/catalogs/{CATALOG_ID}.h5"
CHAINS_DIR = Path(config.get("chains_dir", "chains"))
CONFIGS_DIR = Path(config.get("mcmc_configs_dir", "configs/mcmc"))
JAX_PLATFORM = config.get("jax_platforms", "cuda")

# Campaign expansion happens here rather than in a generated manifest: the
# sweep spec is the only input, so editing it (or any fragment) rebuilds
# exactly the configs that changed. `campaigns` selects a subset; omit it to
# build every campaign in the spec.
SWEEP = load_sweep_config(Path(config.get("sweep_spec", "configs/mcmc.sweeps.toml")))
RUN_FRAGMENTS = run_fragments(SWEEP)

SELECTED_CAMPAIGNS = config.get("campaigns") or list(SWEEP.runs)
_unknown = sorted(set(SELECTED_CAMPAIGNS) - set(SWEEP.runs))
if _unknown:
    raise ValueError(
        f"unknown campaigns {_unknown}; the sweep spec defines {sorted(SWEEP.runs)}"
    )
RUN_FRAGMENTS = {
    key: fragments
    for key, fragments in RUN_FRAGMENTS.items()
    if key[0] in set(SELECTED_CAMPAIGNS)
}

CONFIG_PATTERN = str(CONFIGS_DIR / "{campaign}" / "{run}.json")
CHAIN_PATTERN = str(CHAINS_DIR / CATALOG_ID / "{campaign}" / "{run}.nc")
SIDECAR_PATTERN = str(CHAINS_DIR / CATALOG_ID / "{campaign}" / "{run}.json")
CHAINS = [
    CHAIN_PATTERN.format(campaign=campaign, run=run)
    for campaign, run in RUN_FRAGMENTS
]


def run_config_fragments(wildcards):
    key = (wildcards.campaign, wildcards.run)
    if key not in RUN_FRAGMENTS:
        raise ValueError(
            f"MCMC run {wildcards.campaign}/{wildcards.run} is not part of the "
            "selected campaigns"
        )
    return [str(path) for path in RUN_FRAGMENTS[key]]


wildcard_constraints:
    campaign="|".join(re.escape(campaign) for campaign, _ in RUN_FRAGMENTS),
    run="|".join(re.escape(run) for _, run in RUN_FRAGMENTS),


localrules:
    mcmc,
    mcmc_config,


rule mcmc_config:
    """Merge a run's fragment layers and validate the result.

    `knf` merges left to right, so the order `run_config_fragments` returns is
    the semantic contract: base, priors, network, observation, analysis.
    `--strict` rejects a layer that changes an existing key's type;
    astrogwb-validate-config enforces the RunConfig schema and writes the
    canonical, defaults-filled JSON that config_sha256 identifies runs by.

    Local because `knf` is a standalone binary that need not exist on a
    compute node, and merging costs milliseconds.
    """
    input:
        fragments=run_config_fragments,
    output:
        config=CONFIG_PATTERN,
    shell:
        "knf {input.fragments} --strict -f json"
        " | uv run --package astrogwb-paper astrogwb-validate-config -"
        " --output {output.config:q}"


rule mcmc:
    input:
        CHAINS,


rule run_mcmc:
    input:
        # ancient(): a rebuilt config must not invalidate a protected chain that
        # cost GPU-hours. Whether a chain is stale is decided by content, not
        # mtime -- every sidecar records config_sha256 -- so reformatting a
        # fragment does not trigger a resample. Delete the chain to force one.
        config=ancient(CONFIG_PATTERN),
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
