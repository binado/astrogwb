import re
from pathlib import Path
import tomllib

from astrogwb.config.catalogs import catalog_path, load_catalog_recipes, population_path
from astrogwb.config.sweeps import load_sweep_spec


configfile: "configs/workflow.yaml"


PAPER_CONFIG_PATH = Path(config["paper_config"])

with PAPER_CONFIG_PATH.open("rb") as handle:
    PAPER_CONFIG = tomllib.load(handle)

CATALOG_REGISTRY_PATH = Path(config["catalog"]["registry"])
CATALOG_RECIPES = load_catalog_recipes(CATALOG_REGISTRY_PATH)
DEFAULT_CATALOG = config["catalog"]["id"]
if DEFAULT_CATALOG not in CATALOG_RECIPES:
    raise ValueError(f"workflow catalog {DEFAULT_CATALOG!r} is not in the registry")

SWEEP_SPEC_PATH = "configs/mcmc.sweeps.toml"
SWEEP_SPEC = load_sweep_spec(Path(SWEEP_SPEC_PATH))
SWEEP_CONFIGS = [f"configs/mcmc/{filename}" for filename in SWEEP_SPEC.filenames()]
SWEEP_CAMPAIGNS = tuple(SWEEP_SPEC.campaigns)
CHAINS_DIR = PAPER_CONFIG["paths"]["chains_dir"]
AMPLITUDE_TOY_PDF = config["amplitude_toy"]["output_pdf"]
SNR_BY_DETECTOR_PDF = config["snr_by_detector"]["output_pdf"]
SNR_BY_DETECTOR_CSV = config["snr_by_detector"]["output_csv"]
SNR_BY_DETECTOR_TEX = config["snr_by_detector"]["output_tex"]
SNR_BY_DETECTOR_SIGMAS_CSV = config["snr_by_detector"]["output_sigmas_csv"]
SNR_BY_DETECTOR_SIGMAS_TEX = config["snr_by_detector"]["output_sigmas_tex"]
POSTERIOR_PDF = config["mcmc_compare_posteriors"]["output_pdf"]
POSTERIOR_CSV = config["mcmc_compare_posteriors"]["output_csv"]
POSTERIOR_TEX = config["mcmc_compare_posteriors"]["output_tex"]
POSTERIOR_FIGURE = PAPER_CONFIG["figures"]["mcmc_compare_posteriors"]
POSTERIOR_CHAINS = [
    f"{CHAINS_DIR}/{DEFAULT_CATALOG}/{POSTERIOR_FIGURE['campaign']}/{entry['run']}.nc"
    for entry in POSTERIOR_FIGURE["posteriors"]
]

def campaign_config_dir(campaign):
    base = "configs/mcmc" if campaign in SWEEP_CAMPAIGNS else "configs/mcmc/curated"
    return f"{base}/{campaign}"


def campaign_chains(catalog, campaign):
    if campaign in SWEEP_CAMPAIGNS:
        runs = SWEEP_SPEC.campaign_runs(campaign)
    else:
        runs = sorted(glob_wildcards(campaign_config_dir(campaign) + "/{run}.json").run)
    return [f"{CHAINS_DIR}/{catalog}/{campaign}/{run}.nc" for run in runs]


def run_config_path(wc):
    return Path(campaign_config_dir(wc.campaign)) / f"{wc.run}.json"


wildcard_constraints:
    catalog="|".join(re.escape(catalog_id) for catalog_id in CATALOG_RECIPES),
    campaign="[^/]+",
    run="[^/]+",


# Everything except run_mcmc runs on the submit host under a cluster executor.
localrules:
    paper_figures,
    mcmc_paper_h0,
    mcmc_sweeps,
    amplitude_toy,
    snr_by_detector,
    mcmc_compare_posteriors,


rule paper_figures:
    input:
        AMPLITUDE_TOY_PDF,
        SNR_BY_DETECTOR_PDF,
        POSTERIOR_PDF,


rule generate_sweep_configs:
    input:
        example="configs/mcmc.example.toml",
        generator="scripts/generate_mcmc_configs.py",
        sweep_spec=SWEEP_SPEC_PATH,
    output:
        SWEEP_CONFIGS,
    shell:
        "uv run --extra mcmc python {input.generator} --force"


rule bns_population:
    input:
        population_config=lambda wc: CATALOG_RECIPES[wc.catalog].population_config,
        registry=str(CATALOG_REGISTRY_PATH),
    output:
        "out/populations/{catalog}.h5",
    params:
        recipe=lambda wc: CATALOG_RECIPES[wc.catalog],
        outdir=lambda wc: str(population_path(wc.catalog).parent),
    shell:
        "mkdir -p {params.outdir}\n"
        "uv run gwmock-pop simulate"
        " --config {input.population_config}"
        " --n {params.recipe.n_samples}"
        " --output {output}"
        " --seed {params.recipe.seed}"


rule bns_waveform_catalog:
    input:
        population="out/populations/{catalog}.h5",
        registry=str(CATALOG_REGISTRY_PATH),
    output:
        "out/catalogs/{catalog}.h5",
    params:
        recipe=lambda wc: CATALOG_RECIPES[wc.catalog],
    shell:
        "uv run python scripts/generate_waveform_catalog.py"
        " --population {input.population}"
        " --output {output}"
        " --approximant {params.recipe.approximant}"
        " --sampling-frequency {params.recipe.sampling_frequency}"
        " --minimum-frequency {params.recipe.minimum_frequency}"
        " --maximum-frequency {params.recipe.maximum_frequency}"
        " --reference-frequency {params.recipe.reference_frequency}"
        " --frequency-resolution {params.recipe.frequency_resolution}"
        " --chunk-size {params.recipe.chunk_size}"


rule run_mcmc:
    input:
        config=run_config_path,
        catalog="out/catalogs/{catalog}.h5",
    output:
        chain=protected(CHAINS_DIR + "/{catalog}/{campaign}/{run}.nc"),
        sidecar=protected(CHAINS_DIR + "/{catalog}/{campaign}/{run}.json"),
    params:
        # Override for local smoke tests: --config jax_platforms=cpu
        jax_platforms=config.get("jax_platforms", "cuda"),
        outdir=lambda wc: f"{CHAINS_DIR}/{wc.catalog}/{wc.campaign}",
    resources:
        cpus_per_task=4,
        mem_mb=8000,
        runtime=240,
    shell:
        """
        export OMP_NUM_THREADS={resources.cpus_per_task}
        export JAX_PLATFORMS={params.jax_platforms}
        # job-nanny I/O conventions; harmless when the wrapper is absent.
        export INPUT="*"
        export OUTPUT="*"
        if command -v job-nanny >/dev/null 2>&1; then
            job-nanny uv run --extra mcmc python scripts/run_mcmc.py \
                --config {input.config} --outdir {params.outdir} \
                --label {wildcards.run} --catalog {input.catalog} --force
        else
            uv run --extra mcmc python scripts/run_mcmc.py \
                --config {input.config} --outdir {params.outdir} \
                --label {wildcards.run} --catalog {input.catalog} --force
        fi
        """


rule mcmc_paper_h0:
    input:
        campaign_chains(DEFAULT_CATALOG, "paper-h0"),


rule mcmc_sweeps:
    input:
        [
            chain
            for campaign in SWEEP_CAMPAIGNS
            for chain in campaign_chains(DEFAULT_CATALOG, campaign)
        ],


rule amplitude_toy:
    input:
        catalog=str(catalog_path(DEFAULT_CATALOG)),
        config=str(PAPER_CONFIG_PATH),
    output:
        AMPLITUDE_TOY_PDF,
    shell:
        "uv run python notebooks/amplitude_toy_model.py"
        " --catalog {input.catalog:q}"
        " --config {input.config:q}"
        " --output-pdf {output:q}"


rule snr_by_detector:
    input:
        catalog=str(catalog_path(DEFAULT_CATALOG)),
        config=str(PAPER_CONFIG_PATH),
    output:
        pdf=SNR_BY_DETECTOR_PDF,
        csv=SNR_BY_DETECTOR_CSV,
        tex=SNR_BY_DETECTOR_TEX,
        sigmas_csv=SNR_BY_DETECTOR_SIGMAS_CSV,
        sigmas_tex=SNR_BY_DETECTOR_SIGMAS_TEX,
    shell:
        "uv run python notebooks/snr_by_detector.py"
        " --catalog {input.catalog:q}"
        " --config {input.config:q}"
        " --output-pdf {output.pdf:q}"
        " --output-csv {output.csv:q}"
        " --output-tex {output.tex:q}"
        " --output-sigmas-csv {output.sigmas_csv:q}"
        " --output-sigmas-tex {output.sigmas_tex:q}"


rule mcmc_compare_posteriors:
    input:
        config=str(PAPER_CONFIG_PATH),
        chains=POSTERIOR_CHAINS,
        snr_csv=SNR_BY_DETECTOR_CSV,
    output:
        pdf=POSTERIOR_PDF,
        csv=POSTERIOR_CSV,
        tex=POSTERIOR_TEX,
    shell:
        "uv run --extra mcmc python notebooks/mcmc_compare_posteriors.py"
        " --chains {input.chains:q}"
        " --config {input.config:q}"
        " --snr-csv {input.snr_csv:q}"
        " --output-pdf {output.pdf:q}"
        " --output-csv {output.csv:q}"
        " --output-tex {output.tex:q}"
