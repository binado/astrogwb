import json
from pathlib import Path
import tomllib


configfile: "configs/workflow.yaml"


PAPER_CONFIG_PATH = Path(config["paper_config"])

with PAPER_CONFIG_PATH.open("rb") as handle:
    PAPER_CONFIG = tomllib.load(handle)

CATALOG = PAPER_CONFIG["paths"]["catalog"]
POPULATION = f"out/bns_n={PAPER_CONFIG['population']['n_samples']}.h5"
POPULATION_CONFIG = "examples/bns_population.yaml"
WAVEFORM_CATALOG = PAPER_CONFIG["waveform_catalog"]
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
    f"{CHAINS_DIR}/{POSTERIOR_FIGURE['campaign']}/{entry['run']}.nc"
    for entry in POSTERIOR_FIGURE["posteriors"]
]

# Sweep campaigns keep their generated configs directly under
# configs/mcmc/<campaign>/ (see scripts/generate_mcmc_configs.py); every other
# campaign is hand-curated under configs/mcmc/curated/<campaign>/.
SWEEP_CAMPAIGNS = ("cosmology", "modified-propagation", "astrophysical")


def campaign_config_dir(campaign):
    base = "configs/mcmc" if campaign in SWEEP_CAMPAIGNS else "configs/mcmc/curated"
    return f"{base}/{campaign}"


def campaign_chains(campaign):
    runs = glob_wildcards(campaign_config_dir(campaign) + "/{run}.json").run
    return [f"{CHAINS_DIR}/{campaign}/{run}.nc" for run in sorted(runs)]


def run_config_path(wc):
    return Path(campaign_config_dir(wc.campaign)) / f"{wc.run}.json"


def run_catalog_path(wc):
    """Return the catalog consumed by one run's self-describing config."""
    raw = json.loads(run_config_path(wc).read_text(encoding="utf-8"))
    return raw["catalog"]["path"]


wildcard_constraints:
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


rule bns_population:
    input:
        population_config=POPULATION_CONFIG,
        paper_config=str(PAPER_CONFIG_PATH),
    output:
        POPULATION,
    params:
        n_samples=PAPER_CONFIG["population"]["n_samples"],
        seed=PAPER_CONFIG["population"]["seed"],
        outdir=str(Path(POPULATION).parent),
    shell:
        "mkdir -p {params.outdir}\n"
        "uv run gwmock-pop simulate"
        " --config {input.population_config}"
        " --n {params.n_samples}"
        " --output {output}"
        " --seed {params.seed}"


rule bns_waveform_catalog:
    input:
        population=POPULATION,
        paper_config=str(PAPER_CONFIG_PATH),
    output:
        CATALOG,
    params:
        **WAVEFORM_CATALOG,
    shell:
        "uv run python scripts/generate_waveform_catalog.py"
        " --population {input.population}"
        " --output {output}"
        " --approximant {params.approximant}"
        " --sampling-frequency {params.sampling_frequency}"
        " --minimum-frequency {params.minimum_frequency}"
        " --maximum-frequency {params.maximum_frequency}"
        " --reference-frequency {params.reference_frequency}"
        " --frequency-resolution {params.frequency_resolution}"
        " --chunk-size {params.chunk_size}"


rule run_mcmc:
    input:
        config=run_config_path,
        catalog=run_catalog_path,
    output:
        chain=protected(CHAINS_DIR + "/{campaign}/{run}.nc"),
        sidecar=protected(CHAINS_DIR + "/{campaign}/{run}.json"),
    params:
        # Override for local smoke tests: --config jax_platforms=cpu
        jax_platforms=config.get("jax_platforms", "cuda"),
        outdir=lambda wc: f"{CHAINS_DIR}/{wc.campaign}",
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
                --label {wildcards.run} --force
        else
            uv run --extra mcmc python scripts/run_mcmc.py \
                --config {input.config} --outdir {params.outdir} \
                --label {wildcards.run} --force
        fi
        """


rule mcmc_paper_h0:
    input:
        campaign_chains("paper-h0"),


rule mcmc_sweeps:
    input:
        [chain for campaign in SWEEP_CAMPAIGNS for chain in campaign_chains(campaign)],


rule amplitude_toy:
    input:
        catalog=CATALOG,
        config=str(PAPER_CONFIG_PATH),
    output:
        AMPLITUDE_TOY_PDF,
    shell:
        "uv run python notebooks/amplitude_toy_model.py"
        " --config {input.config}"
        " --output-pdf {output}"


rule snr_by_detector:
    input:
        catalog=CATALOG,
        config=str(PAPER_CONFIG_PATH),
    output:
        pdf=SNR_BY_DETECTOR_PDF,
        csv=SNR_BY_DETECTOR_CSV,
        tex=SNR_BY_DETECTOR_TEX,
        sigmas_csv=SNR_BY_DETECTOR_SIGMAS_CSV,
        sigmas_tex=SNR_BY_DETECTOR_SIGMAS_TEX,
    shell:
        "uv run python notebooks/snr_by_detector.py"
        " --config {input.config}"
        " --output-pdf {output.pdf}"
        " --output-csv {output.csv}"
        " --output-tex {output.tex}"
        " --output-sigmas-csv {output.sigmas_csv}"
        " --output-sigmas-tex {output.sigmas_tex}"


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
        " --config {input.config}"
        " --snr-csv {input.snr_csv}"
        " --output-pdf {output.pdf}"
        " --output-csv {output.csv}"
        " --output-tex {output.tex}"
