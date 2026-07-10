from pathlib import Path
import json
import tomllib


configfile: "configs/workflow.yaml"


PAPER_CONFIG_PATH = Path(config["paper_config"])

with PAPER_CONFIG_PATH.open("rb") as handle:
    PAPER_CONFIG = tomllib.load(handle)

CATALOG = PAPER_CONFIG["paths"]["catalog"]
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
POSTERIOR_LOCK = POSTERIOR_FIGURE["campaign_lock"]
with Path(POSTERIOR_LOCK).open() as handle:
    _POSTERIOR_CAMPAIGN = json.load(handle)
_POSTERIOR_RUNS = {run["id"]: run for run in _POSTERIOR_CAMPAIGN["runs"]}
POSTERIOR_CHAINS = [
    _POSTERIOR_RUNS[entry["run"]]["outputs"]["chain"]
    for entry in POSTERIOR_FIGURE["posteriors"]
]
CAMPAIGN_INVENTORY = config["mcmc_campaign"]["inventory"]
CAMPAIGN_FROZEN_DIR = config["mcmc_campaign"]["frozen_dir"]
CAMPAIGN_MANIFEST = f"{CAMPAIGN_FROZEN_DIR}/array-manifest.txt"


rule paper_figures:
    input:
        AMPLITUDE_TOY_PDF,
        SNR_BY_DETECTOR_PDF,
        POSTERIOR_PDF,


rule mcmc_campaign:
    input:
        inventory=CAMPAIGN_INVENTORY,
        lock=POSTERIOR_LOCK,
    output:
        manifest=CAMPAIGN_MANIFEST,
    params:
        frozen_dir=CAMPAIGN_FROZEN_DIR,
    shell:
        "uv run --extra mcmc python scripts/freeze_mcmc_campaign.py"
        " {input.inventory} --lock {input.lock} --frozen-dir {params.frozen_dir}"
        " --manifest {output.manifest}"


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
        lock=POSTERIOR_LOCK,
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
