from pathlib import Path
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
POSTERIOR_PDF = config["mcmc_compare_posteriors"]["output_pdf"]
POSTERIOR_CHAINS = [
    entry["path"]
    for entry in PAPER_CONFIG["figures"]["mcmc_compare_posteriors"]["posteriors"]
]


rule paper_figures:
    input:
        AMPLITUDE_TOY_PDF,
        SNR_BY_DETECTOR_PDF,
        POSTERIOR_PDF,


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
    shell:
        "uv run python notebooks/snr_by_detector.py"
        " --config {input.config}"
        " --output-pdf {output.pdf}"
        " --output-csv {output.csv}"
        " --output-tex {output.tex}"


rule mcmc_compare_posteriors:
    input:
        config=str(PAPER_CONFIG_PATH),
        chains=POSTERIOR_CHAINS,
    output:
        POSTERIOR_PDF,
    shell:
        "uv run python notebooks/mcmc_compare_posteriors.py"
        " --config {input.config}"
        " --output-pdf {output}"
