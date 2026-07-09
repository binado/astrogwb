from pathlib import Path
import tomllib


CONFIG_PATH = Path("configs/paper.toml")

with CONFIG_PATH.open("rb") as handle:
    PAPER_CONFIG = tomllib.load(handle)

CATALOG = PAPER_CONFIG["paths"]["catalog"]
AMPLITUDE_TOY_PDF = PAPER_CONFIG["figures"]["amplitude_toy"]["output_pdf"]
SNR_BY_DETECTOR_PDF = PAPER_CONFIG["figures"]["snr_by_detector"]["output_pdf"]
SNR_BY_DETECTOR_CSV = PAPER_CONFIG["figures"]["snr_by_detector"]["output_csv"]
POSTERIOR_PDF = PAPER_CONFIG["figures"]["mcmc_compare_posteriors"]["output_pdf"]
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
        config=str(CONFIG_PATH),
    output:
        AMPLITUDE_TOY_PDF,
    shell:
        "uv run python notebooks/amplitude_toy_model.py --config configs/paper.toml"


rule snr_by_detector:
    input:
        catalog=CATALOG,
        config=str(CONFIG_PATH),
    output:
        pdf=SNR_BY_DETECTOR_PDF,
        csv=SNR_BY_DETECTOR_CSV,
    shell:
        "uv run python notebooks/snr_by_detector.py --config configs/paper.toml"


rule mcmc_compare_posteriors:
    input:
        config=str(CONFIG_PATH),
        chains=POSTERIOR_CHAINS,
    output:
        POSTERIOR_PDF,
    shell:
        "uv run python notebooks/mcmc_compare_posteriors.py --config configs/paper.toml"
