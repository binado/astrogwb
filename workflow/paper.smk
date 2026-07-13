from pathlib import Path
import tomllib


configfile: "configs/workflow.yaml"


PAPER_CONFIG_PATH = Path(config["paper_config"])

with PAPER_CONFIG_PATH.open("rb") as handle:
    PAPER_CONFIG = tomllib.load(handle)

PAPER_ANALYSIS = PAPER_CONFIG["analysis"]
PAPER_COSMOLOGY = PAPER_ANALYSIS["cosmology"]
PAPER_FIDUCIALS = PAPER_ANALYSIS["fiducials"]
PAPER_NETWORKS = PAPER_CONFIG["detector_networks"]
PAPER_NETWORK_NAMES = list(PAPER_NETWORKS)
PAPER_NETWORK_ARGS = [
    argument
    for name, detectors in PAPER_NETWORKS.items()
    for argument in ("--network", f"{name}={','.join(detectors)}")
]

CATALOG_ID = config["catalog"]["id"]
CATALOG_PATH = (
    config["catalog"].get("path") or f"out/catalogs/{CATALOG_ID}.h5"
)
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
    f"{CHAINS_DIR}/{CATALOG_ID}/{POSTERIOR_FIGURE['campaign']}/{entry['run']}.nc"
    for entry in POSTERIOR_FIGURE["posteriors"]
]


localrules:
    paper_figures,
    amplitude_toy,
    snr_by_detector,
    mcmc_compare_posteriors,


rule paper_figures:
    input:
        AMPLITUDE_TOY_PDF,
        SNR_BY_DETECTOR_PDF,
        POSTERIOR_PDF,


rule amplitude_toy:
    input:
        catalog=CATALOG_PATH,
        config=str(PAPER_CONFIG_PATH),
    output:
        AMPLITUDE_TOY_PDF,
    params:
        chains_dir=PAPER_CONFIG["paths"]["chains_dir"],
        observation_time=PAPER_ANALYSIS["observation_time"],
        f_min=PAPER_ANALYSIS["f_min"],
        f_max=PAPER_ANALYSIS["f_max"],
    shell:
        "uv run python notebooks/amplitude_toy_model.py"
        " --catalog {input.catalog:q}"
        " --chains-dir {params.chains_dir:q}"
        " --observation-time {params.observation_time}"
        " --f-min {params.f_min}"
        " --f-max {params.f_max}"
        " --output-pdf {output:q}"


rule snr_by_detector:
    input:
        catalog=CATALOG_PATH,
        config=str(PAPER_CONFIG_PATH),
    output:
        pdf=SNR_BY_DETECTOR_PDF,
        csv=SNR_BY_DETECTOR_CSV,
        tex=SNR_BY_DETECTOR_TEX,
        sigmas_csv=SNR_BY_DETECTOR_SIGMAS_CSV,
        sigmas_tex=SNR_BY_DETECTOR_SIGMAS_TEX,
    params:
        observation_time=PAPER_ANALYSIS["observation_time"],
        f_min=PAPER_ANALYSIS["f_min"],
        f_max=PAPER_ANALYSIS["f_max"],
        z_min=PAPER_COSMOLOGY["z_min"],
        z_max=PAPER_COSMOLOGY["z_max"],
        n_grid=PAPER_COSMOLOGY["n_grid"],
        h0=PAPER_FIDUCIALS["H0"],
        omega_m=PAPER_FIDUCIALS["Omega_m"],
        xi_0=PAPER_FIDUCIALS["xi_0"],
        xi_n=PAPER_FIDUCIALS["xi_n"],
        gamma=PAPER_FIDUCIALS["gamma"],
        kappa=PAPER_FIDUCIALS["kappa"],
        z_peak=PAPER_FIDUCIALS["z_peak"],
        local_merger_rate=PAPER_FIDUCIALS["local_merger_rate"],
        network_args=PAPER_NETWORK_ARGS,
        network_names=PAPER_NETWORK_NAMES,
    shell:
        "uv run python notebooks/snr_by_detector.py"
        " --catalog {input.catalog:q}"
        " --observation-time {params.observation_time}"
        " --f-min {params.f_min}"
        " --f-max {params.f_max}"
        " --z-min {params.z_min}"
        " --z-max {params.z_max}"
        " --n-grid {params.n_grid}"
        " --h0 {params.h0}"
        " --omega-m {params.omega_m}"
        " --xi-0 {params.xi_0}"
        " --xi-n {params.xi_n}"
        " --gamma {params.gamma}"
        " --kappa {params.kappa}"
        " --z-peak {params.z_peak}"
        " --local-merger-rate {params.local_merger_rate}"
        " {params.network_args:q}"
        " --networks {params.network_names:q}"
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
