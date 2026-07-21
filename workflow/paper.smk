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
FIDUCIAL_SPECTRUM = config["fiducial_spectrum"]
FIDUCIAL_SPECTRUM_PDF = FIDUCIAL_SPECTRUM["output_pdf"]
FIDUCIAL_SPECTRUM_OMEGA_GW_MIN = FIDUCIAL_SPECTRUM["omega_gw_min"]
COSMOLOGY_OUTPUTS = config["mcmc_cosmological_parameters"]
COSMOLOGY_DETECTOR_PDF = COSMOLOGY_OUTPUTS["output_detector_pdf"]
COSMOLOGY_PRIOR_PDF = COSMOLOGY_OUTPUTS["output_prior_pdf"]
COSMOLOGY_NARROW_CORNER_PDF = COSMOLOGY_OUTPUTS["output_narrow_corner_pdf"]
COSMOLOGY_BROAD_CORNER_PDF = COSMOLOGY_OUTPUTS["output_broad_corner_pdf"]
COSMOLOGY_OMEGA_M_CORNER_PDF = COSMOLOGY_OUTPUTS["output_omega_m_corner_pdf"]
COSMOLOGY_CSV = COSMOLOGY_OUTPUTS["output_csv"]
COSMOLOGY_TEX = COSMOLOGY_OUTPUTS["output_tex"]
COSMOLOGY_FIGURE = PAPER_CONFIG["figures"]["mcmc_cosmological_parameters"]
COSMOLOGY_CAMPAIGN = COSMOLOGY_FIGURE["campaign"]
DETECTOR_POSTERIORS = COSMOLOGY_FIGURE["detector_posteriors"]
DETECTOR_CHAINS = [
    f"{CHAINS_DIR}/{CATALOG_ID}/{COSMOLOGY_CAMPAIGN}/"
    f"{entry['network']}__{COSMOLOGY_FIGURE['detector_analysis']}__baseline.nc"
    for entry in DETECTOR_POSTERIORS
]
DETECTOR_LABELS = [entry["label"] for entry in DETECTOR_POSTERIORS]
PRIOR_POSTERIORS = COSMOLOGY_FIGURE["prior_posteriors"]
PRIOR_CHAINS = [
    f"{CHAINS_DIR}/{CATALOG_ID}/{COSMOLOGY_CAMPAIGN}/"
    f"{COSMOLOGY_FIGURE['prior_network']}__{entry['analysis']}__baseline.nc"
    for entry in PRIOR_POSTERIORS
]
PRIOR_LABELS = [entry["label"] for entry in PRIOR_POSTERIORS]
OMEGA_M_CHAIN = (
    f"{CHAINS_DIR}/{CATALOG_ID}/{COSMOLOGY_CAMPAIGN}/"
    f"{COSMOLOGY_FIGURE['prior_network']}__"
    f"{COSMOLOGY_FIGURE['omega_m_analysis']}__baseline.nc"
)
OMEGA_M_LABEL = r"$H_0 + \Omega_m$"

MODIFIED_PROPAGATION_FIGURE = PAPER_CONFIG["figures"]["mcmc_modified_propagation"]
MODIFIED_PROPAGATION_CAMPAIGN = MODIFIED_PROPAGATION_FIGURE["campaign"]
MODIFIED_PROPAGATION_NETWORK = MODIFIED_PROPAGATION_FIGURE["network"]
MODIFIED_PROPAGATION_XI0_CHAIN = (
    f"{CHAINS_DIR}/{CATALOG_ID}/{MODIFIED_PROPAGATION_CAMPAIGN}/"
    f"{MODIFIED_PROPAGATION_NETWORK}__"
    f"{MODIFIED_PROPAGATION_FIGURE['xi0_analysis']}__baseline.nc"
)
MODIFIED_PROPAGATION_XI0_N_CHAIN = (
    f"{CHAINS_DIR}/{CATALOG_ID}/{MODIFIED_PROPAGATION_CAMPAIGN}/"
    f"{MODIFIED_PROPAGATION_NETWORK}__"
    f"{MODIFIED_PROPAGATION_FIGURE['xi0_n_analysis']}__baseline.nc"
)
MODIFIED_PROPAGATION_H0_CHAIN = (
    f"{CHAINS_DIR}/{CATALOG_ID}/{MODIFIED_PROPAGATION_CAMPAIGN}/"
    f"{MODIFIED_PROPAGATION_NETWORK}__"
    f"{MODIFIED_PROPAGATION_FIGURE['h0_analysis']}__baseline.nc"
)
MODIFIED_PROPAGATION_MARGINAL_LABELS = MODIFIED_PROPAGATION_FIGURE["marginal_labels"]
MODIFIED_PROPAGATION_H0_LABELS = MODIFIED_PROPAGATION_FIGURE["h0_labels"]
MODIFIED_PROPAGATION_OUTPUTS = config["mcmc_modified_propagation"]
MODIFIED_PROPAGATION_XI_N_CORNER_PDF = MODIFIED_PROPAGATION_OUTPUTS[
    "output_xi_n_corner_pdf"
]
MODIFIED_PROPAGATION_XI0_MARGINAL_PDF = MODIFIED_PROPAGATION_OUTPUTS[
    "output_xi0_marginal_pdf"
]
MODIFIED_PROPAGATION_H0_CORNER_PDF = MODIFIED_PROPAGATION_OUTPUTS[
    "output_h0_corner_pdf"
]


localrules:
    paper_figures,
    amplitude_toy,
    fiducial_spectrum,
    mcmc_cosmological_parameters,
    mcmc_modified_propagation,


rule paper_figures:
    input:
        AMPLITUDE_TOY_PDF,
        FIDUCIAL_SPECTRUM_PDF,
        COSMOLOGY_DETECTOR_PDF,
        COSMOLOGY_PRIOR_PDF,
        COSMOLOGY_NARROW_CORNER_PDF,
        COSMOLOGY_BROAD_CORNER_PDF,
        COSMOLOGY_OMEGA_M_CORNER_PDF,
        COSMOLOGY_CSV,
        COSMOLOGY_TEX,
        MODIFIED_PROPAGATION_XI_N_CORNER_PDF,
        MODIFIED_PROPAGATION_XI0_MARGINAL_PDF,
        MODIFIED_PROPAGATION_H0_CORNER_PDF,


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


rule fiducial_spectrum:
    input:
        catalog=CATALOG_PATH,
        config=str(PAPER_CONFIG_PATH),
    output:
        FIDUCIAL_SPECTRUM_PDF,
    params:
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
        omega_gw_min=FIDUCIAL_SPECTRUM_OMEGA_GW_MIN,
    shell:
        "uv run --group plotting"
        " python notebooks/paper/fiducial_spectrum.py"
        " --catalog {input.catalog:q}"
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
        " --omega-gw-min {params.omega_gw_min}"
        " --output-pdf {output:q}"


rule mcmc_cosmological_parameters:
    input:
        catalog=CATALOG_PATH,
        config=str(PAPER_CONFIG_PATH),
        detector_chains=DETECTOR_CHAINS,
        prior_chains=PRIOR_CHAINS,
        omega_m_chain=OMEGA_M_CHAIN,
    output:
        detector_pdf=COSMOLOGY_DETECTOR_PDF,
        prior_pdf=COSMOLOGY_PRIOR_PDF,
        narrow_corner_pdf=COSMOLOGY_NARROW_CORNER_PDF,
        broad_corner_pdf=COSMOLOGY_BROAD_CORNER_PDF,
        omega_m_corner_pdf=COSMOLOGY_OMEGA_M_CORNER_PDF,
        csv=COSMOLOGY_CSV,
        tex=COSMOLOGY_TEX,
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
        detector_labels=lambda wildcards: DETECTOR_LABELS,
        prior_labels=lambda wildcards: PRIOR_LABELS,
        omega_m_label=lambda wildcards: OMEGA_M_LABEL,
    shell:
        "uv run --extra mcmc --group plotting"
        " python notebooks/paper/mcmc_cosmological_parameters.py"
        " --config {input.config:q}"
        " --catalog {input.catalog:q}"
        " --detector-chains {input.detector_chains:q}"
        " --detector-labels {params.detector_labels:q}"
        " --prior-chains {input.prior_chains:q}"
        " --prior-labels {params.prior_labels:q}"
        " --omega-m-chain {input.omega_m_chain:q}"
        " --omega-m-label {params.omega_m_label:q}"
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
        " --output-detector-pdf {output.detector_pdf:q}"
        " --output-prior-pdf {output.prior_pdf:q}"
        " --output-narrow-corner-pdf {output.narrow_corner_pdf:q}"
        " --output-broad-corner-pdf {output.broad_corner_pdf:q}"
        " --output-omega-m-corner-pdf {output.omega_m_corner_pdf:q}"
        " --output-csv {output.csv:q}"
        " --output-tex {output.tex:q}"


rule mcmc_modified_propagation:
    input:
        config=str(PAPER_CONFIG_PATH),
        xi0_chain=MODIFIED_PROPAGATION_XI0_CHAIN,
        xi0_n_chain=MODIFIED_PROPAGATION_XI0_N_CHAIN,
        h0_chain=MODIFIED_PROPAGATION_H0_CHAIN,
    output:
        xi_n_corner_pdf=MODIFIED_PROPAGATION_XI_N_CORNER_PDF,
        xi0_marginal_pdf=MODIFIED_PROPAGATION_XI0_MARGINAL_PDF,
        h0_corner_pdf=MODIFIED_PROPAGATION_H0_CORNER_PDF,
    params:
        marginal_labels=MODIFIED_PROPAGATION_MARGINAL_LABELS,
        h0_labels=MODIFIED_PROPAGATION_H0_LABELS,
        xi_0=PAPER_FIDUCIALS["xi_0"],
        xi_n=PAPER_FIDUCIALS["xi_n"],
        h0=PAPER_FIDUCIALS["H0"],
    shell:
        "uv run --extra mcmc --group plotting"
        " python notebooks/paper/mcmc_modified_propagation.py"
        " --xi0-chain {input.xi0_chain:q}"
        " --xi0-n-chain {input.xi0_n_chain:q}"
        " --h0-chain {input.h0_chain:q}"
        " --marginal-labels {params.marginal_labels:q}"
        " --h0-labels {params.h0_labels:q}"
        " --xi-0 {params.xi_0}"
        " --xi-n {params.xi_n}"
        " --h0 {params.h0}"
        " --output-xi-n-corner-pdf {output.xi_n_corner_pdf:q}"
        " --output-xi0-marginal-pdf {output.xi0_marginal_pdf:q}"
        " --output-h0-corner-pdf {output.h0_corner_pdf:q}"
