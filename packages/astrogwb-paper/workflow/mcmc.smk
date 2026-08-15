import re
import tomllib
from pathlib import Path

from astrogwb_paper.config.experiments import (
    DEFAULT_CATALOG,
    DETECTOR_NETWORKS,
    EXPERIMENTS,
    chain_path,
    chain_paths,
    config_path,
    experiment,
)
from astrogwb_paper.config.loading import load_mapping


BASE_CONFIG = Path("inputs/mcmc.base.toml")
BASE = load_mapping(BASE_CONFIG)
JAX_PLATFORM = config.get("jax_platforms", "cuda")
CATALOGS_DIR = Path(config.get("catalogs_dir", "outputs/catalogs"))

CONFIG_PATTERN = "outputs/configs/{experiment}/{run}.json"
CHAIN_PATTERN = "outputs/chains/{experiment}/{run}.nc"
SIDECAR_PATTERN = "outputs/chains/{experiment}/{run}.json"

EXPERIMENT_PATTERN = "|".join(re.escape(name) for name in EXPERIMENTS)
RUN_PATTERN = "|".join(
    re.escape(run)
    for run in dict.fromkeys(
        run for specification in EXPERIMENTS.values() for run in specification.runs
    )
)


def _load_toml(path):
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _run_spec(wildcards):
    specification = experiment(wildcards.experiment)
    if wildcards.run not in specification.runs:
        raise ValueError(f"unknown run {wildcards.experiment}/{wildcards.run}")
    return specification


def run_config_path(wildcards):
    _run_spec(wildcards)
    return str(config_path(wildcards.experiment, wildcards.run))


def run_catalog_path(wildcards):
    catalog = _run_spec(wildcards).catalog_for(wildcards.run)
    return str(CATALOGS_DIR / catalog.name)


def run_outdir(wildcards):
    return str(Path("outputs/chains") / wildcards.experiment)


def network_args():
    return [
        argument
        for name, detectors in DETECTOR_NETWORKS.items()
        for argument in ("--network", f"{name}={','.join(detectors)}")
    ]


def _chain(experiment_name, run):
    return str(chain_path(experiment_name, run))


def _catalog(name):
    return str(CATALOGS_DIR / name)


def _figure_config(experiment_name):
    return _load_toml(Path("experiments") / experiment_name / "figure.toml")


H0_DETECTOR_CONFIG = _figure_config("H0-all-detectors")
H0_DETECTOR_CHAINS = [
    _chain("H0-all-detectors", entry["run"])
    for entry in H0_DETECTOR_CONFIG["posteriors"]
]
H0_DETECTOR_LABELS = [
    entry["label"] for entry in H0_DETECTOR_CONFIG["posteriors"]
]
H0_DETECTOR_OUTPUTS = [
    H0_DETECTOR_CONFIG["output_pdf"],
    H0_DETECTOR_CONFIG["output_csv"],
    H0_DETECTOR_CONFIG["output_tex"],
]

H0_RATE_CONFIG = _figure_config("H0-merger-rate")
H0_RATE_CHAINS = [
    _chain("H0-merger-rate", H0_RATE_CONFIG["fixed_run"]),
    _chain("H0-merger-rate", H0_RATE_CONFIG["sampled_run"]),
]
H0_RATE_OUTPUTS = [
    H0_RATE_CONFIG["output_prior_pdf"],
    H0_RATE_CONFIG["output_corner_pdf"],
    H0_RATE_CONFIG["output_csv"],
    H0_RATE_CONFIG["output_tex"],
]

H0_OMEGA_CONFIG = _figure_config("H0-omega-m")
H0_OMEGA_CHAIN = _chain("H0-omega-m", H0_OMEGA_CONFIG["run"])
H0_OMEGA_OUTPUTS = [
    H0_OMEGA_CONFIG["output_corner_pdf"],
    H0_OMEGA_CONFIG["output_ess_corner_pdf"],
]

PROPAGATION_CONFIG = _figure_config("modified-propagation-all-detectors")
PROPAGATION_DETECTOR_CHAINS = [
    _chain("modified-propagation-all-detectors", entry["run"])
    for entry in PROPAGATION_CONFIG["detector_posteriors"]
]
PROPAGATION_DETECTOR_LABELS = [
    entry["label"] for entry in PROPAGATION_CONFIG["detector_posteriors"]
]
PROPAGATION_OUTPUTS = [
    PROPAGATION_CONFIG["output_xi_n_corner_pdf"],
    PROPAGATION_CONFIG["output_xi_n_ess_corner_pdf"],
    PROPAGATION_CONFIG["output_xi0_marginal_pdf"],
    PROPAGATION_CONFIG["output_h0_corner_pdf"],
    PROPAGATION_CONFIG["output_csv"],
    PROPAGATION_CONFIG["output_tex"],
]

STANDALONE_CONFIG_PATH = Path("inputs/figures/standalone.toml")
STANDALONE_CONFIG = _load_toml(STANDALONE_CONFIG_PATH)
AMPLITUDE_TOY_PDF = STANDALONE_CONFIG["amplitude_toy"]["output_pdf"]
FIDUCIAL_SPECTRUM = STANDALONE_CONFIG["fiducial_spectrum"]
FIDUCIAL_SPECTRUM_OUTPUTS = [
    FIDUCIAL_SPECTRUM["output_pdf"],
    FIDUCIAL_SPECTRUM["output_effective_psd_pdf"],
]
IMPORTANCE_GRID = STANDALONE_CONFIG["importance_weights_grid"]
IMPORTANCE_GRID_OUTPUTS = [
    IMPORTANCE_GRID["output_h0_omega_m_pdf"],
    IMPORTANCE_GRID["output_xi0_n_pdf"],
]
STANDALONE_OUTPUTS = [
    AMPLITUDE_TOY_PDF,
    *FIDUCIAL_SPECTRUM_OUTPUTS,
    *IMPORTANCE_GRID_OUTPUTS,
]

FIDUCIALS = BASE["fiducials"]
COSMOLOGY = BASE["cosmology"]
ANALYSIS = BASE["analysis"]
NETWORK_ARGS = network_args()
NETWORK_NAMES = list(DETECTOR_NETWORKS)


wildcard_constraints:
    experiment=EXPERIMENT_PATTERN,
    run=RUN_PATTERN,


localrules:
    experiments,
    standalone_figures,
    H0_all_detectors,
    H0_all_detectors_chains,
    modified_propagation_all_detectors,
    modified_propagation_all_detectors_chains,
    H0_merger_rate,
    H0_merger_rate_chains,
    H0_omega_m,
    H0_omega_m_chains,
    astrophysical_parameters,
    astrophysical_parameters_chains,
    star_formation_peak,
    star_formation_peak_chains,
    variable_injection_size,
    variable_injection_size_chains,
    assemble_config,
    plot_H0_all_detectors,
    plot_H0_merger_rate,
    plot_H0_omega_m,
    plot_modified_propagation,
    amplitude_toy,
    fiducial_spectrum,
    importance_weights_grid,


rule experiments:
    input:
        H0_DETECTOR_OUTPUTS,
        PROPAGATION_OUTPUTS,
        H0_RATE_OUTPUTS,
        H0_OMEGA_OUTPUTS,
        chain_paths("astrophysical-parameters"),
        chain_paths("star-formation-peak"),
        chain_paths("variable-injection-size"),


rule standalone_figures:
    input:
        STANDALONE_OUTPUTS,


rule H0_all_detectors:
    input:
        H0_DETECTOR_OUTPUTS,


rule H0_all_detectors_chains:
    input:
        chain_paths("H0-all-detectors"),


rule modified_propagation_all_detectors:
    input:
        PROPAGATION_OUTPUTS,


rule modified_propagation_all_detectors_chains:
    input:
        chain_paths("modified-propagation-all-detectors"),


rule H0_merger_rate:
    input:
        H0_RATE_OUTPUTS,


rule H0_merger_rate_chains:
    input:
        chain_paths("H0-merger-rate"),


rule H0_omega_m:
    input:
        H0_OMEGA_OUTPUTS,


rule H0_omega_m_chains:
    input:
        chain_paths("H0-omega-m"),


rule astrophysical_parameters:
    input:
        chain_paths("astrophysical-parameters"),


rule astrophysical_parameters_chains:
    input:
        chain_paths("astrophysical-parameters"),


rule star_formation_peak:
    input:
        chain_paths("star-formation-peak"),


rule star_formation_peak_chains:
    input:
        chain_paths("star-formation-peak"),


rule variable_injection_size:
    input:
        chain_paths("variable-injection-size"),


rule variable_injection_size_chains:
    input:
        chain_paths("variable-injection-size"),


rule assemble_config:
    input:
        base=str(BASE_CONFIG),
        run=run_config_path,
    output:
        config=CONFIG_PATTERN,
    shell:
        "knf {input.base:q} {input.run:q} --strict -f json"
        " | uv run --package astrogwb-paper astrogwb-validate-config -"
        " --output {output.config:q}"


rule run_mcmc:
    input:
        config=ancient(CONFIG_PATTERN),
        catalog=run_catalog_path,
    output:
        chain=protected(CHAIN_PATTERN),
        sidecar=protected(SIDECAR_PATTERN),
    params:
        platform=JAX_PLATFORM,
        outdir=run_outdir,
    threads: 4
    resources:
        mem_mb=8000,
        runtime=720,
    shell:
        """
        export INPUT="*"
        export OUTPUT="*"
        NANNY=
        if command -v job-nanny >/dev/null 2>&1; then
            NANNY=job-nanny
        fi
        if [ "{params.platform}" = "cpu" ]; then
            RUNTIME_FLAGS="--host-device-count {threads} --cpu-threads 1"
            UV_EXTRAS="--package astrogwb-paper"
        else
            RUNTIME_FLAGS="--cpu-threads {threads}"
            UV_EXTRAS="--package astrogwb-paper --extra cuda"
        fi
        $NANNY uv run $UV_EXTRAS astrogwb-run-mcmc \
            --config {input.config:q} --outdir {params.outdir:q} \
            --label {wildcards.run:q} --catalog {input.catalog:q} \
            --platform {params.platform:q} $RUNTIME_FLAGS --force
        """


rule plot_H0_all_detectors:
    input:
        chains=H0_DETECTOR_CHAINS,
        catalog=_catalog(DEFAULT_CATALOG.name),
        config="experiments/H0-all-detectors/figure.toml",
        base=str(BASE_CONFIG),
    output:
        pdf=H0_DETECTOR_CONFIG["output_pdf"],
        csv=H0_DETECTOR_CONFIG["output_csv"],
        tex=H0_DETECTOR_CONFIG["output_tex"],
    params:
        labels=H0_DETECTOR_LABELS,
        network_args=NETWORK_ARGS,
        networks=NETWORK_NAMES,
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python notebooks/paper/mcmc_cosmological_parameters.py"
        " --section detectors --config {input.config:q}"
        " --base-config {input.base:q}"
        " --catalog {input.catalog:q} --detector-chains {input.chains:q}"
        " --detector-labels {params.labels:q}"
        " {params.network_args:q} --networks {params.networks:q}"
        " --output-detector-pdf {output.pdf:q}"
        " --output-csv {output.csv:q} --output-tex {output.tex:q}"


rule plot_H0_merger_rate:
    input:
        chains=H0_RATE_CHAINS,
        config="experiments/H0-merger-rate/figure.toml",
        base=str(BASE_CONFIG),
    output:
        prior_pdf=H0_RATE_CONFIG["output_prior_pdf"],
        corner_pdf=H0_RATE_CONFIG["output_corner_pdf"],
        csv=H0_RATE_CONFIG["output_csv"],
        tex=H0_RATE_CONFIG["output_tex"],
    params:
        # Callable so snakemake does not treat LaTeX argument braces (e.g. the
        # ``{R}`` in ``\mathcal{R}``) inside label strings as wildcards.
        labels=lambda wildcards: H0_RATE_CONFIG["labels"],
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python notebooks/paper/mcmc_cosmological_parameters.py"
        " --section merger-rate --config {input.config:q}"
        " --base-config {input.base:q}"
        " --prior-chains {input.chains:q} --prior-labels {params.labels:q}"
        " --output-prior-pdf {output.prior_pdf:q}"
        " --output-narrow-corner-pdf {output.corner_pdf:q}"
        " --output-csv {output.csv:q} --output-tex {output.tex:q}"


rule plot_H0_omega_m:
    input:
        chain=H0_OMEGA_CHAIN,
        config="experiments/H0-omega-m/figure.toml",
        base=str(BASE_CONFIG),
    output:
        corner_pdf=H0_OMEGA_CONFIG["output_corner_pdf"],
        ess_corner_pdf=H0_OMEGA_CONFIG["output_ess_corner_pdf"],
    params:
        label=H0_OMEGA_CONFIG["label"],
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python notebooks/paper/mcmc_cosmological_parameters.py"
        " --section omega-m --config {input.config:q}"
        " --base-config {input.base:q}"
        " --omega-m-chain {input.chain:q} --omega-m-label {params.label:q}"
        " --output-omega-m-corner-pdf {output.corner_pdf:q}"
        " --output-omega-m-ess-corner-pdf {output.ess_corner_pdf:q}"


rule plot_modified_propagation:
    input:
        xi0_chain=_chain(
            "modified-propagation-all-detectors", PROPAGATION_CONFIG["xi0_run"]
        ),
        xi0_n_chain=_chain(
            "modified-propagation-all-detectors", PROPAGATION_CONFIG["xi0_n_run"]
        ),
        h0_chain=_chain(
            "modified-propagation-all-detectors", PROPAGATION_CONFIG["h0_run"]
        ),
        detector_chains=PROPAGATION_DETECTOR_CHAINS,
        catalog=_catalog(DEFAULT_CATALOG.name),
        config="experiments/modified-propagation-all-detectors/figure.toml",
        base=str(BASE_CONFIG),
    output:
        xi_n_corner_pdf=PROPAGATION_CONFIG["output_xi_n_corner_pdf"],
        xi_n_ess_corner_pdf=PROPAGATION_CONFIG["output_xi_n_ess_corner_pdf"],
        xi0_marginal_pdf=PROPAGATION_CONFIG["output_xi0_marginal_pdf"],
        h0_corner_pdf=PROPAGATION_CONFIG["output_h0_corner_pdf"],
        csv=PROPAGATION_CONFIG["output_csv"],
        tex=PROPAGATION_CONFIG["output_tex"],
    params:
        marginal_labels=PROPAGATION_CONFIG["marginal_labels"],
        h0_labels=PROPAGATION_CONFIG["h0_labels"],
        detector_labels=PROPAGATION_DETECTOR_LABELS,
        network_args=NETWORK_ARGS,
        networks=NETWORK_NAMES,
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python notebooks/paper/mcmc_modified_propagation.py"
        " --base-config {input.base:q}"
        " --xi0-chain {input.xi0_chain:q} --xi0-n-chain {input.xi0_n_chain:q}"
        " --h0-chain {input.h0_chain:q}"
        " --marginal-labels {params.marginal_labels:q}"
        " --h0-labels {params.h0_labels:q}"
        " --detector-xi0-n-chains {input.detector_chains:q}"
        " --detector-labels {params.detector_labels:q}"
        " {params.network_args:q} --networks {params.networks:q}"
        " --catalog {input.catalog:q}"
        " --output-xi-n-corner-pdf {output.xi_n_corner_pdf:q}"
        " --output-xi-n-ess-corner-pdf {output.xi_n_ess_corner_pdf:q}"
        " --output-xi0-marginal-pdf {output.xi0_marginal_pdf:q}"
        " --output-h0-corner-pdf {output.h0_corner_pdf:q}"
        " --output-xi0-n-csv {output.csv:q} --output-xi0-n-tex {output.tex:q}"


rule amplitude_toy:
    input:
        catalog=_catalog(DEFAULT_CATALOG.name),
        config=str(BASE_CONFIG),
    output:
        AMPLITUDE_TOY_PDF,
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python notebooks/paper/amplitude_toy_model.py"
        " --catalog {input.catalog:q}"
        " --chains-dir outputs/chains/amplitude-toy"
        " --observation-time {BASE[observation_time]}"
        " --f-min {ANALYSIS[f_min]} --f-max {ANALYSIS[f_max]}"
        " --output-pdf {output:q}"


rule fiducial_spectrum:
    input:
        catalog=_catalog(DEFAULT_CATALOG.name),
        config=str(STANDALONE_CONFIG_PATH),
        base=str(BASE_CONFIG),
    output:
        spectrum_pdf=FIDUCIAL_SPECTRUM["output_pdf"],
        effective_psd_pdf=FIDUCIAL_SPECTRUM["output_effective_psd_pdf"],
    params:
        network_args=NETWORK_ARGS,
        networks=NETWORK_NAMES,
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python notebooks/paper/fiducial_spectrum.py"
        " --config {input.config:q} --catalog {input.catalog:q}"
        " --f-min {ANALYSIS[f_min]} --f-max {ANALYSIS[f_max]}"
        " --z-min {COSMOLOGY[z_min]} --z-max {COSMOLOGY[z_max]}"
        " --n-grid {COSMOLOGY[n_grid]}"
        " --h0 {FIDUCIALS[H0]} --omega-m {FIDUCIALS[Omega_m]}"
        " --xi-0 {FIDUCIALS[xi_0]} --xi-n {FIDUCIALS[xi_n]}"
        " --gamma {FIDUCIALS[gamma]} --kappa {FIDUCIALS[kappa]}"
        " --z-peak {FIDUCIALS[z_peak]}"
        " --local-merger-rate {FIDUCIALS[local_merger_rate]}"
        " --omega-gw-min {FIDUCIAL_SPECTRUM[omega_gw_min]}"
        " --output-pdf {output.spectrum_pdf:q}"
        " --output-effective-psd-pdf {output.effective_psd_pdf:q}"


rule importance_weights_grid:
    input:
        catalog=_catalog(DEFAULT_CATALOG.name),
        base=str(BASE_CONFIG),
    output:
        h0_omega_m_pdf=IMPORTANCE_GRID["output_h0_omega_m_pdf"],
        xi0_n_pdf=IMPORTANCE_GRID["output_xi0_n_pdf"],
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python notebooks/paper/importance_weights_grid.py"
        " --config {input.base:q} --catalog {input.catalog:q}"
        " --output-h0-omega-m-pdf {output.h0_omega_m_pdf:q}"
        " --output-xi0-n-pdf {output.xi0_n_pdf:q}"
