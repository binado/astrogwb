import re
import shlex
from pathlib import Path

from astrogwb_paper.config.experiments import (
    DEFAULT_CATALOG,
    chain_path,
    chain_paths,
    config_path,
    experiment,
    load_experiments,
)
from astrogwb_paper.config.figures import figure_networks, load_figure_config
from astrogwb_paper.config.loading import load_mapping


BASE_CONFIG = Path("inputs/mcmc.base.toml")
BASE = load_mapping(BASE_CONFIG)
FIGURES_DIR = Path("inputs/figures")
NETWORK_LABELS = FIGURES_DIR / "detector-networks.toml"
JAX_PLATFORM = config.get("jax_platforms", "cuda")
CATALOGS_DIR = Path(config.get("catalogs_dir", "outputs/catalogs"))

CONFIG_PATTERN = "outputs/configs/{experiment}/{run}.json"
CHAIN_PATTERN = "outputs/chains/{experiment}/{run}.nc"
SIDECAR_PATTERN = "outputs/chains/{experiment}/{run}.json"

experiments = load_experiments()
EXPERIMENT_PATTERN = "|".join(re.escape(name) for name in experiments)
RUN_PATTERN = "|".join(
    re.escape(run)
    for run in dict.fromkeys(
        run for specification in experiments.values() for run in specification.runs
    )
)


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


def _chain(experiment_name, run):
    return str(chain_path(experiment_name, run))


def _catalog(name):
    return str(CATALOGS_DIR / name)


def _figure(name):
    return load_figure_config(FIGURES_DIR / f"{name}.toml")


def _figure_path(name):
    return str(FIGURES_DIR / f"{name}.toml")


def _outputs(figure):
    return [str(value) for key, value in figure.items() if key.startswith("output_")]


def _quote_join(values):
    return " ".join(shlex.quote(str(value)) for value in values)


def _const(value):
    """Disable Snakemake wildcard expansion in params that contain LaTeX braces."""
    return lambda wildcards: value


def _network_cli(networks):
    return " ".join(
        "--network " + shlex.quote(f"{network.name}={','.join(network.detectors)}")
        for network in networks
    )


def _labels_cli(flag, labels):
    return f"{flag} " + _quote_join(labels)


H0_DETECTOR_CONFIG = _figure("H0-all-detectors")
H0_DETECTOR_NETWORKS = figure_networks(H0_DETECTOR_CONFIG, "posteriors")
H0_DETECTOR_CHAINS = [
    _chain("H0-all-detectors", network.name) for network in H0_DETECTOR_NETWORKS
]
H0_DETECTOR_OUTPUTS = _outputs(H0_DETECTOR_CONFIG)

H0_RATE_CONFIG = _figure("H0-merger-rate")
H0_RATE_CHAINS = [
    _chain("H0-merger-rate", H0_RATE_CONFIG["fixed_run"]),
    _chain("H0-merger-rate", H0_RATE_CONFIG["sampled_run"]),
]
H0_RATE_OUTPUTS = _outputs(H0_RATE_CONFIG)

H0_OMEGA_CONFIG = _figure("H0-omega-m")
H0_OMEGA_CHAIN = _chain("H0-omega-m", H0_OMEGA_CONFIG["run"])
H0_OMEGA_OUTPUTS = _outputs(H0_OMEGA_CONFIG)

PROPAGATION_CONFIG = _figure("modified-propagation-all-detectors")
PROPAGATION_NETWORKS = figure_networks(PROPAGATION_CONFIG, "detector_posteriors")
PROPAGATION_DETECTOR_CHAINS = [
    _chain("modified-propagation-all-detectors", network.name)
    for network in PROPAGATION_NETWORKS
]
PROPAGATION_OUTPUTS = _outputs(PROPAGATION_CONFIG)

STANDALONE_CONFIG_PATH = FIGURES_DIR / "standalone.toml"
STANDALONE_CONFIG = load_mapping(STANDALONE_CONFIG_PATH)
AMPLITUDE_TOY_PDF = STANDALONE_CONFIG["amplitude_toy"]["output_pdf"]
FIDUCIAL_SPECTRUM = STANDALONE_CONFIG["fiducial_spectrum"]
FIDUCIAL_SPECTRUM_NETWORKS = figure_networks(FIDUCIAL_SPECTRUM, "networks")
FIDUCIAL_SPECTRUM_EXPERIMENT = "experiments/{}.toml".format(
    FIDUCIAL_SPECTRUM["experiment"]
)
FIDUCIAL_SPECTRUM_OUTPUTS = _outputs(FIDUCIAL_SPECTRUM)
IMPORTANCE_GRID = STANDALONE_CONFIG["importance_weights_grid"]
IMPORTANCE_GRID_OUTPUTS = _outputs(IMPORTANCE_GRID)
STANDALONE_OUTPUTS = [
    AMPLITUDE_TOY_PDF,
    *FIDUCIAL_SPECTRUM_OUTPUTS,
    *IMPORTANCE_GRID_OUTPUTS,
]
FIGURE_OUTPUTS = {
    "H0-all-detectors": H0_DETECTOR_OUTPUTS,
    "H0-merger-rate": H0_RATE_OUTPUTS,
    "H0-omega-m": H0_OMEGA_OUTPUTS,
    "modified-propagation-all-detectors": PROPAGATION_OUTPUTS,
}

FIDUCIALS = BASE["fiducials"]
COSMOLOGY = BASE["cosmology"]
ANALYSIS = BASE["analysis"]
FIDUCIAL_CLI = (
    f"--h0 {FIDUCIALS['H0']} --omega-m {FIDUCIALS['Omega_m']}"
    f" --xi-0 {FIDUCIALS['xi_0']} --xi-n {FIDUCIALS['xi_n']}"
    f" --gamma {FIDUCIALS['gamma']} --kappa {FIDUCIALS['kappa']}"
    f" --z-peak {FIDUCIALS['z_peak']}"
    f" --local-merger-rate {FIDUCIALS['local_merger_rate']}"
)
ANALYSIS_CLI = (
    f"--observation-time {BASE['observation_time']}"
    f" --f-min {ANALYSIS['f_min']} --f-max {ANALYSIS['f_max']}"
    f" --z-min {COSMOLOGY['z_min']} --z-max {COSMOLOGY['z_max']}"
    f" --n-grid {COSMOLOGY['n_grid']}"
)
H0_DETECTOR_LABELS_CLI = _labels_cli(
    "--detector-labels", [network.label for network in H0_DETECTOR_NETWORKS]
)
H0_DETECTOR_NETWORK_CLI = _network_cli(H0_DETECTOR_NETWORKS)
H0_RATE_LABELS_CLI = _labels_cli("--prior-labels", H0_RATE_CONFIG["labels"])
H0_OMEGA_LABEL_CLI = _labels_cli("--omega-m-label", [H0_OMEGA_CONFIG["label"]])
PROPAGATION_MARGINAL_LABELS_CLI = _labels_cli(
    "--marginal-labels", PROPAGATION_CONFIG["marginal_labels"]
)
PROPAGATION_H0_LABELS_CLI = _labels_cli("--h0-labels", PROPAGATION_CONFIG["h0_labels"])
PROPAGATION_DETECTOR_LABELS_CLI = _labels_cli(
    "--detector-labels", [network.label for network in PROPAGATION_NETWORKS]
)
PROPAGATION_NETWORK_CLI = _network_cli(PROPAGATION_NETWORKS)
STANDALONE_NETWORK_CLI = _network_cli(FIDUCIAL_SPECTRUM_NETWORKS)
STANDALONE_LABELS_CLI = _labels_cli(
    "--detector-labels", [network.label for network in FIDUCIAL_SPECTRUM_NETWORKS]
)
EXPERIMENT_TARGET_INPUTS = [
    path
    for name in experiments
    for path in (FIGURE_OUTPUTS.get(name) or chain_paths(name))
]


wildcard_constraints:
    experiment=EXPERIMENT_PATTERN,
    run=RUN_PATTERN,


localrules:
    experiments,
    standalone_figures,
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
        EXPERIMENT_TARGET_INPUTS,


rule standalone_figures:
    input:
        STANDALONE_OUTPUTS,


for specification in experiments.values():

    rule:
        name: specification.chains_target
        localrule: True
        input:
            chain_paths(specification.name),

    rule:
        name: specification.target
        localrule: True
        input:
            FIGURE_OUTPUTS.get(specification.name) or chain_paths(specification.name),


rule assemble_config:
    input:
        base=str(BASE_CONFIG),
        experiment=run_config_path,
    output:
        config=CONFIG_PATTERN,
    shell:
        "uv run --package astrogwb-paper astrogwb-validate-config"
        " --base {input.base:q} --run {wildcards.run:q}"
        " {input.experiment:q} --output {output.config:q}"


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
        config="experiments/H0-all-detectors.toml",
        figure=_figure_path("H0-all-detectors"),
        network_labels=str(NETWORK_LABELS),
        base=str(BASE_CONFIG),
    output:
        pdf=H0_DETECTOR_CONFIG["output_pdf"],
        csv=H0_DETECTOR_CONFIG["output_csv"],
        tex=H0_DETECTOR_CONFIG["output_tex"],
    params:
        labels=_const(H0_DETECTOR_LABELS_CLI),
        networks=_const(H0_DETECTOR_NETWORK_CLI),
        fiducials=FIDUCIAL_CLI,
        analysis=ANALYSIS_CLI,
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python scripts/mcmc_cosmological_parameters.py"
        " --section detectors"
        " --catalog {input.catalog:q} --detector-chains {input.chains:q}"
        " {params.labels} {params.networks}"
        " --output-detector-pdf {output.pdf:q}"
        " --output-csv {output.csv:q} --output-tex {output.tex:q}"
        " {params.fiducials} {params.analysis}"


rule plot_H0_merger_rate:
    input:
        chains=H0_RATE_CHAINS,
        config="experiments/H0-merger-rate.toml",
        figure=_figure_path("H0-merger-rate"),
        base=str(BASE_CONFIG),
    output:
        prior_pdf=H0_RATE_CONFIG["output_prior_pdf"],
        corner_pdf=H0_RATE_CONFIG["output_corner_pdf"],
        csv=H0_RATE_CONFIG["output_csv"],
        tex=H0_RATE_CONFIG["output_tex"],
    params:
        labels=_const(H0_RATE_LABELS_CLI),
        fiducials=FIDUCIAL_CLI,
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python scripts/mcmc_cosmological_parameters.py"
        " --section merger-rate"
        " --prior-chains {input.chains:q} {params.labels}"
        " --output-prior-pdf {output.prior_pdf:q}"
        " --output-narrow-corner-pdf {output.corner_pdf:q}"
        " --output-csv {output.csv:q} --output-tex {output.tex:q}"
        " {params.fiducials}"


rule plot_H0_omega_m:
    input:
        chain=H0_OMEGA_CHAIN,
        config="experiments/H0-omega-m.toml",
        figure=_figure_path("H0-omega-m"),
        base=str(BASE_CONFIG),
    output:
        corner_pdf=H0_OMEGA_CONFIG["output_corner_pdf"],
        ess_corner_pdf=H0_OMEGA_CONFIG["output_ess_corner_pdf"],
    params:
        label=_const(H0_OMEGA_LABEL_CLI),
        fiducials=FIDUCIAL_CLI,
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python scripts/mcmc_cosmological_parameters.py"
        " --section omega-m"
        " --omega-m-chain {input.chain:q} {params.label}"
        " --output-omega-m-corner-pdf {output.corner_pdf:q}"
        " --output-omega-m-ess-corner-pdf {output.ess_corner_pdf:q}"
        " {params.fiducials}"


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
        config="experiments/modified-propagation-all-detectors.toml",
        figure=_figure_path("modified-propagation-all-detectors"),
        network_labels=str(NETWORK_LABELS),
        base=str(BASE_CONFIG),
    output:
        xi_n_corner_pdf=PROPAGATION_CONFIG["output_xi_n_corner_pdf"],
        xi_n_ess_corner_pdf=PROPAGATION_CONFIG["output_xi_n_ess_corner_pdf"],
        xi0_marginal_pdf=PROPAGATION_CONFIG["output_xi0_marginal_pdf"],
        h0_corner_pdf=PROPAGATION_CONFIG["output_h0_corner_pdf"],
        csv=PROPAGATION_CONFIG["output_csv"],
        tex=PROPAGATION_CONFIG["output_tex"],
    params:
        marginal_labels=_const(PROPAGATION_MARGINAL_LABELS_CLI),
        h0_labels=_const(PROPAGATION_H0_LABELS_CLI),
        detector_labels=_const(PROPAGATION_DETECTOR_LABELS_CLI),
        networks=_const(PROPAGATION_NETWORK_CLI),
        fiducials=FIDUCIAL_CLI,
        analysis=ANALYSIS_CLI,
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python scripts/mcmc_modified_propagation.py"
        " --xi0-chain {input.xi0_chain:q} --xi0-n-chain {input.xi0_n_chain:q}"
        " --h0-chain {input.h0_chain:q}"
        " --detector-xi0-n-chains {input.detector_chains:q}"
        " --catalog {input.catalog:q}"
        " {params.marginal_labels} {params.h0_labels} {params.detector_labels}"
        " {params.networks}"
        " --output-xi-n-corner-pdf {output.xi_n_corner_pdf:q}"
        " --output-xi-n-ess-corner-pdf {output.xi_n_ess_corner_pdf:q}"
        " --output-xi0-marginal-pdf {output.xi0_marginal_pdf:q}"
        " --output-h0-corner-pdf {output.h0_corner_pdf:q}"
        " --output-xi0-n-csv {output.csv:q} --output-xi0-n-tex {output.tex:q}"
        " {params.fiducials} {params.analysis}"


rule amplitude_toy:
    input:
        catalog=_catalog(DEFAULT_CATALOG.name),
        config=str(BASE_CONFIG),
    output:
        AMPLITUDE_TOY_PDF,
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python scripts/amplitude_toy_model.py"
        " --catalog {input.catalog:q}"
        " --chains-dir outputs/chains/amplitude-toy"
        " --observation-time {BASE[observation_time]}"
        " --f-min {ANALYSIS[f_min]} --f-max {ANALYSIS[f_max]}"
        " --output-pdf {output:q}"


rule fiducial_spectrum:
    input:
        catalog=_catalog(DEFAULT_CATALOG.name),
        config=str(STANDALONE_CONFIG_PATH),
        experiment_config=FIDUCIAL_SPECTRUM_EXPERIMENT,
        network_labels=str(NETWORK_LABELS),
        base=str(BASE_CONFIG),
    output:
        spectrum_pdf=FIDUCIAL_SPECTRUM["output_pdf"],
        effective_psd_pdf=FIDUCIAL_SPECTRUM["output_effective_psd_pdf"],
    params:
        networks=_const(STANDALONE_NETWORK_CLI),
        labels=_const(STANDALONE_LABELS_CLI),
        fiducials=FIDUCIAL_CLI,
        analysis=ANALYSIS_CLI,
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python scripts/fiducial_spectrum.py"
        " --catalog {input.catalog:q} {params.networks} {params.labels}"
        " {params.analysis} {params.fiducials}"
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
    params:
        fiducials=FIDUCIAL_CLI,
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python scripts/importance_weights_grid.py"
        " --catalog {input.catalog:q} {params.fiducials}"
        " --output-h0-omega-m-pdf {output.h0_omega_m_pdf:q}"
        " --output-xi0-n-pdf {output.xi0_n_pdf:q}"
