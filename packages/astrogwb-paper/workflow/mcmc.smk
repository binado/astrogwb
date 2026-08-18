import re
from pathlib import Path

from astrogwb_paper.config.experiments import (
    DEFAULT_CATALOG,
    chain_paths,
    experiment,
    load_experiments,
    merged_config_path,
)
from astrogwb_paper.plotting import DETECTOR_NETWORK_RUNS


SOURCE_CONFIG = Path("inputs/config.yaml")
JAX_PLATFORM = config.get("jax_platforms", "cuda")
CATALOGS_DIR = Path(config.get("catalogs_dir", "outputs/catalogs"))

CONFIG_PATTERN = "outputs/configs/{experiment}/{run}.json"
CHAIN_PATTERN = "outputs/chains/{experiment}/{run}.nc"
SIDECAR_PATTERN = "outputs/chains/{experiment}/{run}.json"

experiments = load_experiments()
CONFIG_OUTPUTS = [
    str(merged_config_path(specification.name, run))
    for specification in experiments.values()
    for run in specification.runs
]
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


def run_catalog_path(wildcards):
    catalog = _run_spec(wildcards).catalog_for(wildcards.run)
    return str(CATALOGS_DIR / catalog.name)


def run_outdir(wildcards):
    return str(Path("outputs/chains") / wildcards.experiment)


def _catalog(name):
    return str(CATALOGS_DIR / name)


wildcard_constraints:
    experiment=EXPERIMENT_PATTERN,
    run=RUN_PATTERN,


localrules:
    experiments,
    assemble_config,
    plot_cosmological_parameters,
    plot_modified_propagation,
    amplitude_toy,
    fiducial_spectrum,
    importance_weights_grid,


rule assemble_config:
    input:
        config=str(SOURCE_CONFIG),
    output:
        CONFIG_OUTPUTS,
    shell:
        "uv run --package astrogwb-paper astrogwb-validate-config"
        " {input.config:q} --output-dir outputs/configs"


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


rule plot_cosmological_parameters:
    input:
        # Legend order comes from plotting.DETECTOR_NETWORKS, which the
        # script reads too -- one list, so chain and label order cannot drift.
        detector_chains=expand(
            "outputs/chains/cosmological-parameters/{run}.nc",
            run=DETECTOR_NETWORK_RUNS,
        ),
        prior_chains=expand(
            "outputs/chains/cosmological-parameters/{run}.nc",
            run=["fixed", "sampled"],
        ),
        omega_m_chain="outputs/chains/cosmological-parameters/H0-Omega_m.nc",
        catalog=_catalog(DEFAULT_CATALOG.name),
        config=str(SOURCE_CONFIG),
    output:
        detector_pdf="outputs/figures/cosmological-parameters/H0-by-detector.pdf",
        detector_csv="outputs/figures/cosmological-parameters/H0-by-detector.csv",
        detector_tex="outputs/figures/cosmological-parameters/H0-by-detector.tex",
        prior_pdf="outputs/figures/cosmological-parameters/H0-merger-rate-priors.pdf",
        merger_rate_corner_pdf=(
            "outputs/figures/cosmological-parameters/H0-merger-rate-corner.pdf"
        ),
        merger_rate_csv="outputs/figures/cosmological-parameters/H0-merger-rate.csv",
        merger_rate_tex="outputs/figures/cosmological-parameters/H0-merger-rate.tex",
        omega_m_corner_pdf="outputs/figures/cosmological-parameters/H0-Omega_m-corner.pdf",
        omega_m_ess_corner_pdf=(
            "outputs/figures/cosmological-parameters/H0-Omega_m-ess-corner.pdf"
        ),
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python scripts/mcmc_cosmological_parameters.py"
        " --base-config {input.config:q}"
        " --catalog {input.catalog:q}"
        " --detector-chains {input.detector_chains:q}"
        " --prior-chains {input.prior_chains:q}"
        " --omega-m-chain {input.omega_m_chain:q}"
        " --output-detector-pdf {output.detector_pdf:q}"
        " --output-detector-csv {output.detector_csv:q}"
        " --output-detector-tex {output.detector_tex:q}"
        " --output-prior-pdf {output.prior_pdf:q}"
        " --output-narrow-corner-pdf {output.merger_rate_corner_pdf:q}"
        " --output-merger-rate-csv {output.merger_rate_csv:q}"
        " --output-merger-rate-tex {output.merger_rate_tex:q}"
        " --output-omega-m-corner-pdf {output.omega_m_corner_pdf:q}"
        " --output-omega-m-ess-corner-pdf {output.omega_m_ess_corner_pdf:q}"


rule plot_modified_propagation:
    input:
        xi0_chain="outputs/chains/modified-propagation/Xi_0.nc",
        xi0_n_chain=(
            "outputs/chains/modified-propagation"
            "/ET-2L-aligned-CE-Hanford.nc"
        ),
        h0_chain="outputs/chains/modified-propagation/Xi_0-H0.nc",
        detector_chains=expand(
            "outputs/chains/modified-propagation/{run}.nc",
            run=DETECTOR_NETWORK_RUNS,
        ),
        catalog=_catalog(DEFAULT_CATALOG.name),
        config=str(SOURCE_CONFIG),
    output:
        xi_n_corner_pdf="outputs/figures/modified-propagation/Xi0-n-corner.pdf",
        xi_n_ess_corner_pdf="outputs/figures/modified-propagation/Xi0-n-ess-corner.pdf",
        xi0_marginal_pdf="outputs/figures/modified-propagation/Xi0-marginal.pdf",
        h0_corner_pdf="outputs/figures/modified-propagation/Xi0-H0-corner.pdf",
        csv="outputs/figures/modified-propagation/Xi0-n-by-detector.csv",
        tex="outputs/figures/modified-propagation/Xi0-n-by-detector.tex",
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python scripts/mcmc_modified_propagation.py"
        " --base-config {input.config:q}"
        " --xi0-chain {input.xi0_chain:q} --xi0-n-chain {input.xi0_n_chain:q}"
        " --h0-chain {input.h0_chain:q}"
        " --detector-xi0-n-chains {input.detector_chains:q}"
        " --catalog {input.catalog:q}"
        " --output-xi-n-corner-pdf {output.xi_n_corner_pdf:q}"
        " --output-xi-n-ess-corner-pdf {output.xi_n_ess_corner_pdf:q}"
        " --output-xi0-marginal-pdf {output.xi0_marginal_pdf:q}"
        " --output-h0-corner-pdf {output.h0_corner_pdf:q}"
        " --output-xi0-n-csv {output.csv:q} --output-xi0-n-tex {output.tex:q}"


rule amplitude_toy:
    input:
        catalog=_catalog(DEFAULT_CATALOG.name),
        config=str(SOURCE_CONFIG),
    output:
        "outputs/figures/standalone/amplitude_toy_fisher_overlay.pdf",
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python scripts/amplitude_toy_model.py"
        " --base-config {input.config:q}"
        " --catalog {input.catalog:q}"
        " --chains-dir outputs/chains/amplitude-toy"
        " --output-pdf {output:q}"


rule fiducial_spectrum:
    input:
        catalog=_catalog(DEFAULT_CATALOG.name),
        # Borrows the cosmological-parameters networks; reads no chains.
        config=str(SOURCE_CONFIG),
    output:
        spectrum_pdf="outputs/figures/standalone/fiducial_spectrum.pdf",
        effective_psd_pdf=(
            "outputs/figures/standalone/fiducial_effective_psd_by_detector.pdf"
        ),
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python scripts/fiducial_spectrum.py"
        " --base-config {input.config:q}"
        " --catalog {input.catalog:q}"
        " --output-pdf {output.spectrum_pdf:q}"
        " --output-effective-psd-pdf {output.effective_psd_pdf:q}"


rule importance_weights_grid:
    input:
        catalog=_catalog(DEFAULT_CATALOG.name),
        config=str(SOURCE_CONFIG),
    output:
        h0_omega_m_pdf=(
            "outputs/figures/standalone/importance_weights_grid_H0_Omega_m.pdf"
        ),
        xi0_n_pdf="outputs/figures/standalone/importance_weights_grid_Xi0_n.pdf",
    shell:
        "uv run --package astrogwb-paper --group plotting"
        " python scripts/importance_weights_grid.py"
        " --base-config {input.config:q}"
        " --catalog {input.catalog:q}"
        " --output-h0-omega-m-pdf {output.h0_omega_m_pdf:q}"
        " --output-xi0-n-pdf {output.xi0_n_pdf:q}"


# The cosmological experiment feeds one paper section. Its chains-only target
# remains separate from the complete plotting target.
COSMOLOGICAL_PARAMETER_EXPERIMENT = "cosmological-parameters"
FIGURE_OUTPUTS = {
    "modified-propagation": list(rules.plot_modified_propagation.output),
}
EXPERIMENT_OUTPUTS = [
    *rules.plot_cosmological_parameters.output,
    *[
        path
        for name in experiments
        if name != COSMOLOGICAL_PARAMETER_EXPERIMENT
        for path in (FIGURE_OUTPUTS.get(name) or chain_paths(name))
    ],
]


rule experiments:
    input:
        EXPERIMENT_OUTPUTS,


for specification in experiments.values():

    rule:
        name: specification.chains_target
        localrule: True
        input:
            chain_paths(specification.name),

    if specification.name == COSMOLOGICAL_PARAMETER_EXPERIMENT:
        continue

    rule:
        name: specification.target
        localrule: True
        input:
            FIGURE_OUTPUTS.get(specification.name) or chain_paths(specification.name),
