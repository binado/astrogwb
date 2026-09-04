import re
import shlex
from pathlib import Path

from astrogwb.paper.config.banks import discover_banks
from astrogwb.paper.config.runs import (
    base_config_paths,
    discover_runs,
    load_base,
    resolve_bank_names,
    run_config_paths,
    run_target,
)
from astrogwb.paper.plotting import DETECTOR_NETWORK_RUNS

# Neither config module imports JAX, pydantic, or matplotlib at module scope,
# so DAG construction stays cheap: a --dry-run costs ~300 modules rather than
# the ~1200 it took while `assemble_run` reached `config.banks`.


JAX_PLATFORM = config.get("jax_platforms", "cuda")
BANKS_DIR = Path(config.get("banks_dir", "outputs/banks"))
BANK_CONFIG_PATTERN = "config/banks/{bank}.toml"
CHAIN_PATTERN = "outputs/chains/{experiment}/{run}.nc"

# Filenames are the mapping: config/banks/<bank>.toml -> outputs/banks/<bank>.h5,
# config/analysis/runs/<experiment>/<run>.toml -> outputs/chains/<experiment>/<run>.nc.
# Nothing below translates a registry name into a path; it only globs the config
# tree and reads back what a run's own [catalog] block names.
banks = discover_banks()
runs = discover_runs()

BANK_OUTPUTS = [str(BANKS_DIR / f"{name}.h5") for name in banks]
CHAIN_OUTPUTS = [
    f"outputs/chains/{experiment}/{run}.nc"
    for experiment, names in runs.items()
    for run in names
]
RUN_CONFIG_FILES = sorted(
    {
        str(path)
        for experiment, names in runs.items()
        for run in names
        for path in run_config_paths(experiment, run, root=Path("."))
    }
)
BANK_PATTERN = "|".join(re.escape(name) for name in banks)
EXPERIMENT_PATTERN = "|".join(re.escape(name) for name in runs)
RUN_PATTERN = "|".join(
    re.escape(run) for run in dict.fromkeys(r for names in runs.values() for r in names)
)

# The figures read a bank file directly rather than a composed catalog: the
# shared injection composition requests every sample its bank holds, so the bank
# file *is* the composed catalog. Both names are read off the base catalog
# config so they cannot drift from what the runs actually sample.
_BASE_CATALOGS = load_base()["catalog"]
INJECTION_BANK = str(BANKS_DIR / f"{_BASE_CATALOGS['injection']['md_bank']}.h5")
DEFAULT_PROPOSAL_BANK = str(BANKS_DIR / f"{_BASE_CATALOGS['proposal']['md_bank']}.h5")
# Figures report what was sampled, so each one is handed a run's own config
# layers for the shared fiducials and analysis grid. Which run that is used to
# be a constant buried in the library (config.figures.REFERENCE_RUN); it is an
# explicit choice here now. The two chain figures read the layers of a run they
# actually plot; the three chain-free figures fall back to this one.
FIGURE_RUN = ("cosmological-parameters", "ET-2L-aligned-CE-Hanford")
BASE_CONFIGS = [str(path) for path in base_config_paths(Path("."))]


def config_layers(experiment, run):
    """The ordered layer files one run's config is merged from."""
    return [str(path) for path in run_config_paths(experiment, run, root=Path("."))]


def config_flags(experiment, run):
    """`config_layers` as repeated --config flags, in merge order.

    Built in `params:` rather than interpolated from `input:`: a list input
    expands to bare space-separated paths, which argparse's `action="append"`
    would read as one value plus stray positionals.
    """
    return " ".join(
        f"--config {shlex.quote(path)}" for path in config_layers(experiment, run)
    )


def network_run_flags(experiment):
    """--network-run flags in DETECTOR_NETWORKS order, which is legend order."""
    return " ".join(
        f"--network-run {shlex.quote(f'{experiment}/{run}')}"
        for run in DETECTOR_NETWORK_RUNS
    )


def network_config_inputs(experiment):
    """The run TOMLs behind `network_run_flags`, so the DAG edges are real.

    The script re-derives these paths from the run names it is given; declaring
    them here is what makes editing one network's TOML retrigger the figure.
    """
    return [
        f"config/analysis/runs/{experiment}/{run}.toml"
        for run in DETECTOR_NETWORK_RUNS
    ]


def bank_path(name):
    return str(BANKS_DIR / f"{name}.h5")


def bank_population(wildcards):
    """The population graph a bank config names, relative to this workflow's cwd."""
    return str(banks[wildcards.bank].population_path(Path(".")))


def run_bank_names(wildcards):
    """Every distinct bank a run's injection + proposal catalogs draw from.

    Bank names are known only after the three-layer merge, so `run_mcmc` cannot
    declare its inputs without one. `resolve_bank_names` is the library
    implementation; this is only the wildcards adapter.
    """
    return resolve_bank_names(wildcards.experiment, wildcards.run, root=Path("."))


def run_bank_inputs(wildcards):
    return [bank_path(name) for name in run_bank_names(wildcards)]


def run_bank_flags(wildcards):
    return " ".join(
        f"--bank {shlex.quote(f'{name}={bank_path(name)}')}"
        for name in run_bank_names(wildcards)
    )


def run_outdir(wildcards):
    return str(Path("outputs/chains") / wildcards.experiment)


def experiment_chains(experiment):
    return [f"outputs/chains/{experiment}/{run}.nc" for run in runs[experiment]]


wildcard_constraints:
    bank=BANK_PATTERN,
    experiment=EXPERIMENT_PATTERN,
    run=RUN_PATTERN,


localrules:
    waveform_bank,
    validate,
    plot_cosmological_parameters,
    plot_modified_propagation,
    amplitude_toy,
    fiducial_spectrum,
    importance_weights_grid,
    banks,
    experiments,


rule waveform_bank:
    """Population draw + waveform generation, in one process."""
    input:
        config=BANK_CONFIG_PATTERN,
        population=bank_population,
    output:
        str(BANKS_DIR / "{bank}.h5"),
    shell:
        "uv run --extra paper astrogwb-generate-bank"
        " --config {input.config:q} --output {output:q} --force"


rule banks:
    """Aggregate target: build every bank declared in config/banks/."""
    input:
        BANK_OUTPUTS,


# Replaces `assemble_config --all`, whose real value was failing on the first
# invalid run *before any bank was built* -- a bank is a GPU job. Deliberately
# not an input of `run_mcmc`: run it by hand before a campaign. `run_mcmc`
# re-checks its own run's banks anyway.
rule validate:
    """Pre-flight: merge, validate, and bank-check every run, building nothing."""
    input:
        RUN_CONFIG_FILES,
        [f"config/banks/{name}.toml" for name in banks],
    output:
        "outputs/validated-runs.txt",
    shell:
        "uv run --extra paper python scripts/validate_configs.py"
        " --output {output:q}"


rule run_mcmc:
    """Sample one run into outputs/chains/<experiment>/<run>.nc."""
    input:
        # The same three layers `assemble_config` used to declare, so re-run
        # granularity is unchanged: edit a leaf -> one chain; edit
        # base/sampling.toml -> all 26.
        config=lambda w: config_layers(w.experiment, w.run),
        banks=run_bank_inputs,
    output:
        chain=protected(CHAIN_PATTERN),
    params:
        platform=JAX_PLATFORM,
        outdir=run_outdir,
        bank_flags=run_bank_flags,
        config_flags=lambda w: config_flags(w.experiment, w.run),
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
            UV_EXTRAS="--extra paper"
        else
            RUNTIME_FLAGS="--cpu-threads {threads}"
            UV_EXTRAS="--extra paper --extra cuda"
        fi
        # --active --no-sync: with storage-local-copies the cwd is a
        # node-local staged copy holding only this job's declared inputs -- no
        # pyproject.toml -- so plain `uv run` has no project to resolve and
        # ignores VIRTUAL_ENV. The shared-FS venv is already synced (the GPU
        # profile requires --extra cuda), so run directly from it.
        #
        # The original reason was narrower and is now gone: astrogwb used to be
        # a uv *workspace source*, which did not survive staging. With one
        # non-editable package this may no longer be needed at all -- but that
        # can only be settled by a real submission, not locally, so it stays
        # until one is run. UV_EXTRAS is inert under --no-sync; it is kept so
        # the two branches still say which environment each platform wants.
        $NANNY uv run --active --no-sync $UV_EXTRAS astrogwb-run-mcmc \
            {params.config_flags} --outdir {params.outdir:q} \
            --label {wildcards.run:q} \
            {params.bank_flags} \
            --platform {params.platform:q} $RUNTIME_FLAGS --force
        """


rule plot_cosmological_parameters:
    """Cosmological-parameter section: five figures and two table pairs."""
    input:
        # Legend order comes from plotting.DETECTOR_NETWORKS, which the
        # script reads too -- one list, so chain and label order cannot drift.
        detector_chains=expand(
            "outputs/chains/cosmological-parameters/{run}.nc",
            run=DETECTOR_NETWORK_RUNS,
        ),
        # The fixed-R0 reference reuses the ET-2L-aligned-CE-Hanford chain,
        # whose run pins local_merger_rate to its fiducial; the comparison
        # chain samples local_merger_rate with H0 amplitude-marginalized.
        prior_chains=expand(
            "outputs/chains/cosmological-parameters/{run}.nc",
            run=["ET-2L-aligned-CE-Hanford", "H0-merger-rate"],
        ),
        omega_m_chain="outputs/chains/cosmological-parameters/H0-Omega_m.nc",
        catalog=INJECTION_BANK,
        # The layers of a run this figure actually plots, and the run TOMLs
        # behind --network-run.
        config=config_layers("cosmological-parameters", "ET-2L-aligned-CE-Hanford"),
        network_configs=network_config_inputs("cosmological-parameters"),
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
        "uv run --extra notebook"
        " python scripts/mcmc_cosmological_parameters.py"
        f" {config_flags('cosmological-parameters', 'ET-2L-aligned-CE-Hanford')}"
        f" {network_run_flags('cosmological-parameters')}"
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
    """Modified-propagation section: corners, marginal overlay, and tables."""
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
        catalog=INJECTION_BANK,
        config=config_layers("modified-propagation", "ET-2L-aligned-CE-Hanford"),
        network_configs=network_config_inputs("modified-propagation"),
    output:
        xi_n_corner_pdf="outputs/figures/modified-propagation/Xi0-n-corner.pdf",
        xi_n_ess_corner_pdf="outputs/figures/modified-propagation/Xi0-n-ess-corner.pdf",
        xi0_marginal_pdf="outputs/figures/modified-propagation/Xi0-marginal.pdf",
        h0_corner_pdf="outputs/figures/modified-propagation/Xi0-H0-corner.pdf",
        csv="outputs/figures/modified-propagation/Xi0-n-by-detector.csv",
        tex="outputs/figures/modified-propagation/Xi0-n-by-detector.tex",
    shell:
        "uv run --extra notebook"
        " python scripts/mcmc_modified_propagation.py"
        f" {config_flags('modified-propagation', 'ET-2L-aligned-CE-Hanford')}"
        f" {network_run_flags('modified-propagation')}"
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
    """Single-amplitude toy MCMC and its Fisher-overlay figure."""
    input:
        catalog=INJECTION_BANK,
        config=config_layers(*FIGURE_RUN),
    output:
        "outputs/figures/standalone/amplitude_toy_fisher_overlay.pdf",
    shell:
        "uv run --extra notebook"
        " python scripts/amplitude_toy_model.py"
        f" {config_flags(*FIGURE_RUN)}"
        " --catalog {input.catalog:q}"
        " --chains-dir outputs/chains/amplitude-toy"
        " --output-pdf {output:q}"


rule fiducial_spectrum:
    """Fiducial injection spectrum and per-network effective PSDs."""
    input:
        catalog=INJECTION_BANK,
        # Borrows the cosmological-parameters networks; reads no chains. The
        # network TOMLs are declared even though no chain is, so editing one
        # network's detector list retriggers this figure -- which it did not do
        # while the figure resolved everything from one assembled config.
        config=config_layers(*FIGURE_RUN),
        network_configs=network_config_inputs("cosmological-parameters"),
    output:
        spectrum_pdf="outputs/figures/standalone/fiducial_spectrum.pdf",
        effective_psd_pdf=(
            "outputs/figures/standalone/fiducial_effective_psd_by_detector.pdf"
        ),
    shell:
        "uv run --extra notebook"
        " python scripts/fiducial_spectrum.py"
        f" {config_flags(*FIGURE_RUN)}"
        f" {network_run_flags('cosmological-parameters')}"
        " --catalog {input.catalog:q}"
        " --output-pdf {output.spectrum_pdf:q}"
        " --output-effective-psd-pdf {output.effective_psd_pdf:q}"


rule importance_weights_grid:
    """Relative-ESS heatmaps over the H0-Omega_m and Xi0-n prior grids."""
    input:
        catalog=DEFAULT_PROPOSAL_BANK,
        config=config_layers(*FIGURE_RUN),
    output:
        h0_omega_m_pdf=(
            "outputs/figures/standalone/importance_weights_grid_H0_Omega_m.pdf"
        ),
        xi0_n_pdf="outputs/figures/standalone/importance_weights_grid_Xi0_n.pdf",
    shell:
        "uv run --extra notebook"
        " python scripts/importance_weights_grid.py"
        f" {config_flags(*FIGURE_RUN)}"
        " --catalog {input.catalog:q}"
        " --output-h0-omega-m-pdf {output.h0_omega_m_pdf:q}"
        " --output-xi0-n-pdf {output.xi0_n_pdf:q}"


rule experiments:
    """Aggregate target: every chain of every experiment, no figures."""
    input:
        CHAIN_OUTPUTS,


for _experiment in runs:

    rule:
        """Aggregate target: every chain of one experiment."""
        name: run_target(_experiment)
        localrule: True
        input:
            experiment_chains(_experiment),
