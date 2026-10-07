import re
import shlex
from pathlib import Path

from astrogwb.paper.config.catalogs import resolve_run_catalogs
from astrogwb.paper.config.runs import (
    DETECTOR_DEFAULT_PATHS,
    discover_runs,
    run_config_paths,
)
from astrogwb.paper.plotting import DETECTOR_NETWORK_RUNS

# Keying the catalogs reaches pydantic: the key is taken over a validated
# CatalogMetadata, which is the price of one canonical form. Importing the
# records imports JAX (their parent packages do) but does not initialize the
# XLA backend; `catalog_stem` names the file from the metadata and
# the seed alone.


JAX_PLATFORM = config.get("jax_platforms", "cuda")
CATALOGS_DIR = Path(config.get("catalogs_dir", "outputs/catalogs"))
CHAIN_PATTERN = "outputs/chains/{experiment}/{run}.nc"


def catalog_path(stem: str) -> str:
    """The file a catalog stem names, in the catalog directory."""
    return str(CATALOGS_DIR / f"{stem}.h5")


#: `catalog_path` with the stem left as the rule's wildcard; `generate_catalog`
#: refuses any output that is not named by the metadata and seed it is given.
CATALOG_PATTERN = str(CATALOGS_DIR / "{catalog}.h5")


# Filenames are the mapping for runs: config/runs/<experiment>/<run>.toml ->
# outputs/chains/<experiment>/<run>.nc. Catalogs are content-addressed instead:
# each run's [analysis.injection] / [analysis.proposal] resolve to a
# CatalogMetadata, and that record plus the seed it is drawn at (analysis.seeds)
# names outputs/catalogs/polarization_power-<key>-<seed>.h5. Two runs asking
# for the same draw share one file, and any edit to a draw -- or a bump of the
# astrogwb version -- names a new one, so the catalog rule needs no config
# inputs to rebuild correctly.
runs = discover_runs()
run_catalogs = resolve_run_catalogs()

CATALOG_OUTPUTS = [catalog_path(key) for key in run_catalogs.requests]
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
EXPERIMENT_PATTERN = "|".join(re.escape(name) for name in runs)
RUN_PATTERN = "|".join(
    re.escape(run) for run in dict.fromkeys(r for names in runs.values() for r in names)
)

# The figures read the same catalog files a run samples against, resolved off
# that run's own config so they cannot drift from what it actually used.
FIGURE_RUN = ("cosmological-parameters", "ET-2L-aligned-CE-Hanford")
INJECTION_CATALOG = catalog_path(run_catalogs.by_run[FIGURE_RUN]["injection"])
DEFAULT_PROPOSAL_CATALOG = catalog_path(run_catalogs.by_run[FIGURE_RUN]["proposal"])
# Figures report what was sampled, so each one is handed a run's own config
# layers for the shared fiducials and analysis settings. Which run that is used to
# be a constant buried in the library (config.figures.REFERENCE_RUN); it is an
# explicit choice here now (FIGURE_RUN, above). The two chain figures read the
# layers of a run they actually plot; the two chain-free figures fall back to
# this one.


def config_layers(experiment, run):
    """The ordered layer files one run's config is merged from."""
    return [str(path) for path in run_config_paths(experiment, run, root=Path("."))]


def config_flags(experiment, run):
    """`config_layers` as repeated --config flags, in merge order.

    Built in `params:` rather than interpolated from `input:`: a list input
    expands to bare space-separated paths, which argparse's `action="append"`
    would read as one value plus stray positionals.

    How every entrypoint that reads a run's config is handed it, `run_mcmc`
    included: each merges the layers in process, through the one fold in
    `astrogwb.paper.config.runs.merge_config_layers`.
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


#: Packaged geometry and sensitivity. Every run merges these before its four
#: layers, so an edit retriggers every chain.
DETECTOR_INPUTS = [str(path) for path in DETECTOR_DEFAULT_PATHS]


def run_catalog_input(role):
    """The catalog file one role of a run samples against."""

    def resolve(wildcards):
        roles = run_catalogs.by_run[(wildcards.experiment, wildcards.run)]
        return catalog_path(roles[role])

    return resolve


def catalog_request(wildcards):
    """The metadata behind one catalog stem, as the JSON the generator takes."""
    return run_catalogs.requests[wildcards.catalog][0].model_dump_json()


def catalog_seed(wildcards):
    """The seed behind one catalog stem."""
    return int(run_catalogs.requests[wildcards.catalog][1])


def run_outdir(wildcards):
    return str(Path("outputs/chains") / wildcards.experiment)


def experiment_chains(experiment):
    return [f"outputs/chains/{experiment}/{run}.nc" for run in runs[experiment]]


wildcard_constraints:
    catalog="polarization_power-[0-9a-f]{16}-[0-9]+",
    experiment=EXPERIMENT_PATTERN,
    run=RUN_PATTERN,


localrules:
    waveform_catalog,
    validate,
    plot_cosmological_parameters,
    plot_modified_propagation,
    importance_weights_grid,
    catalogs,
    experiments,


rule waveform_catalog:
    """Population draw + waveform generation, in one process.

    The output path is the metadata's key and the seed, so the rule
    declares no config inputs: an edit that changes what a run asks for changes
    the path, and with it the file, rather than invalidating this one. The
    generator re-derives the path from the metadata and seed it is handed and
    refuses an output that disagrees.
    """
    input:
        script="scripts/generate_catalog.py",
    output:
        CATALOG_PATTERN,
    params:
        request=catalog_request,
        seed=catalog_seed,
    shell:
        "uv run --extra paper python {input.script:q}"
        " --request {params.request:q} --seed {params.seed}"
        " --output {output:q} --force"


rule catalogs:
    """Aggregate target: build every catalog any committed run asks for."""
    input:
        CATALOG_OUTPUTS,


# Replaces `assemble_config --all`, whose real value was failing on the first
# invalid run *before any catalog was built* -- a catalog is a GPU job.
# Deliberately not an input of `run_mcmc`: run it by hand before a campaign.
# `run_mcmc` re-checks its own run's catalogs anyway.
rule validate:
    """Pre-flight: merge, validate, and catalog-check every run, building nothing."""
    input:
        RUN_CONFIG_FILES,
        DETECTOR_INPUTS,
    output:
        "outputs/validated-runs.txt",
    shell:
        "uv run --extra paper python scripts/validate_configs.py"
        " --output {output:q}"


rule run_mcmc:
    """Sample one run into outputs/chains/<experiment>/<run>.nc."""
    input:
        script="scripts/run_mcmc.py",
        # The run's six layers: edit a leaf -> one chain; edit an
        # experiment's _base.toml -> that experiment; edit
        # any shared config layer -> all 27. The catalogs are named by key, so an
        # edit to a draw reaches the chain through a new catalog path too.
        config=lambda w: config_layers(w.experiment, w.run),
        psds=DETECTOR_INPUTS,
        injection=run_catalog_input("injection"),
        proposal=run_catalog_input("proposal"),
    output:
        chain=protected(CHAIN_PATTERN),
    params:
        platform=JAX_PLATFORM,
        outdir=run_outdir,
        config_flags=lambda w: config_flags(w.experiment, w.run),
        # The job looks its catalogs up by key in the cache the two inputs
        # above live in, and --cached-only makes a miss an error rather than
        # a generation: catalogs are built by `waveform_catalog`, upstream.
        catalog_dir=str(CATALOGS_DIR),
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
        $NANNY uv run --active --no-sync $UV_EXTRAS python {input.script:q} \
            {params.config_flags} --outdir {params.outdir:q} \
            --label {wildcards.run:q} \
            --catalog-dir {params.catalog_dir:q} --cached-only \
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
        catalog=INJECTION_CATALOG,
        # The layers of a run this figure actually plots. Each network run's
        # own file is an input of its chain, which this rule already requires.
        config=config_layers("cosmological-parameters", "ET-2L-aligned-CE-Hanford"),
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
        catalog=INJECTION_CATALOG,
        config=config_layers("modified-propagation", "ET-2L-aligned-CE-Hanford"),
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


rule importance_weights_grid:
    """Relative-ESS heatmaps over the H0-Omega_m and Xi0-n prior grids."""
    input:
        catalog=DEFAULT_PROPOSAL_CATALOG,
        config=config_layers(*FIGURE_RUN),
        psds=DETECTOR_INPUTS,
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
        name: f"run_experiment_{_experiment.replace('-', '_')}"
        localrule: True
        input:
            experiment_chains(_experiment),
