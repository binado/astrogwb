import re
import shlex
from pathlib import Path

from astrogwb.paper.config.runs import (
    BLOCK_FOLDS,
    catalog_config_paths,
    discover_catalog_names,
    discover_runs,
    load_base,
    resolve_catalog_names,
    run_config_paths,
)
from astrogwb.paper.plotting import DETECTOR_NETWORK_RUNS

# Neither config module imports JAX, pydantic, or matplotlib at module scope,
# so DAG construction stays cheap: a --dry-run costs ~300 modules rather than
# the ~1200 it took while `assemble_run` reached the pydantic layer.


JAX_PLATFORM = config.get("jax_platforms", "cuda")
CATALOGS_DIR = Path(config.get("catalogs_dir", "outputs/catalogs"))
CHAIN_PATTERN = "outputs/chains/{experiment}/{run}.nc"


def catalog_path(name: str) -> str:
    return str(CATALOGS_DIR / f"{name}.h5")


# Filenames are the mapping: config/catalogs/<name>.json ->
# outputs/catalogs/<name>.h5, and config/runs/<experiment>/<run>.json
# -> outputs/chains/<experiment>/<run>.nc. Nothing below translates a registry
# name into a path; it only globs the config tree and reads back the two names
# a run's own [analysis.catalog] block carries.
catalogs = discover_catalog_names()
runs = discover_runs()

CATALOG_OUTPUTS = [catalog_path(name) for name in catalogs]
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
CATALOG_PATTERN = "|".join(re.escape(name) for name in catalogs)
EXPERIMENT_PATTERN = "|".join(re.escape(name) for name in runs)
RUN_PATTERN = "|".join(
    re.escape(run) for run in dict.fromkeys(r for names in runs.values() for r in names)
)

# The figures read the same catalog files the runs sample against. Both names
# are read off the base catalog config so they cannot drift from what the runs
# actually use.
_BASE_CATALOGS = load_base()["analysis"]["catalog"]
INJECTION_CATALOG = catalog_path(_BASE_CATALOGS["injection"])
DEFAULT_PROPOSAL_CATALOG = catalog_path(_BASE_CATALOGS["proposal"])
# Figures report what was sampled, so each one is handed a run's own config
# layers for the shared fiducials and analysis settings. Which run that is used to
# be a constant buried in the library (config.figures.REFERENCE_RUN); it is an
# explicit choice here now. The two chain figures read the layers of a run they
# actually plot; the two chain-free figures fall back to this one.
FIGURE_RUN = ("cosmological-parameters", "ET-2L-aligned-CE-Hanford")


def config_layers(experiment, run):
    """The ordered layer files one run's config is merged from."""
    return [str(path) for path in run_config_paths(experiment, run, root=Path("."))]


def config_flags(experiment, run):
    """`config_layers` as repeated --config flags, in merge order.

    Built in `params:` rather than interpolated from `input:`: a list input
    expands to bare space-separated paths, which argparse's `action="append"`
    would read as one value plus stray positionals.

    Still how the figure scripts are handed a run's config: they want the
    merged mapping for reference values, not one flag per block, and merging in
    process keeps them off `jq`.
    """
    return " ".join(
        f"--config {shlex.quote(path)}" for path in config_layers(experiment, run)
    )


# `BLOCK_FOLDS` -- the jq program per block -- comes from the config layer, so
# the shell fold and the Python fold it must agree with are defined next to
# each other. Interpolated through `params:` rather than written inline, like
# `rule merge_catalog_config`: Snakemake formats the shell string, so a literal
# `{}` in a jq program would be read as a placeholder.
def block_flags(experiment, run):
    """`--<block> "$(jq ...)"` for every shared block, as one shell fragment.

    The layer files are the same list `input:` declares, so the dependency
    edges and the data path stay one list even though the blocks, not the
    paths, are what the script now reads.
    """
    layers = " ".join(shlex.quote(path) for path in config_layers(experiment, run))
    return " ".join(
        f"--{block} \"$(jq -s {shlex.quote(program)} {layers})\""
        for block, program in BLOCK_FOLDS.items()
    )


def network_run_flags(experiment):
    """--network-run flags in DETECTOR_NETWORKS order, which is legend order."""
    return " ".join(
        f"--network-run {shlex.quote(f'{experiment}/{run}')}"
        for run in DETECTOR_NETWORK_RUNS
    )


def network_config_inputs(experiment):
    """The run files behind `network_run_flags`, so the DAG edges are real.

    The script re-derives these paths from the run names it is given; declaring
    them here is what makes editing one network's config retrigger the figure.
    Taken from `run_config_paths`, whose last layer is the run's own file, so
    the path convention lives in one place.
    """
    return [
        str(run_config_paths(experiment, run, root=Path("."))[-1])
        for run in DETECTOR_NETWORK_RUNS
    ]


def catalog_layers(wildcards):
    """One catalog's ordered config layers, relative to this workflow's cwd.

    Declaring the shared layers alongside the def is what makes editing the
    common [waveform], [population] or [fiducials] block invalidate every
    catalog.
    """
    return [
        str(path) for path in catalog_config_paths(wildcards.catalog, root=Path("."))
    ]


def run_catalog_input(role):
    """The catalog file one role of a run samples against.

    The names are known only after the three-layer merge, so `run_mcmc` cannot
    declare its inputs without one. `resolve_catalog_names` is the library
    implementation; this returns the wildcards adapter for a single role, so
    the two files arrive as named inputs the shell block can reference.
    """

    def resolve(wildcards):
        names = resolve_catalog_names(
            wildcards.experiment, wildcards.run, root=Path(".")
        )
        return catalog_path(names[role])

    return resolve


def run_outdir(wildcards):
    return str(Path("outputs/chains") / wildcards.experiment)


def experiment_chains(experiment):
    return [f"outputs/chains/{experiment}/{run}.nc" for run in runs[experiment]]


wildcard_constraints:
    catalog=CATALOG_PATTERN,
    experiment=EXPERIMENT_PATTERN,
    run=RUN_PATTERN,


localrules:
    merge_catalog_config,
    waveform_catalog,
    validate,
    plot_cosmological_parameters,
    plot_modified_propagation,
    importance_weights_grid,
    catalogs,
    experiments,


rule merge_catalog_config:
    """One catalog's layers, folded once into the blocks the generator takes.

    A rule rather than a shell variable so the fold happens exactly once per
    catalog and is cached: `waveform_catalog` then reads five keys out of the
    result, instead of re-folding all four layer files once per flag. `jq`'s
    `*` is a recursive merge, which is `astrogwb.paper.utils.deep_merge`
    exactly; the catalog layers carry no `[priors]` block, so the shallow-merge
    rule the run path needs never applies here, and
    `tests/paper/test_runs.py` pins the two merges agreeing.

    `temp()` because this is a build intermediate, not an artifact: nothing
    downstream of the generated catalog reads it, and the `.h5` records its own
    provenance. The dependency edge on the layer files lives here now, and
    `waveform_catalog` inherits it transitively -- editing any shared layer
    still rebuilds every catalog.
    """
    input:
        catalog_layers,
    output:
        temp(str(CATALOGS_DIR / "{catalog}.merged.json")),
    params:
        merge="reduce .[] as $layer ({}; . * $layer)",
    shell:
        "jq -s '{params.merge}' {input:q} > {output:q}"


rule waveform_catalog:
    """Population draw + waveform generation, in one process.

    The generator is handed the merged blocks rather than a list of paths, so
    nothing re-reads the config tree downstream. A `jq` that failed would
    substitute an empty argument, which `generate_catalog.py` rejects as
    invalid JSON rather than acting on.
    """
    input:
        script="scripts/generate_catalog.py",
        merged=str(CATALOGS_DIR / "{catalog}.merged.json"),
    output:
        catalog_path("{catalog}"),
    shell:
        "uv run --extra paper python {input.script:q}"
        " --name {wildcards.catalog:q}"
        " --population \"$(jq -c .population {input.merged:q})\""
        " --fiducials \"$(jq -c .fiducials {input.merged:q})\""
        " --waveform \"$(jq -c .waveform {input.merged:q})\""
        " --seed \"$(jq -r .seed {input.merged:q})\""
        " --num-samples \"$(jq -r .num_samples {input.merged:q})\""
        " --output {output:q} --force"


rule catalogs:
    """Aggregate target: build every catalog declared in config/catalogs/."""
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
        sorted({str(path) for name in catalogs for path in catalog_config_paths(name, root=Path("."))}),
    output:
        "outputs/validated-runs.txt",
    shell:
        "uv run --extra paper python scripts/validate_configs.py"
        " --output {output:q}"


rule run_mcmc:
    """Sample one run into outputs/chains/<experiment>/<run>.nc."""
    input:
        script="scripts/run_mcmc.py",
        # The same layers `assemble_config` used to declare, so re-run
        # granularity is unchanged: edit a leaf -> one chain; edit
        # config/sampler.json -> all 27.
        config=lambda w: config_layers(w.experiment, w.run),
        injection=run_catalog_input("injection"),
        proposal=run_catalog_input("proposal"),
    output:
        chain=protected(CHAIN_PATTERN),
    params:
        platform=JAX_PLATFORM,
        outdir=run_outdir,
        block_flags=lambda w: block_flags(w.experiment, w.run),
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
            {params.block_flags} --outdir {params.outdir:q} \
            --label {wildcards.run:q} \
            --injection-catalog {input.injection:q} \
            --proposal-catalog {input.proposal:q} \
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
        catalog=INJECTION_CATALOG,
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


rule importance_weights_grid:
    """Relative-ESS heatmaps over the H0-Omega_m and Xi0-n prior grids."""
    input:
        catalog=DEFAULT_PROPOSAL_CATALOG,
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
        name: f"run_experiment_{_experiment.replace('-', '_')}"
        localrule: True
        input:
            experiment_chains(_experiment),
