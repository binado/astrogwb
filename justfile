# Operator shortcuts for catalog / MCMC / paper Snakemake workflows.
# Scientific config stays in configs/; resources stay in profiles/.
#
# Snakemake recipes default to --dry-run. Opt into a real run:
#   just dry=0 mcmc-slurm configs/mcmc/manifests/mcmc.batch.cosmology.json
#   just dry=0 paper

snakemake := "uv run snakemake"
catalog_smk := "workflow/catalog.smk"
mcmc_smk := "workflow/mcmc.smk"
paper_smk := "workflow/paper.smk"

# Default dry=1 (pass --dry-run). Set dry=0 for a real run/submit.
dry := "1"
dry_run := if dry == "0" { "" } else { "--dry-run" }

# List recipes: just --list

# Generate sweep JSON configs and gitignored batch manifests.
gen-configs *args:
    uv run --extra mcmc python scripts/generate_mcmc_configs.py --write-manifests {{args}}

# Build a catalog target (population first if stale), e.g.:
#   just dry=0 catalog out/catalogs/bns-n16384-df1.h5
catalog target *args:
    {{snakemake}} --snakefile {{catalog_smk}} --cores 1 {{dry_run}} {{target}} {{args}}

# --- MCMC (profile stays explicit) ---

#   just mcmc profiles/slurm configs/mcmc/manifests/mcmc.batch.cosmology.json
#   just dry=0 mcmc profiles/slurm configs/mcmc/manifests/mcmc.batch.cosmology.json
mcmc profile configfile *args:
    {{snakemake}} --snakefile {{mcmc_smk}} --profile {{profile}} \
        --configfile {{configfile}} {{dry_run}} mcmc {{args}}

mcmc-local configfile cores="8" *args:
    just dry={{dry}} mcmc profiles/local {{configfile}} --cores {{cores}} {{args}}

mcmc-slurm configfile *args:
    just dry={{dry}} mcmc profiles/slurm {{configfile}} {{args}}

mcmc-slurm-cpu configfile *args:
    just dry={{dry}} mcmc profiles/slurm-cpu {{configfile}} {{args}}

# --- Paper ---

# Default dry-run; build with dry=0:
#   just paper
#   just dry=0 paper
#   just dry=0 paper figures/fiducial_spectrum.pdf
paper *target:
    {{snakemake}} --snakefile {{paper_smk}} --cores 1 {{dry_run}} \
        {{ if target == "" { "paper_figures" } else { target } }}
