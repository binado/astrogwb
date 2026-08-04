from pathlib import Path

from astrogwb_paper.config.catalogs import load_catalog_recipe


RECIPE_DIR = Path("configs/catalogs")
_RECIPE_CACHE = {}


def recipe_path(wildcards):
    return RECIPE_DIR / f"{wildcards.catalog}.toml"


def recipe(wildcards):
    catalog = wildcards.catalog
    if catalog not in _RECIPE_CACHE:
        _RECIPE_CACHE[catalog] = load_catalog_recipe(recipe_path(wildcards))
    return _RECIPE_CACHE[catalog]


wildcard_constraints:
    catalog="[A-Za-z0-9][A-Za-z0-9._-]*",


localrules:
    bns_population,
    bns_waveform_catalog,


rule bns_population:
    input:
        recipe=recipe_path,
        population_config=lambda wc: recipe(wc).population_config,
    output:
        "out/populations/{catalog}.h5",
    params:
        n_samples=lambda wc: recipe(wc).n_samples,
        seed=lambda wc: recipe(wc).seed,
        outdir=lambda wc: str(Path("out/populations")),
    shell:
        "mkdir -p {params.outdir:q}\n"
        "uv run --package astrogwb-paper gwmock-pop simulate"
        " --config {input.population_config:q}"
        " --n {params.n_samples}"
        " --output {output:q}"
        " --seed {params.seed}"


rule bns_waveform_catalog:
    input:
        population="out/populations/{catalog}.h5",
        recipe=recipe_path,
    output:
        "out/catalogs/{catalog}.h5",
    params:
        approximant=lambda wc: recipe(wc).approximant,
        sampling_frequency=lambda wc: recipe(wc).sampling_frequency,
        minimum_frequency=lambda wc: recipe(wc).minimum_frequency,
        maximum_frequency=lambda wc: recipe(wc).maximum_frequency,
        reference_frequency=lambda wc: recipe(wc).reference_frequency,
        frequency_resolution=lambda wc: recipe(wc).frequency_resolution,
        chunk_size=lambda wc: recipe(wc).chunk_size,
    shell:
        "uv run --package astrogwb-paper astrogwb-generate-waveform-catalog"
        " --population {input.population:q}"
        " --output {output:q}"
        " --approximant {params.approximant:q}"
        " --sampling-frequency {params.sampling_frequency}"
        " --minimum-frequency {params.minimum_frequency}"
        " --maximum-frequency {params.maximum_frequency}"
        " --reference-frequency {params.reference_frequency}"
        " --frequency-resolution {params.frequency_resolution}"
        " --chunk-size {params.chunk_size}"
