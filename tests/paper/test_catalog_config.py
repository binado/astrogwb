"""The committed catalog configs, and the population declarations they name.

These files describe a catalog only until it exists. Afterwards the *file* is
authoritative -- it records its own model, construction settings,
hyperparameters and included density factors -- so nothing here is re-read at
analysis time and no run config restates any of it. What is left to check is
that every committed declaration can actually be built.

That replaced a much larger surface: a descriptor extracted from a gwmock graph
YAML, a mixture-of-proposals model, a derived proposal-density config, and an
exact-float-equality check reconciling a run's fiducials against five of the
eight parameters the catalog was drawn at.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from numpyro import handlers
from repo import REPO_ROOT

from astrogwb.paper.config.catalogs import (
    CatalogDefinition,
    check_rate_model,
    check_source_model,
    discover_catalogs,
    load_catalog_layers,
)
from astrogwb.populations import (
    Population,
    build_population,
    known_merger_rate_models,
    known_source_models,
)


def _definitions() -> dict[str, CatalogDefinition]:
    return discover_catalogs(REPO_ROOT)


def _build(population) -> Population:
    return build_population(
        source_model=population.source_model,
        rate_model=population.rate_model,
        settings=population.kwargs,
        source_kwargs=population.source_kwargs,
    )


# --------------------------------------------------------------------------- #
# The committed declarations
# --------------------------------------------------------------------------- #
def test_every_committed_catalog_names_a_registered_model() -> None:
    """Caught pre-flight, not at the top of a queued GPU generation job."""
    for name, definition in _definitions().items():
        check_source_model(
            definition.population.source_model, label=f"catalog {name!r}"
        )
        check_rate_model(definition.population.rate_model, label=f"catalog {name!r}")


def test_every_committed_catalog_can_build_its_population() -> None:
    """The construction settings and parameters must actually fit the model.

    A typo in ``population.kwargs`` is otherwise invisible until generation
    runs, and generation is the expensive step this pre-flight exists to
    protect.
    """
    for name, definition in _definitions().items():
        population = definition.population
        model = _build(population)
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model.source).get_trace(population.params)
        assert trace["redshift"]["type"] == "sample", name
        assert trace["luminosity_distance"]["type"] == "deterministic", name


def test_every_declared_density_factor_is_a_real_sample_site() -> None:
    for name, definition in _definitions().items():
        model = _build(definition.population)
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model.source).get_trace(definition.population.params)
        for site in model.source.density_sites:
            assert trace[site]["type"] == "sample", name
        assert "redshift" in model.source.density_sites, name


def test_the_retired_population_graphs_are_gone() -> None:
    """Populations are model declarations now, not gwmock graph YAML."""
    assert not (REPO_ROOT / "config/populations").exists()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_an_unregistered_source_model_name_lists_the_known_set() -> None:
    with pytest.raises(ValueError) as error:
        check_source_model("no_such_population", label="catalog 'toy'")
    message = str(error.value)
    assert "catalog 'toy'" in message
    for name in known_source_models():
        assert name in message


def test_an_unregistered_rate_model_name_lists_the_known_set() -> None:
    with pytest.raises(ValueError) as error:
        check_rate_model("no_such_rate", label="catalog 'toy'")
    message = str(error.value)
    assert "catalog 'toy'" in message
    for name in known_merger_rate_models():
        assert name in message


def test_layers_must_declare_a_population(tmp_path: Path) -> None:
    path = tmp_path / "toy.toml"
    path.write_text("num_samples = 8\nseed = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="population"):
        load_catalog_layers([path])


def test_an_inverted_redshift_window_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "toy.toml"
    path.write_text(
        """
num_samples = 8
seed = 1

[population]
source_model = "bns_md_cosmological"
rate_model = "madau_dickinson"

[population.kwargs]
z_min = 20.0
z_max = 0.0
n_grid = 256

[population.params]
H0 = 67.66

[waveform]
approximant = "Toy"
sampling_frequency = 128.0
minimum_frequency = 10.0
maximum_frequency = 50.0
reference_frequency = 20.0
frequency_resolution = 1.0
chunk_size = 4
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="z_min must be less than"):
        load_catalog_layers([path])


def test_no_catalog_layers_is_rejected() -> None:
    with pytest.raises(ValueError, match="no catalog config layers"):
        load_catalog_layers([])
