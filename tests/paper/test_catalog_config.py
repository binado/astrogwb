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
    check_population_model,
    discover_catalogs,
    load_catalog_layers,
)
from astrogwb.populations import (
    known_population_models,
    population_model,
)


def _definitions() -> dict[str, CatalogDefinition]:
    return discover_catalogs(REPO_ROOT)


# --------------------------------------------------------------------------- #
# The committed declarations
# --------------------------------------------------------------------------- #
def test_every_committed_catalog_names_a_registered_model() -> None:
    """Caught pre-flight, not at the top of a queued GPU generation job."""
    for name, definition in _definitions().items():
        check_population_model(definition.population.model, label=f"catalog {name!r}")


def test_every_committed_catalog_can_build_its_population() -> None:
    """The construction settings and parameters must actually fit the model.

    A typo in ``population.kwargs`` is otherwise invisible until generation
    runs, and generation is the expensive step this pre-flight exists to
    protect.
    """
    for name, definition in _definitions().items():
        population = definition.population
        model = population_model(population.model)(**population.kwargs)
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model).get_trace(population.params)
        assert trace["redshift"]["type"] == "sample", name
        assert trace["luminosity_distance"]["type"] == "deterministic", name


def test_every_declared_density_factor_is_a_real_sample_site() -> None:
    for name, definition in _definitions().items():
        model = population_model(definition.population.model)(
            **definition.population.kwargs
        )
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model).get_trace(definition.population.params)
        for site in model.density_sites:
            assert trace[site]["type"] == "sample", name
        assert "redshift" in model.density_sites, name


def test_every_catalog_shares_one_generation_support() -> None:
    """A proposal narrowed to the analysis window must contain it.

    The window is shared by every run, so a catalog generated over a smaller
    span could not serve as a proposal for it -- and would only say so once the
    file existed.
    """
    windows = {
        (definition.population.kwargs["z_min"], definition.population.kwargs["z_max"])
        for definition in _definitions().values()
    }
    assert windows == {(0.0, 20.0)}


def test_only_the_guard_catalogs_use_the_mixture_population() -> None:
    """The guard fraction is the population's, not a post-hoc blend of draws."""
    definitions = _definitions()
    mixtures = {
        name
        for name, definition in definitions.items()
        if definition.population.model == "bns_md_uniform_mixture"
    }
    assert mixtures == {
        "md-uniform-imrphenom-s61-n16384-eps1e-1",
        "md-uniform-imrphenom-s62-n16384-eps1e-2",
        "md-uniform-imrphenom-s63-n16384-eps1e-3",
    }
    fractions = {
        name: definitions[name].population.kwargs["uniform_mixing_fraction"]
        for name in mixtures
    }
    assert set(fractions.values()) == {0.1, 0.01, 0.001}
    for name, fraction in fractions.items():
        # The eps in the filename is the eps in the config.
        assert name.endswith(f"eps{fraction:.0e}".replace("e-0", "e-")), name


def test_the_three_md_s42_catalogs_are_one_nested_series() -> None:
    """Same population, same seed, three sizes: prefixes of one draw.

    ``Predictive`` allocates per-draw keys with ``jax.random.split``, which is
    prefix-stable, so the sizes are nested rather than unrelated. That is what
    makes ``variable-catalog-size`` a clean series; the RNG property itself is
    pinned in ``tests/core/test_populations.py``.
    """
    definitions = _definitions()
    series = {
        name: definition
        for name, definition in definitions.items()
        if name.startswith("md-imrphenom-s42-")
    }
    assert len(series) == 3
    assert {d.seed for d in series.values()} == {42}
    assert {d.population.model for d in series.values()} == {"bns_md_cosmological"}
    assert {d.num_samples for d in series.values()} == {8192, 16384, 32768}


def test_the_shared_layers_are_declared_once() -> None:
    """Editing a base layer must invalidate every catalog, so it is shared."""
    base = sorted(path.name for path in (REPO_ROOT / "config/catalogs/base").glob("*"))
    assert base == ["population.toml", "waveform.toml"]

    definition = load_catalog_layers(
        [
            REPO_ROOT / "config/catalogs/base/population.toml",
            REPO_ROOT / "config/catalogs/base/waveform.toml",
            REPO_ROOT / "config/catalogs/defs/md-imrphenom-s41-n32768.toml",
        ]
    )
    assert definition.name == "md-imrphenom-s41-n32768"
    assert definition.waveform.approximant == "IMRPhenomXAS_NRTidalv3"
    assert definition.population.model == "bns_md_cosmological"


def test_the_retired_population_graphs_are_gone() -> None:
    """Populations are model declarations now, not gwmock graph YAML."""
    assert not (REPO_ROOT / "config/populations").exists()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_an_unregistered_model_name_lists_the_known_set() -> None:
    with pytest.raises(ValueError) as error:
        check_population_model("no_such_population", label="catalog 'toy'")
    message = str(error.value)
    assert "catalog 'toy'" in message
    for name in known_population_models():
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
model = "bns_md_cosmological"

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
