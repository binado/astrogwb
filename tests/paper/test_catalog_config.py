"""The committed catalog configs, and the population declarations they name.

These files describe a catalog only until it exists. Afterwards the *file* is
authoritative -- it records its own model, construction kwargs,
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

from astrogwb.constants import ISCO_ALPHA
from astrogwb.paper.config.catalogs import (
    CatalogDefinition,
    check_population_model,
    discover_catalogs,
    load_catalog_layers,
)
from astrogwb.populations import (
    DEFAULT_DENSITY_SITES,
    Population,
    build_population,
    known_populations,
)
from astrogwb.waveform import AnalyticInspiralGenerator


def _definitions() -> dict[str, CatalogDefinition]:
    return discover_catalogs(REPO_ROOT)


def _build(population) -> Population:
    return build_population(population.model, **population.kwargs)


# --------------------------------------------------------------------------- #
# The committed declarations
# --------------------------------------------------------------------------- #
def test_every_committed_catalog_names_a_registered_population() -> None:
    """Caught pre-flight, not at the top of a queued GPU generation job."""
    for name, definition in _definitions().items():
        check_population_model(
            definition.population.model,
            label=f"catalog {name!r}",
            kwargs=definition.population.kwargs,
        )


def test_every_committed_catalog_can_build_its_population() -> None:
    """The construction kwargs and parameters must actually fit the population.

    A typo in ``population.kwargs`` is otherwise invisible until generation
    runs, and generation is the expensive step this pre-flight exists to
    protect. The kwargs mapping now reaches the population whole, so a key it
    does not take fails here rather than being filtered on its way to one of
    two separately built callables.
    """
    for name, definition in _definitions().items():
        population = definition.population
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(_build(population).source_model).get_trace(
                population.params
            )
        assert trace["redshift"]["type"] == "sample", name
        assert trace["luminosity_distance"]["type"] == "deterministic", name


def test_only_the_guarded_proposals_declare_no_merger_rate() -> None:
    """A guard mixture is a sampling density; every other catalog is physical.

    The mixture density is not normalized by the Madau-Dickinson total rate, so
    pairing the two -- which the old two-name record allowed, and every guarded
    def did -- recorded a rate that was never the one its samples imply.
    """
    for name, definition in _definitions().items():
        merger_rate_fn = _build(definition.population).merger_rate_fn
        expected_none = "uniform_mixture" in definition.population.model
        assert (merger_rate_fn is None) is expected_none, name


def test_every_declared_density_factor_is_a_real_sample_site() -> None:
    """Generation records ``DEFAULT_DENSITY_SITES``; each must be a sample site."""
    assert "redshift" in DEFAULT_DENSITY_SITES
    for name, definition in _definitions().items():
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(
                _build(definition.population).source_model
            ).get_trace(definition.population.params)
        for site in DEFAULT_DENSITY_SITES:
            assert trace[site]["type"] == "sample", name


def test_the_retired_population_graphs_are_gone() -> None:
    """Populations are model declarations now, not gwmock graph YAML."""
    assert not (REPO_ROOT / "config/populations").exists()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_an_unregistered_population_name_lists_the_known_set() -> None:
    with pytest.raises(ValueError) as error:
        check_population_model("no_such_population", label="catalog 'toy'")
    message = str(error.value)
    assert "catalog 'toy'" in message
    for name in known_populations():
        assert name in message


def test_a_kwarg_the_population_does_not_take_is_rejected() -> None:
    """The factory signature is the kwargs schema, so a stale key fails here."""
    with pytest.raises(ValueError, match="uniform_mixing_fraction"):
        check_population_model(
            "bns_md_cosmological",
            label="catalog 'toy'",
            kwargs={
                "minimum_redshift": 0.0,
                "maximum_redshift": 20.0,
                "n_grid": 256,
                "uniform_mixing_fraction": 0.1,
            },
        )


def test_a_proposal_density_is_rejected_as_an_analysis_target() -> None:
    """An analysis target reconstructs an observed rate, so it must declare one."""
    with pytest.raises(ValueError, match="declares no merger rate"):
        check_population_model(
            "bns_md_uniform_mixture",
            label="run 'toy' analysis.population_model",
            kwargs={
                "minimum_redshift": 0.3,
                "maximum_redshift": 20.0,
                "n_grid": 256,
                "uniform_mixing_fraction": 0.1,
            },
            requires_merger_rate=True,
        )


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
minimum_redshift = 20.0
maximum_redshift = 0.0
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
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="minimum_redshift must be less than"):
        load_catalog_layers([path])


_TOY_DEF = """
num_samples = 8
seed = 1

[population]
model = "bns_md_cosmological"

[population.kwargs]
minimum_redshift = 0.0
maximum_redshift = 20.0
n_grid = 256

[population.params]
H0 = 67.66

[waveform]
approximant = "{approximant}"
sampling_frequency = 128.0
minimum_frequency = 10.0
maximum_frequency = 50.0
reference_frequency = 20.0
frequency_resolution = 1.0
{alpha}
"""


def test_alpha_is_rejected_for_a_non_analytical_approximant(tmp_path: Path) -> None:
    """``alpha`` terminates the closed-form inspiral and means nothing to Ripple.

    Silently ignoring it would let a def look like it set a termination
    frequency that the generated catalog does not honour.
    """
    path = tmp_path / "toy.toml"
    path.write_text(
        _TOY_DEF.format(approximant="TaylorF2", alpha="alpha = 0.02"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="waveform.alpha is only valid"):
        load_catalog_layers([path])


def test_a_declared_alpha_reaches_the_analytical_generator(tmp_path: Path) -> None:
    path = tmp_path / "toy.toml"
    path.write_text(
        _TOY_DEF.format(approximant="analytical", alpha="alpha = 0.02"),
        encoding="utf-8",
    )
    generator = load_catalog_layers([path]).waveform.build()

    assert isinstance(generator, AnalyticInspiralGenerator)
    assert generator.alpha == 0.02


def test_an_omitted_alpha_defaults_to_isco(tmp_path: Path) -> None:
    path = tmp_path / "toy.toml"
    path.write_text(
        _TOY_DEF.format(approximant="analytical", alpha=""), encoding="utf-8"
    )
    generator = load_catalog_layers([path]).waveform.build()

    assert isinstance(generator, AnalyticInspiralGenerator)
    assert generator.alpha == ISCO_ALPHA


def test_no_catalog_layers_is_rejected() -> None:
    with pytest.raises(ValueError, match="no catalog config layers"):
        load_catalog_layers([])
