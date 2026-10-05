"""The catalogs the committed runs ask for, and the populations they name.

A run's ``[analysis.injection]`` and ``[analysis.proposal]`` are requests; once a catalog is
built, its *file* is authoritative about what it holds, and ``run_mcmc`` checks
it against the request. What is left to check here is that every committed
request can actually be drawn.

That replaced a much larger surface: a descriptor extracted from a gwmock graph
YAML, a mixture-of-proposals model, a derived proposal-density config, and an
exact-float-equality check reconciling a run's fiducials against five of the
eight parameters the catalog was drawn at.
"""

from __future__ import annotations

import pytest
from numpyro import handlers
from repo import REPO_ROOT

from astrogwb.constants import ISCO_ALPHA
from astrogwb.paper.config.catalogs import (
    check_population_model,
    resolve_run_catalogs,
)
from astrogwb.populations import (
    DEFAULT_DENSITY_SITES,
    ComponentMetadata,
    Population,
    PopulationMetadata,
    known_populations,
)
from astrogwb.simulators.polarization_power import CatalogMetadata
from astrogwb.waveform import AnalyticInspiralGenerator


def _requests() -> dict[str, CatalogMetadata]:
    """Every distinct catalog record the committed runs ask for, by file stem."""
    return {
        stem: metadata
        for stem, (metadata, _) in resolve_run_catalogs(REPO_ROOT).requests.items()
    }


def _build(request: CatalogMetadata) -> Population:
    """The population a request declares, built. ``PopulationMetadata.build`` is
    the same call generation makes, so this exercises the production path."""
    return request.population.build()


# --------------------------------------------------------------------------- #
# The committed declarations
# --------------------------------------------------------------------------- #
def test_every_committed_catalog_names_a_registered_population() -> None:
    """Caught pre-flight, not at the top of a queued GPU generation job."""
    for key, request in _requests().items():
        check_population_model(request.population, label=f"catalog {key}")


def test_every_committed_catalog_can_build_its_population() -> None:
    """The construction kwargs and parameters must actually fit the population.

    A typo in a population or sub-model kwarg is otherwise invisible until
    generation runs, and generation is the expensive step this pre-flight exists
    to protect. Each kwargs mapping reaches its factory whole, so a key it does
    not take fails here rather than being filtered away.
    """
    for key, request in _requests().items():
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(_build(request).source_model).get_trace(
                request.fiducials
            )
        assert trace["redshift"]["type"] == "sample", key
        assert trace["luminosity_distance"]["type"] == "deterministic", key


def test_only_the_guarded_proposals_declare_no_merger_rate() -> None:
    """A guard mixture is a sampling density; every other catalog is physical.

    The mixture density is not normalized by the Madau-Dickinson total rate, so
    pairing the two -- which the old two-name record allowed, and every guarded
    def did -- recorded a rate that was never the one its samples imply.
    """
    for key, request in _requests().items():
        merger_rate_fn = _build(request).merger_rate_fn
        expected_none = "uniform_guard" in request.population.redshift.model
        assert (merger_rate_fn is None) is expected_none, key


def test_every_declared_density_factor_is_a_real_sample_site() -> None:
    """Generation records ``DEFAULT_DENSITY_SITES``; each must be a sample site."""
    assert "redshift" in DEFAULT_DENSITY_SITES
    for key, request in _requests().items():
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(_build(request).source_model).get_trace(
                request.fiducials
            )
        for site in DEFAULT_DENSITY_SITES:
            assert trace[site]["type"] == "sample", key


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _population(
    *, redshift: str = "madau_dickinson", **redshift_kwargs: float
) -> PopulationMetadata:
    return PopulationMetadata(
        model_name="bns_madau_dickinson",
        redshift=ComponentMetadata(
            model=redshift,
            kwargs={
                "minimum_redshift": 0.3,
                "maximum_redshift": 20.0,
                "n_grid": 256,
                **redshift_kwargs,
            },
        ),
        mass=ComponentMetadata(model="ordered_uniform"),
    )


def test_an_unregistered_population_name_lists_the_known_set() -> None:
    unknown = PopulationMetadata(
        model_name="no_such_population",
        redshift=_population().redshift,
        mass=_population().mass,
    )
    with pytest.raises(ValueError) as error:
        check_population_model(unknown, label="catalog 'toy'")
    message = str(error.value)
    assert "catalog 'toy'" in message
    for name in known_populations():
        assert name in message


def test_a_kwarg_the_component_does_not_take_is_rejected() -> None:
    """Each factory signature is its kwargs schema, so a stale key fails here."""
    with pytest.raises(ValueError, match="uniform_mixing_fraction"):
        check_population_model(
            _population(uniform_mixing_fraction=0.1), label="catalog 'toy'"
        )


def test_a_proposal_density_is_rejected_as_an_analysis_target() -> None:
    """An analysis target reconstructs an observed rate, so it must declare one."""
    with pytest.raises(ValueError, match="declares no merger rate"):
        check_population_model(
            _population(
                redshift="madau_dickinson_uniform_guard", uniform_mixing_fraction=0.1
            ),
            label="run 'toy' analysis.population",
            requires_merger_rate=True,
        )


def test_a_time_delayed_population_is_a_valid_analysis_target() -> None:
    check_population_model(
        _population(
            redshift="madau_dickinson_time_delayed",
            minimum_delay=0.02,
            maximum_formation_redshift=20.0,
            n_delay_nodes=48,
        ),
        label="run 'toy' analysis.population",
        requires_merger_rate=True,
    )


def _waveform(approximant: str, alpha: float | None = None) -> dict[str, object]:
    waveform: dict[str, object] = {
        "approximant": approximant,
        "sampling_frequency": 128.0,
        "minimum_frequency": 10.0,
        "maximum_frequency": 50.0,
        "reference_frequency": 20.0,
        "frequency_resolution": 1.0,
    }
    if alpha is not None:
        waveform["alpha"] = alpha
    return waveform


def _request(waveform: dict[str, object]) -> CatalogMetadata:
    return CatalogMetadata.model_validate(
        {
            "population": {
                "model_name": "bns_madau_dickinson",
                "redshift": {
                    "model": "madau_dickinson",
                    "kwargs": {
                        "minimum_redshift": 0.0,
                        "maximum_redshift": 20.0,
                        "n_grid": 256,
                    },
                },
                "mass": {"model": "ordered_uniform"},
            },
            "waveform": waveform,
            "fiducials": {"H0": 67.66},
            "num_samples": 8,
        }
    )


def test_request_with_alpha_for_ripple_approximant_raises() -> None:
    """``alpha`` terminates the closed-form inspiral and means nothing to Ripple.

    Silently ignoring it would let a request look like it set a termination
    frequency that the generated catalog does not honour.
    """
    with pytest.raises(
        ValueError, match="alpha is only valid for the AnalyticInspiral"
    ):
        _request(_waveform("TaylorF2", alpha=0.02))


def test_request_with_declared_alpha_reaches_analytic_generator() -> None:
    generator = _request(_waveform("AnalyticInspiral", alpha=0.02)).waveform
    built = generator.build()

    assert isinstance(built, AnalyticInspiralGenerator)
    assert built.metadata.alpha == 0.02


def test_request_without_alpha_defaults_to_isco() -> None:
    built = _request(_waveform("AnalyticInspiral")).waveform.build()

    assert isinstance(built, AnalyticInspiralGenerator)
    assert built.metadata.alpha == ISCO_ALPHA
