"""Tests for shared config loading and deep-merge overrides."""

from __future__ import annotations

from pathlib import Path

import numpyro.distributions as dist
import pytest
from config_fixtures import example_raw
from pydantic import ValidationError
from repo import REPO_ROOT

from astrogwb.constants import ISCO_ALPHA
from astrogwb.paper.config import fiducials, networks, priors, waveform_generator
from astrogwb.paper.config.mcmc import (
    DEFAULT_DENSITY_SITES,
    AmplitudeParameter,
    build_run_config,
    prior_to_spec,
)
from astrogwb.paper.config.runs import (
    assemble_run,
    discover_runs,
    load_base,
)
from astrogwb.paper.utils import deep_merge, load_mapping
from astrogwb.waveform import AnalyticInspiralGenerator, RippleGenerator

PAPER_ROOT = REPO_ROOT


def test_deep_merge_nested_dicts_and_list_replacement() -> None:
    # `detector_ids` is a stand-in list key, deliberately not named `networks`:
    # that is a real config section now, and it is a *mapping*, so reusing the
    # name here would advertise the wrong merge rule for it.
    base = {
        "seed": 1,
        "figures": {"compare": {"var_name": "H0", "figure_dpi": 300}},
        "detector_ids": ["A", "B"],
    }
    override = {
        "figures": {"compare": {"var_name": "Omega_m"}},
        "detector_ids": ["C"],
    }

    merged = deep_merge(base, override)

    assert merged == {
        "seed": 1,
        "figures": {"compare": {"var_name": "Omega_m", "figure_dpi": 300}},
        "detector_ids": ["C"],
    }
    # Inputs are not mutated.
    figures = base["figures"]
    assert isinstance(figures, dict)
    compare = figures["compare"]
    assert isinstance(compare, dict)
    assert compare["var_name"] == "H0"
    assert base["detector_ids"] == ["A", "B"]


def test_load_mapping_rejects_a_non_json_layer(tmp_path: Path) -> None:
    """One format, so "every layer is jq-readable" holds by construction."""
    path = tmp_path / "config.toml"
    path.write_text("[analysis]\nnetwork = 'demo'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config layers are JSON"):
        load_mapping(path)


def test_build_run_config_deep_merges_extra_overrides() -> None:
    raw = example_raw()
    config = build_run_config(
        raw,
        seed=99,
        sampler={"num_warmup": 11, "num_samples": 13},
    )

    assert config.sampler.seed == 99
    assert config.sampler.num_warmup == 11
    assert config.sampler.num_samples == 13
    # Unrelated sampler fields keep their file values.
    assert config.sampler.target_accept == raw["sampler"]["target_accept"]


def test_analysis_population_declares_the_required_redshift_grid() -> None:
    config = build_run_config(example_raw())
    kwargs = config.analysis.population.model_kwargs
    expected = example_raw()["analysis"]["population"]["model_kwargs"]
    assert {"minimum_redshift", "maximum_redshift", "n_grid"} <= set(kwargs)
    assert kwargs["minimum_redshift"] == expected["minimum_redshift"]
    assert kwargs["maximum_redshift"] == expected["maximum_redshift"]
    assert kwargs["n_grid"] == expected["n_grid"]


def test_run_config_carries_no_proposal_density() -> None:
    """The density is the proposal catalog's own record, not a config input.

    Window equality with the analysis population support used to need a
    validator; it now holds by construction, because
    `PolarizationPowerCatalog.restrict_redshift` is handed the run's own window
    and moves the samples and the recorded density together.
    """
    config = build_run_config(example_raw())

    assert not hasattr(config, "proposal")
    assert "proposal" not in config.model_dump(mode="json")
    # `catalog.proposal` is the catalog, not the density -- it stays.
    assert config.analysis.catalog.proposal


def test_derived_analysis_values_are_not_serialized(tmp_path) -> None:
    """Derived properties stay out of normalized workflow configurations."""
    config = build_run_config(example_raw())
    assert "grid" not in config.model_dump(mode="json")["analysis"]
    assert "fixed_params" not in config.model_dump(mode="json")

    path = tmp_path / "run.json"
    config.save(path)
    assert "grid" not in load_mapping(path)["analysis"]

    reloaded = build_run_config(load_mapping(path))
    assert reloaded.model_dump(mode="json") == config.model_dump(mode="json")


def test_every_experiment_run_assembles_into_a_valid_config() -> None:
    """Every run the workflow can build must validate without a runtime.

    This replaces a check over two committed example configs. Assembling each
    experiment run is both wider coverage and the thing that actually ships.
    """
    assert "runtime" not in load_base()

    runs = discover_runs()
    assert runs, "no experiments discovered"
    for experiment, names in runs.items():
        for run in names:
            raw = assemble_run(experiment, run)
            assert "runtime" not in raw, (
                f"{experiment}/{run} declares a runtime section"
            )
            build_run_config(raw)


def test_build_run_config_rejects_legacy_runtime_section() -> None:
    raw = example_raw()
    raw["runtime"] = {"platform": "cpu"}

    with pytest.raises(ValidationError, match="runtime"):
        build_run_config(raw)


# --------------------------------------------------------------------------- #
# Amplitude-marginalized likelihood
# --------------------------------------------------------------------------- #


def _marginalized_raw() -> dict:
    """The assembled run config, switched to marginalize H0 out entirely."""
    raw = example_raw()
    raw["analysis"] = {
        **raw["analysis"],
        "likelihood": "amplitude_marginalized",
        "amplitude_parameter": "H0",
        "sampled_params": [],
    }
    return raw


def test_marginalized_config_round_trips_through_save(tmp_path) -> None:
    """assemble_config writes normalized configs; run_mcmc must reload them.

    The amplitude prior lives in ``priors``, so the round trip is symmetric:
    a saved JSON reloads unchanged without any dedicated amplitude field.
    """
    config = build_run_config(_marginalized_raw())
    path = tmp_path / "run.json"
    config.save(path)

    assert "fixed_params" not in load_mapping(path)

    reloaded = build_run_config(load_mapping(path))

    # Distributions have no value equality; compare their wire-format specs.
    assert {name: prior_to_spec(prior) for name, prior in reloaded.priors.items()} == {
        name: prior_to_spec(prior) for name, prior in config.priors.items()
    }
    assert reloaded.analysis.sampled_params == config.analysis.sampled_params
    assert reloaded.fixed_params == config.fixed_params
    assert reloaded.model_dump(mode="json") == config.model_dump(mode="json")


def test_reloaded_marginalized_config_still_rejects_amplitude_parameter_sampled(
    tmp_path,
) -> None:
    config = build_run_config(_marginalized_raw())
    path = tmp_path / "run.json"
    config.save(path)
    raw = load_mapping(path)
    raw["analysis"]["sampled_params"] = [
        *raw["analysis"]["sampled_params"],
        "H0",
    ]

    with pytest.raises(ValidationError, match="cannot also appear in sampled_params"):
        build_run_config(raw)


def test_marginalized_config_rejects_amplitude_parameter_also_sampled() -> None:
    raw = _marginalized_raw()
    raw["analysis"]["sampled_params"] = ["H0"]

    with pytest.raises(ValidationError, match="cannot also appear in sampled_params"):
        build_run_config(raw)


def test_marginalized_config_rejects_amplitude_parameter_without_prior_table() -> None:
    raw = _marginalized_raw()
    # The assembled base gives every fiducial a prior; drop one to create the
    # amplitude-parameter-without-a-prior case this test is about.
    del raw["priors"]["local_merger_rate"]
    raw["analysis"]["amplitude_parameter"] = "local_merger_rate"

    with pytest.raises(ValidationError, match=r"needs a \[priors\.\*\] table"):
        build_run_config(raw)


def test_marginalized_config_rejects_amplitude_parameter_missing_fiducial() -> None:
    raw = _marginalized_raw()
    raw["analysis"]["amplitude_parameter"] = "local_merger_rate"
    raw["priors"] = {
        **raw["priors"],
        "local_merger_rate": {
            "dist": "Uniform",
            "kwargs": {"low": 50.0, "high": 300.0},
        },
    }
    del raw["fiducials"]["local_merger_rate"]

    with pytest.raises(ValidationError, match=r"missing from \[fiducials\]"):
        build_run_config(raw)


def test_marginalized_config_rejects_unsupported_amplitude_parameter_name() -> None:
    raw = _marginalized_raw()
    raw["analysis"]["amplitude_parameter"] = "Omega_m"
    raw["priors"] = {
        **raw["priors"],
        "Omega_m": {"dist": "Uniform", "kwargs": {"low": 0.05, "high": 0.95}},
    }

    with pytest.raises(ValidationError):
        build_run_config(raw)


def test_default_likelihood_rejects_amplitude_parameter() -> None:
    raw = example_raw()
    raw["analysis"] = {**raw["analysis"], "amplitude_parameter": "H0"}

    with pytest.raises(ValidationError, match="only valid when"):
        build_run_config(raw)


# --------------------------------------------------------------------------- #
# Prior specs
# --------------------------------------------------------------------------- #
def test_config_rejects_a_fiducial_without_a_prior() -> None:
    raw = example_raw()
    del raw["priors"]["gamma"]

    with pytest.raises(ValidationError, match="fiducials without"):
        build_run_config(raw)


def test_config_rejects_a_prior_without_a_fiducial() -> None:
    raw = example_raw()
    raw["priors"]["unused"] = {"dist": "Normal", "kwargs": {"loc": 0.0, "scale": 1.0}}

    with pytest.raises(ValidationError, match="priors missing from"):
        build_run_config(raw)


def test_prior_spec_rejects_stale_keys_from_a_cross_type_override() -> None:
    raw = example_raw()
    polluted = deep_merge(
        raw,
        {
            "priors": {
                "H0": {"dist": "Normal", "kwargs": {"loc": 67.66, "scale": 0.6766}}
            }
        },
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        build_run_config(polluted)


def test_prior_spec_rejects_an_unknown_distribution_name() -> None:
    raw = example_raw()
    raw["priors"]["H0"] = {"dist": "Lognormal", "kwargs": {"loc": 1.0, "scale": 1.0}}

    with pytest.raises(ValidationError, match="not a numpyro distribution"):
        build_run_config(raw)


def test_prior_spec_rejects_a_name_that_is_not_a_distribution() -> None:
    """The guard on the `getattr`, not just the lookup.

    `dist` is a config-supplied string indexed into a live module, so a name
    that resolves to something other than a Distribution subclass -- here the
    `constraints` submodule -- must be refused rather than called.
    """
    raw = example_raw()
    raw["priors"]["H0"] = {"dist": "constraints", "kwargs": {}}

    with pytest.raises(ValidationError, match="not a numpyro distribution"):
        build_run_config(raw)


def test_prior_spec_rejects_missing_required_key() -> None:
    raw = example_raw()
    raw["priors"]["H0"] = {"dist": "Normal", "kwargs": {"loc": 67.66, "scal": 0.6766}}

    with pytest.raises(ValidationError, match="scale"):
        build_run_config(raw)


def test_waveform_generator_defaults_to_the_committed_ripple() -> None:
    generator = waveform_generator(REPO_ROOT)

    assert isinstance(generator, RippleGenerator)
    assert generator.metadata.approximant == "IMRPhenomXAS_NRTidalv3"
    assert generator.metadata.minimum_frequency == 2.0
    assert generator.metadata.maximum_frequency == 2048.0


def test_waveform_generator_can_select_untapered_nrtidal() -> None:
    generator = waveform_generator(REPO_ROOT, use_taper_in_tidal_corrections=False)

    assert isinstance(generator, RippleGenerator)
    assert generator.metadata.use_taper_in_tidal_corrections is False


def test_waveform_generator_kwargs_select_the_analytical_inspiral() -> None:
    generator = waveform_generator(REPO_ROOT, approximant="AnalyticInspiral")

    assert isinstance(generator, AnalyticInspiralGenerator)
    assert generator.metadata.approximant == "AnalyticInspiral"
    assert generator.metadata.alpha == ISCO_ALPHA
    assert generator.metadata.minimum_frequency == 2.0


def test_waveform_generator_overrides_are_validated_not_trusted() -> None:
    """Overrides go through ``WaveformMetadata``, so a bad one fails here.

    Before the accessor shared a path with a catalog def, every keyword was
    coerced with a bare ``float()`` and an override that made no sense for the
    named approximant was passed straight through.
    """
    generator = waveform_generator(
        REPO_ROOT, approximant="AnalyticInspiral", alpha=0.02
    )
    assert isinstance(generator, AnalyticInspiralGenerator)
    assert generator.metadata.alpha == 0.02

    with pytest.raises(ValidationError, match="alpha is only valid"):
        waveform_generator(REPO_ROOT, alpha=0.02)


@pytest.mark.parametrize(
    "approximant", ["analytical", "analytic", "Analytic", "AnalyticalInspiral"]
)
def test_a_near_miss_analytical_approximant_is_rejected(approximant: str) -> None:
    """A spelling close to the canonical one must not be read as a Ripple name.

    Anything that is not ``"AnalyticInspiral"`` selects the Ripple backend, so
    without this the failure is silent in the direction that matters: the
    closed-form inspiral a caller asked for is quietly swapped for a waveform
    approximant, and the error -- if any -- surfaces from inside ripple.
    """
    with pytest.raises(ValidationError, match="is not a Ripple approximant"):
        waveform_generator(REPO_ROOT, approximant=approximant)


# --------------------------------------------------------------------------- #
# Accessor kwargs overrides
# --------------------------------------------------------------------------- #
def test_fiducials_kwargs_override_the_file() -> None:
    from_file = fiducials(REPO_ROOT)
    overridden = fiducials(REPO_ROOT, H0=70.0)

    assert overridden["H0"] == 70.0
    # Every fiducial the override did not name keeps its file value.
    assert set(overridden) == set(from_file)
    assert {k: v for k, v in overridden.items() if k != "H0"} == {
        k: v for k, v in from_file.items() if k != "H0"
    }


def test_accessor_kwargs_may_add_an_entry() -> None:
    """A merge, not a rejection: waveform_generator already behaves this way."""
    assert fiducials(REPO_ROOT, demo=1)["demo"] == 1.0
    assert len(fiducials(REPO_ROOT, demo=1)) == len(fiducials(REPO_ROOT)) + 1


def test_networks_kwargs_override_and_coerce_to_a_tuple() -> None:
    # Hyphenated names cannot be literal keywords; unpack a mapping, as the
    # docstring shows.
    overridden = networks(REPO_ROOT, **{"ET-2L-aligned": ["E1", "E2"]})

    assert overridden["ET-2L-aligned"] == ("E1", "E2")
    assert isinstance(overridden["ET-2L-aligned"], tuple)
    assert networks(REPO_ROOT, **{"demo-net": ["A"]})["demo-net"] == ("A",)


def test_priors_kwargs_accept_a_wire_spec() -> None:
    overridden = priors(
        REPO_ROOT,
        H0={"dist": "Uniform", "kwargs": {"low": 21.0, "high": 139.0}},
    )

    assert prior_to_spec(overridden["H0"]) == {
        "dist": "Uniform",
        "kwargs": {"low": 21.0, "high": 139.0},
    }
    # Untouched entries are still materialized from the file.
    assert len(overridden) == len(priors(REPO_ROOT))


def test_priors_kwargs_accept_a_live_distribution() -> None:
    prior = dist.Normal(0.0, 1.0)

    assert priors(REPO_ROOT, H0=prior)["H0"] is prior


def test_accessor_kwargs_do_not_poison_the_cache() -> None:
    """The merge happens after the cached parse, so no-arg calls are unaffected."""
    fiducials(REPO_ROOT, H0=999.0)
    networks(REPO_ROOT, **{"ET-2L-aligned": ("nope",)})
    priors(REPO_ROOT, H0=dist.Normal(0.0, 1.0))

    assert fiducials(REPO_ROOT)["H0"] == 67.66
    assert networks(REPO_ROOT)["ET-2L-aligned"] == ("S1", "R1")
    assert prior_to_spec(priors(REPO_ROOT)["H0"]) == {
        "dist": "Uniform",
        "kwargs": {"low": 20.0, "high": 140.0},
    }


# --------------------------------------------------------------------------- #
# The constants `mcmc` restates rather than imports
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_the_restated_population_constants_match_the_registry() -> None:
    """`config.mcmc` copies two tuples out of the population layer.

    It has to: `astrogwb.populations.registry` imports JAX at module scope, and
    `tests/paper/test_cli.py` pins that importing the config layer does not --
    every `snakemake --dry-run` would otherwise pay for it. Copies drift, so
    this is the cross-check, and it is marked `integration` because asserting
    it is what imports JAX.
    """
    from typing import get_args

    from astrogwb.populations import AMPLITUDE_PARAMETERS
    from astrogwb.populations import DEFAULT_DENSITY_SITES as registered_sites

    assert get_args(AmplitudeParameter) == AMPLITUDE_PARAMETERS
    assert DEFAULT_DENSITY_SITES == registered_sites
