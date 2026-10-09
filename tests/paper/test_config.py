"""Tests for shared config loading and deep-merge overrides."""

from __future__ import annotations

from pathlib import Path

import numpyro.distributions as dist
import pytest
from pydantic import ValidationError
from repo import REPO_ROOT

from astrogwb.constants import ISCO_ALPHA
from astrogwb.distributions.config import DistributionConfig
from astrogwb.paper.config import fiducials, networks, priors, waveform_generator
from astrogwb.paper.config.detectors import load_detector_config
from astrogwb.paper.config.runs import load_base
from astrogwb.paper.utils import deep_merge, load_mapping
from astrogwb.waveform import (
    AnalyticInspiralGenerator,
    RippleGenerator,
    WaveformMetadata,
)

PAPER_ROOT = REPO_ROOT


def test_detector_loader_merges_packaged_definitions_and_ordered_overrides(
    tmp_path: Path,
) -> None:
    packaged = load_detector_config([])
    assert packaged.networks == {}
    base = tmp_path / "networks.toml"
    base.write_text(
        '[networks]\nreference = ["S1", "R1"]\n'
        'comparison = "${networks.reference}"\n'
        '[detectors.custom]\nextends = "${detectors.S1}"\nlabel = "Custom"\n',
        encoding="utf-8",
    )
    override = tmp_path / "override.toml"
    override.write_text(
        '[networks]\nreference = ["S1", "C1"]\n'
        "[detectors.custom.geometry]\nxarm_azimuth_rad = 0.5\n"
        '[detectors.custom]\npsd_reference = "${detectors.C1.psd_reference}"\n',
        encoding="utf-8",
    )
    registry = load_detector_config([base, override])
    assert registry.networks["reference"] == ("S1", "C1")
    assert registry.networks["comparison"] == ("S1", "C1")
    assert registry.detectors["custom"].geometry.xarm_azimuth_rad == 0.5
    assert (
        registry.detectors["custom"].geometry.latitude_rad
        == packaged.detectors["S1"].geometry.latitude_rad
    )
    assert (
        registry.detectors["custom"].psd_reference
        == packaged.detectors["C1"].psd_reference
    )
    assert registry.detectors["custom"].label == "Custom"


def test_detector_loader_rejects_spectrum_settings(tmp_path: Path) -> None:
    layer = tmp_path / "wrong-group.toml"
    layer.write_text("[spectra]\nnum_draws = 5\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_detector_config([layer])


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


def test_load_mapping_rejects_a_non_toml_layer(tmp_path: Path) -> None:
    """Every run-config layer is TOML; a JSON one is refused, not parsed."""
    path = tmp_path / "config.json"
    path.write_text('{"analysis": {"network": "demo"}}', encoding="utf-8")

    with pytest.raises(ValueError, match="config layers are TOML"):
        load_mapping(path)


def test_priors_materialize_to_live_distributions() -> None:
    materialized = priors(REPO_ROOT)

    assert isinstance(materialized["H0"], dist.Uniform)
    assert isinstance(materialized["Omega_m"], dist.Normal)


def test_waveform_generator_defaults_to_the_committed_ripple() -> None:
    generator = waveform_generator(REPO_ROOT)

    assert isinstance(generator, RippleGenerator)
    assert generator.metadata.approximant == "IMRPhenomXAS_NRTidalv3"
    assert generator.metadata.minimum_frequency == 2.0
    assert generator.metadata.maximum_frequency == 2048.0


def test_committed_waveform_uses_the_loglinear_grid() -> None:
    metadata = waveform_generator(REPO_ROOT).metadata

    assert metadata.frequency_spacing == "loglinear"
    assert metadata.frequency_resolution == 1.0
    assert metadata.turnover_frequency == 100.0


def test_committed_waveform_variants_share_the_default_band_and_differ_in_grid() -> (
    None
):
    waveforms = {
        name: WaveformMetadata.model_validate(record)
        for name, record in load_base(REPO_ROOT)["waveforms"].items()
    }
    default = waveforms["default"]

    assert waveforms["linear"].frequency_spacing == "linear"
    assert waveforms["linear"].turnover_frequency is None
    assert (
        waveforms["linear"].model_copy(
            update={
                "frequency_spacing": default.frequency_spacing,
                "turnover_frequency": default.turnover_frequency,
            }
        )
        == default
    )
    assert waveforms["TaylorF2"].approximant == "TaylorF2"
    assert waveforms["TaylorF2"].frequency_spacing == default.frequency_spacing
    assert waveforms["TaylorF2"].turnover_frequency == default.turnover_frequency


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

    assert DistributionConfig.from_distribution(overridden["H0"]).model_dump() == {
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
    assert DistributionConfig.from_distribution(
        priors(REPO_ROOT)["H0"]
    ).model_dump() == {
        "dist": "Uniform",
        "kwargs": {"low": 20.0, "high": 140.0},
    }
