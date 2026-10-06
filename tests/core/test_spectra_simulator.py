"""The spectra simulator: one population draw, reduced through a waveform."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from astrogwb.constants import INCLINATION_AVERAGE_TO_FACE_ON_RATIO
from astrogwb.distributions.config import DistributionConfig
from astrogwb.populations import PopulationMetadata
from astrogwb.simulators.core import batch_keys
from astrogwb.simulators.population import PopulationSimulator
from astrogwb.simulators.spectra import SpectraMetadata, SpectraSimulator
from astrogwb.utils import years_to_seconds
from astrogwb.waveform import WaveformMetadata

FIDUCIALS = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "minimum_mass": 1.0,
    "mass_width": 1.5,
}
CHUNK = 8
EXPECTED_EVENTS = 12.0

type MetadataFactory = Callable[..., SpectraMetadata]


@pytest.fixture
def make_metadata() -> MetadataFactory:
    """A tiny closed-form-inspiral record; sampled rate so draws differ in count."""
    waveform = WaveformMetadata(
        approximant="AnalyticInspiral",
        minimum_frequency=20.0,
        maximum_frequency=50.0,
        reference_frequency=20.0,
        sampling_frequency=128.0,
        frequency_resolution=10.0,
    )

    def make(sample_inclination: bool = True, **overrides: Any) -> SpectraMetadata:
        population_metadata = PopulationMetadata(
            model_name="bns_md_cosmological",
            model_kwargs={
                "minimum_redshift": 0.0,
                "maximum_redshift": 5.0,
                "n_grid": 64,
                "sample_inclination": sample_inclination,
            },
        )
        built = population_metadata.build()
        assert built.merger_rate_fn is not None
        rate = float(built.merger_rate_fn({**FIDUCIALS, "local_merger_rate": 770.0}))
        fields: dict[str, Any] = {
            "waveform": waveform,
            "population": population_metadata,
            "hyperparameters": {
                **FIDUCIALS,
                "local_merger_rate": DistributionConfig(
                    dist="Uniform", kwargs={"low": 400.0, "high": 1200.0}
                ),
            },
            "observation_time": EXPECTED_EVENTS / (rate * years_to_seconds(1.0)),
            "count": "poisson",
        }
        fields.update(overrides)
        return SpectraMetadata.model_validate(fields)

    return make


def _spectra(metadata: SpectraMetadata, keys: Any, **settings: Any) -> Any:
    simulator = SpectraSimulator(metadata, chunk_size=CHUNK, **settings)
    return simulator.simulate_batch(keys)


def test_spectra_shapes_and_columns(make_metadata: MetadataFactory) -> None:
    out = _spectra(make_metadata(), batch_keys(41, 5))
    assert out["spectral_density"].shape == (5, out["frequencies"].size)
    assert out["n_events"].shape == out["total_merger_rate"].shape == (5,)
    assert set(out["hyperparameters"]) == {*FIDUCIALS, "local_merger_rate"}
    assert np.all(out["spectral_density"] >= 0.0)


def test_fixed_counts_report_the_recorded_count(make_metadata: MetadataFactory) -> None:
    out = _spectra(make_metadata(count="fixed", num_events=5), batch_keys(41, 3))
    assert_array_equal(out["n_events"], [5, 5, 5])


def test_a_draw_depends_on_its_own_seed_alone(make_metadata: MetadataFactory) -> None:
    metadata = make_metadata()
    keys = batch_keys(41, 6)
    batch = _spectra(metadata, keys)
    alone = _spectra(metadata, keys[2:3])
    assert batch["n_events"][2] == alone["n_events"][0]
    assert_allclose(
        batch["spectral_density"][2], alone["spectral_density"][0], rtol=1e-10
    )


def test_superbatch_changes_cost_not_the_spectra(
    make_metadata: MetadataFactory,
) -> None:
    metadata = make_metadata()
    keys = batch_keys(41, 5)
    whole = _spectra(metadata, keys, superbatch=5)
    pieces = _spectra(metadata, keys, superbatch=2)
    assert_array_equal(whole["n_events"], pieces["n_events"])
    assert_allclose(whole["spectral_density"], pieces["spectral_density"], rtol=1e-10)


@pytest.mark.parametrize(
    ("sample_inclination", "inclination_factor"),
    [(False, INCLINATION_AVERAGE_TO_FACE_ON_RATIO), (True, 1.0)],
)
def test_fixed_spectra_with_one_source_scale_the_power_by_rate_and_inclination(
    make_metadata: MetadataFactory,
    sample_inclination: bool,
    inclination_factor: float,
) -> None:
    """With one source per draw, ``S = A_inc * R * P``.

    ``A_inc`` is the isotropic average of face-on power when the population
    omits inclination, and one when each source's orientation is already in its
    power.
    """
    metadata = make_metadata(
        sample_inclination=sample_inclination, count="fixed", num_events=1
    )
    keys = batch_keys(41, 3)
    out = _spectra(metadata, keys)
    draws = PopulationSimulator(metadata.sources, chunk_size=CHUNK).simulate_batch(keys)
    assert ("inclination" in draws["source_parameters"]) == sample_inclination
    generator = metadata.waveform.build()
    for draw in range(3):
        sources = {k: v[draw : draw + 1] for k, v in draws["source_parameters"].items()}
        power = np.asarray(generator.generate_batch(sources))[:, 0]
        expected = inclination_factor * draws["total_merger_rate"][draw] * power
        assert_allclose(out["spectral_density"][draw], expected, rtol=1e-10)


def test_reducing_a_persisted_population_matches_drawing_in_one_go(
    make_metadata: MetadataFactory,
) -> None:
    metadata = make_metadata()
    keys = batch_keys(41, 4)
    drawn = _spectra(metadata, keys, superbatch=4)
    persisted = PopulationSimulator(metadata.sources, chunk_size=CHUNK).simulate_batch(
        keys
    )
    reduced = cast(
        dict[str, Any], SpectraSimulator(metadata, chunk_size=CHUNK).reduce(persisted)
    )
    assert_array_equal(reduced["n_events"], drawn["n_events"])
    assert_allclose(reduced["spectral_density"], drawn["spectral_density"], rtol=1e-10)


def test_the_same_population_feeds_two_waveforms(
    make_metadata: MetadataFactory,
) -> None:
    """The population key ignores the waveform, so one draw serves both."""
    first = make_metadata()
    other_waveform = first.waveform.model_copy(update={"frequency_resolution": 5.0})
    second = make_metadata(waveform=other_waveform)
    assert first.key() != second.key()
    assert first.sources.key() == second.sources.key()


def test_simulate_batch_is_the_batch_of_simulate(
    make_metadata: MetadataFactory,
) -> None:
    metadata = make_metadata()
    simulator = SpectraSimulator(metadata, chunk_size=CHUNK)
    keys = batch_keys(41, 4)
    batch = simulator.simulate_batch(keys)
    for i in range(keys.shape[0]):
        single = simulator.simulate(keys[i])
        assert single["n_events"] == batch["n_events"][i]
        assert_allclose(
            single["spectral_density"], batch["spectral_density"][i], rtol=1e-10
        )
        for name, column in batch["hyperparameters"].items():
            assert_allclose(single["hyperparameters"][name], column[i], rtol=1e-12)


def test_spectra_rejects_a_stale_version(make_metadata: MetadataFactory) -> None:
    with pytest.raises(ValueError, match="is installed"):
        _spectra(make_metadata(version="0.0.1"), batch_keys(41, 1))


def test_spectra_columns_are_numpy_not_jax(make_metadata: MetadataFactory) -> None:
    out = cast(dict[str, Any], _spectra(make_metadata(), batch_keys(41, 2)))
    assert isinstance(out["spectral_density"], np.ndarray)
