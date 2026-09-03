"""Tests for array-native catalog metadata, generation, and simulation."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from astrogwb.catalog import (
    AnalyticInspiralGenerator,
    Catalog,
    FrequencyDomainWaveformMetadata,
    PopulationMetadata,
    simulate_population,
)
from astrogwb.constants import ISCO_ALPHA
from astrogwb.waveform import inspiral_polarization_power
from numpy.typing import ArrayLike


@pytest.fixture
def source_parameters() -> dict[str, np.ndarray]:
    return {
        "source_frame_mass_1": np.array([1.4, 1.2]),
        "source_frame_mass_2": np.array([1.3, 1.1]),
        "redshift": np.array([0.1, 0.2]),
        "luminosity_distance": np.array([400.0, 900.0]),
        "inclination": np.array([0.0, 1.0]),
        "integer_label": np.array([7, 8], dtype=np.int16),
    }


def _waveform_metadata() -> FrequencyDomainWaveformMetadata:
    return FrequencyDomainWaveformMetadata(
        frequencies=np.array([10.0, 12.0]),
        approximant="FakeWaveform",
        minimum_frequency=10.0,
        maximum_frequency=12.5,
        reference_frequency=11.0,
        sampling_frequency=64.0,
        df=2.0,
    )


def _population_metadata(
    *, num_samples: int = 2, provenance: Mapping[str, str | int | float] | None = None
) -> PopulationMetadata:
    return PopulationMetadata(
        name="test-population",
        seed=42,
        num_samples=num_samples,
        source_type="bns",
        provenance={} if provenance is None else provenance,
    )


@pytest.mark.parametrize(
    ("maximum_frequency", "expected"),
    [
        (20.0, np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])),
        (19.0, np.array([10.0, 12.0, 14.0, 16.0, 18.0])),
    ],
)
def test_waveform_metadata_from_bounds_includes_largest_in_band_bin(
    maximum_frequency: float, expected: np.ndarray
) -> None:
    metadata = FrequencyDomainWaveformMetadata.from_bounds(
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=maximum_frequency,
        reference_frequency=10.0,
        sampling_frequency=64.0,
        df=2.0,
    )

    np.testing.assert_array_equal(metadata.frequencies, expected)
    assert metadata.maximum_frequency == maximum_frequency


@pytest.mark.parametrize(
    ("frequencies", "df", "message"),
    [
        ([], 1.0, "at least one bin"),
        ([1.0, np.inf], 1.0, "finite"),
        ([1.0, 1.0], 1.0, "strictly increasing"),
        ([2.0, 1.0], 1.0, "strictly increasing"),
        ([1.0, 2.5, 3.0], 1.0, "uniformly spaced"),
        ([1.0, 2.0], 0.0, "finite positive"),
    ],
)
def test_waveform_metadata_rejects_invalid_grid(
    frequencies: list[float], df: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FrequencyDomainWaveformMetadata(
            frequencies=np.asarray(frequencies),
            approximant="Toy",
            minimum_frequency=1.0,
            maximum_frequency=3.0,
            reference_frequency=1.0,
            sampling_frequency=8.0,
            df=df,
        )


def test_waveform_metadata_accepts_float64_fft_roundoff() -> None:
    frequencies = np.array([10.0, 20.0, 30.0, 40.0])
    frequencies[2] += 32.0 * np.finfo(np.float64).eps * frequencies[-1]

    FrequencyDomainWaveformMetadata(
        frequencies=frequencies,
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=40.0,
        reference_frequency=20.0,
        sampling_frequency=128.0,
        df=10.0,
    )


def test_from_generator_passes_exact_grid_and_preserves_parameter_dtypes(
    source_parameters: dict[str, np.ndarray],
) -> None:
    expected_power = np.array([[1.0, 2.0], [3.0, 4.0]])
    waveform = _waveform_metadata()
    population = _population_metadata(provenance={"producer": "test"})

    class FakeGenerator:
        received_parameters: Mapping[str, ArrayLike] | None = None
        received_metadata: FrequencyDomainWaveformMetadata | None = None

        def __call__(
            self,
            source_parameters: Mapping[str, ArrayLike],
            waveform_metadata: FrequencyDomainWaveformMetadata,
        ) -> np.ndarray:
            self.received_parameters = source_parameters
            self.received_metadata = waveform_metadata
            return expected_power

    generator = FakeGenerator()
    catalog = Catalog.from_generator(
        source_parameters,
        generator=generator,
        waveform_metadata=waveform,
        population_metadata=population,
    )

    assert generator.received_parameters is source_parameters
    assert generator.received_metadata is waveform
    assert catalog.waveform_metadata is waveform
    assert catalog.population_metadata is population
    np.testing.assert_array_equal(catalog.polarization_power, expected_power)
    assert catalog.source_parameters["integer_label"].dtype == np.int16


def test_analytic_generator_evaluates_on_exact_metadata_grid(
    source_parameters: dict[str, np.ndarray],
) -> None:
    waveform = FrequencyDomainWaveformMetadata(
        frequencies=np.array([10.0, 12.0, 14.0]),
        approximant="AnalyticInspiral",
        minimum_frequency=9.5,
        maximum_frequency=14.5,
        reference_frequency=10.0,
        sampling_frequency=32.0,
        df=2.0,
    )
    generator = AnalyticInspiralGenerator(alpha=ISCO_ALPHA)

    actual = generator(source_parameters, waveform)
    expected = np.asarray(
        inspiral_polarization_power(
            waveform.frequencies, source_parameters, alpha=ISCO_ALPHA
        )
    ).T

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("power", "message"),
    [
        (np.ones(2), "two-dimensional"),
        (np.ones((3, 2)), "frequency axis"),
        (np.ones((2, 3)), "num_samples"),
        (np.ones((2, 2), dtype=np.complex128), "real-valued"),
    ],
)
def test_catalog_rejects_malformed_power(power: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Catalog(
            source_parameters={"redshift": np.array([0.1, 0.2])},
            polarization_power=power,
            waveform_metadata=_waveform_metadata(),
            population_metadata=_population_metadata(),
        )


@pytest.mark.parametrize("values", [np.ones((2, 1)), np.ones(1), np.ones(3)])
def test_catalog_rejects_malformed_source_parameters(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="source parameter"):
        Catalog(
            source_parameters={"redshift": values},
            polarization_power=np.ones((2, 2)),
            waveform_metadata=_waveform_metadata(),
            population_metadata=_population_metadata(),
        )


@pytest.mark.parametrize("value", [True, {"nested": 1}, [1], None, np.int64(1)])
def test_population_metadata_rejects_non_scalar_provenance(value: object) -> None:
    with pytest.raises(TypeError, match="provenance"):
        PopulationMetadata(
            name="test",
            seed=1,
            num_samples=2,
            provenance={"invalid": value},  # ty: ignore[invalid-argument-type]
        )


def test_simulate_population_uses_metadata_and_is_prefix_stable(tmp_path: Path) -> None:
    graph = {
        "draw": {
            "sampler": {
                "function": "uniform",
                "arguments": {"minimum": 0.0, "maximum": 1.0},
            }
        }
    }
    config_path = tmp_path / "population.yaml"
    config_path.write_text(
        """\
name: test-population
parameters:
  draw:
    sampler:
      function: uniform
      arguments:
        minimum: 0.0
        maximum: 1.0
""",
        encoding="utf-8",
    )
    metadata = PopulationMetadata(
        name="test-population", seed=123, num_samples=4, source_type="bns"
    )
    longer_metadata = PopulationMetadata(
        name="test-population", seed=123, num_samples=8, source_type="bns"
    )

    mapping_draw = simulate_population(graph, metadata=metadata)
    repeated_draw = simulate_population(graph, metadata=metadata)
    longer_draw = simulate_population(graph, metadata=longer_metadata)
    path_draw = simulate_population(config_path, metadata=metadata)

    assert type(mapping_draw) is dict
    assert isinstance(mapping_draw["draw"], np.ndarray)
    np.testing.assert_array_equal(mapping_draw["draw"], repeated_draw["draw"])
    np.testing.assert_array_equal(mapping_draw["draw"], longer_draw["draw"][:4])
    np.testing.assert_array_equal(mapping_draw["draw"], path_draw["draw"])


def test_population_simulation_missing_extra_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "gwmock_pop", None)
    metadata = PopulationMetadata(name="test", seed=0, num_samples=1)

    with pytest.raises(ImportError, match=r"pip install astrogwb\[simulation\]"):
        simulate_population({}, metadata=metadata)
