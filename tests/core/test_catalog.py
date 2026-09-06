"""Tests for array-native catalog generation and simulation."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from astrogwb.catalog import Catalog, PopulationMetadata, simulate_population
from astrogwb.constants import ISCO_ALPHA
from astrogwb.waveform import (
    AnalyticInspiralGenerator,
    PolarizationPowerGenerator,
    inspiral_polarization_power,
)


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


def _waveform_generator() -> PolarizationPowerGenerator:
    return PolarizationPowerGenerator(
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
def test_generator_from_bounds_includes_largest_in_band_bin(
    maximum_frequency: float, expected: np.ndarray
) -> None:
    generator = PolarizationPowerGenerator.from_bounds(
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=maximum_frequency,
        reference_frequency=10.0,
        sampling_frequency=64.0,
        df=2.0,
    )

    np.testing.assert_array_equal(generator.frequencies, expected)
    assert generator.maximum_frequency == maximum_frequency


def test_from_generator_uses_generator_descriptor_and_preserves_parameter_dtypes(
    source_parameters: dict[str, np.ndarray],
) -> None:
    generator = AnalyticInspiralGenerator.from_bounds(
        alpha=ISCO_ALPHA,
        approximant="AnalyticInspiral",
        minimum_frequency=10.0,
        maximum_frequency=12.0,
        reference_frequency=10.0,
        sampling_frequency=32.0,
        df=2.0,
    )
    population = _population_metadata(provenance={"producer": "test"})
    catalog = Catalog.from_generator(
        source_parameters,
        generator=generator,
        population_metadata=population,
    )

    assert catalog.waveform_metadata is generator
    assert catalog.population_metadata is population
    assert catalog.source_parameters["integer_label"].dtype == np.int16


def test_analytic_generator_evaluates_on_exact_metadata_grid(
    source_parameters: dict[str, np.ndarray],
) -> None:
    generator = AnalyticInspiralGenerator.from_bounds(
        alpha=ISCO_ALPHA,
        approximant="AnalyticInspiral",
        minimum_frequency=9.5,
        maximum_frequency=14.5,
        reference_frequency=10.0,
        sampling_frequency=32.0,
        df=2.0,
    )

    actual = generator(source_parameters)
    expected = np.asarray(
        inspiral_polarization_power(
            generator.frequencies, source_parameters, alpha=ISCO_ALPHA
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
            waveform_metadata=_waveform_generator(),
            population_metadata=_population_metadata(),
        )


@pytest.mark.parametrize("values", [np.ones((2, 1)), np.ones(1), np.ones(3)])
def test_catalog_rejects_malformed_source_parameters(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="source parameter"):
        Catalog(
            source_parameters={"redshift": values},
            polarization_power=np.ones((2, 2)),
            waveform_metadata=_waveform_generator(),
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


@pytest.mark.integration
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
