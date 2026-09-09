"""Tests for the array-native catalog container and its population record."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

from astrogwb.catalog import Catalog
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


#: The population record every direct construction has to carry. The catalog
#: is the record of the density that drew it, so there is no valid catalog
#: without one.
POPULATION_RECORD: dict[str, Any] = {
    "model_name": "bns_md_cosmological",
    "model_kwargs": {"z_min": 0.0, "z_max": 20.0, "n_grid": 256},
    "fiducials": {
        "H0": 67.66,
        "Omega_m": 0.3096,
        "gamma": 1.42,
        "kappa": 4.62,
        "z_peak": 1.84,
        "local_merger_rate": 770.0,
    },
    "density_sites": ("redshift",),
}
PRIVATE_RECORD: dict[str, Any] = {
    f"_{name}": value for name, value in POPULATION_RECORD.items()
}
CATALOG_DEFAULTS: dict[str, Any] = {
    **PRIVATE_RECORD,
    "seed": 42,
}


@pytest.mark.parametrize(
    ("maximum_frequency", "expected"),
    [
        (20.0, np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])),
        (19.0, np.array([10.0, 12.0, 14.0, 16.0, 18.0])),
    ],
)
def test_generator_includes_largest_in_band_bin(
    maximum_frequency: float, expected: np.ndarray
) -> None:
    generator = PolarizationPowerGenerator(
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
    generator = AnalyticInspiralGenerator(
        alpha=ISCO_ALPHA,
        approximant="AnalyticInspiral",
        minimum_frequency=10.0,
        maximum_frequency=12.0,
        reference_frequency=10.0,
        sampling_frequency=32.0,
        df=2.0,
    )
    catalog = Catalog.from_generator(
        source_parameters,
        generator=generator,
        seed=42,
        **POPULATION_RECORD,
    )

    assert catalog.waveform_metadata is generator
    assert catalog.seed == 42
    assert catalog.num_samples == 2
    assert catalog.source_parameters["integer_label"].dtype == np.int16


def test_analytic_generator_evaluates_on_exact_metadata_grid(
    source_parameters: dict[str, np.ndarray],
) -> None:
    generator = AnalyticInspiralGenerator(
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

    assert isinstance(actual, jax.Array)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("power", "message"),
    [
        (np.ones(2), "two-dimensional"),
        (np.ones((3, 2)), "frequency axis"),
        (np.ones((2, 3)), "source parameter"),
        (np.ones((2, 0)), "at least one sample"),
        (np.ones((2, 2), dtype=np.complex128), "real-valued"),
    ],
)
def test_catalog_rejects_malformed_power(power: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Catalog(
            source_parameters={"redshift": np.array([0.1, 0.2])},
            polarization_power=power,
            waveform_metadata=_waveform_generator(),
            **CATALOG_DEFAULTS,
        )


@pytest.mark.parametrize("values", [np.ones((2, 1)), np.ones(1), np.ones(3)])
def test_catalog_rejects_malformed_source_parameters(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="source parameter"):
        Catalog(
            source_parameters={"redshift": values},
            polarization_power=np.ones((2, 2)),
            waveform_metadata=_waveform_generator(),
            **CATALOG_DEFAULTS,
        )


def test_catalog_rejects_non_int_seed() -> None:
    with pytest.raises(TypeError, match="seed"):
        Catalog(
            source_parameters={"redshift": np.array([0.1, 0.2])},
            polarization_power=np.ones((2, 2)),
            waveform_metadata=_waveform_generator(),
            seed="not_an_int",  # ty: ignore[invalid-argument-type]
            **PRIVATE_RECORD,
        )


def test_catalog_requires_a_redshift_column() -> None:
    """The one source parameter whose density never cancels in a weight."""
    with pytest.raises(ValueError, match="redshift"):
        Catalog(
            source_parameters={"source_frame_mass_1": np.array([1.4, 1.3])},
            polarization_power=np.ones((2, 2)),
            waveform_metadata=_waveform_generator(),
            **CATALOG_DEFAULTS,
        )


def _catalog(redshift: np.ndarray) -> Catalog:
    num_samples = redshift.size
    return Catalog(
        source_parameters={
            "redshift": redshift,
            "luminosity_distance": 1e3 * (1.0 + redshift),
        },
        polarization_power=np.arange(2 * num_samples, dtype=np.float64).reshape(
            2, num_samples
        ),
        waveform_metadata=_waveform_generator(),
        **CATALOG_DEFAULTS,
    )


def test_get_population_model_binds_construction_settings_only() -> None:
    """Generating hyperparameters must not be captured in the bound callable.

    They describe how the catalog was made; a target evaluation supplies its
    own, and binding the generating ones here would silently pin them.
    """
    from astrogwb.populations.bns_madau_dickinson import bns_md_cosmological

    model = _catalog(np.array([0.5, 1.5])).get_population_model()
    assert model.fn is bns_md_cosmological
    assert dict(model.settings) == POPULATION_RECORD["model_kwargs"]
    assert model.density_sites == POPULATION_RECORD["density_sites"]


def test_unknown_population_model_names_fail_clearly() -> None:
    catalog = _catalog(np.array([0.5, 1.5]))
    object.__setattr__(catalog, "_model_name", "no_such_population")
    with pytest.raises(KeyError, match="bns_md_cosmological"):
        catalog.get_population_model()


def test_restrict_redshift_narrows_the_samples_and_the_population_together() -> None:
    """Truncating changes the density's *normalization*, so both must move."""
    catalog = _catalog(np.array([0.1, 0.5, 1.5, 19.0]))
    restricted = catalog.restrict_redshift(0.3, 2.0)

    np.testing.assert_array_equal(
        restricted.source_parameters["redshift"], np.array([0.5, 1.5])
    )
    np.testing.assert_array_equal(
        restricted.polarization_power, catalog.polarization_power[:, [1, 2]]
    )
    assert restricted.num_samples == 2
    assert restricted.population_model_kwargs["z_min"] == 0.3
    assert restricted.population_model_kwargs["z_max"] == 2.0
    # Everything else about the record travels unchanged.
    assert restricted.fiducials == catalog.fiducials
    assert restricted.density_sites == catalog.density_sites


def test_restrict_redshift_leaves_the_original_untouched() -> None:
    catalog = _catalog(np.array([0.1, 0.5, 1.5, 19.0]))
    catalog.restrict_redshift(0.3, 2.0)

    assert catalog.num_samples == 4
    assert catalog.polarization_power.shape == (2, 4)
    assert catalog.population_model_kwargs["z_min"] == 0.0


def test_restrict_redshift_rejects_a_window_outside_the_generation_support() -> None:
    catalog = _catalog(np.array([0.5, 1.5]))
    with pytest.raises(ValueError, match="must lie within"):
        catalog.restrict_redshift(0.3, 25.0)


def test_restrict_redshift_rejects_an_empty_window() -> None:
    catalog = _catalog(np.array([0.5, 1.5]))
    with pytest.raises(ValueError, match="no samples"):
        catalog.restrict_redshift(5.0, 10.0)
