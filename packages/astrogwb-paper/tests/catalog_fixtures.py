"""Small paper-format catalog builders shared by tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xarray as xr
from astrogwb.catalog import (
    Catalog,
    FrequencyDomainWaveformMetadata,
    PopulationMetadata,
)
from astrogwb.catalog.io import (
    catalog_from_dataset,
    catalog_to_dataset,
)
from astrogwb.catalog.io import (
    save_catalog as save_core_catalog,
)


def make_catalog(
    *,
    frequencies: np.ndarray,
    polarization_power: np.ndarray,
    source_parameters: Mapping[str, np.ndarray],
    approximant: str,
    minimum_frequency: float,
    maximum_frequency: float,
    reference_frequency: float,
    sampling_frequency: float,
    df: float,
    extra_attrs: Mapping[str, str | int | float] | None = None,
    population_metadata: PopulationMetadata | None = None,
) -> xr.Dataset:
    """Build the xarray representation used by paper runtime tests."""
    num_samples = np.asarray(polarization_power).shape[1]
    population = population_metadata or PopulationMetadata(
        name="test-population",
        seed=0,
        num_samples=num_samples,
        provenance={} if extra_attrs is None else extra_attrs,
    )
    core_catalog = Catalog(
        source_parameters={
            name: np.asarray(value) for name, value in source_parameters.items()
        },
        polarization_power=np.asarray(polarization_power),
        waveform_metadata=FrequencyDomainWaveformMetadata(
            frequencies=np.asarray(frequencies),
            approximant=approximant,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            reference_frequency=reference_frequency,
            sampling_frequency=sampling_frequency,
            df=df,
        ),
        population_metadata=population,
    )
    return catalog_to_dataset(core_catalog)


def save_catalog(
    path: str | Path,
    catalog: Catalog | xr.Dataset,
    *,
    compression: str | None = None,
) -> None:
    """Save either representation through the production paper serializer."""
    core_catalog = (
        catalog_from_dataset(catalog) if isinstance(catalog, xr.Dataset) else catalog
    )
    save_core_catalog(path, core_catalog, compression=compression)
