"""Small paper-format catalog builders shared by tests.

Test catalogs are *real* catalogs: they carry a registered population, and
every derived column is computed by that population from the stochastic ones.
That is not ceremony -- :meth:`Catalog.load` re-executes the recorded model and
compares its derived columns against the file, so a hand-assembled catalog with
invented distances would fail to load, exactly as a drifted production one
would.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from astrogwb.catalog import Catalog, PopulationMetadata
from astrogwb.populations import (
    population_model,
)
from astrogwb.waveform import PolarizationPowerGenerator

#: The population every fixture catalog is drawn from, matching what
#: ``config/catalogs/base/population.toml`` commits.
PAPER_MODEL = "bns_md_cosmological"
PAPER_MODEL_KWARGS: dict[str, float | int] = {
    "z_min": 0.0,
    "z_max": 20.0,
    "n_grid": 256,
}
PAPER_POPULATION_PARAMS: dict[str, float] = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
}


def source_parameters(
    redshift: np.ndarray,
    *,
    model_name: str = PAPER_MODEL,
    model_kwargs: Mapping[str, float | int] | None = None,
    population_params: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Complete a redshift ladder into every column the population declares."""
    model = population_model(model_name)(**model_kwargs or PAPER_MODEL_KWARGS)
    ones = np.ones_like(redshift)
    columns = model.derive_sources(
        population_params or PAPER_POPULATION_PARAMS,
        {
            "redshift": redshift,
            "source_frame_mass_1": 1.4 * ones,
            "source_frame_mass_2": 1.3 * ones,
            "spin_1z": 0.0 * ones,
            "spin_2z": 0.0 * ones,
            "lambda_1": 400.0 * ones,
            "lambda_2": 300.0 * ones,
        },
    )
    return {name: np.asarray(values) for name, values in columns.items()}


def make_catalog(
    *,
    redshift: np.ndarray,
    polarization_power: np.ndarray | None = None,
    num_frequencies: int = 5,
    approximant: str = "Toy",
    minimum_frequency: float = 10.0,
    reference_frequency: float = 20.0,
    sampling_frequency: float = 128.0,
    df: float = 10.0,
    name: str = PAPER_MODEL,
    seed: int = 41,
    source_type: str | None = "bns",
    provenance: Mapping[str, str | int | float] | None = None,
    model_name: str = PAPER_MODEL,
    model_kwargs: Mapping[str, float | int] | None = None,
    population_params: Mapping[str, float] | None = None,
    density_sites: tuple[str, ...] = ("redshift",),
    extra_source_parameters: Mapping[str, np.ndarray] | None = None,
) -> Catalog:
    """Build a valid paper-format catalog over a chosen redshift ladder."""
    redshift = np.asarray(redshift, dtype=np.float64)
    num_samples = redshift.size
    if polarization_power is None:
        polarization_power = np.arange(
            num_frequencies * num_samples, dtype=np.float64
        ).reshape(num_frequencies, num_samples)
    polarization_power = np.asarray(polarization_power, dtype=np.float64)
    num_frequencies = polarization_power.shape[0]

    parameters = source_parameters(
        redshift,
        model_name=model_name,
        model_kwargs=model_kwargs,
        population_params=population_params,
    )
    if extra_source_parameters is not None:
        parameters.update(
            {
                name: np.asarray(values)
                for name, values in extra_source_parameters.items()
            }
        )

    return Catalog(
        source_parameters=parameters,
        polarization_power=polarization_power,
        waveform_metadata=PolarizationPowerGenerator(
            approximant=approximant,
            minimum_frequency=minimum_frequency,
            maximum_frequency=minimum_frequency + df * (num_frequencies - 1),
            reference_frequency=reference_frequency,
            sampling_frequency=sampling_frequency,
            df=df,
        ),
        population_metadata=PopulationMetadata(
            name=name,
            seed=seed,
            num_samples=num_samples,
            source_type=source_type,
            provenance={} if provenance is None else dict(provenance),
        ),
        _model_name=model_name,
        _model_kwargs=dict(model_kwargs or PAPER_MODEL_KWARGS),
        _population_params=dict(population_params or PAPER_POPULATION_PARAMS),
        _density_sites=density_sites,
    )


def save_catalog(
    path: str | Path, catalog: Catalog, *, compression: str | None = None
) -> None:
    """Persist a fixture catalog through the production serializer."""
    catalog.save(path, compression=compression)
