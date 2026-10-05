"""Small paper-format catalog builders shared by tests.

Test catalogs are *real* catalogs: they carry a registered population, and
every derived column is computed by that population from the stochastic ones,
matching what any later evaluation recomputes from the stored samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from repo import REPO_ROOT

from astrogwb.paper.config import fiducials, population_metadata
from astrogwb.populations import PopulationMetadata
from astrogwb.populations.evaluation import evaluate_sources
from astrogwb.simulators.polarization_power import (
    CatalogMetadata,
    PolarizationPowerCatalog,
)
from astrogwb.waveform import WaveformMetadata

#: The population every fixture catalog is drawn from: the committed one, read
#: rather than restated, so a fixture cannot drift from what the runs sample
#: against. ``n_grid`` is the one deliberate difference -- 256 keeps the
#: cosmology integrals cheap enough for a unit test -- and it is written as an
#: override so the difference is visible instead of buried in a retyped table.
PAPER_POPULATION = population_metadata(REPO_ROOT, n_grid=256)

#: The hyperparameters fixtures draw at: the shared ``[fiducials]``, which is
#: what a real catalog inherits. It carries ``xi_0 = 1`` and ``xi_n``, so the
#: modified-propagation factor is applied and is exactly one, as it is in
#: generation.
PAPER_POPULATION_PARAMS: dict[str, float] = fiducials(REPO_ROOT)


def _derived_columns(model, params, sources):
    """Replay a source model at fixed source values, returning declared outputs.

    The same isolated, plated pass generation and every later evaluation take,
    so stored deterministics match their recomputation bit for bit.
    """
    _, outputs = evaluate_sources(model, params, sources, density_sites=())
    return outputs


def source_parameters(
    redshift: np.ndarray,
    *,
    population: PopulationMetadata = PAPER_POPULATION,
    fiducials: Mapping[str, float] | None = None,
    population_params: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Complete a redshift ladder into every column the population declares."""
    if fiducials is None:
        fiducials = population_params
    model = population.build().source_model
    ones = np.ones_like(redshift)
    inclination = (
        {"inclination": 0.75 * ones}
        if population.model_kwargs.get("sample_inclination", True)
        else {}
    )
    columns = _derived_columns(
        model,
        fiducials or PAPER_POPULATION_PARAMS,
        {
            "redshift": redshift,
            "source_frame_mass_1": 1.4 * ones,
            "source_frame_mass_2": 1.3 * ones,
            "spin_1z": 0.0 * ones,
            "spin_2z": 0.0 * ones,
            "lambda_1": 400.0 * ones,
            "lambda_2": 300.0 * ones,
            **inclination,
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
    population: PopulationMetadata = PAPER_POPULATION,
    fiducials: Mapping[str, float] | None = None,
    population_params: Mapping[str, float] | None = None,
    extra_source_parameters: Mapping[str, np.ndarray] | None = None,
) -> PolarizationPowerCatalog:
    """Build a valid paper-format catalog over a chosen redshift ladder."""
    if fiducials is None:
        fiducials = population_params
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
        population=population,
        fiducials=fiducials,
    )
    if extra_source_parameters is not None:
        parameters.update(
            {
                name: np.asarray(values)
                for name, values in extra_source_parameters.items()
            }
        )

    return PolarizationPowerCatalog(
        source_parameters=parameters,
        polarization_power=polarization_power,
        frequencies=minimum_frequency
        + df * np.arange(num_frequencies, dtype=np.float64),
        _metadata=CatalogMetadata(
            waveform=WaveformMetadata(
                approximant=approximant,
                minimum_frequency=minimum_frequency,
                maximum_frequency=minimum_frequency + df * (num_frequencies - 1),
                reference_frequency=reference_frequency,
                sampling_frequency=sampling_frequency,
                frequency_resolution=df,
            ),
            population=population,
            fiducials={
                name: float(value)
                for name, value in (fiducials or PAPER_POPULATION_PARAMS).items()
            },
            num_samples=int(np.shape(polarization_power)[1]),
        ),
    )


def save_catalog(
    catalog: PolarizationPowerCatalog, path: Path, *, seed: int = 41
) -> None:
    """Write ``catalog`` in the cache's file format, as ``polarization_power`` would.

    Tests that hand a run a catalog *by path* need a file; this is the same
    writer the cache uses, fed the catalog's arrays and an explicit seed.
    """
    from astrogwb.simulators.core.cache import _save_atomically

    _save_atomically(
        path,
        name="polarization_power",
        inputs={"seed": np.uint64(seed)},
        outputs={
            "frequencies": catalog.frequencies,
            "polarization_power": catalog.polarization_power,
            "source_parameters": dict(catalog.source_parameters),
        },
        metadata=catalog.metadata,
    )
