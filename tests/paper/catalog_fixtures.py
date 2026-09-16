"""Small paper-format catalog builders shared by tests.

Test catalogs are *real* catalogs: they carry a registered population, and
every derived column is computed by that population from the stochastic ones,
matching what any later evaluation recomputes from the stored samples.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from repo import REPO_ROOT

from astrogwb.catalog import PolarizationPowerCatalog
from astrogwb.metadata import CatalogMetadata, PopulationMetadata, WaveformMetadata
from astrogwb.paper.config import fiducials, population_metadata
from astrogwb.populations import build_population
from astrogwb.utils.sampling import evaluate_sources

#: The population every fixture catalog is drawn from: the committed one, read
#: rather than restated, so a fixture cannot drift from what the runs sample
#: against. ``n_grid`` is the one deliberate difference -- 256 keeps the
#: cosmology integrals cheap enough for a unit test -- and it is written as an
#: override so the difference is visible instead of buried in a retyped table.
#: The seed is a fixture detail; ``make_catalog`` takes its own.
PAPER_POPULATION = population_metadata(REPO_ROOT, seed=41, n_grid=256)
PAPER_MODEL = PAPER_POPULATION.model_name
PAPER_MODEL_KWARGS: dict[str, float | int] = dict(PAPER_POPULATION.model_kwargs)

#: The hyperparameters fixtures draw at: ``config/fiducials.json``, which is
#: what a real catalog inherits. It carries ``xi_0`` / ``xi_n`` that
#: ``bns_md_cosmological`` never reads -- source models index ``params`` by
#: name, so the extra entries are inert here exactly as they are in generation.
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
    model_name: str = PAPER_MODEL,
    model_kwargs: Mapping[str, float | int] | None = None,
    fiducials: Mapping[str, float] | None = None,
    population_params: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Complete a redshift ladder into every column the population declares."""
    if fiducials is None:
        fiducials = population_params
    model = build_population(
        model_name, **(model_kwargs or PAPER_MODEL_KWARGS)
    ).source_model
    ones = np.ones_like(redshift)
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
    seed: int = 41,
    model_name: str = PAPER_MODEL,
    model_kwargs: Mapping[str, float | int] | None = None,
    fiducials: Mapping[str, float] | None = None,
    population_params: Mapping[str, float] | None = None,
    density_sites: tuple[str, ...] = (
        "redshift",
        "source_frame_mass_1",
        "source_frame_mass_2",
    ),
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
        model_name=model_name,
        model_kwargs=model_kwargs,
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
            population=PopulationMetadata(
                model_name=model_name,
                model_kwargs=dict(model_kwargs or PAPER_MODEL_KWARGS),
                density_sites=density_sites,
                seed=seed,
            ),
        ),
        _fiducials=dict(fiducials or PAPER_POPULATION_PARAMS),
    )
