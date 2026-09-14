"""Small paper-format catalog builders shared by tests.

Test catalogs are *real* catalogs: they carry a registered population, and
every derived column is computed by that population from the stochastic ones,
matching what any later evaluation recomputes from the stored samples.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from astrogwb.catalog import PolarizationPowerCatalog
from astrogwb.populations import PopulationRecord, build_source_model
from astrogwb.utils.sampling import evaluate_sources
from astrogwb.waveform import PolarizationPowerGenerator

#: The population every fixture catalog is drawn from, matching what
#: ``config/catalogs/base/population.toml`` commits.
PAPER_MODEL = "bns_md_cosmological"
PAPER_RATE_MODEL = "madau_dickinson"
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
    "minimum_mass": 1.0,
    "mass_width": 1.5,
}


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
    model = build_source_model(model_name, settings=model_kwargs or PAPER_MODEL_KWARGS)
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
        waveform_metadata=PolarizationPowerGenerator(
            approximant=approximant,
            minimum_frequency=minimum_frequency,
            maximum_frequency=minimum_frequency + df * (num_frequencies - 1),
            reference_frequency=reference_frequency,
            sampling_frequency=sampling_frequency,
            frequency_resolution=df,
        ),
        _population=PopulationRecord(
            source_model_name=model_name,
            rate_model_name=PAPER_RATE_MODEL,
            model_kwargs=dict(model_kwargs or PAPER_MODEL_KWARGS),
            density_sites=density_sites,
            seed=seed,
        ),
        _fiducials=dict(fiducials or PAPER_POPULATION_PARAMS),
    )
