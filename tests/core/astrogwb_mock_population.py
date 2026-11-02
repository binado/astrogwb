"""The mock BNS population, and what the test suite builds from it.

The module name is unique because pytest's ``prepend`` import mode puts each
test directory on ``sys.path``. ``conftest.py`` re-exports the loaders as
fixtures; test modules may import the constants and helpers directly.

The draw runs the same registered NumPyro population the analysis evaluates
densities against, so it is seeded and a sample from exactly the density the
tests reweight with.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from astrogwb.constants import ISCO_ALPHA
from astrogwb.gwb.catalog import (
    CatalogMetadata,
    PolarizationPowerData,
)
from astrogwb.gwb.importance import (
    reference_catalog,
)
from astrogwb.inference import ImportanceGaussianLikelihood
from astrogwb.populations import (
    DEFAULT_DENSITY_SITES,
    Population,
    PopulationMetadata,
    build_population,
    joint_model,
)
from astrogwb.utils.numpyro import sample_and_logprob, sample_sources_qmc
from astrogwb.waveform import AnalyticInspiralGenerator, WaveformMetadata


def derived_columns(
    population: Population,
    params: Mapping[str, ArrayLike],
    sources: Mapping[str, ArrayLike],
) -> dict[str, jax.Array]:
    """Replay a population's model at fixed source values, returning declared outputs.

    Sample sites take the supplied values and deterministic outputs are
    recomputed through :func:`astrogwb.utils.numpyro.sample_and_logprob`.
    """
    model = joint_model(*population(params))
    outputs, _ = sample_and_logprob(model, density_sites=(), values=sources)
    return outputs


#: Hyperparameters the mock injection is drawn at and built at.
#: ``local_merger_rate`` is in Gpc^-3 yr^-1; the rest feed the Madau-Dickinson
#: rate shape and the flat-LambdaCDM cosmology. ``xi_0 = 1`` reduces modified
#: propagation exactly to GR, so a catalog drawn without ``xi_0`` is its own
#: proposal against a target evaluated with it.
FIDUCIALS: dict[str, float] = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "xi_0": 1.0,
    "xi_n": 1.91,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
    "minimum_mass": 1.0,
    "mass_width": 1.5,
}

#: The parameters the *generating* population takes: the modified-propagation
#: ones are target-side only.
POPULATION_PARAMS: dict[str, float] = {
    name: value for name, value in FIDUCIALS.items() if name not in {"xi_0", "xi_n"}
}

#: Redshift window and grid shared by the catalog factory and every test that
#: reweights against it: ``log w`` is exactly zero only on the same grid.
Z_MIN = 0.3
Z_MAX = 20.0
N_GRID = 256

#: Source-frame component-mass bounds of the mock population, in solar masses.
MOCK_MINIMUM_COMPONENT_MASS = 1.0
MOCK_MAXIMUM_COMPONENT_MASS = 2.5

#: Seed the mock draw is made at.
MOCK_POPULATION_SEED = 41

#: Frequency band shared by the core test suite's mock-catalog analyses.
F_MIN = 2.0
F_MAX = 2048.0


def make_redshift_grid(n_grid: int = N_GRID) -> jax.Array:
    """Build the redshift grid the cosmology integrals run on."""
    return jnp.linspace(Z_MIN, Z_MAX, n_grid)


def mock_population(n_grid: int = N_GRID) -> Population:
    """The mock BNS population: Madau-Dickinson, uniform masses.

    The generating population and the reweighting target are the same
    registered model; they differ only in whether ``xi_0`` is among the
    hyperparameters it is called with.
    """
    return build_population(
        "bns_coba",
        mass_model="uniform",
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        n_grid=n_grid,
    )


def load_mock_population(num_sources: int = 1024) -> dict[str, np.ndarray]:
    """Draw a Sobol population for stable spectrum integration checks."""
    model = joint_model(*mock_population()(POPULATION_PARAMS))
    samples = sample_sources_qmc(
        model,
        jax.random.PRNGKey(MOCK_POPULATION_SEED),
        num_samples=num_sources,
    )
    return {name: np.asarray(values) for name, values in samples.items()}


def mock_catalog(
    source_parameters: dict[str, np.ndarray],
    *,
    generator: AnalyticInspiralGenerator,
) -> tuple[PolarizationPowerData, CatalogMetadata]:
    """Pair a mock draw with the metadata of the population that produced it."""
    power = np.asarray(jax.jit(generator.generate_batch)(source_parameters))
    data = PolarizationPowerData(
        frequencies=np.asarray(generator.frequencies),
        polarization_power=power,
        source_parameters={
            name: np.asarray(values) for name, values in source_parameters.items()
        },
    )
    metadata = CatalogMetadata(
        waveform=generator.metadata,
        population=PopulationMetadata(
            model_name="bns_coba",
            model_kwargs={
                "mass_model": "uniform",
                "minimum_redshift": Z_MIN,
                "maximum_redshift": Z_MAX,
                "n_grid": N_GRID,
            },
        ),
        fiducials={name: float(value) for name, value in POPULATION_PARAMS.items()},
        num_samples=int(power.shape[-1]),
    )
    return data, metadata


def build_mock_catalog(
    population: dict[str, np.ndarray],
    *,
    num_sources: int = 1024,
    f_min: float = 2.0,
    f_max: float = 4096.0,
    frequency_resolution: float = 8.0,
) -> tuple[PolarizationPowerData, CatalogMetadata]:
    """Build a ``(data, metadata)`` pair from the mock population draw.

    The power comes from :class:`~astrogwb.waveform.AnalyticInspiralGenerator`.

    """
    available_sources = min(values.shape[0] for values in population.values())
    if num_sources > available_sources:
        raise ValueError(
            f"requested {num_sources} sources, but the population contains only "
            f"{available_sources}"
        )

    # A prefix, not a random subsample, so catalog construction is deterministic.
    parameters = {name: values[:num_sources] for name, values in population.items()}
    return mock_catalog(
        parameters,
        generator=AnalyticInspiralGenerator(
            WaveformMetadata(
                alpha=ISCO_ALPHA,
                approximant="AnalyticInspiral",
                minimum_frequency=f_min,
                maximum_frequency=f_max,
                reference_frequency=f_min,
                sampling_frequency=2.0 * f_max,
                frequency_resolution=frequency_resolution,
            )
        ),
    )


def catalog_samples(data: PolarizationPowerData) -> dict[str, jax.Array]:
    """The draw's source parameters as JAX arrays, keyed by name."""
    return {
        name: jnp.asarray(values) for name, values in data["source_parameters"].items()
    }


#: The density factors a reference-catalog spectrum may weight: the default
#: sites minus redshift, which the quadrature integrates instead.
INTRINSIC_DENSITY_SITES: tuple[str, ...] = tuple(
    site for site in DEFAULT_DENSITY_SITES if site != "redshift"
)

#: Redshift nodes of the mock spectra: enough for a smooth kernel on the window.
NUM_REDSHIFT_NODES = 24


def build_reference_catalog(
    num_sources: int = 256,
    *,
    f_min: float = 2.0,
    f_max: float = 4096.0,
    frequency_resolution: float = 8.0,
) -> tuple[PolarizationPowerData, CatalogMetadata]:
    """A closed-form reference catalog of the mock population.

    Every draw sits at the window's lower edge, as
    :func:`~astrogwb.gwb.importance.reference_catalog` places them; the power
    comes from :class:`~astrogwb.waveform.AnalyticInspiralGenerator`.
    """
    generator = AnalyticInspiralGenerator(
        WaveformMetadata(
            alpha=ISCO_ALPHA,
            approximant="AnalyticInspiral",
            minimum_frequency=f_min,
            maximum_frequency=f_max,
            reference_frequency=f_min,
            sampling_frequency=2.0 * f_max,
            frequency_resolution=frequency_resolution,
        )
    )
    metadata = CatalogMetadata(
        waveform=generator.metadata,
        population=PopulationMetadata(
            model_name="bns_coba",
            model_kwargs={
                "mass_model": "uniform",
                "minimum_redshift": Z_MIN,
                "maximum_redshift": Z_MAX,
                "n_grid": N_GRID,
            },
        ),
        fiducials={name: float(value) for name, value in POPULATION_PARAMS.items()},
        num_samples=num_sources,
    )
    return (
        reference_catalog(metadata, jax.random.PRNGKey(MOCK_POPULATION_SEED)),
        metadata,
    )


def build_reference_spectrum(
    data: PolarizationPowerData,
    metadata: CatalogMetadata,
    *,
    frequencies: ArrayLike | None = None,
    density_sites: tuple[str, ...] = INTRINSIC_DENSITY_SITES,
    population: Population | None = None,
    **likelihood_kwargs: Any,
) -> ImportanceGaussianLikelihood:
    """The mock reference catalog bound to the mock target, spectrum only.

    ``frequencies`` is the observed grid, the catalog's own by default.
    ``likelihood_kwargs`` are ``from_catalog``'s data and shot-noise arguments.
    """
    return ImportanceGaussianLikelihood.from_catalog(
        data,
        metadata,
        population=mock_population() if population is None else population,
        frequencies=data["frequencies"] if frequencies is None else frequencies,
        num_redshift_nodes=NUM_REDSHIFT_NODES,
        density_sites=density_sites,
        **likelihood_kwargs,
    )
