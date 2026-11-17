"""The mock BNS population, and what the test suite builds from it.

The module name is unique because pytest's ``prepend`` import mode puts each
test directory on ``sys.path``. ``conftest.py`` re-exports the loaders as
fixtures; test modules may import the constants and helpers directly.

The draw runs the same registered NumPyro population the analysis evaluates
densities against, so it is seeded and a sample from exactly the density the
tests reweight with.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from astrogwb.constants import ISCO_ALPHA, SECONDS_PER_YEAR
from astrogwb.distributions.orientation import UniformCosineDistribution
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.importance.catalog import (
    CatalogMetadata,
    IntensityData,
)
from astrogwb.importance.spectrum import (
    reference_catalog,
)
from astrogwb.inference import CatalogModel, ReferenceCatalog
from astrogwb.populations import (
    BNSCobaPopulation,
    Population,
)
from astrogwb.populations._types import SourceModel
from astrogwb.utils.numpyro import sample_and_logprob, sample_sources_qmc
from astrogwb.waveform import AnalyticInspiralGenerator, WaveformMetadata

#: The extrinsic sites of a physical source.
EXTRINSIC_SITES = (
    "declination",
    "right_ascension",
    "polarization_angle",
    "coa_phase",
    "coa_time",
)


def extrinsic_priors(
    coa_time_bounds: tuple[float, float] = (-0.5, 0.5),
) -> dict[str, dist.Distribution]:
    """The prior of each :data:`EXTRINSIC_SITES` entry.

    Declination is the cosine law shifted from colatitude to
    :math:`[-\\pi/2, \\pi/2]`.

    Parameters
    ----------
    coa_time_bounds
        Lower and upper geocentric arrival times in seconds.
    """
    return {
        "declination": dist.TransformedDistribution(
            UniformCosineDistribution(),
            dist.transforms.AffineTransform(-math.pi / 2.0, 1.0),
        ),
        "right_ascension": dist.Uniform(0.0, 2.0 * math.pi),
        "polarization_angle": dist.Uniform(0.0, math.pi),
        "coa_phase": dist.Uniform(0.0, 2.0 * math.pi),
        "coa_time": dist.Uniform(*coa_time_bounds),
    }


def distance_model(redshift: RedshiftDistribution) -> Callable[[], None]:
    """A NumPyro model of a source's redshift and GW luminosity distance."""

    def model() -> None:
        z = jnp.asarray(numpyro.sample("redshift", redshift))
        gw_distance = redshift.luminosity_distance(z) * redshift.distance_ratio(z)
        numpyro.deterministic("luminosity_distance", gw_distance)

    return model


def source_draw_model(
    redshift: RedshiftDistribution,
    source_model: SourceModel,
    *,
    coa_time_bounds: tuple[float, float] = (-0.5, 0.5),
) -> Callable[[], None]:
    """Compose source sites for density and replay tests."""

    def model() -> None:
        distance_model(redshift)()
        source_model()
        for name, prior in extrinsic_priors(coa_time_bounds).items():
            numpyro.sample(name, prior)

    return model


def population_models(
    population: Population,
    params: Mapping[str, ArrayLike],
    *,
    minimum_redshift: float = 0.3,
    maximum_redshift: float = 20.0,
    n_grid: int = 256,
) -> tuple[RedshiftDistribution, SourceModel]:
    """Build the redshift sampler and source model for replay tests."""
    return (
        RedshiftDistribution(
            params=params,
            source_frame_distribution=population.merger_rate,
            minimum_redshift=minimum_redshift,
            maximum_redshift=maximum_redshift,
            n_grid=n_grid,
        ),
        population.source_model(params),
    )


def derived_columns(
    population: Population,
    params: Mapping[str, ArrayLike],
    sources: Mapping[str, ArrayLike],
) -> dict[str, jax.Array]:
    """Replay a population's model at fixed source values, returning declared outputs.

    Sample sites take the supplied values and deterministic outputs are
    recomputed through :func:`astrogwb.utils.numpyro.sample_and_logprob`.
    """
    model = source_draw_model(*population_models(population, params))
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
    return BNSCobaPopulation(
        mass_model="uniform",
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        n_grid=n_grid,
    ).build()


def load_mock_population(num_sources: int = 1024) -> dict[str, np.ndarray]:
    """Draw a Sobol population for stable spectrum integration checks."""
    model = source_draw_model(*population_models(mock_population(), POPULATION_PARAMS))
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
) -> tuple[IntensityData, CatalogMetadata]:
    """Pair a mock draw with the metadata of the population that produced it."""
    power = np.asarray(jax.jit(generator.generate_batch)(source_parameters))
    data = IntensityData(
        frequencies=np.asarray(generator.frequencies),
        intensity=power,
        source_parameters={
            name: np.asarray(values) for name, values in source_parameters.items()
        },
    )
    metadata = CatalogMetadata(
        waveform=generator.metadata,
        population=BNSCobaPopulation(
            mass_model="uniform",
            minimum_redshift=Z_MIN,
            maximum_redshift=Z_MAX,
            n_grid=N_GRID,
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
) -> tuple[IntensityData, CatalogMetadata]:
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


def catalog_samples(data: IntensityData) -> dict[str, jax.Array]:
    """The draw's source parameters as JAX arrays, keyed by name."""
    return {
        name: jnp.asarray(values) for name, values in data["source_parameters"].items()
    }


#: The density factors a reference-catalog spectrum may weight: the intrinsic
#: mass pair. Redshift is integrated by the quadrature instead.
INTRINSIC_DENSITY_SITES: tuple[str, ...] = (
    "source_frame_mass_1",
    "source_frame_mass_2",
)

#: Redshift nodes of the mock spectra: enough for a smooth kernel on the window.
NUM_REDSHIFT_NODES = 24


def build_reference_catalog(
    num_sources: int = 256,
    *,
    f_min: float = 2.0,
    f_max: float = 4096.0,
    frequency_resolution: float = 8.0,
) -> tuple[IntensityData, CatalogMetadata]:
    """A closed-form reference catalog of the mock population.

    Every draw sits at the window's lower edge, as
    :func:`~astrogwb.importance.spectrum.reference_catalog` places them; the power
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
        population=BNSCobaPopulation(
            mass_model="uniform",
            minimum_redshift=Z_MIN,
            maximum_redshift=Z_MAX,
            n_grid=N_GRID,
        ),
        fiducials={name: float(value) for name, value in POPULATION_PARAMS.items()},
        num_samples=num_sources,
    )
    return (
        reference_catalog(metadata, jax.random.PRNGKey(MOCK_POPULATION_SEED)),
        metadata,
    )


def uniform_bns_log_prob(
    intrinsic: Mapping[str, jax.Array], params: Mapping[str, ArrayLike]
) -> jax.Array:
    """Joint density of two ordered, independently uniform component masses."""
    primary, secondary = (intrinsic[name] for name in INTRINSIC_DENSITY_SITES)
    low, width = params["minimum_mass"], params["mass_width"]
    support = (low <= secondary) & (secondary <= primary) & (primary <= low + width)
    return jnp.where(support, jnp.log(2.0) - 2.0 * jnp.log(width), -jnp.inf)


def bns_reference_catalog(
    data: IntensityData, proposal: Mapping[str, ArrayLike]
) -> ReferenceCatalog:
    """Bind the ordered mass draws and their joint proposal log density."""
    columns = data["source_parameters"]
    intrinsic = {name: jnp.asarray(columns[name]) for name in INTRINSIC_DENSITY_SITES}
    return ReferenceCatalog(
        frequencies=data["frequencies"],
        intensity=data["intensity"],
        intrinsic=intrinsic,
        proposal_log_prob=uniform_bns_log_prob(intrinsic, proposal),
        reference_redshift=float(columns["redshift"][0]),
        reference_distance=float(columns["luminosity_distance"][0]),
    )


def build_reference_spectrum(
    data: IntensityData,
    metadata: CatalogMetadata,
    *,
    frequencies: ArrayLike | None = None,
    reweight_masses: bool = True,
    observation_time: float = 1.0,
    num_redshift_nodes: int = NUM_REDSHIFT_NODES,
    minimum_redshift: float | None = None,
) -> CatalogModel:
    """The mock reference catalog bound to its ordered uniform BNS target."""
    return CatalogModel.from_catalog(
        bns_reference_catalog(data, metadata.fiducials),
        merger_rate=metadata.population.build().merger_rate,
        frequencies=data["frequencies"] if frequencies is None else frequencies,
        num_redshift_nodes=num_redshift_nodes,
        minimum_redshift=(
            metadata.population.minimum_redshift
            if minimum_redshift is None
            else minimum_redshift
        ),
        maximum_redshift=metadata.population.maximum_redshift,
        observation_seconds=observation_time * SECONDS_PER_YEAR,
        log_prob=uniform_bns_log_prob if reweight_masses else None,
        target_matches_proposal=not reweight_masses,
    )
