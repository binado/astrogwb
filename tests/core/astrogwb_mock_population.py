"""The mock BNS population, and what the test suite builds from it.

A plain helper module rather than part of ``conftest.py``. Its
``astrogwb_mock_population`` name is unique across the workspace, avoiding a
generic top-level helper name while pytest's default ``prepend`` import mode
places each unpackaged test directory on ``sys.path``.

``conftest.py`` re-exports the loaders here as fixtures; test modules may
import the constants and helpers directly.

The draw used to be a committed CSV, produced from a gwmock graph by a script,
because the isolated core suite must not depend on another package's RNG
details. It is drawn in process now: generation runs the same registered
NumPyro population the analysis evaluates densities against, so the whole draw
is astrogwb and JAX, reproducible from a seed, and guaranteed to be a sample
from exactly the density the tests reweight with.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from astrogwb.catalog import Catalog
from astrogwb.constants import ISCO_ALPHA
from astrogwb.importance.estimator import SpectralDensityImportanceEstimator
from astrogwb.populations import Population, build_population
from astrogwb.waveform import AnalyticInspiralGenerator


def derived_columns(
    model: Population,
    params: Mapping[str, ArrayLike],
    sources: Mapping[str, ArrayLike],
) -> dict[str, jax.Array]:
    """Replay a population at fixed source values, returning declared outputs.

    The test-side counterpart of the batched replay inside
    :meth:`astrogwb.populations.Population.sample`: sample sites take the
    supplied values, deterministic outputs are the model's recomputation.
    """
    trace = model.trace(params, sources)
    return {name: jnp.asarray(trace[name]["value"]) for name in model.source_sites}


#: Hyperparameters the mock injection is drawn at and built at.
#: ``local_merger_rate`` is in Gpc^-3 yr^-1; the rest feed the Madau-Dickinson
#: rate shape and the flat-LambdaCDM cosmology. ``xi_0 = 1`` makes the modified
#: propagation law reduce exactly to the cosmological one, which is what lets a
#: catalog drawn from ``bns_md_cosmological`` serve as its own proposal against
#: a ``bns_md_modified_propagation`` target.
FIDUCIALS: dict[str, float] = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "xi_0": 1.0,
    "xi_n": 1.91,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
}

#: The parameters the *generating* population takes: the modified-propagation
#: ones are target-side only.
POPULATION_PARAMS: dict[str, float] = {
    name: value for name, value in FIDUCIALS.items() if name not in {"xi_0", "xi_n"}
}

#: Redshift window and grid resolution shared by the catalog factory and every
#: test that reweights against it. They must agree: ``log w`` is exactly zero
#: only when the proposal density was evaluated on the *same* grid as the
#: target, so a test that built its own grid at a different resolution would
#: silently pick up interpolation-level weights.
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


def mock_population_model(n_grid: int = N_GRID) -> Population:
    """The generating population: Madau-Dickinson, standard propagation."""
    return build_population(
        "bns_md_cosmological",
        settings={"z_min": Z_MIN, "z_max": Z_MAX, "n_grid": n_grid},
    )


def mock_target_model(n_grid: int = N_GRID) -> Population:
    """The target population the mock catalog is reweighted to."""
    return build_population(
        "bns_md_modified_propagation",
        settings={"z_min": Z_MIN, "z_max": Z_MAX, "n_grid": n_grid},
    )


def load_mock_population(num_sources: int = 1024) -> dict[str, np.ndarray]:
    """Draw the mock population as plain ``(N,)`` float64 arrays."""
    samples = mock_population_model().sample(
        jax.random.PRNGKey(MOCK_POPULATION_SEED),
        POPULATION_PARAMS,
        num_samples=num_sources,
    )
    return {name: np.asarray(values) for name, values in samples.items()}


def mock_catalog(
    source_parameters: dict[str, np.ndarray],
    *,
    generator: AnalyticInspiralGenerator,
) -> Catalog:
    """Wrap a mock draw in a catalog carrying the population that produced it."""
    return Catalog.from_generator(
        source_parameters,
        generator=generator,
        model_name="bns_md_cosmological",
        model_kwargs={"z_min": Z_MIN, "z_max": Z_MAX, "n_grid": N_GRID},
        fiducials=POPULATION_PARAMS,
        density_sites=("redshift",),
        seed=MOCK_POPULATION_SEED,
    )


def build_mock_catalog(
    population: dict[str, np.ndarray],
    *,
    num_sources: int = 1024,
    f_min: float = 2.0,
    f_max: float = 4096.0,
    df: float = 8.0,
) -> Catalog:
    """Build a real ``Catalog`` from the mock population draw.

    The polarization power comes from
    :class:`~astrogwb.waveform.AnalyticInspiralGenerator`, so the catalog is
    a genuine closed-form inspiral bank -- no Ripple backend, no persisted
    file -- and :meth:`~astrogwb.catalog.Catalog.from_generator` self-validates,
    so a malformed mock fails at construction rather than deep inside a model.

    ``inclination`` is a column of exact zeros, declared that way by the
    population. That is not a defect -- it pairs exactly with
    ``average_mode="analytic_inclination"``, whose
    ``INCLINATION_AVERAGE_TO_FACE_ON_RATIO = <g>/g(0) = 0.4`` converts face-on
    power into the inclination average.
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
            alpha=ISCO_ALPHA,
            approximant="AnalyticInspiral",
            minimum_frequency=f_min,
            maximum_frequency=f_max,
            reference_frequency=f_min,
            sampling_frequency=2.0 * f_max,
            df=df,
        ),
    )


def catalog_samples(catalog: Catalog) -> dict[str, jax.Array]:
    """The catalog's source parameters as JAX arrays, keyed by name.

    ``Catalog.source_parameters`` is already a ``Mapping[str, NDArray]`` keyed by
    name, so this only crosses into JAX -- which every model in the suite wants
    and no test should have to restate.
    """
    return {
        name: jnp.asarray(values) for name, values in catalog.source_parameters.items()
    }


def synthetic_source_parameters(n_samples: int = 16) -> dict[str, jax.Array]:
    """Sources on an evenly spaced redshift ladder, with derived columns.

    Deliberately not a draw: an even ladder spans the whole window with a
    handful of sources, so a test that reweights it exercises the tails as well
    as the bulk. The derived columns come from the generating model itself, so
    the stored ``luminosity_distance`` is bit-identical to what any later
    evaluation recomputes -- which is what keeps the self-proposal log weights
    at exactly zero rather than at rounding noise.
    """
    ladder = jnp.linspace(Z_MIN, Z_MAX, n_samples)
    constant = jnp.ones(n_samples)
    stochastic = {
        "redshift": ladder,
        "source_frame_mass_1": 1.4 * constant,
        "source_frame_mass_2": 1.3 * constant,
        "spin_1z": 0.0 * constant,
        "spin_2z": 0.0 * constant,
        "lambda_1": 400.0 * constant,
        "lambda_2": 300.0 * constant,
    }
    return derived_columns(mock_population_model(), POPULATION_PARAMS, stochastic)


def build_synthetic_estimator(
    n_samples: int = 16,
    *,
    polarization_power: jax.Array | None = None,
    model: Population | None = None,
) -> tuple[SpectralDensityImportanceEstimator, dict[str, jax.Array]]:
    """Build an estimator whose proposal *is* its target at the fiducials.

    Shared by ``test_importance.py`` and ``test_amplitude_scalings.py``, which
    both need every log-weight to be exactly zero at ``FIDUCIALS``, so that any
    departure is attributable to the parameter under test rather than to the
    catalog. :meth:`SpectralDensityImportanceEstimator.from_catalog` is what
    makes that exact rather than approximate: the cached proposal density and
    reference distances are the *same expressions*, on the same inputs, that
    the target side will evaluate.

    ``polarization_power`` defaults to a single unit-power frequency bin --
    callers that only want rates and weights need no waveforms. Its sample axis
    must be ``n_samples``.
    """
    samples = synthetic_source_parameters(n_samples)
    if polarization_power is None:
        polarization_power = jnp.ones((1, n_samples))
    # A descriptor sized to whatever power the caller supplied: `Catalog`
    # checks the two against each other, and the frequencies themselves are
    # never used by anything reweighting this catalog.
    num_frequencies = int(jnp.shape(polarization_power)[0])
    generator = AnalyticInspiralGenerator(
        alpha=ISCO_ALPHA,
        approximant="AnalyticInspiral",
        minimum_frequency=F_MIN,
        maximum_frequency=F_MIN * num_frequencies,
        reference_frequency=F_MIN,
        sampling_frequency=2.0 * F_MAX,
        df=F_MIN,
    )
    catalog = Catalog(
        source_parameters={
            name: np.asarray(values) for name, values in samples.items()
        },
        polarization_power=np.asarray(polarization_power),
        waveform_metadata=generator,
        _model_name="bns_md_cosmological",
        _model_kwargs={"z_min": Z_MIN, "z_max": Z_MAX, "n_grid": N_GRID},
        _fiducials=POPULATION_PARAMS,
        _density_sites=("redshift",),
        seed=MOCK_POPULATION_SEED,
    )
    estimator = SpectralDensityImportanceEstimator.from_catalog(
        catalog,
        model=mock_target_model() if model is None else model,
        average_mode="analytic_inclination",
    )
    return estimator, samples
