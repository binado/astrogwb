"""The committed mock BNS population, and what the test suite builds from it.

A plain helper module rather than part of ``conftest.py``. Its
``astrogwb_mock_population`` name is unique across the workspace, avoiding a
generic top-level helper name while pytest's default ``prepend`` import mode
places each unpackaged test directory on ``sys.path``.

``conftest.py`` re-exports the loaders here as fixtures; test modules may
import the constants and helpers directly.

The fixture itself is produced from the population graph committed beside it
by ``scripts/generate_mock_population_fixture.py``. See that script for why the
draw is a committed file rather than an in-process ``GraphSimulator`` call.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from astrogwb.catalog import (
    AnalyticInspiralGenerator,
    Catalog,
    FrequencyDomainWaveformMetadata,
    PopulationMetadata,
)
from astrogwb.catalog.importance import ImportanceCatalog
from astrogwb.constants import ISCO_ALPHA
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    bns_population,
    compute_merger_rate_distance_and_logprob,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

#: The committed frozen draw from the core test population graph.
MOCK_POPULATION_PATH = FIXTURES_DIR / "mock_bns_population.csv"

#: Hyperparameters the mock injection is built at.
#: ``local_merger_rate`` is in Gpc^-3 yr^-1; the rest feed the Madau-Dickinson
#: rate shape and the flat-LambdaCDM cosmology. They also match the population
#: graph the fixture was drawn from, which is what lets the catalog serve as
#: its own importance-sampling proposal.
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

#: Redshift window and grid resolution shared by the catalog factory and every
#: test that reweights against it. They must agree: ``log w`` is exactly zero
#: only when the proposal log-density was evaluated on the *same* grid as the
#: target, so a test that built its own grid at a different resolution would
#: silently pick up interpolation-level weights.
Z_MIN = 0.3
Z_MAX = 20.0
N_GRID = 256

#: Source-frame component-mass bounds of the pinned graph's
#: ``joint_uniform_mass_pair`` block, in solar masses.
MOCK_MINIMUM_COMPONENT_MASS = 1.0
MOCK_MAXIMUM_COMPONENT_MASS = 2.5

#: Seed used to generate the committed fixture.
MOCK_POPULATION_SEED = 41

#: Frequency band shared by the core test suite's mock-catalog analyses.
F_MIN = 2.0
F_MAX = 2048.0


def make_redshift_grid(n_grid: int = N_GRID) -> jax.Array:
    """Build the redshift grid the cosmology integrals run on."""
    return jnp.linspace(Z_MIN, Z_MAX, n_grid)


def load_mock_population() -> dict[str, np.ndarray]:
    """Read the committed draw as plain ``(N,)`` float64 arrays.

    ``luminosity_distance`` is deliberately absent from the file: it is
    recomputed by :func:`build_mock_catalog` from the fiducial cosmology, so the
    injection and the proposal share one source of truth instead of agreeing
    only to CSV round-trip precision.
    """
    table = np.genfromtxt(
        MOCK_POPULATION_PATH, names=True, delimiter=",", skip_header=1
    )
    names = table.dtype.names
    if names is None:
        raise ValueError(f"{MOCK_POPULATION_PATH}: missing the column-name row")
    return {name: np.asarray(table[name], dtype=np.float64) for name in names}


def build_mock_catalog(
    population: dict[str, np.ndarray],
    *,
    num_sources: int = 1024,
    f_min: float = 2.0,
    f_max: float = 4096.0,
    df: float = 8.0,
) -> Catalog:
    """Build a real ``Catalog`` from the committed population draw.

    The polarization power comes from
    :class:`~astrogwb.catalog.AnalyticInspiralGenerator`, so the catalog is
    a genuine closed-form inspiral bank -- no Ripple backend, no persisted
    file -- and :meth:`~astrogwb.catalog.Catalog.from_generator` self-validates,
    so a malformed mock fails at construction rather than deep inside a model.

    ``inclination`` is a column of exact zeros: the pinned graph sets
    ``constant_like(@redshift, 0.0)``. That is not a defect -- it pairs exactly
    with ``average_mode="analytic_inclination"``, whose
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
    _, luminosity_distance, _ = compute_merger_rate_distance_and_logprob(
        FIDUCIALS,
        {"redshift": jnp.asarray(parameters["redshift"])},
        redshift_grid=make_redshift_grid(),
    )
    parameters["luminosity_distance"] = np.asarray(luminosity_distance)

    return Catalog.from_generator(
        parameters,
        generator=AnalyticInspiralGenerator(alpha=ISCO_ALPHA),
        waveform_metadata=FrequencyDomainWaveformMetadata.from_bounds(
            approximant="AnalyticInspiral",
            minimum_frequency=f_min,
            maximum_frequency=f_max,
            reference_frequency=f_min,
            sampling_frequency=2.0 * f_max,
            df=df,
        ),
        population_metadata=PopulationMetadata(
            name="madau-dickinson",
            seed=MOCK_POPULATION_SEED,
            num_samples=num_sources,
            source_type="bns",
            provenance={
                "fixture": MOCK_POPULATION_PATH.name,
                "termination_alpha": ISCO_ALPHA,
                **{f"fiducial_{name}": value for name, value in FIDUCIALS.items()},
            },
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


def build_synthetic_importance_catalog(
    n_samples: int = 16,
    *,
    polarization_power: jax.Array | None = None,
) -> tuple[ImportanceCatalog, dict[str, jax.Array]]:
    """Build a catalog whose proposal *is* its target at the fiducials.

    Shared by ``test_importance.py`` and ``test_amplitude_scalings.py``, which
    both need every log-weight to be exactly zero at ``FIDUCIALS``, so that any
    departure is attributable to the parameter under test rather than to the
    catalog. ``from_population`` is what makes that exact rather than
    approximate: the cached proposal density and reference distance are the
    *same expressions*, evaluated on the same inputs, that
    ``compute_population_terms`` will produce on the target side.

    ``polarization_power`` defaults to a single unit-power frequency bin --
    callers that only want rates and weights need no waveforms. Its sample axis
    must be ``n_samples``.
    """
    redshift_grid = make_redshift_grid()
    samples: dict[str, jax.Array] = {"redshift": jnp.linspace(Z_MIN, Z_MAX, n_samples)}
    _, luminosity_distance, _ = compute_merger_rate_distance_and_logprob(
        FIDUCIALS, samples, redshift_grid=redshift_grid
    )
    # The EM distance, kept on the samples because tests that build their own
    # reference distances still read it. The catalog below is given the
    # *effective* distance instead.
    samples["luminosity_distance"] = luminosity_distance

    population = bns_population(FIDUCIALS, redshift_grid=redshift_grid)
    if polarization_power is None:
        polarization_power = jnp.ones((1, n_samples))
    catalog = ImportanceCatalog.from_population(
        population=population,
        source_parameters=samples,
        polarization_power=polarization_power,
        luminosity_distance=population.luminosity_distance(samples["redshift"]),
    )
    return catalog, samples
