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
import xarray as xr
from astrogwb.constants import ISCO_ALPHA
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
    make_merger_rate_and_log_weights_fn,
)
from astrogwb.importance.protocol import MergerRateAndLogWeightsFn
from astrogwb.waveform import inspiral_polarization_power, make_catalog

FIXTURES_DIR = Path(__file__).parent / "fixtures"

#: The committed frozen draw from the core test population graph.
MOCK_POPULATION_PATH = FIXTURES_DIR / "mock_bns_population.csv"

#: Hyperparameters the mock injection is built at, verbatim from
#: ``examples/h0_mcmc.py`` and ``examples/amplitude_marginalized_model.py``.
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
) -> xr.Dataset:
    """Build a real ``WaveformCatalog`` from the committed population draw.

    The polarization power comes from
    :func:`~astrogwb.waveform.inspiral_polarization_power`, so the catalog is a
    genuine closed-form inspiral bank -- no Ripple backend, no persisted file --
    and :func:`~astrogwb.waveform.make_catalog` self-validates, so a malformed
    mock fails at construction rather than deep inside a model.

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
    # Form every bin from its integer index rather than accumulating ``df``.
    # The grid starts at ``f_min`` even when it is not an integer multiple of
    # ``df`` (the default 2 Hz lower bound with df=8 Hz is the motivating case)
    # and ends at the greatest grid point not exceeding ``f_max``.
    num_frequency_bins = int(np.floor((f_max - f_min) / df)) + 1
    frequencies = np.asarray(
        f_min + df * np.arange(num_frequency_bins), dtype=np.float64
    )

    _, luminosity_distance, _ = compute_merger_rate_distance_and_logprob(
        FIDUCIALS,
        {"redshift": jnp.asarray(parameters["redshift"])},
        redshift_grid=make_redshift_grid(),
    )
    parameters["luminosity_distance"] = np.asarray(luminosity_distance)

    # inspiral_polarization_power lays out (sample, frequency); the catalog
    # format is frequency-first.
    power = np.asarray(
        inspiral_polarization_power(frequencies, parameters, alpha=ISCO_ALPHA)
    ).T

    return make_catalog(
        frequencies=frequencies,
        polarization_power=power,
        source_parameters=parameters,
        approximant="AnalyticInspiral",
        minimum_frequency=f_min,
        maximum_frequency=f_max,
        reference_frequency=f_min,
        sampling_frequency=2.0 * f_max,
        df=df,
        extra_attrs={
            "fixture": MOCK_POPULATION_PATH.name,
            "population": "madau-dickinson",
            "population_seed": MOCK_POPULATION_SEED,
            "num_sources": num_sources,
            "termination_alpha": ISCO_ALPHA,
            **{f"fiducial_{name}": value for name, value in FIDUCIALS.items()},
        },
    )


def build_synthetic_weights_callback(
    n_samples: int = 16,
) -> tuple[MergerRateAndLogWeightsFn, dict[str, jax.Array]]:
    """Build the reference weights callback over an evenly spaced redshift set.

    Shared by ``test_importance.py`` and ``test_amplitude_scalings.py``, which
    both need a callback whose proposal *is* its target at the fiducials, so
    every log-weight is exactly zero and any departure is attributable to the
    parameter under test rather than to the catalog.
    """
    redshift_grid = make_redshift_grid()
    samples: dict[str, jax.Array] = {"redshift": jnp.linspace(Z_MIN, Z_MAX, n_samples)}
    _, luminosity_distance, proposal_logprob = compute_merger_rate_distance_and_logprob(
        FIDUCIALS, samples, redshift_grid=redshift_grid
    )
    samples["luminosity_distance"] = luminosity_distance
    fn = make_merger_rate_and_log_weights_fn(
        fiducials=FIDUCIALS,
        redshift_grid=redshift_grid,
        proposal_logprob=proposal_logprob,
    )
    return fn, samples
