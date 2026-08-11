"""Validate the H0^3/H0^2 and local_merger_rate amplitude scalings against the
real merger-rate + importance-weights callback.

Every other test in the module trusts the named amplitude / merger-rate
scaling functions' exponents; this is the one that checks them against
:func:`~astrogwb.importance.models.bns_madau_dickinson_modified_propagation.make_merger_rate_and_log_weights_fn`
on a synthetic catalog, rather than against a restatement of the same
formulas. If the cosmology or the importance weights ever change, this test
-- not a documentation comment -- is what catches a drifted exponent.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.gwb import spectral_density
from astrogwb.importance.models import bns_madau_dickinson_modified_propagation as mod
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    AMPLITUDE_PARAMETERS,
    amplitude_H0_fn,
    amplitude_local_merger_rate_fn,
    compute_merger_rate_distance_and_logprob,
    make_merger_rate_and_log_weights_fn,
    merger_rate_H0_fn,
    merger_rate_local_merger_rate_fn,
)

FIDUCIALS = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "xi_0": 1.0,
    "xi_n": 1.91,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
}

Z_MIN = 0.0
Z_MAX = 20.0
N_GRID = 256
N_SAMPLES = 16

_SCALINGS = {
    "H0": (amplitude_H0_fn, merger_rate_H0_fn),
    "local_merger_rate": (
        amplitude_local_merger_rate_fn,
        merger_rate_local_merger_rate_fn,
    ),
}


def _build_synthetic_callback():
    z_grid = jnp.linspace(Z_MIN, Z_MAX, N_GRID)
    z_samples = jnp.linspace(0.01, Z_MAX - 0.01, N_SAMPLES)
    samples = {"redshift": z_samples}

    _, luminosity_distance, proposal_logprob = compute_merger_rate_distance_and_logprob(
        FIDUCIALS, samples, redshift_grid=z_grid
    )
    samples = {**samples, "luminosity_distance": luminosity_distance}
    fn = make_merger_rate_and_log_weights_fn(
        fiducials=FIDUCIALS,
        redshift_grid=z_grid,
        proposal_logprob=proposal_logprob,
    )
    return fn, samples


@pytest.mark.parametrize("parameter", AMPLITUDE_PARAMETERS)
def test_merger_rate_amplitude_matches_the_real_callback(parameter: str) -> None:
    fn, samples = _build_synthetic_callback()
    fiducial = FIDUCIALS[parameter]
    _amplitude_fn, merger_rate_fn = _SCALINGS[parameter]

    fiducial_rate, _ = fn(FIDUCIALS, samples)

    for phi in [0.5 * fiducial, 0.8 * fiducial, 1.3 * fiducial, 2.0 * fiducial]:
        params = {**FIDUCIALS, parameter: phi}
        rate, _ = fn(params, samples)
        # The scalings are absolute, so the physical claim is about the ratio
        # to the fiducial -- exactly what AmplitudeConditional forms.
        ratio = float(merger_rate_fn(jnp.asarray(phi))) / float(
            merger_rate_fn(jnp.asarray(fiducial))
        )
        np.testing.assert_allclose(
            float(rate), ratio * float(fiducial_rate), rtol=1e-10
        )


@pytest.mark.parametrize("parameter", AMPLITUDE_PARAMETERS)
def test_amplitude_factorization_matches_the_real_spectral_density(
    parameter: str,
) -> None:
    fn, samples = _build_synthetic_callback()
    fiducial = FIDUCIALS[parameter]
    amplitude_fn, _merger_rate_fn = _SCALINGS[parameter]

    rng = np.random.default_rng(0)
    polarization_power = jnp.asarray(rng.uniform(0.5, 1.5, size=(5, N_SAMPLES)))

    fiducial_rate, fiducial_log_weights = fn(FIDUCIALS, samples)
    fiducial_spectral_density = spectral_density(
        polarization_power,
        jnp.exp(fiducial_log_weights),
        fiducial_rate,
        average_mode="catalog_inclination",
    )

    for phi in [0.5 * fiducial, 0.8 * fiducial, 1.3 * fiducial, 2.0 * fiducial]:
        params = {**FIDUCIALS, parameter: phi}
        rate, log_weights = fn(params, samples)
        actual_spectral_density = spectral_density(
            polarization_power,
            jnp.exp(log_weights),
            rate,
            average_mode="catalog_inclination",
        )

        amplitude = float(amplitude_fn(jnp.asarray(phi))) / float(
            amplitude_fn(jnp.asarray(fiducial))
        )
        np.testing.assert_allclose(
            np.asarray(actual_spectral_density),
            amplitude * np.asarray(fiducial_spectral_density),
            rtol=1e-8,
        )


def test_amplitude_scalings_are_hashable_singletons() -> None:
    """``AmplitudeConditional`` carries the scaling as pytree *aux* data.

    JAX hashes aux data into the jit cache key, so a scaling that is not
    identity-stable across calls silently retraces the model every step. A
    lambda or a ``functools.partial`` over the fiducial would fail this.
    """
    pairs = (
        (amplitude_H0_fn, mod.amplitude_H0_fn),
        (merger_rate_H0_fn, mod.merger_rate_H0_fn),
        (amplitude_local_merger_rate_fn, mod.amplitude_local_merger_rate_fn),
        (merger_rate_local_merger_rate_fn, mod.merger_rate_local_merger_rate_fn),
    )
    for imported, via_module in pairs:
        assert imported is via_module
        assert hash(imported) == hash(via_module)
