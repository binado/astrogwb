"""Validate the H0^3/H0^2 and local_merger_rate amplitude scalings against the
real importance-sampled spectral density.

Every other test in the module trusts the named amplitude / merger-rate
scaling functions' exponents; this is the one that checks them against
:func:`~astrogwb.importance.spectral.importance_spectral_density`
over a synthetic catalog, rather than against a restatement of the same
formulas. If the cosmology or the importance weights ever change, this test
-- not a documentation comment -- is what catches a drifted exponent.
"""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest

# The `synthetic_importance` fixture builds its catalog at these fiducials; a
# second copy here would let the two drift apart silently.
from astrogwb_mock_population import FIDUCIALS

from astrogwb.importance.spectral import importance_spectral_density
from astrogwb.populations import (
    AMPLITUDE_PARAMETERS,
    amplitude_H0_fn,
    amplitude_local_merger_rate_fn,
    merger_rate_H0_fn,
    merger_rate_local_merger_rate_fn,
)

_SCALINGS = {
    "H0": (amplitude_H0_fn, merger_rate_H0_fn),
    "local_merger_rate": (
        amplitude_local_merger_rate_fn,
        merger_rate_local_merger_rate_fn,
    ),
}


_PHI_FACTORS = (0.5, 0.8, 1.3, 2.0)


def _spectrum(synthetic_importance, n_samples: int = 16):
    """The real spectrum over a synthetic catalog that is its own proposal.

    ``catalog_inclination`` keeps the contraction a plain weighted mean, so a
    drifted exponent shows up undivided by the 0.4 analytic factor.
    """
    rng = np.random.default_rng(0)
    importance, _ = synthetic_importance(
        n_samples,
        polarization_power=jnp.asarray(rng.uniform(0.5, 1.5, size=(5, n_samples))),
    )
    return partial(
        importance_spectral_density,
        **importance,
    )


@pytest.mark.parametrize("parameter", AMPLITUDE_PARAMETERS)
def test_merger_rate_amplitude_matches_the_real_spectrum(
    parameter: str, synthetic_importance
) -> None:
    estimator = _spectrum(synthetic_importance)
    fiducial = FIDUCIALS[parameter]
    _amplitude_fn, merger_rate_fn = _SCALINGS[parameter]

    _, fiducial_extras = estimator(FIDUCIALS)

    for factor in _PHI_FACTORS:
        phi = factor * fiducial
        _, extras = estimator({**FIDUCIALS, parameter: phi})
        # The scalings are absolute, so the physical claim is about the ratio
        # to the fiducial -- exactly what AmplitudeConditional forms.
        ratio = float(merger_rate_fn(jnp.asarray(phi))) / float(
            merger_rate_fn(jnp.asarray(fiducial))
        )
        np.testing.assert_allclose(
            float(extras["total_merger_rate"]),
            ratio * float(fiducial_extras["total_merger_rate"]),
            rtol=1e-10,
        )


@pytest.mark.parametrize("parameter", AMPLITUDE_PARAMETERS)
def test_amplitude_factorization_matches_the_real_spectral_density(
    parameter: str, synthetic_importance
) -> None:
    estimator = _spectrum(synthetic_importance)
    fiducial = FIDUCIALS[parameter]
    amplitude_fn, _merger_rate_fn = _SCALINGS[parameter]

    fiducial_spectral_density, _ = estimator(FIDUCIALS)

    for factor in _PHI_FACTORS:
        phi = factor * fiducial
        actual_spectral_density, _ = estimator({**FIDUCIALS, parameter: phi})

        amplitude = float(amplitude_fn(jnp.asarray(phi))) / float(
            amplitude_fn(jnp.asarray(fiducial))
        )
        np.testing.assert_allclose(
            np.asarray(actual_spectral_density),
            amplitude * np.asarray(fiducial_spectral_density),
            rtol=1e-8,
        )
