"""Validate the H0 and local_merger_rate amplitude transforms against the
real importance-sampled spectral density.

Every other test in the module trusts the named amplitude transforms; this is
the one that checks them against
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
from astrogwb.inference import (
    amplitude_H0_transform,
    amplitude_local_merger_rate_transform,
)
from astrogwb.populations import AMPLITUDE_PARAMETERS

_TRANSFORMS = {
    "H0": amplitude_H0_transform,
    "local_merger_rate": amplitude_local_merger_rate_transform,
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
def test_amplitude_factorization_matches_the_real_spectral_density(
    parameter: str, synthetic_importance
) -> None:
    estimator = _spectrum(synthetic_importance)
    fiducial = FIDUCIALS[parameter]
    transform = _TRANSFORMS[parameter](fiducial)

    np.testing.assert_allclose(np.asarray(transform(jnp.asarray(fiducial))), 1.0)

    fiducial_spectral_density, _ = estimator(FIDUCIALS)

    for factor in _PHI_FACTORS:
        phi = factor * fiducial
        actual_spectral_density, _ = estimator({**FIDUCIALS, parameter: phi})

        amplitude = np.asarray(transform(jnp.asarray(phi)))
        np.testing.assert_allclose(
            np.asarray(actual_spectral_density),
            amplitude * np.asarray(fiducial_spectral_density),
            rtol=1e-8,
        )
