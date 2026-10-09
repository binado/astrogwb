"""Validate the H0 and local_merger_rate amplitude transforms against the
real rescaled quadrature spectrum.

Every other test in the module trusts the named amplitude transforms; this is
the one that checks them against
:func:`~astrogwb.gwb.importance.build_rescaled_spectrum` over a reference
catalog, rather than against a restatement of the same formulas. If the
cosmology or the redshift kernel ever changes, this test -- not a
documentation comment -- is what catches a drifted exponent.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

# The mock reference catalog is built at these fiducials; a second copy here
# would let the two drift apart silently.
from astrogwb_mock_population import (
    FIDUCIALS,
    build_reference_catalog,
    build_reference_spectrum,
)

from astrogwb.inference import (
    amplitude_H0_transform,
    amplitude_local_merger_rate_transform,
)

_TRANSFORMS = {
    "H0": amplitude_H0_transform,
    "local_merger_rate": amplitude_local_merger_rate_transform,
}


_PHI_FACTORS = (0.5, 0.8, 1.3, 2.0)


@pytest.mark.parametrize("parameter", tuple(_TRANSFORMS))
def test_amplitude_factorization_matches_the_real_spectral_density(
    parameter: str,
) -> None:
    estimator, _ = build_reference_spectrum(*build_reference_catalog(num_sources=16))
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
