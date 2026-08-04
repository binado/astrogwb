from __future__ import annotations

import numpy as np
from astrogwb.cosmology import H0_SI, hubble_constant_si


def test_hubble_constant_si_matches_h0_si_default() -> None:
    np.testing.assert_allclose(hubble_constant_si(67.74), H0_SI)
    np.testing.assert_allclose(
        hubble_constant_si(67.66),
        H0_SI * (67.66 / 67.74),
    )
