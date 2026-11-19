"""Tests for quadrupolar Stokes factors."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from tolerances import assert_close

from astrogwb.waveform import quadrupolar_stokes_factors
from astrogwb.waveform.intensity import QUADRUPOLE_STOKES_WEIGHTS, StokesPair


@pytest.mark.parametrize(
    ("cos_inclination", "expected"),
    [
        (1.0, (2.0, 0.0, 0.0, -2.0)),
        (-1.0, (2.0, 0.0, 0.0, 2.0)),
        (0.0, (0.25, 0.25, 0.0, 0.0)),
    ],
    ids=["face-on", "face-off", "edge-on"],
)
def test_quadrupolar_stokes_factors_at_special_inclinations(
    cos_inclination: float, expected: tuple[float, ...]
) -> None:
    """Face-on and face-off are circular with opposite V, edge-on is linear."""
    factors = quadrupolar_stokes_factors(jnp.array([cos_inclination]))

    assert_close(factors[:, 0], expected)


@pytest.mark.parametrize(
    ("pair", "moment"),
    [
        ("QQ", lambda s: s[1] ** 2 / 2.0),
        ("UU", lambda s: s[1] ** 2 / 2.0),
        ("VV", lambda s: s[3] ** 2),
    ],
)
def test_quadrupole_stokes_weights_are_the_inclination_and_polarization_averages(
    pair: StokesPair, moment: Callable[[np.ndarray], np.ndarray]
) -> None:
    r"""The polarization average halves :math:`\langle Q^2 \rangle` between QQ and UU."""
    nodes, weights = np.polynomial.legendre.leggauss(8)
    factors = np.asarray(quadrupolar_stokes_factors(jnp.asarray(nodes)))

    assert QUADRUPOLE_STOKES_WEIGHTS[pair] == pytest.approx(
        float(weights @ moment(factors) / (weights @ factors[0] ** 2)), rel=1e-12
    )
