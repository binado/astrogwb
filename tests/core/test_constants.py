"""Tests for the shared physical constants.

These are the regression tests for the class of bug that motivated collecting
the constants in one place: two modules independently encoding the same
physics, in silently incompatible units. The identities asserted here are what
keep :mod:`astrogwb.gwb.analytic` and :mod:`astrogwb.waveform.generator.analytical`
speaking the same language.
"""

from __future__ import annotations

import math

import pytest

from astrogwb.constants import (
    EARTH_MEAN_RADIUS_IN_METERS,
    FACE_ON_INCLINATION_FACTOR,
    GPC_IN_METERS,
    GRAVITATIONAL_CONSTANT,
    INCLINATION_AVERAGE_TO_FACE_ON_RATIO,
    ISCO_ALPHA,
    MEAN_INCLINATION_FACTOR,
    MPC_IN_METERS,
    MPC_IN_SECONDS,
    SECONDS_PER_YEAR,
    SOLAR_MASS_IN_KILOGRAMS,
    SOLAR_MASS_IN_METERS,
    SOLAR_MASS_IN_SECONDS,
    SPEED_OF_LIGHT,
)
from astrogwb.waveform import termination_frequency


def test_lal_solar_mass_triple_is_self_consistent() -> None:
    """The three solar-mass units must agree to float64 round-off.

    They are tabulated independently from LALSuite rather than derived, so
    nothing but this test stops a transcription error in one of them from going
    unnoticed. The relations hold bitwise for the current literals; `approx`
    leaves room for a future revision of ``GRAVITATIONAL_CONSTANT``.
    """
    assert GRAVITATIONAL_CONSTANT * SOLAR_MASS_IN_KILOGRAMS / SPEED_OF_LIGHT**3 == (
        pytest.approx(SOLAR_MASS_IN_SECONDS, rel=1e-16)
    )
    assert GRAVITATIONAL_CONSTANT * SOLAR_MASS_IN_KILOGRAMS / SPEED_OF_LIGHT**2 == (
        pytest.approx(SOLAR_MASS_IN_METERS, rel=1e-16)
    )
    assert SOLAR_MASS_IN_METERS / SPEED_OF_LIGHT == pytest.approx(
        SOLAR_MASS_IN_SECONDS, rel=1e-16
    )


def test_derived_conversions_match_their_definitions() -> None:
    """The derived constants are exactly their defining expressions."""
    assert GPC_IN_METERS == 1.0e3 * MPC_IN_METERS
    assert MPC_IN_SECONDS == MPC_IN_METERS / SPEED_OF_LIGHT
    assert SECONDS_PER_YEAR == 365.25 * 24.0 * 3600.0
    assert INCLINATION_AVERAGE_TO_FACE_ON_RATIO == (
        MEAN_INCLINATION_FACTOR / FACE_ON_INCLINATION_FACTOR
    )


def test_isco_alpha_is_dimensionless() -> None:
    """``ISCO_ALPHA`` is the geometrized coefficient, not the Hz-solar-mass one.

    :mod:`astrogwb.gwb.analytic` previously exported a constant of the same
    name equal to the second value below. Pinning both numbers here documents
    the factor separating the two conventions.
    """
    assert ISCO_ALPHA == pytest.approx(1.0 / (math.pi * 6.0**1.5), rel=1e-15)
    assert 0.02165 < ISCO_ALPHA < 0.02166
    assert ISCO_ALPHA / SOLAR_MASS_IN_SECONDS == pytest.approx(4397.2, rel=1e-4)


def test_isco_alpha_is_the_schwarzschild_isco_gw_frequency() -> None:
    """For a 1 solar-mass binary the cutoff is the Schwarzschild ISCO frequency.

    Twice the orbital frequency of a test particle at ``r = 6 G M / c^2``.
    """
    expected = SPEED_OF_LIGHT**3 / (
        6.0 ** (3.0 / 2.0) * math.pi * GRAVITATIONAL_CONSTANT * SOLAR_MASS_IN_KILOGRAMS
    )

    assert ISCO_ALPHA / SOLAR_MASS_IN_SECONDS == pytest.approx(expected, rel=1e-15)


def test_inclination_ratio_is_two_fifths() -> None:
    """The ratio ``<g> / g(0)`` is the factor applied to face-on catalogs."""
    assert INCLINATION_AVERAGE_TO_FACE_ON_RATIO == 0.4


def test_earth_radius_is_the_mean_not_equatorial_radius() -> None:
    """The overlap reduction fixtures assume the IUGG mean radius.

    ``LAL_REARTH_SI`` is the equatorial radius, 6378136.6 m. Substituting it
    would shift every detector separation angle and break the bundled gwfast
    reference comparison.
    """
    assert EARTH_MEAN_RADIUS_IN_METERS == 6.371e6


# `termination_frequency` returns a JAX array; this round-trip is only
# meaningful in x64.
def test_gwb_and_waveform_alpha_conventions_agree() -> None:
    """The two modules must invert the same cutoff relation.

    :func:`astrogwb.waveform.termination_frequency` maps a total mass to a
    cutoff frequency; :mod:`astrogwb.gwb.analytic` inverts that map to get the
    total mass contributing at each frequency. Feeding one into the other has
    to return the mass it started from -- which it did not when the two modules
    each owned an ``ISCO_ALPHA`` in different units.
    """
    mass_1, mass_2, redshift = 20.0, 15.0, 0.3

    cutoff = float(termination_frequency(mass_1, mass_2, redshift, alpha=ISCO_ALPHA))
    recovered = ISCO_ALPHA / (cutoff * (1.0 + redshift) * SOLAR_MASS_IN_SECONDS)

    assert recovered == pytest.approx(mass_1 + mass_2, rel=1e-12)
