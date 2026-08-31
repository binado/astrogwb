"""Physical constants and unit conversions shared across astrogwb.

A leaf module: it imports nothing from :mod:`astrogwb`, so every other module
may import it without risk of a cycle. All values are SI unless a name says
otherwise, and the tabulated ones are the full-precision LALSuite literals
(``lal/LALConstants.h``) so that astrogwb, ripple, and LALSimulation agree to
the last representable bit. Values that follow exactly from those literals are
derived here rather than re-typed, so they cannot drift apart.
"""

import math

__all__ = [
    "EARTH_MEAN_RADIUS_IN_METERS",
    "FACE_ON_INCLINATION_FACTOR",
    "GPC_IN_METERS",
    "GRAVITATIONAL_CONSTANT",
    "INCLINATION_AVERAGE_TO_FACE_ON_RATIO",
    "ISCO_ALPHA",
    "MEAN_INCLINATION_FACTOR",
    "MPC_IN_METERS",
    "MPC_IN_SECONDS",
    "SECONDS_PER_YEAR",
    "SOLAR_MASS_IN_KILOGRAMS",
    "SOLAR_MASS_IN_METERS",
    "SOLAR_MASS_IN_SECONDS",
    "SPEED_OF_LIGHT",
]

# --- Fundamental constants ---------------------------------------------------

# Speed of light in vacuum (LAL_C_SI)
SPEED_OF_LIGHT: float = 299792458.0  # m / s

# Newton's gravitational constant (LAL_G_SI)
GRAVITATIONAL_CONSTANT: float = 6.6743e-11  # m^3 / kg / s^2

# --- Solar mass, in the three units the inspiral formulae need ---------------
#
# The three are consistent to the last bit: GRAVITATIONAL_CONSTANT *
# SOLAR_MASS_IN_KILOGRAMS / SPEED_OF_LIGHT**3 reproduces SOLAR_MASS_IN_SECONDS
# exactly in float64, and likewise / SPEED_OF_LIGHT**2 for
# SOLAR_MASS_IN_METERS. They are tabulated rather than derived so that a value
# quoted from astrogwb is bit-identical to one quoted from LALSuite. Note that
# LAL fixes the mass from the IAU 2015 nominal GM_sun = 1.3271244e20
# m^3 s^-2; the older 1.32712440018e20 differs by 1.4e-10 relative.

# Solar mass (LAL_MSUN_SI)
SOLAR_MASS_IN_KILOGRAMS: float = 1.988409870698050731911960804878414216e30  # kg

# Geometrized nominal solar mass, G M_sun / c^2 (LAL_MRSUN_SI)
SOLAR_MASS_IN_METERS: float = 1.476625038050124729627979840144936351e3  # m

# Geometrized nominal solar mass time, G M_sun / c^3 (LAL_MTSUN_SI)
SOLAR_MASS_IN_SECONDS: float = 4.925490947641266978197229498498379006e-6  # s

# --- Distance ----------------------------------------------------------------

# Megaparsec (LAL_PC_SI x 1e6)
MPC_IN_METERS: float = 3.085677581491367278913937957796471611e22  # m

# Gigaparsec
GPC_IN_METERS: float = 1.0e3 * MPC_IN_METERS  # m

# Megaparsec light-travel time, Mpc / c
MPC_IN_SECONDS: float = MPC_IN_METERS / SPEED_OF_LIGHT  # s

# --- Time --------------------------------------------------------------------

# Julian year, 365.25 x 86400 s (LAL_YRJUL_SI)
SECONDS_PER_YEAR: float = 31_557_600.0  # s

# --- Earth geometry ----------------------------------------------------------

# IUGG mean Earth radius R_1. This is deliberately NOT LAL_REARTH_SI
# (6378136.6 m), which is the equatorial radius: the bundled gwfast overlap
# reduction function reference fixtures were generated with the mean radius,
# and the detector chord distances that feed them assume a sphere.
EARTH_MEAN_RADIUS_IN_METERS: float = 6.371e6  # m

# --- Inspiral model constants (dimensionless) --------------------------------

# Schwarzschild test-particle ISCO truncation coefficient, alpha = 1/(pi 6^3/2).
# Truncates the inspiral at the dominant gravitational-wave frequency -- twice
# the orbital frequency -- of a test particle at the innermost stable circular
# orbit of a Schwarzschild black hole of detector-frame total mass M:
#
#     f_end = alpha / (M_det * SOLAR_MASS_IN_SECONDS)   [Hz, M_det in Msun]
#
# The convention is *dimensionless*: alpha multiplies an inverse geometrized
# mass, never an inverse mass in solar masses. Multiply by
# 1 / SOLAR_MASS_IN_SECONDS = 2.0303e5 to recover the Hz-solar-mass form.
ISCO_ALPHA: float = 1.0 / (math.pi * 6.0**1.5)  # dimensionless

# Face-on quadrupolar inclination factor, g(0) = 2, the maximum of
# g(iota) = ((1 + cos^2 iota)/2)^2 + cos^2 iota.
FACE_ON_INCLINATION_FACTOR: float = 2.0  # dimensionless

# Inclination-averaged quadrupolar factor, <g> = 4/5 for cos iota uniform on
# [-1, 1]: <((1 + cos^2 iota)/2)^2> = 7/15 and <cos^2 iota> = 1/3.
MEAN_INCLINATION_FACTOR: float = 0.8  # dimensionless

# <g> / g(0) = 2/5. The factor by which a catalog of face-on sources must be
# scaled to represent an inclination-averaged population.
INCLINATION_AVERAGE_TO_FACE_ON_RATIO: float = (
    MEAN_INCLINATION_FACTOR / FACE_ON_INCLINATION_FACTOR
)  # dimensionless
