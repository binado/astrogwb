"""Physical constants and unit conversions shared across astrogwb.

A leaf module: it imports nothing from :mod:`astrogwb`. Values are SI unless a
name says otherwise; tabulated ones are the full-precision LALSuite literals
(``lal/LALConstants.h``), so astrogwb, ripple and LALSimulation agree bit for
bit.
"""

import math

__all__ = [
    "GPC_IN_METERS",
    "GRAVITATIONAL_CONSTANT",
    "ISCO_ALPHA",
    "MPC_IN_METERS",
    "MPC_IN_SECONDS",
    "SECONDS_PER_YEAR",
    "SOLAR_MASS_IN_KILOGRAMS",
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
# Tabulated rather than derived, to stay bit-identical to LALSuite (which uses
# the IAU 2015 nominal GM_sun = 1.3271244e20 m^3 s^-2). They are mutually
# consistent in float64.

# Solar mass (LAL_MSUN_SI)
SOLAR_MASS_IN_KILOGRAMS: float = 1.988409870698050731911960804878414216e30  # kg

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

# --- Inspiral model constants (dimensionless) --------------------------------

# Schwarzschild ISCO coefficient, alpha = 1/(pi 6^3/2): the dominant GW
# frequency at the ISCO of detector-frame total mass M_det (in Msun) is
#
#     f_end = alpha / (M_det * SOLAR_MASS_IN_SECONDS)   [Hz]
ISCO_ALPHA: float = 1.0 / (math.pi * 6.0**1.5)  # dimensionless
