r"""Closed-form quadrupolar, inspiral-only polarization power.

The leading-order stationary-phase amplitude of a circular, non-spinning
compact binary gives the polarization power in closed form:

.. math::

    |\tilde h_+(f)|^2 + |\tilde h_\times(f)|^2
    = \frac{5}{24\,\pi^{4/3}\,r^2 c^3}\,
      \frac{(G M_c)^{5/3}}{f^{7/3}}\, g(\iota),
    \qquad
    g(\iota) = \left(\frac{1 + \cos^2\iota}{2}\right)^2 + \cos^2\iota,

with :math:`M_c` the *detector-frame* chirp mass and :math:`r` the luminosity
distance. The inspiral is terminated at

.. math:: f_{\rm end} = \frac{\alpha}{(1 + z)\, M}

in geometrized units -- equivalently :math:`f_{\rm end} = \alpha c^3 / (G
M_{\rm det})` in Hz -- for a dimensionless :math:`\alpha` that the caller
chooses. :data:`ISCO_ALPHA` is the value putting :math:`f_{\rm end}` at the
Schwarzschild test-particle ISCO; nothing here defaults to it.

Everything is reduced to seconds internally. Writing :math:`(G M_c)^{5/3}/c^3
= (G M_c/c^3)^{5/3} c^2` turns the prefactor into

.. math::

    \frac{5}{24}\,\pi^{-4/3}\,
    \frac{M_c[\mathrm{s}]^{5/3}}{(r/c)[\mathrm{s}]^2},

so the result comes out in :math:`\mathrm{s}^2 = \mathrm{Hz}^{-2}` with no
explicit :math:`G` anywhere -- the same units as the Ripple-generated power
that :func:`astrogwb.waveform.polarization_power` reduces.

These functions are JAX-only and JAX-traceable: they evaluate through
:mod:`jax.numpy` and return :class:`jax.Array`, so the same code runs at
catalog-build time and inside a jitted NumPyro model on traced arrays.

.. warning::

    The power is of order :math:`10^{-47}\,\mathrm{Hz}^{-2}` for a BNS at a
    few hundred Mpc -- nine orders of magnitude below the smallest normal
    float32 (:math:`1.2\times10^{-38}`). ``jax_enable_x64`` must be on or the
    result underflows to zeros *silently*, with no warning and no NaN.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from astrogwb.cosmology import MPC_IN_METERS, SPEED_OF_LIGHT

__all__ = [
    "FACE_ON_INCLINATION_FACTOR",
    "ISCO_ALPHA",
    "MEAN_INCLINATION_FACTOR",
    "MPC_IN_SECONDS",
    "SOLAR_MASS_IN_SECONDS",
    "chirp_mass",
    "inclination_factor",
    "inspiral_polarization_power",
    "termination_frequency",
]

#: One solar mass in seconds, $G M_\odot / c^3$, from the IAU nominal
#: $G M_\odot = 1.32712440018 \times 10^{20}\,\mathrm{m^3\,s^{-2}}$. The
#: product is quoted rather than $G$ and $M_\odot$ separately because it is
#: known to far better precision than either factor.
SOLAR_MASS_IN_SECONDS: float = 1.32712440018e20 / SPEED_OF_LIGHT**3

#: One megaparsec in seconds, $\mathrm{Mpc}/c$.
MPC_IN_SECONDS: float = MPC_IN_METERS / SPEED_OF_LIGHT

#: $g(0) = 2$ -- the face-on inclination factor, the maximum of $g$.
FACE_ON_INCLINATION_FACTOR: float = 2.0

#: $\langle g(\iota)\rangle = 4/5$ for $\cos\iota$ uniform on $[-1, 1]$:
#: $\langle((1+\cos^2\iota)/2)^2\rangle = 7/15$ and
#: $\langle\cos^2\iota\rangle = 1/3$. The ratio to
#: :data:`FACE_ON_INCLINATION_FACTOR` is exactly the ``0.4`` that
#: :func:`astrogwb.gwb.spectral_density` applies in ``"analytic_inclination"``
#: mode, which is what makes that constant correct for the face-on catalogs
#: the population graphs generate.
MEAN_INCLINATION_FACTOR: float = 0.8

#: $\alpha = 1/(\pi\,6^{3/2}) \approx 0.02166$, placing $f_{\rm end}$ at the
#: dominant *gravitational-wave* frequency -- twice the orbital frequency -- of
#: a test particle at the innermost stable circular orbit of a Schwarzschild
#: black hole of mass $M$. A physical reference point for $\alpha$, not a
#: default: callers pass $\alpha$ explicitly.
ISCO_ALPHA: float = 1.0 / (math.pi * 6.0**1.5)

#: $\tfrac{5}{24}\pi^{-4/3}$, the amplitude-squared prefactor in geometrized units.
_AMPLITUDE_PREFACTOR: float = (5.0 / 24.0) * math.pi ** (-4.0 / 3.0)


def inclination_factor(inclination: jax.Array | float) -> jax.Array:
    r"""Quadrupolar inclination factor $g(\iota)$.

    .. math::

        g(\iota) = \left(\frac{1 + \cos^2\iota}{2}\right)^2 + \cos^2\iota

    The two terms are the plus and cross contributions to
    $|\tilde h_+|^2 + |\tilde h_\times|^2$. It runs from $g(\pi/2) = 1/4$
    edge-on to $g(0) = g(\pi) = 2$ face-on/face-off.

    Parameters
    ----------
    inclination:
        Inclination angle in radians.
    """
    cos_squared = jnp.cos(inclination) ** 2
    return ((1.0 + cos_squared) / 2.0) ** 2 + cos_squared


def chirp_mass(mass_1: jax.Array | float, mass_2: jax.Array | float) -> jax.Array:
    r"""Chirp mass $M_c = (m_1 m_2)^{3/5} / (m_1 + m_2)^{1/5}$.

    Frame-agnostic and unit-agnostic: the result carries whatever mass unit
    and frame the components came in. Scaling both components by $(1 + z)$
    scales $M_c$ by $(1 + z)$, which is how
    :func:`inspiral_polarization_power` reaches the detector-frame chirp mass.
    """
    mass_1, mass_2 = jnp.asarray(mass_1), jnp.asarray(mass_2)
    return (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2


def termination_frequency(
    mass_1: jax.Array | float,
    mass_2: jax.Array | float,
    redshift: jax.Array | float,
    *,
    alpha: jax.Array | float,
) -> jax.Array:
    r"""Frequency in Hz at which the inspiral is truncated.

    .. math:: f_{\rm end} = \frac{\alpha}{(1 + z)\, M}

    in geometrized units, i.e. $\alpha c^3 / (G M_{\rm det})$ in Hz with
    $M_{\rm det} = (1 + z)(m_1 + m_2)$ the detector-frame total mass.

    Parameters
    ----------
    mass_1, mass_2:
        Source-frame component masses in solar masses.
    redshift:
        Source redshift.
    alpha:
        Dimensionless truncation parameter. See :data:`ISCO_ALPHA` for the
        value corresponding to the Schwarzschild test-particle ISCO.

    Returns
    -------
    jax.Array
        The truncation frequency in Hz.
    """
    total_mass = jnp.asarray(mass_1) + mass_2
    return alpha / ((1.0 + redshift) * total_mass * SOLAR_MASS_IN_SECONDS)


def inspiral_polarization_power(
    frequencies: jax.Array,
    *,
    mass_1: jax.Array,
    mass_2: jax.Array,
    redshift: jax.Array,
    luminosity_distance: jax.Array,
    inclination: jax.Array | float,
    alpha: jax.Array | float,
) -> jax.Array:
    r"""Polarization power $|\tilde h_+|^2 + |\tilde h_\times|^2$ on an ``(N, F)`` grid.

    Evaluates the module's closed form at every ``(source, frequency)`` pair
    and zeroes the bins above each source's :func:`termination_frequency`.
    Output is in $\mathrm{Hz}^{-2}$, laid out ``(sample, frequency)`` -- the
    same orientation the Ripple backend produces before
    :func:`astrogwb.waveform.polarization_power` transposes it. The on-disk
    catalog format is ``(frequency, sample)``, so
    :func:`astrogwb.waveform.make_catalog` takes the transpose::

        make_catalog(..., polarization_power=power.T)

    .. warning::

        The truncation is a hard mask, so the power is not differentiable in
        ``alpha``: its gradient with respect to ``alpha`` is zero wherever it
        exists. ``alpha`` is a fixed or grid-scanned choice, not something to
        put behind a ``numpyro.sample`` site, unless a smooth taper replaces
        the mask.

    Parameters
    ----------
    frequencies:
        Frequency grid in Hz, shape ``(F,)``. A zero bin returns zero power
        rather than the divergence of $f^{-7/3}$.
    mass_1, mass_2:
        Source-frame component masses in solar masses, broadcasting to
        ``(N,)`` against the other source parameters.
    redshift:
        Source redshift. Enters only by redshifting the masses -- both the
        chirp mass in the amplitude and the total mass in the cutoff.
    luminosity_distance:
        Luminosity distance in Mpc.
    inclination:
        Inclination angle in radians; see :func:`inclination_factor`.
    alpha:
        Dimensionless truncation parameter; see :func:`termination_frequency`.

    Returns
    -------
    jax.Array
        Polarization power of shape ``(N, F)`` in $\mathrm{Hz}^{-2}$.
    """
    detector_frame_chirp_mass_seconds = (
        chirp_mass(mass_1, mass_2) * (1.0 + redshift) * SOLAR_MASS_IN_SECONDS
    )
    distance_seconds = luminosity_distance * MPC_IN_SECONDS
    amplitude_squared = (
        _AMPLITUDE_PREFACTOR
        * detector_frame_chirp_mass_seconds ** (5.0 / 3.0)
        / distance_seconds**2
        * inclination_factor(inclination)
    )
    cutoff = termination_frequency(mass_1, mass_2, redshift, alpha=alpha)

    # (N, 1) against (1, F) builds the (N, F) layout directly -- no full-size
    # transpose is ever materialized. `reshape` rather than `[:, None]` so a
    # single-source catalog may pass 0-d source parameters.
    frequency_row = jnp.reshape(frequencies, (1, -1))
    amplitude_column = jnp.reshape(amplitude_squared, (-1, 1))
    cutoff_column = jnp.reshape(cutoff, (-1, 1))

    # f = 0 would make f**(-7/3) infinite, and `where` evaluates both branches:
    # inf * 0 is NaN, and under reverse-mode JAX that NaN propagates into the
    # gradient even though the bin is masked away. Substitute a safe frequency
    # before taking the power, then mask.
    inside_band = (frequency_row > 0.0) & (frequency_row <= cutoff_column)
    safe_frequency = jnp.where(frequency_row > 0.0, frequency_row, 1.0)
    power = amplitude_column * safe_frequency ** (-7.0 / 3.0)
    return jnp.where(inside_band, power, 0.0)
