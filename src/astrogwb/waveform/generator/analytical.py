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
chooses. :data:`~astrogwb.constants.ISCO_ALPHA` is the value putting :math:`f_{\rm end}` at the
Schwarzschild test-particle ISCO; nothing here defaults to it.

.. warning::

    The power is of order :math:`10^{-47}\,\mathrm{Hz}^{-2}` for a BNS at a
    few hundred Mpc -- nine orders of magnitude below the smallest normal
    float32 (:math:`1.2\times10^{-38}`). ``jax_enable_x64`` must be on or the
    result underflows to zeros *silently*, with no warning and no NaN;
    :func:`inspiral_polarization_power` therefore raises
    :class:`RuntimeError` unless it is.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike as JaxArrayLike
from numpy.typing import ArrayLike

from astrogwb.constants import MPC_IN_SECONDS, SOLAR_MASS_IN_SECONDS
from astrogwb.frequency import uniform_frequency_grid
from astrogwb.utils import require_x64
from astrogwb.waveform.generator.base import PolarizationPowerGenerator

__all__ = [
    "AnalyticInspiralGenerator",
    "chirp_mass",
    "inclination_factor",
    "inspiral_polarization_power",
    "termination_frequency",
]

#: $\tfrac{5}{24}\pi^{-4/3}$, the amplitude-squared prefactor in geometrized units.
_AMPLITUDE_PREFACTOR: float = (5.0 / 24.0) * math.pi ** (-4.0 / 3.0)


def inclination_factor(inclination: JaxArrayLike) -> jax.Array:
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


def chirp_mass(mass_1: JaxArrayLike, mass_2: JaxArrayLike) -> jax.Array:
    r"""Chirp mass $M_c = (m_1 m_2)^{3/5} / (m_1 + m_2)^{1/5}$.

    Frame-agnostic and unit-agnostic: the result carries whatever mass unit
    and frame the components came in. Scaling both components by $(1 + z)$
    scales $M_c$ by $(1 + z)$, which is how
    :func:`inspiral_polarization_power` reaches the detector-frame chirp mass.
    """
    mass_1, mass_2 = jnp.asarray(mass_1), jnp.asarray(mass_2)
    return (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2


def _require_scalar_alpha(alpha: JaxArrayLike) -> jax.Array:
    alpha_value = jnp.asarray(alpha)
    if alpha_value.ndim != 0:
        msg = (
            f"alpha must be a scalar; received an array with shape {alpha_value.shape}"
        )
        raise ValueError(msg)
    return alpha_value


def termination_frequency(
    mass_1: JaxArrayLike,
    mass_2: JaxArrayLike,
    redshift: JaxArrayLike,
    *,
    alpha: JaxArrayLike,
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
        Scalar dimensionless truncation parameter. See
        :data:`~astrogwb.constants.ISCO_ALPHA` for
        the value corresponding to the Schwarzschild test-particle ISCO.

    Returns
    -------
    jax.Array
        The truncation frequency in Hz.
    """
    total_mass = jnp.asarray(mass_1) + mass_2
    alpha_value = _require_scalar_alpha(alpha)
    return alpha_value / ((1.0 + redshift) * total_mass * SOLAR_MASS_IN_SECONDS)


def _require_frequency_grid(frequencies: JaxArrayLike) -> jax.Array:
    frequency_grid = jnp.asarray(frequencies)
    if frequency_grid.ndim != 1:
        msg = (
            "frequencies must have shape (F,); "
            f"received an array with shape {frequency_grid.shape}"
        )
        raise ValueError(msg)
    return frequency_grid


def _normalize_source_parameter(name: str, value: JaxArrayLike) -> jax.Array:
    parameter = jnp.asarray(value)
    if parameter.ndim > 1:
        msg = (
            f"{name} must be a scalar or have shape (N,); "
            f"received an array with shape {parameter.shape}"
        )
        raise ValueError(msg)
    if parameter.ndim == 0:
        return jnp.reshape(parameter, (1,))
    return parameter


def _normalize_source_parameters(
    *,
    source_frame_mass_1: JaxArrayLike,
    source_frame_mass_2: JaxArrayLike,
    redshift: JaxArrayLike,
    luminosity_distance: JaxArrayLike,
    inclination: JaxArrayLike | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    parameters = (
        _normalize_source_parameter("source_frame_mass_1", source_frame_mass_1),
        _normalize_source_parameter("source_frame_mass_2", source_frame_mass_2),
        _normalize_source_parameter("redshift", redshift),
        _normalize_source_parameter("luminosity_distance", luminosity_distance),
        _normalize_source_parameter(
            "inclination", 0.0 if inclination is None else inclination
        ),
    )
    (
        source_frame_mass_1_array,
        source_frame_mass_2_array,
        redshift_array,
        distance_array,
        iota_array,
    ) = jnp.broadcast_arrays(*parameters)
    return (
        source_frame_mass_1_array,
        source_frame_mass_2_array,
        redshift_array,
        distance_array,
        iota_array,
    )


def _single_source_inspiral_power(
    frequencies: jax.Array,
    source_frame_mass_1: jax.Array,
    source_frame_mass_2: jax.Array,
    redshift: jax.Array,
    luminosity_distance: jax.Array,
    inclination: jax.Array,
    alpha: jax.Array,
) -> jax.Array:
    detector_frame_chirp_mass_seconds = (
        chirp_mass(source_frame_mass_1, source_frame_mass_2)
        * (1.0 + redshift)
        * SOLAR_MASS_IN_SECONDS
    )
    distance_seconds = luminosity_distance * MPC_IN_SECONDS
    amplitude_squared = (
        _AMPLITUDE_PREFACTOR
        * detector_frame_chirp_mass_seconds ** (5.0 / 3.0)
        / distance_seconds**2
        * inclination_factor(inclination)
    )
    cutoff = termination_frequency(
        source_frame_mass_1, source_frame_mass_2, redshift, alpha=alpha
    )

    # f = 0 would make f**(-7/3) infinite, and `where` evaluates both branches:
    # inf * 0 is NaN, and under reverse-mode JAX that NaN propagates into the
    # gradient even though the bin is masked away. Substitute a safe frequency
    # before taking the power, then mask.
    inside_band = (frequencies > 0.0) & (frequencies <= cutoff)
    safe_frequency = jnp.where(frequencies > 0.0, frequencies, 1.0)
    power = amplitude_squared * safe_frequency ** (-7.0 / 3.0)
    return jnp.where(inside_band, power, 0.0)


_batched_inspiral_power = jax.vmap(
    _single_source_inspiral_power,
    in_axes=(None, 0, 0, 0, 0, 0, None),
)


@require_x64
def inspiral_polarization_power(
    frequencies: JaxArrayLike,
    parameters: Mapping[str, JaxArrayLike],
    *,
    alpha: JaxArrayLike,
) -> jax.Array:
    r"""Polarization power $|\tilde h_+|^2 + |\tilde h_\times|^2$ on an ``(N, F)`` grid.

    Evaluates the module's closed form at every ``(source, frequency)`` pair
    and zeroes the bins above each source's :func:`termination_frequency`.
    Required source parameters are read from a gwmock-compatible mapping.
    Each may be a scalar or a one-dimensional ``(N,)`` array; they are jointly
    broadcast to ``(N,)``. Higher-rank source arrays are rejected, and
    unrelated mapping entries are ignored. Output is in $\mathrm{Hz}^{-2}$,
    laid out ``(sample, frequency)``.

    .. warning::

        The truncation is a hard mask, so the power is not differentiable in
        ``alpha``: its gradient with respect to ``alpha`` is zero wherever it
        exists.

    Parameters
    ----------
    frequencies:
        Frequency grid in Hz, shape ``(F,)``. A zero bin returns zero power
        rather than the divergence of $f^{-7/3}$.
    parameters:
        Source population mapping with ``source_frame_mass_1``,
        ``source_frame_mass_2``, ``redshift``, ``luminosity_distance``, and
        entries. ``inclination`` is optional and defaults to face-on. The component masses are in solar masses,
        luminosity distance is in Mpc, and inclination is in radians. Redshift
        enters only by redshifting the masses -- both the chirp mass in the
        amplitude and the total mass in the cutoff.
    alpha:
        Scalar dimensionless truncation parameter shared by every source; see
        :func:`termination_frequency`.

    Returns
    -------
    jax.Array
        Polarization power of shape ``(N, F)`` in $\mathrm{Hz}^{-2}$.
    """
    frequency_grid = _require_frequency_grid(frequencies)
    alpha_value = _require_scalar_alpha(alpha)
    source_parameters = _normalize_source_parameters(
        source_frame_mass_1=parameters["source_frame_mass_1"],
        source_frame_mass_2=parameters["source_frame_mass_2"],
        redshift=parameters["redshift"],
        luminosity_distance=parameters["luminosity_distance"],
        inclination=parameters.get("inclination"),
    )
    return _batched_inspiral_power(frequency_grid, *source_parameters, alpha_value)


@dataclass(frozen=True, slots=True)
class AnalyticInspiralGenerator(PolarizationPowerGenerator):
    """Generate inspiral-only polarization power on the descriptor grid."""

    alpha: float

    @property
    def frequencies(self) -> np.ndarray:
        return uniform_frequency_grid(
            self.minimum_frequency, self.maximum_frequency, self.frequency_resolution
        )

    def generate_batch(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        prepared_parameters = {
            name: jnp.asarray(values) for name, values in source_parameters.items()
        }
        frequencies = self.frequencies
        return inspiral_polarization_power(
            frequencies, prepared_parameters, alpha=self.alpha
        ).T

    def __call__(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        return jnp.asarray(self.frequencies), self.generate_batch(source_parameters)
