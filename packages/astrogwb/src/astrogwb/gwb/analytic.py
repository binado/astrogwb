r"""Analytic Newtonian-inspiral stochastic-background spectrum.

This module evaluates the one-sided strain spectral density implied by
Eqs. 2, 3, and 9 of Cousins et al., *The Stochastic Siren* (2026), for a
population separable in source-frame total mass ``M`` and ordered mass ratio
``q = m2 / m1 <= 1``. The mass-ratio moment factors out, while the upper total
mass depends on source-frame frequency through ``f_s = f * (1 + z)``.

All integrations use a fixed Gauss-Legendre rule. ``quadrature_order`` is
therefore static and JAX-friendly; double it until the result is stable to the
accuracy required by the application. Realistic strain spectra require JAX
x64 mode because their values underflow in float32.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import core
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from astrogwb.cosmology import (
    MPC_IN_METERS,
    SPEED_OF_LIGHT,
    hubble_constant_si,
    normalized_hubble_parameter,
)
from astrogwb.utils import SECONDS_PER_YEAR

_GRAVITATIONAL_CONSTANT_SI: float = 6.67430e-11
_SOLAR_MASS_KG: float = 1.988409870698051e30
_GPC_IN_METERS: float = 1.0e3 * MPC_IN_METERS

# Converts a rate density in Gpc^-3 yr^-1 and a total-mass moment in
# solar-mass^(5/3) into the SI coefficient of S_h. Combining these scales
# before entering JAX avoids separately materializing ~1e50 and ~1e-84 terms.
_ASTROPHYSICAL_STRAIN_COEFFICIENT: float = (
    2.0
    * (_GRAVITATIONAL_CONSTANT_SI * _SOLAR_MASS_KG) ** (5.0 / 3.0)
    / (
        3.0
        * math.pi ** (1.0 / 3.0)
        * SPEED_OF_LIGHT**2
        * _GPC_IN_METERS**3
        * SECONDS_PER_YEAR
    )
)

ISCO_ALPHA: float = SPEED_OF_LIGHT**3 / (
    6.0 ** (3.0 / 2.0) * math.pi * _GRAVITATIONAL_CONSTANT_SI * _SOLAR_MASS_KG
)
"""Schwarzschild-ISCO cutoff coefficient in Hz solar-mass."""

PopulationFunction = Callable[[jax.Array, Mapping[str, Any]], jax.Array]
"""Vectorized population rate or probability-density callback."""


@cache
def _gauss_legendre_rule(
    order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return cached float64 nodes and weights on ``[-1, 1]``."""
    nodes, weights = leggauss(order)
    return nodes.astype(np.float64), weights.astype(np.float64)


def _mapped_rule(
    nodes: jax.Array,
    weights: jax.Array,
    lower: float | jax.Array,
    upper: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Map a Gauss-Legendre rule onto one or many intervals."""
    lower = jnp.asarray(lower, dtype=nodes.dtype)
    upper = jnp.asarray(upper, dtype=nodes.dtype)
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    return (
        midpoint[..., None] + half_width[..., None] * nodes,
        half_width[..., None] * weights,
    )


def _validate_static_inputs(
    hyperparameters: Mapping[str, Any],
    *,
    z_min: float,
    z_max: float,
    total_mass_min: float,
    total_mass_max: float,
    q_min: float,
    alpha: float,
    quadrature_order: int,
) -> None:
    missing = {"H0", "Omega_m"}.difference(hyperparameters)
    if missing:
        names = ", ".join(sorted(missing))
        raise KeyError(f"hyperparameters missing required key(s): {names}")

    if not math.isfinite(z_min) or not math.isfinite(z_max):
        raise ValueError("redshift bounds must be finite")
    if z_min < 0.0 or z_max <= z_min:
        raise ValueError("redshift bounds must satisfy 0 <= z_min < z_max")

    if not math.isfinite(total_mass_min) or not math.isfinite(total_mass_max):
        raise ValueError("total-mass bounds must be finite")
    if total_mass_min <= 0.0 or total_mass_max <= total_mass_min:
        raise ValueError(
            "total-mass bounds must satisfy 0 < total_mass_min < total_mass_max"
        )

    if not math.isfinite(q_min) or not 0.0 <= q_min < 1.0:
        raise ValueError("q_min must satisfy 0 <= q_min < 1")
    if math.isnan(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if (
        isinstance(quadrature_order, bool)
        or not isinstance(quadrature_order, int)
        or quadrature_order <= 0
    ):
        raise ValueError("quadrature_order must be a positive integer")


def _validate_concrete_frequencies(frequencies: jax.Array) -> None:
    """Validate values outside tracing; traced callers retain shape checks."""
    if frequencies.ndim != 1:
        raise ValueError("frequencies must be a one-dimensional array")
    if frequencies.shape[0] == 0:
        raise ValueError("frequencies must contain at least one value")
    if isinstance(frequencies, core.Tracer):
        return
    values = np.asarray(frequencies)
    if not np.all(np.isfinite(values)) or not np.all(values > 0.0):
        raise ValueError("frequencies must be finite and strictly positive")


def analytic_spectral_density(
    frequencies: jax.Array,
    hyperparameters: Mapping[str, Any],
    merger_rate_fn: PopulationFunction,
    total_mass_prior_fn: PopulationFunction,
    mass_ratio_prior_fn: PopulationFunction,
    *,
    z_min: float,
    z_max: float,
    total_mass_min: float,
    total_mass_max: float,
    q_min: float = 0.0,
    alpha: float = ISCO_ALPHA,
    quadrature_order: int = 64,
) -> jax.Array:
    r"""Evaluate the analytic inspiral-only one-sided strain PSD ``S_h(f)``.

    Parameters
    ----------
    frequencies:
        One-dimensional observer-frame frequency array in Hz. Values must be
        finite and strictly positive.
    hyperparameters:
        Population/cosmology parameters supplied unchanged to every callback.
        ``H0`` (km s^-1 Mpc^-1) and ``Omega_m`` are required for the
        flat-LCDM expansion history.
    merger_rate_fn:
        ``fn(redshift, hyperparameters)`` returning the absolute source-frame
        merger-rate density in Gpc^-3 yr^-1.
    total_mass_prior_fn:
        ``fn(total_mass, hyperparameters)`` returning a normalized density in
        inverse solar masses. Its support must lie within the mass bounds.
    mass_ratio_prior_fn:
        ``fn(q, hyperparameters)`` returning a normalized density with respect
        to ``dq``. Its support must lie within ``[q_min, 1]``.
    z_min, z_max:
        Redshift integration bounds.
    total_mass_min, total_mass_max:
        Source-frame total-mass bounds in solar masses.
    q_min:
        Lower ordered mass-ratio bound; the upper bound is one.
    alpha:
        Source-frame cutoff coefficient in Hz solar-mass, defining
        ``f_max = alpha / M``. Defaults to :data:`ISCO_ALPHA`; use ``inf`` for
        no cutoff.
    quadrature_order:
        Shared Gauss-Legendre order for the redshift, mass, and mass-ratio
        integrations. This is a static configuration value. For convergence
        testing, compare against a run with twice the order.

    Returns
    -------
    jax.Array
        One-sided strain spectral density in Hz^-1, with the same shape as
        ``frequencies``.

    Notes
    -----
    The priors are densities in ``dM dq``. No ``(m1, m2) -> (M, q)`` Jacobian
    is applied: including ``M / (1 + q)^2`` would double-count a transformation
    already encoded by this public density contract.

    Direct calls validate frequency values. Under :func:`jax.jit`, frequency
    values are traced and therefore must be validated by the caller before the
    compiled invocation; their rank and non-empty shape are still checked.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError(
            "analytic_spectral_density requires JAX x64 mode because realistic "
            "strain spectral densities underflow in float32; call "
            "jax.config.update('jax_enable_x64', True) before creating arrays"
        )

    _validate_static_inputs(
        hyperparameters,
        z_min=z_min,
        z_max=z_max,
        total_mass_min=total_mass_min,
        total_mass_max=total_mass_max,
        q_min=q_min,
        alpha=alpha,
        quadrature_order=quadrature_order,
    )

    frequencies = jnp.asarray(frequencies, dtype=jnp.float64)
    _validate_concrete_frequencies(frequencies)

    host_nodes, host_weights = _gauss_legendre_rule(quadrature_order)
    nodes = jnp.asarray(host_nodes, dtype=jnp.float64)
    weights = jnp.asarray(host_weights, dtype=jnp.float64)

    redshift, redshift_weights = _mapped_rule(nodes, weights, z_min, z_max)
    mass_ratio, mass_ratio_weights = _mapped_rule(nodes, weights, q_min, 1.0)

    mass_ratio_density = jnp.broadcast_to(
        jnp.asarray(
            mass_ratio_prior_fn(mass_ratio, hyperparameters), dtype=jnp.float64
        ),
        mass_ratio.shape,
    )
    mass_ratio_moment = jnp.sum(
        mass_ratio_weights * mass_ratio_density * mass_ratio / (1.0 + mass_ratio) ** 2
    )

    # The cutoff is a bound, not a mask over a fixed mass grid. Inactive
    # intervals are evaluated on the ordinary full support (which keeps prior
    # callbacks away from a potentially singular boundary) and selected to
    # zero afterwards.
    source_frequency = frequencies[:, None] * (1.0 + redshift[None, :])
    mass_upper = jnp.minimum(total_mass_max, alpha / source_frequency)
    mass_interval_active = mass_upper > total_mass_min
    safe_mass_upper = jnp.where(mass_interval_active, mass_upper, total_mass_max)
    total_mass, total_mass_weights = _mapped_rule(
        nodes,
        weights,
        total_mass_min,
        safe_mass_upper,
    )
    total_mass_density = jnp.broadcast_to(
        jnp.asarray(
            total_mass_prior_fn(total_mass, hyperparameters), dtype=jnp.float64
        ),
        total_mass.shape,
    )
    total_mass_moment = jnp.sum(
        total_mass_weights * total_mass_density * total_mass ** (5.0 / 3.0),
        axis=-1,
    )
    total_mass_moment = jnp.where(mass_interval_active, total_mass_moment, 0.0)

    merger_rate = jnp.broadcast_to(
        jnp.asarray(merger_rate_fn(redshift, hyperparameters), dtype=jnp.float64),
        redshift.shape,
    )
    expansion = normalized_hubble_parameter(redshift, hyperparameters["Omega_m"])
    redshift_integrand = merger_rate / (expansion * (1.0 + redshift) ** (4.0 / 3.0))
    redshift_mass_moment = jnp.sum(
        redshift_weights[None, :] * redshift_integrand[None, :] * total_mass_moment,
        axis=-1,
    )

    coefficient = _ASTROPHYSICAL_STRAIN_COEFFICIENT / hubble_constant_si(
        hyperparameters["H0"]
    )
    return (
        coefficient
        * frequencies ** (-7.0 / 3.0)
        * mass_ratio_moment
        * redshift_mass_moment
    )
