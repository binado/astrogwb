r"""Analytic Newtonian-inspiral stochastic-background spectrum.

This module evaluates the one-sided strain spectral density implied by
Eqs. 2, 3, and 9 of Cousins et al., *The Stochastic Siren* (2026), for an
ordered joint source-frame component-mass population ``p(m1, m2)`` with
``m1 >= m2``. Integration is performed in total mass ``M = m1 + m2`` and mass
ratio ``q = m2 / m1``. The mass-ratio integral is evaluated first on a fixed
total-mass grid, then cumulatively integrated in ``M``. Linear interpolation of
that cumulative makes the source-frequency cutoff cheap to evaluate and allows
the mass calculation to be precomputed independently of redshift and cosmology.

All numerical orders and grid sizes are static and JAX-friendly. Double the
relevant order or refine the interpolation grid until the result is stable to
the accuracy required by the application. Realistic strain spectra require JAX
x64 mode because their values underflow in float32.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import cache
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import core
from jax.typing import ArrayLike
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from astrogwb.cosmology import (
    MPC_IN_METERS,
    SPEED_OF_LIGHT,
    hubble_constant_si,
    normalized_hubble_parameter,
)
from astrogwb.utils import SECONDS_PER_YEAR, cumulative_trapezoid, require_x64

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


class PopulationFunction(Protocol):
    """Vectorized one-dimensional population callback.

    Callback arguments are positional, so implementations may name their
    parameters freely.
    """

    def __call__(
        self, redshift: jax.Array, hyperparameters: Mapping[str, ArrayLike], /
    ) -> jax.Array: ...


class JointMassFunction(Protocol):
    """Vectorized ordered joint component-mass probability-density callback.

    Callback arguments are positional, so implementations may name their
    parameters freely.
    """

    def __call__(
        self,
        mass_1: jax.Array,
        mass_2: jax.Array,
        hyperparameters: Mapping[str, ArrayLike],
        /,
    ) -> jax.Array: ...


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


def _validate_cosmology_hyperparameters(
    hyperparameters: Mapping[str, ArrayLike],
) -> None:
    missing = {"H0", "Omega_m"}.difference(hyperparameters)
    if missing:
        names = ", ".join(sorted(missing))
        raise KeyError(f"hyperparameters missing required key(s): {names}")


def _validate_redshift_rule(
    *,
    z_min: float,
    z_max: float,
    quadrature_order: int,
    order_name: str,
) -> None:
    if not math.isfinite(z_min) or not math.isfinite(z_max):
        raise ValueError("redshift bounds must be finite")
    if z_min < 0.0 or z_max <= z_min:
        raise ValueError("redshift bounds must satisfy 0 <= z_min < z_max")

    _validate_positive_integer(quadrature_order, order_name)


def _validate_mass_construction(
    *,
    component_mass_min: float,
    component_mass_max: float,
    mass_ratio_quadrature_order: int,
    n_interp_grid: int,
) -> None:
    if not math.isfinite(component_mass_min) or not math.isfinite(component_mass_max):
        raise ValueError("component-mass bounds must be finite")
    if component_mass_min <= 0.0 or component_mass_max <= component_mass_min:
        raise ValueError(
            "component-mass bounds must satisfy "
            "0 < component_mass_min < component_mass_max"
        )

    _validate_positive_integer(
        mass_ratio_quadrature_order, "mass_ratio_quadrature_order"
    )
    _validate_positive_integer(n_interp_grid, "n_interp_grid")
    if n_interp_grid < 3 or n_interp_grid % 2 == 0:
        raise ValueError("n_interp_grid must be an odd integer >= 3")


def _validate_alpha(alpha: float) -> None:
    if math.isnan(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be positive")


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


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


def _cumulative_mass_moment_grid(
    hyperparameters: Mapping[str, ArrayLike],
    joint_mass_prior_fn: JointMassFunction,
    *,
    component_mass_min: float,
    component_mass_max: float,
    mass_ratio_quadrature_order: int,
    n_interp_grid: int,
) -> tuple[jax.Array, jax.Array]:
    r"""Tabulate the cumulative source-frame chirp-mass moment in total mass.

    The ordered component masses are written in total mass and mass ratio,

    .. math::

        m_1 = \frac{M}{1+q}, \qquad m_2 = \frac{qM}{1+q}.

    Multiplying :math:`\mathcal{M}^{5/3}` by the coordinate Jacobian gives the
    fixed-total-mass integrand

    .. math::

        g(M,q;\theta) = p(m_1,m_2\mid\theta)
            \frac{qM^{8/3}}{(1+q)^4}.

    For component masses in :math:`[m_{\min},m_{\max}]`, the allowed mass-ratio
    interval at fixed ``M`` is :math:`[q_-(M),1]`, where

    .. math::

        q_-(M) =
        \begin{cases}
        m_{\min}/(M-m_{\min}), & 2m_{\min}\le M\le m_{\min}+m_{\max},\\
        M/m_{\max}-1, & m_{\min}+m_{\max}\le M\le2m_{\max}.
        \end{cases}

    The mass-ratio integral is evaluated first,

    .. math::

        H(M;\theta)=\int_{q_-(M)}^1 g(M,q;\theta)\,dq,

    and sampled on a uniform total-mass grid. A cumulative trapezoid gives

    .. math::

        C(u;\theta)=\int_{2m_{\min}}^{
        \operatorname{clip}(u,2m_{\min},2m_{\max})}H(M;\theta)\,dM.

    The odd grid size places the support transition
    :math:`M=m_{\min}+m_{\max}` exactly at the central node.
    """
    total_mass = jnp.linspace(
        2.0 * component_mass_min,
        2.0 * component_mass_max,
        n_interp_grid,
        dtype=jnp.float64,
    )
    transition = component_mass_min + component_mass_max
    mass_ratio_lower = jnp.where(
        total_mass <= transition,
        component_mass_min / (total_mass - component_mass_min),
        total_mass / component_mass_max - 1.0,
    )

    host_nodes, host_weights = _gauss_legendre_rule(mass_ratio_quadrature_order)
    nodes = jnp.asarray(host_nodes, dtype=jnp.float64)
    weights = jnp.asarray(host_weights, dtype=jnp.float64)
    mass_ratio, mass_ratio_weights = _mapped_rule(nodes, weights, mass_ratio_lower, 1.0)

    total_mass_grid = total_mass[..., None]
    one_plus_mass_ratio = 1.0 + mass_ratio
    mass_1 = total_mass_grid / one_plus_mass_ratio
    mass_2 = total_mass_grid * mass_ratio / one_plus_mass_ratio
    joint_mass_density = jnp.broadcast_to(
        jnp.asarray(
            joint_mass_prior_fn(mass_1, mass_2, hyperparameters),
            dtype=jnp.float64,
        ),
        mass_ratio.shape,
    )
    fixed_total_mass_integrand = (
        joint_mass_density
        * mass_ratio
        * total_mass_grid ** (8.0 / 3.0)
        / one_plus_mass_ratio**4
    )
    mass_ratio_integral = jnp.sum(
        mass_ratio_weights * fixed_total_mass_integrand,
        axis=-1,
    )
    return total_mass, cumulative_trapezoid(mass_ratio_integral, total_mass)


@require_x64
def precompute_cumulative_mass_moments(
    frequencies: jax.Array,
    hyperparameters: Mapping[str, ArrayLike],
    joint_mass_prior_fn: JointMassFunction,
    *,
    z_min: float,
    z_max: float,
    component_mass_min: float,
    component_mass_max: float,
    alpha: float = ISCO_ALPHA,
    mass_ratio_quadrature_order: int = 64,
    n_interp_grid: int = 2049,
    redshift_quadrature_order: int = 64,
) -> jax.Array:
    r"""Evaluate fixed mass moments at every frequency-redshift cutoff.

    For redshift quadrature nodes :math:`z_a`, this function constructs

    .. math::

        U_{fa}=\frac{\alpha}{f_f(1+z_a)}, \qquad M_{fa}=C(U_{fa}),

    and linearly interpolates :math:`C(U_{fa})` from a uniform total-mass grid.
    The result has shape ``(frequency, redshift_node)`` and can be reused while
    sampling cosmology or the redshift distribution with a fixed mass
    population. Values below the physical total-mass support are exactly zero;
    values above it are exactly the full mass moment.
    """
    _validate_redshift_rule(
        z_min=z_min,
        z_max=z_max,
        quadrature_order=redshift_quadrature_order,
        order_name="redshift_quadrature_order",
    )
    _validate_mass_construction(
        component_mass_min=component_mass_min,
        component_mass_max=component_mass_max,
        mass_ratio_quadrature_order=mass_ratio_quadrature_order,
        n_interp_grid=n_interp_grid,
    )
    _validate_alpha(alpha)

    frequencies = jnp.asarray(frequencies, dtype=jnp.float64)
    _validate_concrete_frequencies(frequencies)
    host_nodes, host_weights = _gauss_legendre_rule(redshift_quadrature_order)
    nodes = jnp.asarray(host_nodes, dtype=jnp.float64)
    weights = jnp.asarray(host_weights, dtype=jnp.float64)
    redshift, _ = _mapped_rule(nodes, weights, z_min, z_max)
    total_mass_upper = alpha / (frequencies[:, None] * (1.0 + redshift[None, :]))
    total_mass, cumulative_mass_moment = _cumulative_mass_moment_grid(
        hyperparameters,
        joint_mass_prior_fn,
        component_mass_min=component_mass_min,
        component_mass_max=component_mass_max,
        mass_ratio_quadrature_order=mass_ratio_quadrature_order,
        n_interp_grid=n_interp_grid,
    )
    return jnp.interp(total_mass_upper, total_mass, cumulative_mass_moment)


@require_x64
def analytic_spectral_density_from_mass_moments(
    frequencies: jax.Array,
    hyperparameters: Mapping[str, ArrayLike],
    merger_rate_fn: PopulationFunction,
    cumulative_mass_moments: jax.Array,
    *,
    z_min: float,
    z_max: float,
    redshift_quadrature_order: int = 64,
) -> jax.Array:
    r"""Evaluate the strain PSD from fixed cumulative mass moments.

    ``cumulative_mass_moments[f, a]`` must have been prepared at the same
    frequency ``f`` and redshift quadrature node ``z_a`` used here. The spectrum
    is the matrix-vector contraction

    .. math::

        S_h(f_f;\theta)=\frac{C_{\rm astro}}{H_0} f_f^{-7/3}
        \sum_a w_a\frac{R(z_a;\theta)}
        {E(z_a;\Omega_m)(1+z_a)^{4/3}}M_{fa}.

    This stage contains no mass-population evaluation or cumulative lookup, so
    a fixed ``(F, Z)`` array can be reused throughout an MCMC over cosmology or
    the redshift distribution.
    """
    _validate_cosmology_hyperparameters(hyperparameters)
    _validate_redshift_rule(
        z_min=z_min,
        z_max=z_max,
        quadrature_order=redshift_quadrature_order,
        order_name="redshift_quadrature_order",
    )

    frequencies = jnp.asarray(frequencies, dtype=jnp.float64)
    _validate_concrete_frequencies(frequencies)
    cumulative_mass_moments = jnp.asarray(cumulative_mass_moments, dtype=jnp.float64)
    expected_shape = (frequencies.shape[0], redshift_quadrature_order)
    if cumulative_mass_moments.shape != expected_shape:
        raise ValueError(
            "cumulative_mass_moments must have shape "
            f"{expected_shape}, got {cumulative_mass_moments.shape}"
        )

    host_nodes, host_weights = _gauss_legendre_rule(redshift_quadrature_order)
    nodes = jnp.asarray(host_nodes, dtype=jnp.float64)
    weights = jnp.asarray(host_weights, dtype=jnp.float64)
    redshift, redshift_weights = _mapped_rule(nodes, weights, z_min, z_max)
    merger_rate = jnp.broadcast_to(
        jnp.asarray(merger_rate_fn(redshift, hyperparameters), dtype=jnp.float64),
        redshift.shape,
    )
    expansion = normalized_hubble_parameter(
        redshift, jnp.asarray(hyperparameters["Omega_m"], dtype=jnp.float64)
    )
    redshift_integrand = merger_rate / (expansion * (1.0 + redshift) ** (4.0 / 3.0))
    redshift_mass_moments = cumulative_mass_moments @ (
        redshift_weights * redshift_integrand
    )

    coefficient = _ASTROPHYSICAL_STRAIN_COEFFICIENT / hubble_constant_si(
        jnp.asarray(hyperparameters["H0"], dtype=jnp.float64)
    )
    return coefficient * frequencies ** (-7.0 / 3.0) * redshift_mass_moments


@require_x64
def analytic_spectral_density(
    frequencies: jax.Array,
    hyperparameters: Mapping[str, ArrayLike],
    merger_rate_fn: PopulationFunction,
    joint_mass_prior_fn: JointMassFunction,
    *,
    z_min: float,
    z_max: float,
    component_mass_min: float,
    component_mass_max: float,
    alpha: float = ISCO_ALPHA,
    quadrature_order: int = 64,
    n_interp_grid: int = 2049,
) -> jax.Array:
    r"""Evaluate the analytic inspiral-only one-sided strain PSD ``S_h(f)``.

    This convenience API performs both stages in one call: it tabulates the
    cumulative mass moment, interpolates it at the frequency-redshift cutoffs,
    and contracts those fixed mass moments with the redshift-dependent merger
    rate and cosmology. When the mass population is fixed during MCMC, use
    :func:`precompute_cumulative_mass_moments` and
    :func:`analytic_spectral_density_from_mass_moments` separately instead.

    Parameters
    ----------
    frequencies:
        One-dimensional observer-frame frequency array in Hz. Values must be
        finite and strictly positive.
    hyperparameters:
        Population/cosmology parameters supplied unchanged to every callback.
        ``H0`` (km s^-1 Mpc^-1) and ``Omega_m`` are required.
    merger_rate_fn:
        ``fn(redshift, hyperparameters)`` returning the absolute source-frame
        merger-rate density in Gpc^-3 yr^-1.
    joint_mass_prior_fn:
        ``fn(mass_1, mass_2, hyperparameters)`` returning a normalized ordered
        density with respect to ``dmass_1 dmass_2``.
    z_min, z_max:
        Redshift integration bounds.
    component_mass_min, component_mass_max:
        Shared source-frame component-mass bounds in solar masses.
    alpha:
        Source-frame cutoff coefficient in Hz solar-mass, defining
        ``f_max = alpha / M``. Use ``inf`` for no cutoff.
    quadrature_order:
        Shared Gauss-Legendre order for redshift and mass ratio.
    n_interp_grid:
        Odd number of uniform total-mass interpolation nodes. The default gives
        2048 intervals; use 1025, 2049, and 4097 for grid-refinement checks.

    Returns
    -------
    jax.Array
        One-sided strain spectral density in Hz^-1, with the same shape as
        ``frequencies``.

    Notes
    -----
    Direct calls validate frequency values. Under :func:`jax.jit`, frequency
    values are traced and therefore must be validated by the caller before the
    compiled invocation; their rank and non-empty shape are still checked.
    """
    _validate_cosmology_hyperparameters(hyperparameters)
    _validate_redshift_rule(
        z_min=z_min,
        z_max=z_max,
        quadrature_order=quadrature_order,
        order_name="quadrature_order",
    )
    _validate_mass_construction(
        component_mass_min=component_mass_min,
        component_mass_max=component_mass_max,
        mass_ratio_quadrature_order=quadrature_order,
        n_interp_grid=n_interp_grid,
    )
    _validate_alpha(alpha)

    frequencies = jnp.asarray(frequencies, dtype=jnp.float64)
    _validate_concrete_frequencies(frequencies)
    cumulative_mass_moments = precompute_cumulative_mass_moments(
        frequencies,
        hyperparameters,
        joint_mass_prior_fn,
        z_min=z_min,
        z_max=z_max,
        component_mass_min=component_mass_min,
        component_mass_max=component_mass_max,
        alpha=alpha,
        mass_ratio_quadrature_order=quadrature_order,
        n_interp_grid=n_interp_grid,
        redshift_quadrature_order=quadrature_order,
    )
    return analytic_spectral_density_from_mass_moments(
        frequencies,
        hyperparameters,
        merger_rate_fn,
        cumulative_mass_moments,
        z_min=z_min,
        z_max=z_max,
        redshift_quadrature_order=quadrature_order,
    )
