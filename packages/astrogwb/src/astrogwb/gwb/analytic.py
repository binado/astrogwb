r"""Analytic Newtonian-inspiral stochastic-background spectrum.

This module evaluates the one-sided strain spectral density implied by
Eqs. 2, 3, and 9 of Cousins et al., *The Stochastic Siren* (2026), for an
ordered joint source-frame component-mass population ``p(m1, m2)`` with
``m1 >= m2``. Integration is performed in total mass ``M = m1 + m2`` and mass
ratio ``q = m2 / m1``. The mass-ratio integral is evaluated first at fixed
total mass, then represented by a piecewise Legendre cumulative in ``M``. This
makes the source-frequency cutoff a cheap cumulative query and allows the mass
calculation to be precomputed independently of redshift and cosmology.

All numerical orders are static and JAX-friendly. Double the relevant order or
total-mass interval count until the result is stable to the accuracy required
by the application. Realistic strain spectra require JAX x64 mode because
their values underflow in float32.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import core
from numpy.polynomial.legendre import leggauss, legvander
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
"""Vectorized one-dimensional population callback."""

JointMassFunction = Callable[[jax.Array, jax.Array, Mapping[str, Any]], jax.Array]
"""Vectorized ordered joint component-mass probability-density callback."""

CumulativeMassMomentFunction = Callable[[jax.Array], jax.Array]
"""Vectorized cumulative source-frame chirp-mass-moment callback."""


@dataclass(frozen=True, slots=True, eq=False)
class _PiecewiseLegendreScheme:
    """Static host-side geometry for a piecewise Legendre cumulative."""

    edges: NDArray[np.float64]
    centers: NDArray[np.float64]
    half_widths: NDArray[np.float64]
    nodes: NDArray[np.float64]
    coefficient_transform: NDArray[np.float64]
    order: int


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


@cache
def _piecewise_legendre_scheme(
    component_mass_min: float,
    component_mass_max: float,
    interval_count: int,
    order: int,
) -> _PiecewiseLegendreScheme:
    """Build cached float64 geometry for uniform total-mass intervals."""
    total_mass_min = 2.0 * component_mass_min
    total_mass_max = 2.0 * component_mass_max
    transition = component_mass_min + component_mass_max

    edges = np.linspace(total_mass_min, total_mass_max, interval_count + 1)
    edges[interval_count // 2] = transition
    centers = 0.5 * (edges[:-1] + edges[1:])
    half_widths = 0.5 * (edges[1:] - edges[:-1])

    reference_nodes, reference_weights = _gauss_legendre_rule(order)
    nodes = centers[:, None] + half_widths[:, None] * reference_nodes
    basis = legvander(reference_nodes, order - 1).T
    degrees = np.arange(order, dtype=np.float64)
    coefficient_transform = (
        0.5 * (2.0 * degrees[:, None] + 1.0) * basis * reference_weights[None, :]
    )
    return _PiecewiseLegendreScheme(
        edges=edges.astype(np.float64),
        centers=centers.astype(np.float64),
        half_widths=half_widths.astype(np.float64),
        nodes=nodes.astype(np.float64),
        coefficient_transform=coefficient_transform.astype(np.float64),
        order=order,
    )


def _piecewise_legendre_coefficients(
    values: jax.Array,
    scheme: _PiecewiseLegendreScheme,
) -> tuple[jax.Array, jax.Array]:
    """Return interval coefficients and their exclusive cumulative totals."""
    transform = jnp.asarray(scheme.coefficient_transform, dtype=values.dtype)
    coefficients = jnp.einsum("ji,ki->kj", transform, values)
    half_widths = jnp.asarray(scheme.half_widths, dtype=values.dtype)
    interval_integrals = 2.0 * half_widths * coefficients[:, 0]
    cumulative = jnp.concatenate(
        [jnp.zeros((1,), dtype=values.dtype), jnp.cumsum(interval_integrals)]
    )
    return coefficients, cumulative


def _evaluate_piecewise_cumulative(
    coefficients: jax.Array,
    cumulative: jax.Array,
    scheme: _PiecewiseLegendreScheme,
    upper: jax.Array,
) -> jax.Array:
    r"""Evaluate the piecewise cumulative without a query-by-order temporary.

    Within interval ``k``, the represented integrand is

    .. math::

        H(c_k + h_k x) \approx \sum_{j=0}^{r-1} a_{kj} P_j(x).

    The cumulative is the sum of complete earlier intervals plus

    .. math::

        h_k \sum_j a_{kj} I_j(x), \qquad
        I_0(x)=x+1, \quad
        I_j(x)=\frac{P_{j+1}(x)-P_{j-1}(x)}{2j+1}.

    A recurrence gathers one coefficient degree at a time, keeping live query
    storage independent of the Legendre order.
    """
    upper = jnp.asarray(upper, dtype=coefficients.dtype)
    edges = jnp.asarray(scheme.edges, dtype=coefficients.dtype)
    centers = jnp.asarray(scheme.centers, dtype=coefficients.dtype)
    half_widths = jnp.asarray(scheme.half_widths, dtype=coefficients.dtype)

    clipped = jnp.clip(upper, edges[0], edges[-1])
    interval = jnp.searchsorted(edges, clipped, side="right") - 1
    interval = jnp.clip(interval, 0, coefficients.shape[0] - 1)
    x = (clipped - centers[interval]) / half_widths[interval]

    initial_integral = coefficients[interval, 0] * (x + 1.0)

    def add_degree(
        degree: int,
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        polynomial_previous, polynomial, integral = state
        degree_value = jnp.asarray(degree, dtype=coefficients.dtype)
        polynomial_next = (
            (2.0 * degree_value + 1.0) * x * polynomial
            - degree_value * polynomial_previous
        ) / (degree_value + 1.0)
        antiderivative = (polynomial_next - polynomial_previous) / (
            2.0 * degree_value + 1.0
        )
        integral = integral + coefficients[interval, degree] * antiderivative
        return polynomial, polynomial_next, integral

    _, _, partial_reference_integral = jax.lax.fori_loop(
        1,
        scheme.order,
        add_degree,
        (jnp.ones_like(x), x, initial_integral),
    )
    result = cumulative[interval] + half_widths[interval] * partial_reference_integral
    return jnp.where(
        upper <= edges[0],
        jnp.zeros_like(result),
        jnp.where(upper >= edges[-1], cumulative[-1], result),
    )


def _require_x64(function_name: str) -> None:
    if not jax.config.x64_enabled:
        raise RuntimeError(
            f"{function_name} requires JAX x64 mode because realistic strain "
            "spectral densities underflow in float32; call "
            "jax.config.update('jax_enable_x64', True) before creating arrays"
        )


def _validate_cosmology_hyperparameters(hyperparameters: Mapping[str, Any]) -> None:
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
    total_mass_interval_count: int,
    total_mass_legendre_order: int,
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
    _validate_positive_integer(total_mass_interval_count, "total_mass_interval_count")
    if total_mass_interval_count < 2 or total_mass_interval_count % 2 != 0:
        raise ValueError("total_mass_interval_count must be an even integer >= 2")
    _validate_positive_integer(total_mass_legendre_order, "total_mass_legendre_order")


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


def make_cumulative_mass_moment_fn(
    hyperparameters: Mapping[str, Any],
    joint_mass_prior_fn: JointMassFunction,
    *,
    component_mass_min: float,
    component_mass_max: float,
    mass_ratio_quadrature_order: int = 64,
    total_mass_interval_count: int = 32,
    total_mass_legendre_order: int = 16,
) -> CumulativeMassMomentFunction:
    r"""Build a cumulative source-frame chirp-mass-moment function.

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

    and represented on uniform total-mass intervals in a Legendre basis. The
    returned function evaluates the cumulative

    .. math::

        C(u;\theta)=\int_{2m_{\min}}^{
        \operatorname{clip}(u,2m_{\min},2m_{\max})}H(M;\theta)\,dM.

    The closed-form antiderivatives integrate each interval's polynomial
    exactly; approximating ``H`` by that polynomial is the numerical error.
    When the mass population is fixed during MCMC, construct this function once
    outside the model and reuse the same function object.

    Parameters
    ----------
    hyperparameters:
        Parameters supplied unchanged to ``joint_mass_prior_fn``. Cosmological
        parameters are not required unless the mass callback itself uses them.
    joint_mass_prior_fn:
        Normalized ordered joint density with respect to ``dmass_1 dmass_2``.
    component_mass_min, component_mass_max:
        Shared source-frame component-mass bounds in solar masses.
    mass_ratio_quadrature_order:
        Gauss-Legendre order used for the inner mass-ratio integral.
    total_mass_interval_count:
        Even number of uniform total-mass intervals. The central edge is fixed
        at ``component_mass_min + component_mass_max``.
    total_mass_legendre_order:
        Number of nodes and Legendre coefficients within each total-mass
        interval.
    """
    _require_x64("make_cumulative_mass_moment_fn")
    _validate_mass_construction(
        component_mass_min=component_mass_min,
        component_mass_max=component_mass_max,
        mass_ratio_quadrature_order=mass_ratio_quadrature_order,
        total_mass_interval_count=total_mass_interval_count,
        total_mass_legendre_order=total_mass_legendre_order,
    )

    scheme = _piecewise_legendre_scheme(
        component_mass_min,
        component_mass_max,
        total_mass_interval_count,
        total_mass_legendre_order,
    )
    total_mass = jnp.asarray(scheme.nodes, dtype=jnp.float64)
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
    coefficients, cumulative = _piecewise_legendre_coefficients(
        mass_ratio_integral, scheme
    )

    def cumulative_mass_moment(total_mass_upper: jax.Array) -> jax.Array:
        return _evaluate_piecewise_cumulative(
            coefficients,
            cumulative,
            scheme,
            total_mass_upper,
        )

    return cumulative_mass_moment


def precompute_cumulative_mass_moments(
    frequencies: jax.Array,
    cumulative_mass_moment_fn: CumulativeMassMomentFunction,
    *,
    z_min: float,
    z_max: float,
    alpha: float = ISCO_ALPHA,
    redshift_quadrature_order: int = 64,
) -> jax.Array:
    r"""Evaluate fixed mass moments at every frequency-redshift cutoff.

    For redshift quadrature nodes :math:`z_a`, this function constructs

    .. math::

        U_{fa}=\frac{\alpha}{f_f(1+z_a)}, \qquad M_{fa}=C(U_{fa}),

    and returns ``M`` with shape ``(frequency, redshift_node)``. These are the
    exact queries required by the later redshift integral, so no interpolation
    grid is introduced. Frequencies, redshift bounds, quadrature order, and
    ``alpha`` are consequently encoded in the returned values by convention.
    """
    _require_x64("precompute_cumulative_mass_moments")
    _validate_redshift_rule(
        z_min=z_min,
        z_max=z_max,
        quadrature_order=redshift_quadrature_order,
        order_name="redshift_quadrature_order",
    )
    _validate_alpha(alpha)

    frequencies = jnp.asarray(frequencies, dtype=jnp.float64)
    _validate_concrete_frequencies(frequencies)
    host_nodes, host_weights = _gauss_legendre_rule(redshift_quadrature_order)
    nodes = jnp.asarray(host_nodes, dtype=jnp.float64)
    weights = jnp.asarray(host_weights, dtype=jnp.float64)
    redshift, _ = _mapped_rule(nodes, weights, z_min, z_max)
    total_mass_upper = alpha / (frequencies[:, None] * (1.0 + redshift[None, :]))
    return jnp.broadcast_to(
        jnp.asarray(
            cumulative_mass_moment_fn(total_mass_upper),
            dtype=jnp.float64,
        ),
        total_mass_upper.shape,
    )


def analytic_spectral_density_from_mass_moments(
    frequencies: jax.Array,
    hyperparameters: Mapping[str, Any],
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
    _require_x64("analytic_spectral_density_from_mass_moments")
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
    expansion = normalized_hubble_parameter(redshift, hyperparameters["Omega_m"])
    redshift_integrand = merger_rate / (expansion * (1.0 + redshift) ** (4.0 / 3.0))
    redshift_mass_moments = cumulative_mass_moments @ (
        redshift_weights * redshift_integrand
    )

    coefficient = _ASTROPHYSICAL_STRAIN_COEFFICIENT / hubble_constant_si(
        hyperparameters["H0"]
    )
    return coefficient * frequencies ** (-7.0 / 3.0) * redshift_mass_moments


def analytic_spectral_density(
    frequencies: jax.Array,
    hyperparameters: Mapping[str, Any],
    merger_rate_fn: PopulationFunction,
    joint_mass_prior_fn: JointMassFunction,
    *,
    z_min: float,
    z_max: float,
    component_mass_min: float,
    component_mass_max: float,
    alpha: float = ISCO_ALPHA,
    quadrature_order: int = 64,
    total_mass_interval_count: int = 32,
    total_mass_legendre_order: int = 16,
) -> jax.Array:
    r"""Evaluate the analytic inspiral-only one-sided strain PSD ``S_h(f)``.

    This convenience API performs all three stages in one call: it builds the
    piecewise Legendre mass cumulative, evaluates it at the exact
    frequency-redshift cutoffs, and contracts those fixed mass moments with the
    redshift-dependent merger rate and cosmology. When the mass population is
    fixed during MCMC, use :func:`make_cumulative_mass_moment_fn`,
    :func:`precompute_cumulative_mass_moments`, and
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
    total_mass_interval_count:
        Even number of uniform intervals in the piecewise mass cumulative.
    total_mass_legendre_order:
        Legendre order within each total-mass interval.

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
    _require_x64("analytic_spectral_density")
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
        total_mass_interval_count=total_mass_interval_count,
        total_mass_legendre_order=total_mass_legendre_order,
    )
    _validate_alpha(alpha)

    frequencies = jnp.asarray(frequencies, dtype=jnp.float64)
    _validate_concrete_frequencies(frequencies)
    cumulative_mass_moment_fn = make_cumulative_mass_moment_fn(
        hyperparameters,
        joint_mass_prior_fn,
        component_mass_min=component_mass_min,
        component_mass_max=component_mass_max,
        mass_ratio_quadrature_order=quadrature_order,
        total_mass_interval_count=total_mass_interval_count,
        total_mass_legendre_order=total_mass_legendre_order,
    )
    cumulative_mass_moments = precompute_cumulative_mass_moments(
        frequencies,
        cumulative_mass_moment_fn,
        z_min=z_min,
        z_max=z_max,
        alpha=alpha,
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
